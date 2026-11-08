import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export noisy reads, BCJRFormer parameters, and model outputs "
            "as three ordered hardware-oriented CSV files."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--code-config", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_noisy_reads(trace_path: Path, destination: Path) -> int:
    with np.load(trace_path, allow_pickle=False) as trace:
        required = {"raw_received_reads", "raw_read_lengths"}
        if not required.issubset(trace.files):
            raise ValueError(
                "Trace does not contain raw channel reads; regenerate it with "
                "extract_attention_trace.py"
            )
        reads = trace["raw_received_reads"]
        lengths = trace["raw_read_lengths"]
        if reads.ndim != 3 or lengths.shape != reads.shape[:2]:
            raise ValueError("Raw read and length arrays have incompatible shapes")

        row_count = 0
        with destination.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                ["sample", "read_index", "read_length", "noisy_received_bits"]
            )
            for sample_index in range(reads.shape[0]):
                for read_index in range(reads.shape[1]):
                    read_length = int(lengths[sample_index, read_index])
                    if read_length == 0:
                        continue
                    if not 0 < read_length <= reads.shape[2]:
                        raise ValueError(
                            f"Invalid read length for sample {sample_index}, "
                            f"read {read_index}"
                        )
                    symbols = reads[sample_index, read_index, :read_length]
                    if np.any((symbols != 0) & (symbols != 1)):
                        raise ValueError(
                            f"Expected a 0/1 channel read at sample {sample_index}, "
                            f"read {read_index}"
                        )
                    writer.writerow(
                        [
                            sample_index,
                            read_index,
                            read_length,
                            "".join(str(int(symbol)) for symbol in symbols),
                        ]
                    )
                    row_count += 1
    return row_count


def write_model_inputs_and_mask(trace_path: Path, destination: Path) -> int:
    with np.load(trace_path, allow_pickle=False) as trace:
        required = {"model_input", "padding_mask"}
        if not required.issubset(trace.files):
            raise ValueError(f"Trace missing arrays: {required - set(trace.files)}")
        model_input = trace["model_input"]
        padding_mask = trace["padding_mask"]
        if model_input.ndim != 5 or padding_mask.shape != model_input.shape[:3]:
            raise ValueError(
                "Expected model_input [batch, sequence, token, window, feature] "
                "and matching padding_mask [batch, sequence, token]"
            )
        batch, sequences, tokens = model_input.shape[:3]
        model_input = model_input.reshape(batch, sequences * tokens, -1)
        padding_mask = padding_mask.reshape(batch, sequences * tokens)
        feature_count = model_input.shape[-1]

        with destination.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "sample",
                    "token_index",
                    "sequence_index",
                    "code_position",
                    "padding_mask",
                    *[f"window_feature_{i:03d}" for i in range(feature_count)],
                ]
            )
            for sample_index in range(batch):
                for token_index in range(sequences * tokens):
                    sequence_index, code_position = divmod(token_index, tokens)
                    writer.writerow(
                        [
                            sample_index,
                            token_index,
                            sequence_index,
                            code_position,
                            int(padding_mask[sample_index, token_index]),
                            *[
                                format(float(value), ".9g")
                                for value in model_input[sample_index, token_index]
                            ],
                        ]
                    )
        return batch * sequences * tokens


def write_parameters(
    checkpoint_path: Path,
    checkpoint_epoch: int | None,
    destination: Path,
    trace_path: Path,
    trace_manifest: dict,
) -> int:
    # Checkpoints produced by main.py pickle the project config. Only load
    # checkpoints from trusted local training runs.
    import main  # noqa: F401

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("epoch") != checkpoint_epoch:
        raise ValueError(
            f"Checkpoint epoch {checkpoint.get('epoch')} does not match "
            f"trace epoch {checkpoint_epoch}"
        )

    model_config = checkpoint.get("config")
    heads = trace_manifest["model_config"]["attention_heads"]
    if model_config is None or model_config.d_model % heads:
        raise ValueError("Checkpoint config is missing or has invalid head count")
    for name in ("d_model", "N_dec", "h"):
        config_key = {
            "d_model": "d_model",
            "N_dec": "decoder_layers",
            "h": "attention_heads",
        }[name]
        if getattr(model_config, name) != trace_manifest["model_config"][config_key]:
            raise ValueError(
                f"Checkpoint and trace config mismatch for {name}"
            )
    head_dim = model_config.d_model // heads
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain model_state_dict")

    with np.load(trace_path, allow_pickle=False) as trace:
        for layer in trace_manifest["layers"]:
            for projection in ("q", "k", "v"):
                trace_projection = layer["projections"][projection]
                weight_name = trace_projection["weight_array"]
                bias_name = trace_projection["bias_array"]
                checkpoint_prefix = (
                    f"decoder.layers.{layer['layer_index']}.attn.to_{projection}"
                )
                for trace_name, state_name in (
                    (weight_name, f"{checkpoint_prefix}.weight"),
                    (bias_name, f"{checkpoint_prefix}.bias"),
                ):
                    if trace_name is None:
                        continue
                    if state_name not in state or not np.array_equal(
                        trace[trace_name],
                        state[state_name].detach().cpu().numpy(),
                    ):
                        raise ValueError(
                            f"Checkpoint parameter {state_name} does not match "
                            "the attention trace"
                        )

    qkv_pattern = re.compile(
        r"^decoder\.layers\.(\d+)\.attn\.to_([qkv])\.(weight|bias)$"
    )
    total_values = 0
    with destination.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "parameter_name",
                "shape",
                "flat_index",
                "output_channel",
                "input_channel",
                "attention_layer",
                "projection",
                "head_index",
                "value",
            ]
        )
        for name, tensor in state.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("Expected model_state_dict entries to be named tensors")
            values = tensor.detach().cpu().contiguous()
            shape = tuple(values.shape)
            flat_values = values.reshape(-1).tolist()
            match = qkv_pattern.match(name)
            layer_index = int(match.group(1)) if match else ""
            projection = match.group(2).upper() if match else ""
            is_matrix = values.ndim >= 2
            input_width = values.shape[-1] if is_matrix else 1

            for flat_index, value in enumerate(flat_values):
                output_channel = flat_index // input_width if is_matrix else flat_index
                input_channel = flat_index % input_width if is_matrix else ""
                head_index = (
                    output_channel // head_dim
                    if match
                    else ""
                )
                writer.writerow(
                    [
                        name,
                        json.dumps(shape, separators=(",", ":")),
                        flat_index,
                        output_channel,
                        input_channel,
                        layer_index,
                        projection,
                        head_index,
                        format(float(value), ".9g"),
                    ]
                )
            total_values += len(flat_values)
    return total_values


def write_outputs(
    trace_path: Path,
    manifest_path: Path,
    destination: Path,
) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as trace:
        required = {"model_output", "target"}
        if not required.issubset(trace.files):
            raise ValueError(f"Trace missing arrays: {required - set(trace.files)}")
        logits = trace["model_output"]
        targets = trace["target"]
        if logits.shape != targets.shape:
            raise ValueError(
                f"Model output shape {logits.shape} does not match target "
                f"shape {targets.shape}"
            )
        if logits.shape[0] != manifest["sample_count"]:
            raise ValueError("Trace output sample count does not match manifest")

        with destination.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "sample",
                    "output_index",
                    "logit",
                    "probability",
                    "predicted_bit",
                    "target_bit",
                    "correct",
                ]
            )
            for sample_index in range(logits.shape[0]):
                for output_index in range(logits.shape[1]):
                    logit = float(logits[sample_index, output_index])
                    probability = 1.0 / (1.0 + np.exp(-logit))
                    predicted_bit = int(logit >= 0)
                    target_bit = int(targets[sample_index, output_index])
                    writer.writerow(
                        [
                            sample_index,
                            output_index,
                            format(logit, ".9g"),
                            format(float(probability), ".9g"),
                            predicted_bit,
                            target_bit,
                            int(predicted_bit == target_bit),
                        ]
                    )
        return int(logits.size)


def main_export() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint
    trace_dir = args.trace_dir
    trace_path = trace_dir / "attention_trace.npz"
    manifest_path = trace_dir / "manifest.json"
    for path in (
        checkpoint_path,
        trace_path,
        manifest_path,
        args.model_config,
        args.code_config,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    noisy_output = args.output_dir / "01_noisy_reads.csv"
    parameter_output = args.output_dir / "02_bcjrformer_parameters.csv"
    model_output = args.output_dir / "03_bcjrformer_outputs.csv"
    input_output = args.output_dir / "04_model_inputs_and_mask.csv"

    read_count = write_noisy_reads(trace_path, noisy_output)
    parameter_count = write_parameters(
        checkpoint_path,
        manifest.get("checkpoint_epoch"),
        parameter_output,
        trace_path,
        manifest,
    )
    output_count = write_outputs(trace_path, manifest_path, model_output)
    input_count = write_model_inputs_and_mask(trace_path, input_output)

    import main  # noqa: F401

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_config = checkpoint["config"]
    effective_config = {
        key: value
        for key, value in vars(checkpoint_config).items()
        if key not in {"specific_model", "ids_channel", "inner_model_config"}
    }
    effective_config["specific_model"] = type(
        checkpoint_config.specific_model
    ).__name__
    effective_config["ids_channel"] = vars(checkpoint_config.ids_channel)
    effective_config["inner_model_config"] = vars(
        checkpoint_config.inner_model_config
    )
    effective_config_path = args.output_dir / "effective_model_config.json"
    effective_config_path.write_text(
        json.dumps(effective_config, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    files = [
        noisy_output,
        parameter_output,
        model_output,
        input_output,
        effective_config_path,
    ]
    repository_root = Path(__file__).resolve().parent
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("extract_attention_trace.py"),
        repository_root / "main.py",
        repository_root / "bcjrformer" / "datasets" / "bcjrformer_dataset.py",
        repository_root / "bcjrformer" / "models" / "bcjrformer.py",
        repository_root / "bcjrformer" / "models" / "modules.py",
    ]
    if any(not path.is_file() for path in source_paths):
        raise FileNotFoundError("One or more source files required for provenance are missing")

    source_dir = args.output_dir / "source"
    for path in source_paths:
        relative_path = path.relative_to(repository_root)
        destination = source_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    bundle_manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": manifest.get("checkpoint_epoch"),
        "attention_trace": trace_path.name,
        "attention_trace_sha256": sha256_file(trace_path),
        "trace_manifest_sha256": sha256_file(manifest_path),
        "model_config": str(args.model_config),
        "model_config_sha256": sha256_file(args.model_config),
        "note": (
            "The YAML config is the checked-in template; effective values, "
            "including command-line overrides, are in effective_model_config.json."
        ),
        "code_config": str(args.code_config),
        "code_config_sha256": sha256_file(args.code_config),
        "source_files": {
            path.relative_to(repository_root).as_posix(): {
                "sha256": sha256_file(path),
                "bundle_path": (
                    Path("source") / path.relative_to(repository_root)
                ).as_posix(),
            }
            for path in source_paths
        },
        "sample_count": manifest["sample_count"],
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
        "semantics": {
            noisy_output.name: (
                "Raw reads from the same NPZ as model inputs and outputs."
            ),
            parameter_output.name: (
                "All checkpoint parameters; Q/K/V rows include layer, "
                "projection, and head indices. Q/K/V weights are checked "
                "against tensors embedded in the trace."
            ),
            model_output.name: (
                "Logits, probabilities, predictions, and targets directly "
                "from the same NPZ."
            ),
            input_output.name: (
                "Flattened model_input [batch, sequence, token, window, feature] "
                "and padding mask, reshaped as BCJRFormer.forward does."
            ),
        },
    }
    bundle_manifest_path = args.output_dir / "bundle_manifest.json"
    bundle_manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    shutil.copy2(trace_path, args.output_dir / trace_path.name)
    shutil.copy2(manifest_path, args.output_dir / manifest_path.name)
    shutil.copy2(args.model_config, args.output_dir / args.model_config.name)
    shutil.copy2(args.code_config, args.output_dir / args.code_config.name)
    shutil.copy2(checkpoint_path, args.output_dir / checkpoint_path.name)

    zip_path = args.output_dir.with_suffix(".zip")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.output_dir.parent))

    print(f"1. Noisy input reads: {noisy_output} ({read_count} reads)")
    print(f"2. Model weights and Q/K/V head mapping: {parameter_output}")
    print(f"   Parameter values: {parameter_count}")
    print(f"3. BCJRFormer logits and decoded outputs: {model_output}")
    print(f"   Output values: {output_count}")
    print(f"4. Model window inputs and padding mask: {input_output}")
    print(f"   Input rows: {input_count}")
    print(f"Consistent bundle: {zip_path}")


if __name__ == "__main__":
    main_export()
