import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture BCJRFormer attention Q/K/V parameters and per-head "
            "linear input/output vectors from a checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--code-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    import main as training

    # Training checkpoints pickle the project config object and must come from
    # a trusted local run.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    sys.argv = [
        "main.py",
        "-c",
        str(args.model_config),
        str(args.code_config),
    ]
    parsed_model_config, code_config, _ = training.parse_args()
    model_config = checkpoint.get("config")
    if not isinstance(model_config, training.ModelConfig):
        raise TypeError("Checkpoint does not contain a compatible ModelConfig")
    channel_config = model_config.ids_channel
    training.validate_model_code_compatibility(
        model_config.specific_model,
        code_config.inner_code_config,
    )
    training.set_seed(model_config.seed)

    logger = logging.getLogger("attention_trace")
    logger.addHandler(logging.NullHandler())
    builder = training.model_builder_factory(model_config.specific_model)
    train_dataset, test_dataset, trainer = builder.build_for_train(
        model_config,
        channel_config,
        code_config.outer_code_config,
        code_config.inner_code_config,
        torch.device("cpu"),
        logger,
    )
    del train_dataset

    trained_config = checkpoint.get("config")
    if type(trained_config.specific_model) is not type(
        parsed_model_config.specific_model
    ):
        raise ValueError("Selected model config does not match the checkpoint model")
    for name in ("d_model", "N_dec", "h"):
        if getattr(trained_config, name, None) != getattr(
            parsed_model_config, name
        ):
            raise ValueError(
                f"Model config mismatch for {name}: checkpoint has "
                f"{getattr(trained_config, name, None)}, selected config has "
                f"{getattr(parsed_model_config, name)}"
            )

    model = trainer.model.cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    modules = dict(model.named_modules())
    projection_modules: dict[str, torch.nn.Linear] = {}
    captures: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    for name, module in modules.items():
        if (
            name.startswith("decoder.layers.")
            and name.rsplit(".", 1)[-1] in {"to_q", "to_k", "to_v"}
            and isinstance(module, torch.nn.Linear)
        ):
            projection_modules[name] = module

            def capture(
                current_module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                module_name: str = name,
            ) -> None:
                del current_module
                captures[module_name] = (
                    inputs[0].detach().cpu(),
                    output.detach().cpu(),
                )

            handles.append(module.register_forward_hook(capture))

    if not projection_modules:
        raise ValueError("Checkpoint model has no decoder Q/K/V projection layers")

    if args.samples > len(test_dataset):
        raise ValueError(
            f"Requested {args.samples} samples, but configured test dataset "
            f"contains only {len(test_dataset)}"
        )

    if not hasattr(test_dataset, "get_item_with_raw_reads"):
        raise TypeError(
            "The selected test dataset does not support raw marker-read capture"
        )
    sample_items = [
        test_dataset.get_item_with_raw_reads(index) for index in range(args.samples)
    ]
    if any(len(item) != 8 for item in sample_items):
        raise ValueError("Expected marker samples with six model tensors and raw reads")
    batch = default_collate([item[:6] for item in sample_items])
    message, codeword, target, _, model_input, padding_mask = batch
    raw_read_items = [item[6].numpy() for item in sample_items]
    raw_length_items = [item[7].numpy() for item in sample_items]
    maximum_sequences = max(reads.shape[0] for reads in raw_read_items)
    maximum_read_length = max(reads.shape[1] for reads in raw_read_items)
    raw_received_reads = np.full(
        (args.samples, maximum_sequences, maximum_read_length),
        fill_value=-1,
        dtype=np.int64,
    )
    raw_read_lengths = np.zeros((args.samples, maximum_sequences), dtype=np.int64)
    for sample_index, (reads, lengths) in enumerate(
        zip(raw_read_items, raw_length_items, strict=True)
    ):
        raw_received_reads[
            sample_index, : reads.shape[0], : reads.shape[1]
        ] = reads
        raw_read_lengths[sample_index, : len(lengths)] = lengths

    with torch.inference_mode():
        model_output = model(model_input, padding_mask)

    for handle in handles:
        handle.remove()

    arrays: dict[str, np.ndarray] = {
        "model_input": tensor_to_numpy(model_input),
        "padding_mask": tensor_to_numpy(padding_mask),
        "message": tensor_to_numpy(message),
        "codeword": tensor_to_numpy(codeword),
        "target": tensor_to_numpy(target),
        "model_output": tensor_to_numpy(model_output),
        "raw_received_reads": raw_received_reads,
        "raw_read_lengths": raw_read_lengths,
    }
    layers: dict[int, dict[str, Any]] = {}
    d_model = model_config.d_model
    num_heads = model_config.h
    head_dim = d_model // num_heads
    maximum_reconstruction_error = 0.0

    for module_name, module in sorted(projection_modules.items()):
        if module_name not in captures:
            raise RuntimeError(f"Projection hook did not capture {module_name}")
        layer_number = int(module_name.split(".")[2])
        projection_name = module_name.rsplit(".", 1)[-1].removeprefix("to_")
        projection_input, projection_output = captures[module_name]

        layer = layers.setdefault(
            layer_number,
            {
                "layer_index": layer_number,
                "projections": {},
            },
        )
        input_key = f"layer_{layer_number:02d}_projection_input"
        if input_key not in arrays:
            arrays[input_key] = tensor_to_numpy(projection_input)
        elif not torch.equal(
            torch.from_numpy(arrays[input_key]), projection_input
        ):
            raise RuntimeError(
                f"Q/K/V projection inputs differ in decoder layer {layer_number}"
            )

        projection: dict[str, Any] = {
            "input_array": input_key,
            "equation": "output = input @ weight.T + bias",
            "weight_array": f"layer_{layer_number:02d}_{projection_name}_weight",
            "bias_array": (
                f"layer_{layer_number:02d}_{projection_name}_bias"
                if module.bias is not None
                else None
            ),
            "weight_shape": list(module.weight.shape),
            "bias_shape": list(module.bias.shape) if module.bias is not None else None,
            "heads": [],
        }
        arrays[f"layer_{layer_number:02d}_{projection_name}_weight"] = (
            tensor_to_numpy(module.weight)
        )
        if module.bias is not None:
            arrays[f"layer_{layer_number:02d}_{projection_name}_bias"] = (
                tensor_to_numpy(module.bias)
            )

        for head_index in range(num_heads):
            start = head_index * head_dim
            stop = start + head_dim
            actual_output = projection_output[..., start:stop]
            head_weight = module.weight[start:stop]
            head_bias = module.bias[start:stop] if module.bias is not None else None
            with torch.inference_mode():
                expected_output = torch.nn.functional.linear(
                    projection_input,
                    head_weight,
                    head_bias,
                )
            torch.testing.assert_close(
                expected_output,
                actual_output,
                rtol=1e-5,
                atol=1e-6,
            )
            if actual_output.numel():
                maximum_reconstruction_error = max(
                    maximum_reconstruction_error,
                    float((expected_output - actual_output).abs().max().item()),
                )

            prefix = (
                f"layer_{layer_number:02d}_head_{head_index:02d}_"
                f"{projection_name}"
            )
            weight_key = f"{prefix}_weight"
            bias_key = f"{prefix}_bias"
            output_key = f"{prefix}_output"
            arrays[weight_key] = tensor_to_numpy(head_weight)
            if head_bias is not None:
                arrays[bias_key] = tensor_to_numpy(head_bias)
            arrays[output_key] = tensor_to_numpy(actual_output)
            projection["heads"].append(
                {
                    "head_index": head_index,
                    "output_channels": [start, stop],
                    "weight_array": weight_key,
                    "bias_array": bias_key if head_bias is not None else None,
                    "output_array": output_key,
                }
            )

        layer["projections"][projection_name] = projection

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "attention_trace.npz"
    np.savez_compressed(trace_path, **arrays)

    inner_config = code_config.inner_code_config
    inner_parameters = {
        name: getattr(inner_config, name)
        for name in ("marker", "N_c", "p", "k", "g", "random_offset")
        if hasattr(inner_config, name)
    }
    linear_config = code_config.outer_code_config
    manifest = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": checkpoint.get("loss"),
        "sample_count": int(model_input.shape[0]),
        "model_config": {
            "architecture": type(model_config.specific_model).__name__,
            "d_model": d_model,
            "decoder_layers": model_config.N_dec,
            "attention_heads": num_heads,
            "head_dimension": head_dim,
            "seed": model_config.seed,
            "test_batch_size": model_config.test_batch_size,
            "test_batches_per_epoch": model_config.test_batches_per_epoch,
            "inner_sequence_min": model_config.inner_model_config.n_sequence_min,
            "inner_sequence_max": model_config.inner_model_config.n_sequence_max,
        },
        "data_config": {
            "channel": {
                "p_i": channel_config.p_i,
                "p_d": channel_config.p_d,
                "p_s": channel_config.p_s,
                "fixed_del_count": channel_config.fixed_del_count,
            },
            "outer_code": {
                "type": str(linear_config.code_type),
                "n": linear_config.n,
                "k": linear_config.k,
                "q": linear_config.q,
                "custom_file_name": linear_config.custom_file_name,
                "file_extension": (
                    str(linear_config.file_extension)
                    if linear_config.file_extension is not None
                    else None
                ),
            },
            "inner_code": {
                "type": type(inner_config).__name__,
                "parameters": inner_parameters,
            },
        },
        "trace_semantics": {
            "raw_received_reads": (
                "Integer 0/1 output symbols directly from the simulated IDS "
                "channel, before trellis window conversion. -1 denotes padding."
            ),
            "raw_read_lengths": (
                "Valid symbol count per sample and read; use it to trim "
                "raw_received_reads."
            ),
            "target_array": (
                "Transmitted marker-coded bit sequence that the corresponding "
                "noisy read was sampled from."
            ),
            "projection_input": (
                "The pre-normalized decoder token sequence passed to the Q, K, "
                "and V linear layers."
            ),
            "head_output": (
                "The corresponding linear projection output after slicing the "
                "linear output channels for that head and before attention reshape."
            ),
            "weight_layout": (
                "PyTorch nn.Linear layout [out_features, in_features]; "
                "head weights select contiguous output-channel rows."
            ),
            "source_input_array": "model_input",
            "padding_mask_array": "padding_mask",
            "target_array": "target",
            "model_output_array": "model_output",
        },
        "maximum_linear_reconstruction_absolute_error": (
            maximum_reconstruction_error
        ),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "layers": [layers[index] for index in sorted(layers)],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    noisy_reads_path = args.output_dir / "noisy_reads.csv"
    with noisy_reads_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        import csv

        writer = csv.writer(file)
        writer.writerow(
            [
                "sample",
                "read_index",
                "read_length",
                "noisy_received_bits",
                "transmitted_inner_bits",
                "outer_codeword_bits",
                "message_bits",
            ]
        )
        for sample_index in range(args.samples):
            transmitted_bits = "".join(
                str(int(bit)) for bit in arrays["target"][sample_index].reshape(-1)
            )
            codeword_bits = "".join(
                str(int(bit)) for bit in arrays["codeword"][sample_index].reshape(-1)
            )
            message_bits = "".join(
                str(int(bit)) for bit in arrays["message"][sample_index].reshape(-1)
            )
            for read_index, read_length in enumerate(
                raw_read_lengths[sample_index]
            ):
                if read_length == 0:
                    continue
                received_bits = "".join(
                    str(int(bit))
                    for bit in raw_received_reads[
                        sample_index, read_index, :read_length
                    ]
                )
                writer.writerow(
                    [
                        sample_index,
                        read_index,
                        int(read_length),
                        received_bits,
                        transmitted_bits,
                        codeword_bits,
                        message_bits,
                    ]
                )

    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Samples captured: {model_input.shape[0]}")
    print(f"Attention layers: {len(layers)}")
    print(f"Q/K/V projection/head pairs: {sum(len(p['heads']) for l in layers.values() for p in l['projections'].values())}")
    print(f"Maximum Q/K/V linear reconstruction error: {maximum_reconstruction_error:.3g}")
    print(f"Trace arrays: {trace_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Raw noisy channel reads: {noisy_reads_path}")


if __name__ == "__main__":
    main()
