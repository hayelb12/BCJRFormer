import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML and CSV report from an attention trace."
    )
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--example", type=int, default=0)
    return parser.parse_args()


def write_matrix_csv(
    path: Path,
    values: np.ndarray,
    row_labels: list[str],
    column_prefix: str,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row", *[f"{column_prefix}_{i:03d}" for i in range(values.shape[1])]])
        for label, row in zip(row_labels, values, strict=True):
            writer.writerow([label, *[f"{float(value):.9g}" for value in row]])


def write_input_csv(path: Path, values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["sample", "token", *[f"feature_{i:03d}" for i in range(values.shape[-1])]]
        )
        for sample in range(values.shape[0]):
            for token in range(values.shape[1]):
                writer.writerow(
                    [
                        sample,
                        token,
                        *[f"{float(value):.9g}" for value in values[sample, token]],
                    ]
                )


def write_output_csv(path: Path, values: np.ndarray, heads: int, head_dim: int) -> None:
    columns = [
        f"head_{head:02d}_dim_{dimension:02d}"
        for head in range(heads)
        for dimension in range(head_dim)
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample", "token", *columns])
        for sample in range(values.shape[0]):
            for token in range(values.shape[1]):
                writer.writerow(
                    [
                        sample,
                        token,
                        *[f"{float(value):.9g}" for value in values[sample, token]],
                    ]
                )


def save_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    title: str,
    *,
    symmetric: bool = False,
) -> None:
    if symmetric:
        limit = float(np.max(np.abs(values))) if values.size else 1.0
        limit = limit or 1.0
        image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("Feature / input dimension")
    axis.set_ylabel("Token / output dimension")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir
    trace_path = trace_dir / "attention_trace.npz"
    manifest_path = trace_dir / "manifest.json"
    if not trace_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expected attention_trace.npz and manifest.json in {trace_dir}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not 0 <= args.example < manifest["sample_count"]:
        raise ValueError(
            f"--example must be between 0 and {manifest['sample_count'] - 1}"
        )

    report_dir = trace_dir / "human_readable"
    csv_dir = report_dir / "csv"
    image_dir = report_dir / "images"
    csv_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    config = manifest["model_config"]
    heads = config["attention_heads"]
    head_dim = config["head_dimension"]
    layer_sections: list[str] = []

    with np.load(trace_path, allow_pickle=False) as trace:
        for layer in manifest["layers"]:
            index = layer["layer_index"]
            prefix = f"layer_{index:02d}"
            input_array_name = layer["projections"]["q"]["input_array"]
            layer_input = trace[input_array_name]
            if layer_input.ndim != 3:
                raise ValueError(
                    f"Expected [sample, token, feature] input for layer {index}; "
                    f"got {layer_input.shape}"
                )
            write_input_csv(csv_dir / f"{prefix}_input.csv", layer_input)

            figure, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
            save_heatmap(
                axes[0, 0],
                layer_input[args.example],
                f"Layer {index} input (sample {args.example})",
            )

            projection_links: list[str] = []
            for position, name in enumerate(("q", "k", "v"), start=1):
                projection = layer["projections"][name]
                weight = trace[projection["weight_array"]]
                bias = trace[projection["bias_array"]]
                output_arrays = [
                    trace[head["output_array"]]
                    for head in projection["heads"]
                ]
                output = np.concatenate(output_arrays, axis=-1)

                if weight.shape != (config["d_model"], config["d_model"]):
                    raise ValueError(
                        f"Unexpected {name.upper()} weight shape in layer {index}: "
                        f"{weight.shape}"
                    )
                if output.shape[:2] != layer_input.shape[:2]:
                    raise ValueError(
                        f"Input/output token dimensions differ in layer {index} "
                        f"for {name.upper()}"
                    )

                row_labels = [
                    f"head_{head:02d}_out_{dimension:02d}"
                    for head in range(heads)
                    for dimension in range(head_dim)
                ]
                weight_with_bias = np.concatenate([weight, bias[:, None]], axis=1)
                write_matrix_csv(
                    csv_dir / f"{prefix}_{name}_weights_and_bias.csv",
                    weight_with_bias,
                    row_labels,
                    "input",
                )
                write_output_csv(
                    csv_dir / f"{prefix}_{name}_outputs_all_samples.csv",
                    output,
                    heads,
                    head_dim,
                )

                save_heatmap(
                    axes[0, position],
                    weight,
                    f"Layer {index} {name.upper()} weights",
                    symmetric=True,
                )
                save_heatmap(
                    axes[1, position - 1],
                    output[args.example],
                    f"Layer {index} {name.upper()} output (sample {args.example})",
                    symmetric=True,
                )

                projection_links.append(
                    "<li><strong>{name}</strong>: "
                    '<a href="csv/{weights}">weights + bias</a> | '
                    '<a href="csv/{outputs}">outputs for all samples</a></li>'.format(
                        name=name.upper(),
                        weights=f"{prefix}_{name}_weights_and_bias.csv",
                        outputs=f"{prefix}_{name}_outputs_all_samples.csv",
                    )
                )

            axes[1, 3].axis("off")
            axes[1, 3].text(
                0,
                0.95,
                f"Layer input: {layer_input.shape[1]} tokens x "
                f"{layer_input.shape[2]} features\n"
                f"Attention: {heads} heads x {head_dim} features/head\n"
                f"Heatmaps show sample {args.example}; CSV output files contain "
                "all captured samples.\n\n"
                "Colors show value magnitude and sign; exact values are in CSV.",
                va="top",
                fontsize=11,
            )
            figure.suptitle(f"Decoder layer {index}", fontsize=18)
            image_name = f"{prefix}_overview.png"
            figure.savefig(image_dir / image_name, dpi=125)
            plt.close(figure)

            layer_sections.append(
                f"<section><h2>Decoder layer {index}</h2>"
                f'<p><a href="csv/{prefix}_input.csv">Layer input vectors '
                f"(all samples)</a></p>"
                f'<img class="overview" src="images/{image_name}" '
                f'alt="Layer {index} input, weights, and outputs">'
                f"<h3>Exact numeric data</h3><ul>"
                f"{''.join(projection_links)}</ul></section>"
            )

    safe_checkpoint = html.escape(Path(manifest["checkpoint"]).name)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BCJRFormer attention report</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; color: #172033; margin: 0 auto;
       max-width: 1400px; padding: 1.5rem; background: #f5f7fb; }}
h1, h2, h3 {{ color: #15284b; }}
section {{ background: white; padding: 1rem 1.5rem; margin: 1.5rem 0;
           border-radius: 12px; box-shadow: 0 2px 10px #17203314; }}
.overview {{ display: block; width: 100%; height: auto; border: 1px solid #d9e0ec; }}
a {{ color: #0759a5; }}
.meta {{ overflow-wrap: anywhere; }}
code {{ background: #edf1f7; padding: .1rem .3rem; border-radius: 4px; }}
</style>
</head>
<body>
<h1>BCJRFormer attention report</h1>
<section class="meta">
<p><strong>Checkpoint:</strong> {safe_checkpoint}</p>
<p><strong>Best checkpoint epoch:</strong> {manifest["checkpoint_epoch"]};
<strong>loss:</strong> {manifest["checkpoint_loss"]}</p>
<p><strong>Model:</strong> {html.escape(config["architecture"])};
d_model={config["d_model"]}, {config["decoder_layers"]} layers,
{heads} heads x {head_dim} features/head.</p>
<p><strong>Displayed example:</strong> {args.example} of
{manifest["sample_count"]}. CSV output links include all captured samples.</p>
<p><strong>Original capture:</strong> <a href="../attention_trace.npz">
Download the NPZ data</a>; complete array definitions and run settings:
<a href="../manifest.json">open the capture manifest</a>.</p>
<p><strong>Raw noisy reads:</strong>
<a href="../noisy_reads.csv">download received 0/1 sequences and their
transmitted source sequences (CSV)</a>. Read lengths exclude padding.</p>
<p>Each weight row is one head/output feature; its columns are input features
followed by bias. Each output CSV row is one sample/token; columns identify
head and feature. For each head projection:
<code>output = input @ weight.T + bias</code>.</p>
</section>
{''.join(layer_sections)}
</body>
</html>
"""
    report_path = report_dir / "index.html"
    report_path.write_text(page, encoding="utf-8")
    print(f"Human-readable report: {report_path}")
    print(f"Layers: {len(layer_sections)}; displayed sample: {args.example}")
    print(f"Exact numeric CSV files: {len(list(csv_dir.glob('*.csv')))}")
    print(f"Heatmap images: {len(list(image_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
