<p align="center">
    <img src = "./figs/tikzModel.svg">
</p>

# BCJRFormer
Codebase for my master thesis on **Transformer-Based Decoding in Concatenated Coding Schemes Under Synchronization Errors**. 

We investigate the usage of transformer-based decoders to address problems with transmissions via an insertion, deletion, and substitution (IDS) channel in the context of DNA data storage. 

**Contributions:**
- *BCJRFormer*: A transformer for coded [Multiple sequence alignment](https://en.wikipedia.org/wiki/Multiple_sequence_alignment). Performance scales similar to the BCJR algorithm, but the model can be used for larger cluster sizes due to quadratic complexity (compared to BCJR's exponential complexity) in the number of codeword transmissions.
- *ConvBCJRFormer*: A cross-attending transformer to decode convolutional codes. Initial step towards including linear outer code information within the synchronization process, which is for more general linear codes infeasible within BCJR. Our model yields slightly higher error rates compared to BCJR.
- *ECCT for Outer Decoding*: We provide an implementation of a dynamic dataset which uses ECCT as an outer decoder on top of BCJR or a pretained *BCJRFormer* model. By combining inner and outer transformer-based decoders, we show that lower error rates can be achieved compared to combinations involving BCJR or Believe Propagation.  


Both BCJR and Believe Propagation are implemented with [numba](https://github.com/numba/numba) compilation, permitting parallelizable joint decoding of nonbinary marker/convolutional codes with up to $M=4$ transmissions. 

**Disclaimer:** This codebase was somewhat hastily refactored - Some variable names might still be misleading (especially in the context of the thesis).

# Setup
To install the project, you need to have [Poetry](https://python-poetry.org/) installed. At least Python 3.12 is required. Once you have Poetry, you can install all the dependencies by running the following command:

```sh
poetry install
```

If you have multiple Python versions you can specify the Python version to use with:
```sh
poetry env use python3.x
```
Scripts can then be run via `poetry run python foo.py`. If you have specific CUDA requirements, you'll have to recreate the poetry environment by adjusting the `pyproject.toml` file.


# Running code
### Training
Training is dispatched via the `main.py` script. For a quick overview on all possible commands, call:
```sh
poetry run python main.py --help
```

We use [Spock](https://github.com/fidelity/spock) for config management (sorry!). Sometimes CLI commands can be a bit unintuitive. In general, use: 
```sh
poetry run python main.py -c model_config.yaml code_config.yaml --ModelConfig.foo bar
```  
to load model and code config, then you can overwrite key-value pairs (in this case the model_config's key `foo` with value `bar`).

### Extracting attention traces for verification
For a trained BCJRFormer checkpoint, capture each decoder layer's Q/K/V
projection weights, biases, normalized projection inputs, and per-head
projection outputs alongside the model input and output:
```sh
poetry run python extract_attention_trace.py \
  --checkpoint path/to/best_model.pth \
  --model-config configs/model/joint_bcjrformer_config.yaml \
  --code-config configs/code/code_config_ldpc_96_48_marker.yaml \
  --output-dir path/to/attention_fev \
  --samples 10
```
The output directory contains `attention_trace.npz` and `manifest.json`. For
each head, the manifest maps the arrays for its weights, bias, and observed
output to the shared layer projection input. The projection is
`output = input @ weight.T + bias`; weights use PyTorch's
`[out_features, in_features]` layout. The script uses the training config
stored in the checkpoint and a deterministic sample generated from the
selected code config. The capture also writes `noisy_reads.csv`, which lists
the raw received 0/1 reads before trellis conversion alongside the transmitted
inner-code bits, outer codeword, and message bits. Read lengths exclude
padding. Only load checkpoints from trusted training runs.

To view a captured trace as layer-by-layer heatmaps with exact CSV downloads,
generate and open the local HTML report:
```sh
poetry run python make_attention_report.py \
  --trace-dir path/to/attention_fev \
  --example 0
```
The report is written to `path/to/attention_fev/human_readable/index.html`.
It displays input vectors, Q/K/V weight matrices, and projected outputs for
the selected example. Its CSV links contain exact weights and all captured
sample outputs; it also links the raw `noisy_reads.csv` from the capture.

For a reproducible hardware handoff, export noisy reads first, model
parameters second, and model outputs third. A fourth CSV contains the
windowed tensors and padding masks actually consumed by the network:
```sh
poetry run python export_hardware_csv.py \
  --checkpoint path/to/best_model.pth \
  --trace-dir path/to/attention_fev \
  --output-dir path/to/hardware_bundle \
  --model-config configs/model/joint_bcjrformer_config.yaml \
  --code-config configs/code/code_config_ldpc_96_48_marker.yaml
```
This creates a matching `hardware_bundle.zip` with the checkpoint, trace,
config files, exporter scripts, CSVs, and a hash-based `bundle_manifest.json`.
The exporter reads the noisy reads, model inputs/masks, targets, and outputs
from the same trace file and checks the trace's Q/K/V weights against the
selected checkpoint.
The parameter CSV contains every checkpoint parameter as one row per scalar.
For decoder Q/K/V parameters, `attention_layer`, `projection`, and
`head_index` identify which attention head owns the output channel; rows use
PyTorch's row-major linear-weight layout. The output CSV contains each logit,
its sigmoid probability, predicted bit, target bit, and correctness flag. The
fourth CSV flattens `model_input` exactly as `BCJRFormer.forward` does and
includes the corresponding padding-mask bit.

### Evaluation
Write/Use an evaluate config and use the script `evaluate.py`, e. g.:
```sh
poetry run python eval.py -c evaluate_config.yaml code_config.yaml
```

### Optional: Weights & Biases (wandb) Logging
We track experiments using Weights & Biases (wandb). Wandb support can be enabled via the configuration file.

### Known (benign) warnings
- Spock prints `SyntaxWarning: Overriding an already set parameter ...` when a key is set in a config file and also overridden on the command line. This is expected and harmless - the CLI value wins.
- On macOS, `poetry install` prints a block about wheels skipped for the `torch-cu118` source ("environment does not support the identified abi tags"). This is expected: the CUDA wheels only exist for Linux/Windows, and Poetry correctly falls back to the plain CPU/MPS `torch` build on macOS (`torch.version.cuda` is `None`). On Linux/Windows the `+cu118` CUDA build is installed instead.

# Project Structure

<pre>
.
├── bcjrformer/                         :   Project folder
│   ├── <b>bcjr/</b>                           :   <b>CLI for BCJR</b> & BP (with parallelization support)
│   │   ├── cc_bcjr_ids_ldpc.py         :   Convolutional Code BCJR 
│   │   └── marker_bcjr_ids_ldpc.py     :   Marker Code BCJR 
│   ├── code_db/                        :   Linear Code (.alist) database
│   ├── codes/                      
│   │   ├── convolutional_code.py       :   Implementation of BCJR & general conv. codes
│   │   ├── galois_field.py             :   Numba-compatible finite field utils (adjusted from Galois library)
│   │   ├── linear_code.py              :   Linear Codes & Implementation of Believe Propagation
│   │   ├── marker_code.py              :   Marker Codes
│   │   └── ...
│   ├── configs/                        :   Definition of Spock-configs
│   │   └── ...
│   ├── datasets/                       :   Dynamic datasets for inner / outer codeword generation
│   ├── models/
│   │   ├── <b>bcjrformer.py</b>               :   <b>Proposed inner decoder for (nonbinary) marker codes</b>
│   │   ├── <b>convbcjrformer.py</b>           :   <b>Proposed inner decoder for convolutional codes </b>
│   │   ├── ids_ecct_outer_model.py     :   ECCT model of https://github.com/yoniLc
│   │   └── ...
│   └── trainers/                       :   "Trainer" classes (but they also perform evaluation)
│
├── configs/                            :   Some pre-crafted .yaml configs for training
├── notebooks/                          :   Notebooks to evaluate models / generate figures
├── thesis/                             :   All Figures / Numbers / Latex of my master thesis
├── eval_scripts/                       :   Scripts to evaluate models
├── ...
├── eval.py                               :   Evaluation only of trained models   
└── <b>main.py/</b>                            :   <b>Main Training Script</b>
</pre>
