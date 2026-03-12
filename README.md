# BOA Artifact
This repository contains the artifact for the MLSys 2026 paper:

**Toward Principled LLM Safety Testing: Solving the Jailbreak Oracle Problem**

It includes the BOA implementation, benchmark assets, and runnable scripts for minimum, lightweight, and full experiment workflows.

## Table of Contents

- [1 Quick Start](#1-quick-start)
- [2 Reproducibility Levels](#2-reproducibility-levels)
- [3 Requirements](#3-requirements)
- [4 Running Experiments](#4-running-experiments)
- [5 Expected Outputs](#5-expected-outputs)
- [6 Pipeline Overview](#6-pipeline-overview)
- [7 Benchmark](#7-benchmark)
- [8 Threshold (`tau`)](#8-threshold-tau)
- [9 Configuration](#9-configuration)
- [10 Repository Structure](#10-repository-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## 1 Quick Start

This section provides a minimum sanity check, then a real-model run.

Note: Quick Start does not call the external API judger (Layer 4), so it will not produce API-based unsafe/compliance decisions.

Step 1: Clone and enter the repository.

```bash
git clone https://github.com/shuyilinn/BOA.git
cd BOA
```

Step 2: Install dependencies.

```bash
pip install -r requirements.txt
```

Step 3: Run the minimum mock test (pipeline sanity check without loading an LLM).

```bash
python experiment/run_demo.py
```

Expected output (runtime is typically a few seconds):

```text
MINIMUM REPRODUCIBILITY TEST: PASSED
Status: minimum test passed
```

Step 4: (Optional) Log in to HuggingFace.
This is only needed when running open-source models that require gated/download access.

```bash
huggingface-cli login
```

Step 5: Launch a lightweight BOA run with a real model.

```bash
python experiment/run_light_model.py
```

Expected runtime: ~2 minutes.

Expected log:

```text
LIGHTWEIGHT REAL-MODEL TEST: PASSED
Status: light mode passed
```

Step 6: Confirm the run finished.

- Runner logs are written to `experiment/logs/`.
- BOA run logs are written to `logs/`.
- Result artifacts are written to `results/<run_id>/`.

You should see files like:

```text
results/<run_id>/
├── metadata.json
├── metadata.txt
├── runs.jsonl
├── runs.txt
└── trees/
    ├── prompt_0001_tree.txt
    └── prompt_0001_tree.json
```

## 2 Reproducibility Levels

- Minimum (mock, CPU-only): `python experiment/run_demo.py`
- Lightweight (single GPU): `python experiment/run_light_model.py`
- Full reproduction (paper-scale): `python experiment/run_model.py`

## 3 Requirements

- Linux
- Python >= 3.9
- Dependencies in `requirements.txt`

Optional:

- HuggingFace access for model download
- OpenAI API key for optional API-based judger

## 4 Running Experiments

Run the full experiment entry point. To add more experiment sweeps, add `SweepConfig` entries to `MODEL_CONFIGS` in `experiment/run_model.py`.

```bash
python experiment/run_model.py
```

Run lightweight single-task real-model check:

```bash
python experiment/run_light_model.py
```

Run mock pipeline check:

```bash
python experiment/run_demo.py
```

## 5 Expected Outputs

Per-run outputs are saved under:

- `results/<run_id>/`

Typical files:

```text
results/<run_id>/
├── metadata.json
├── metadata.txt
├── runs.jsonl
├── runs.txt
└── trees/
    ├── prompt_0001_tree.txt
    └── prompt_0001_tree.json
```

Log files:

- `logs/`
- `experiment/logs/`

## 6 Pipeline Overview

The BOA pipeline consists of three stages:

1. Load benchmark prompts (see [Section 7 Benchmark](#7-benchmark)).
2. Build or load threshold (`tau`) (see [Section 8 Threshold](#8-threshold-tau)):
   - If a threshold file already exists, BOA reuses it.
   - Otherwise, BOA builds the threshold file and saves it for future runs (this can take tens of minutes, depending on setup).
3. Run tree search for each harmful prompt in the configured prompt range.

## 7 Benchmark

Default harmful benchmark:

- `benchmark/boa_benchmark/jailbreak_oracle_benchmark.json`

Sources:

- JailbreakBench
- HarmBench (chemical/biological category)

Regenerate benchmark:

```bash
python benchmark/generate_benchmark.py
```

## 8 Threshold (`tau`)

`tau` is a log-probability threshold baseline used by BOA during search-time pruning/filtering.

- BOA first tries to load a cached threshold file.
- If no cache is found, BOA builds the baseline from benign prompts and saves it.
- Reusing a cached threshold avoids repeated calibration cost.
- Threshold cache is model/sampling specific. The cache filename includes:
  - `target_model`
  - `target_engine_name`
  - `top_p`
  - `top_k`
  - `temperature`
  - `threshold_baseline_sequences_per_prompt`
  - `threshold_baseline_generation_length`

## 9 Configuration

Primary configuration files:

- `config.py`
- `experiment/run_model.py`
- `experiment/run_light_model.py`
- `experiment/run_demo.py`

Typical knobs:

- target model and engine
- judger engine and GPU memory utilization
- workload-specific options under `workload_configs` (for example `benchmark_path`, `judger_profile`, `clean_response`)
- prompt range (`harmful_prompt_start/end`)
- search/sampling parameters (`top_p`, `top_k`, `temperature`, `chunk_size`, `chunk_width`)
- runtime budget (`time_limit_sec`)

`workload_configs[workload_name]["clean_response"]` controls whether `TreeGuideJudger` strips fictional multi-turn continuation markers such as `Human:` or `Assistant:` before judging. The default configuration keeps this enabled for `single_turn` and `multi_turn`, and disables it for `agent`.

### Models Configuration in Paper

| model name | top-p | top-k | temperature |
| --- | --- | --- | --- |
| `lmsys/vicuna-7b-v1.5` | `0.6` | `-1` | `0.9` |
| `meta-llama/Llama-2-7b-chat-hf` | `0.9` | `-1` | `0.6` |
| `meta-llama/Llama-3.1-8B-Instruct` | `0.9` | `-1` | `0.6` |
| `google/gemma-3-4b-it` | `0.95` | `64` | `1.0` |
| `Qwen/Qwen3-8B` | `0.95` | `20` | `0.6` |

## 10 Repository Structure

```text
BOA/
├── benchmark/
├── experiment/
├── executor/
├── judgers/
├── sampler/
├── searchers/
├── reporters/
├── results/
├── run.py
└── config.py
```

## Citation

Coming Soon

## Acknowledgements

We acknowledge the open-source community and benchmark/model providers including JailbreakBench, HarmBench, HuggingFace, Meta, Qwen, and Google.

## License

This repository is released under the MIT License.
