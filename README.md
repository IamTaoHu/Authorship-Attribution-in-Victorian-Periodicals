# Authorship Attribution in Victorian Periodicals

This repository contains a reproducible research pipeline for authorship attribution and topic modelling experiments on Victorian periodicals, with emphasis on the Saturday Review / PERIAD corpus.

Phase 0 creates only the project scaffold. Training, evaluation, data loading, and topic modelling logic will be added in later phases.

## Project Structure

```text
data/
  raw/              Original source data. Keep large corpus files out of Git.
  processed/        Cleaned and transformed datasets ready for modelling.
  cache/            Cached datasets, embeddings, tokenizer outputs, and temporary artifacts.
notebooks/          Exploratory notebooks and analysis drafts.
src/
  data/             Future data loading, cleaning, and splitting code.
  models/           Future model definitions and wrappers.
  training/         Future training loops and experiment runners.
  evaluation/       Future metrics, reports, and error analysis code.
  topic_modeling/   Future LDA, BERTopic, and visualization code.
  utils/            Shared utilities.
configs/            YAML/Hydra configuration files for experiments.
outputs/
  checkpoints/      Model checkpoints and adapter weights.
  logs/             Training and experiment logs.
  metrics/          Structured metric tables and evaluation summaries.
  plots/            Figures and interactive visualizations.
  predictions/      Model predictions and attribution outputs.
scripts/            Command-line entrypoints for reproducible runs.
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running Future Scripts

Future experiments should be run from the project root with explicit configuration files, for example:

```bash
python scripts/<script_name>.py --config configs/<config_name>.yaml
```

Keep model choices, data paths, random seeds, and training parameters in `configs/` so experiments are easy to rerun and compare.

## Output Conventions

- Store raw corpus files in `data/raw/`.
- Store cleaned datasets and train/validation/test splits in `data/processed/`.
- Store reusable caches in `data/cache/`.
- Store experiment configuration files in `configs/`.
- Store model checkpoints in `outputs/checkpoints/`.
- Store logs in `outputs/logs/`.
- Store metrics tables in `outputs/metrics/`.
- Store plots and topic-model visualizations in `outputs/plots/`.
- Store model predictions in `outputs/predictions/`.

Generated data and experiment outputs are ignored by default to avoid committing large or transient files. If a paper-ready deliverable is produced under an ignored output directory, either add it intentionally with `git add -f` or move it to a tracked results directory introduced in a later phase.

## Phase 0 Utilities

Phase 0 includes small utilities for reproducible experiment setup. These utilities do not train models, load datasets, run evaluation, or perform topic modelling.

- `src/utils/reproducibility.py` provides `set_seed()` to fix Python, NumPy, and PyTorch random seeds.
- `configs/default.yaml` stores default experiment, model, training, data, logging, and evaluation settings.
- `src/utils/config.py` provides `load_config()` for loading YAML configs with basic validation.
- `src/utils/logging.py` provides `setup_logger()` for console logging and optional file logging.
- `src/utils/experiment.py` provides experiment naming and output path creation.

Experiment names use the convention:

```text
<model_short_name>_seed<seed>_lr<learning_rate>_epoch<epochs>
```

For the default config, the generated experiment name is:

```text
deberta_seed42_lr2e5_epoch5
```

Run the Phase 0 smoke test from the project root:

```bash
python scripts/check_setup.py
```

The smoke test loads `configs/default.yaml`, sets the seed, builds the experiment name, creates the experiment output folders, and writes a setup log under `outputs/<experiment_name>/logs/`.

`wandb` is the preferred logging backend for future experiments. `tensorboard` can be used as a fallback when Weights & Biases is unavailable or not desired. Logging backends are not initialized in Phase 0.

Future experiment runs should log:

- training loss
- validation loss
- accuracy
- macro F1
- runtime
- GPU memory
