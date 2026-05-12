# Current State

The repository is being reset and refactored for a unified local + Colab workflow. The goal is to make the repo a clean source-controlled project while moving large artifacts, datasets, checkpoints, and exports outside Git.

## Background

The earlier Kaggle QLoRA attempt exposed several workflow issues:

- GPU quota limits
- Checkpoint loss
- Adapter path mismatch
- Plotting failures caused by malformed training logs
- Gated model access issues for Gemma

These issues motivated a cleaner storage and execution model.

## Current Decision

- Use Colab Pro for heavy training.
- Use local storage as the primary artifact and checkpoint store.
- Use Hugging Face for datasets and pretrained models.
- Use GitHub for source control only.
- Use VS Code/local machine for development, configs, notebooks, docs, lightweight checks, aggregation, and plotting.

## Implementation Status

- Workspace structure has been created outside the repo:
  - `artifacts/`
  - `backups/`
  - `checkpoints/`
  - `datasets/`
  - `exports/`
- Repo structure is being standardized around:
  - `configs/`
  - `docs/`
  - `notebooks/`
  - `prompts/`
  - `scripts/`
  - `src/`
  - `tests/`
- A path manager has been added or is planned at `src/utils/paths.py`.
- Dataset and model configs have been added or are planned:
  - `configs/datasets.yaml`
  - `configs/models.yaml`
- No final Phase 5 QLoRA results should be claimed unless validated artifacts exist.

## Immediate Next Task

Phase 0 refactor and validation:

- Finalize repo structure.
- Validate path configuration.
- Verify local artifact/checkpoint storage.
- Confirm `configs/paths.local.yaml` is ignored by Git.
- Run the project structure check.
