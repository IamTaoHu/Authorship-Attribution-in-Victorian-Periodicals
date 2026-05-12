# GPT Context Summary

## Project Purpose

This project follows the internship plan **Authorship Attribution in Victorian Periodicals: Advancing LLM Classification and Thematic Modelling for the Saturday Review (1855-1875)**.

The goal is to improve authorship classification on Victorian periodicals beyond BERT-base and the original Mistral-7B setup, then connect classification errors and authorship patterns with topic modelling.

## Current Architecture

Local development and source control are separated from generated artifacts.

Workspace root:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution
```

Repo root:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/repo/Authorship-Attribution-in-Victorian-Periodicals
```

Local non-git storage:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/artifacts
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/checkpoints
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/datasets
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/exports
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/backups
```

Repo folders:

```text
configs/
docs/
notebooks/local/
notebooks/colab/
notebooks/archive/
prompts/
scripts/
src/
tests/
```

## Path Conventions

- Open VS Code at the workspace root if convenient.
- Write source code inside the repo under `repo/Authorship-Attribution-in-Victorian-Periodicals/`.
- Run Python scripts from the repo root unless a script says otherwise.
- Use `configs/paths.example.yaml` as the tracked template.
- Use `configs/paths.local.yaml` for local machine paths; do not commit it.
- Use `src/utils/paths.py` to resolve repo, dataset, artifact, checkpoint, and export paths.

## Dataset And Model Sources

Hugging Face is the primary source for datasets and pretrained models.

Datasets:

- `celvaigh/periad`
- `NicholasSynovic/ModifiedVEAA`

Encoders:

- `bert-base-uncased`
- `roberta-base`
- `roberta-large`
- `microsoft/deberta-v3-base`
- `answerdotai/ModernBERT-base`

Decoders:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b-it`

Embeddings:

- `sentence-transformers/all-MiniLM-L6-v2`

Llama and Gemma may require gated access and a valid `HF_TOKEN`.

## Phase Roadmap

- Phase 0: Infrastructure refactor.
- Phase 1: Dataset pipeline.
- Phase 2: Encoder baselines.
- Phase 3: Advanced encoders.
- Phase 4: Decoder prompting.
- Phase 5: QLoRA decoder fine-tuning.
- Phase 6: Ensemble.
- Phase 7: LDA topic modelling.
- Phase 8: BERTopic.
- Phase 9: Topic-aware classification.
- Phase 10: Final research package.

## Current State

The repository is being reset/refactored for the unified local + Colab workflow.

Previous Kaggle QLoRA work exposed:

- GPU quota limits
- Checkpoint loss
- Adapter path mismatch
- Plotting failures from malformed training logs
- Gated model access issues for Gemma

Current decisions:

- Use VS Code/local machine for development, configs, notebooks, docs, lightweight checks, aggregation, and plotting.
- Use Colab Pro for heavy training.
- Use Hugging Face for datasets and pretrained models.
- Use local storage outside the repo as permanent artifact/checkpoint storage.
- Use GitHub for source control only.

No final Phase 5 QLoRA results should be claimed unless validated artifacts exist.

## Immediate Next Tasks

- Complete Phase 0 refactor and validation.
- Confirm `configs/paths.local.yaml` is ignored by Git.
- Validate project structure.
- Implement Phase 1 dataset loading from Hugging Face.
- Validate PERIAD expected counts: train=8279, test=3549.
- Validate six canonical authors.

## Rules

- Do not store outputs, checkpoints, datasets, or large model files in Git.
- Do not commit `configs/paths.local.yaml`.
- Run scripts from the repo root.
- Use local `artifacts/` and `checkpoints/` as permanent storage.
- Use Colab Pro for heavy training.
- Use Hugging Face for datasets and models.
- Do not claim completed final experiments without validated artifacts.
