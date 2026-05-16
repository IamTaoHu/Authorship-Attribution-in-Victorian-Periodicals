# Current State

This document records the current verified project state. It distinguishes implemented work from planned work so future documentation and GPT context do not overstate progress.

Related docs: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md), [TASK_BOARD.md](TASK_BOARD.md), [ROADMAP_PHASE0_TO_PHASE10.md](ROADMAP_PHASE0_TO_PHASE10.md), [GPT_CONTEXT_SUMMARY.md](GPT_CONTEXT_SUMMARY.md).

## Workflow Refactor Status

The project has moved to a unified workflow:

- local VS Code for development, configs, notebooks, lightweight checks, aggregation, plotting, and documentation
- GitHub for source control only
- Colab Pro for heavy GPU training, decoder inference, QLoRA, and large-model evaluation
- Hugging Face for datasets and pretrained model downloads
- local storage outside `repo/` for datasets, artifacts, exports, and backups; Phase 5 active experiment outputs use ignored `outputs/phase5` and `checkpoints/phase5`

Phase 0 infrastructure refactor is mostly complete. Path configuration, external storage conventions, Colab notebooks, and phase check scripts are in place.

## Why the Workflow Changed

Earlier external-compute attempts exposed several practical risks:

- GPU quota limits interrupted heavy jobs.
- Checkpoints could be lost if not copied out of temporary runtimes.
- Adapter path mismatches made QLoRA recovery fragile.
- Malformed logs complicated debugging.
- Plotting failed when aggregate tables were stale or mixed with diagnostics.
- Gated model access issues affected Llama and Gemma runs until authentication/access was handled.

The current decision is to use Colab Pro for heavy compute and keep permanent outputs in local external workspace storage.

## Current Phase Status

| Phase | Status | Evidence |
|---|---|---|
| Phase 0 - Infrastructure Refactor | Mostly complete | workspace layout, path configs, check scripts, Colab workflow scaffolding |
| Phase 1 - Dataset Pipeline | Complete | `artifacts/phase1` reports, tables, and plots exist |
| Phase 2 - Encoder Baselines | Complete | `artifacts/phase2` BERT/RoBERTa runs, tables, report, and plots exist |
| Phase 3 - Advanced Encoders | Complete | `artifacts/phase3` DeBERTa/ModernBERT final runs and visualizations exist |
| Phase 4 - Decoder Prompting | Complete | six final decoder runs plus final-only tables/report/plots exist |
| Phase 5 - QLoRA Decoder Fine-Tuning | Pipeline added / diagnostic-only locally | configs, scripts, validation, and Colab workflow added; full 7B-9B results are not validated until Colab runs finish |
| Phase 6 - Ensemble | Planned | no implemented final artifact set |
| Phase 7 - LDA Topic Modelling | Planned | dependencies exist, final phase implementation not present |
| Phase 8 - BERTopic | Planned | dependencies exist, final phase implementation not present |
| Phase 9 - Topic-Aware Classification | Planned | no implemented final artifact set |
| Phase 10 - Final Research Package | Planned | depends on completed benchmark and topic phases |

## Phase 4 Reporting State

Phase 4 decoder prompting is implemented and validated. It evaluates:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b-it`

Each model has zero-shot and few-shot runs. Final-only reporting is implemented in:

- `src/evaluation/aggregate_phase4_results.py`
- `src/visualization/plot_phase4_results.py`
- `scripts/check_phase4_outputs.py`

Final-only reporting excludes:

- `tinyllama_smoke`
- diagnostic folders
- invalid runs
- runs with `n_samples <= 0`
- runs with non-empty `invalid_reason`

TinyLlama exists only as a diagnostic smoke run and must not be treated as a final benchmark.

## Phase 5 Pipeline State

Phase 5 adds QLoRA decoder fine-tuning infrastructure for:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b-it`

Local Phase 5 work is diagnostic-only for RTX 3050 4GB. The diagnostic config uses a tiny sample budget and writes to `outputs/phase5/diagnostic` plus `checkpoints/phase5/diagnostic`. Full 7B-9B runs are intended for Colab and must not be represented as complete until real Colab training/evaluation artifacts exist. The Llama 3 8B and Gemma 2 9B full QLoRA defaults are tuned for Colab L4 24GB, with documented lower-memory fallback settings if L4 OOM occurs.

## Immediate Next Task

The next major task is running and validating the Phase 5 full QLoRA experiments on Colab. Do not claim final QLoRA results exist until `scripts/check_phase5_outputs.py --require_full_runs` passes.
