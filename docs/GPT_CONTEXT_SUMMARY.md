# GPT Context Summary

Project: Authorship Attribution in Victorian Periodicals: Advancing LLM Classification and Thematic Modelling for the Saturday Review (1855-1875).

Use this file as concise context for future GPT work. Related docs: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md), [CURRENT_STATE.md](CURRENT_STATE.md), [PROJECT_MAP.md](PROJECT_MAP.md), [ROADMAP_PHASE0_TO_PHASE10.md](ROADMAP_PHASE0_TO_PHASE10.md).

## Purpose

The project studies authorship attribution in Victorian periodicals using the PERIAD corpus and related Hugging Face resources. Goals include improving beyond BERT-base and original decoder baselines, evaluating modern encoders, evaluating decoder prompting, fine-tuning decoders with QLoRA, building ensembles, running topic modelling, and preparing paper-ready outputs.

## Architecture

- Local VS Code: development, configs, notebooks, lightweight validation, aggregation, plotting, documentation.
- GitHub: source control only.
- Colab Pro: heavy training, decoder inference, QLoRA, and large-model evaluation.
- Hugging Face: datasets, pretrained models, tokenizers.
- External local storage: permanent artifacts, checkpoints, datasets, exports, and backups.

## Workspace Layout

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/
  artifacts/
  backups/
  checkpoints/
  datasets/
  exports/
  repo/Authorship-Attribution-in-Victorian-Periodicals/
```

Run normal project commands from:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/repo/Authorship-Attribution-in-Victorian-Periodicals
```

## Repo Layout

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

`configs/paths.example.yaml` is portable. `configs/paths.local.yaml` is machine-specific and should not be committed.

## Data and Model Sources

Datasets:

- `celvaigh/periad`
- `NicholasSynovic/ModifiedVEAA`

Models:

- `bert-base-uncased`
- `roberta-base`
- `roberta-large`
- `microsoft/deberta-v3-base`
- `answerdotai/ModernBERT-base`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b-it`
- `sentence-transformers/all-MiniLM-L6-v2`

Llama and Gemma are gated and require Hugging Face access plus `HF_TOKEN`.

## Phase Roadmap and Current Status

- Phase 0 - Infrastructure Refactor: mostly complete.
- Phase 1 - Dataset Pipeline: complete.
- Phase 2 - Encoder Baselines: complete.
- Phase 3 - Advanced Encoders: complete.
- Phase 4 - Decoder Prompting: complete.
- Phase 5 - QLoRA Decoder Fine-Tuning: planned / next.
- Phase 6 - Ensemble: planned.
- Phase 7 - LDA Topic Modelling: planned.
- Phase 8 - BERTopic: planned.
- Phase 9 - Topic-Aware Classification: planned.
- Phase 10 - Final Research Package: planned.

Do not claim Phase 5+ results are complete unless validated artifacts exist.

## Phase 4 Reporting Rule

Use final-only reporting for Phase 4 benchmarks. Final-only reporting includes exactly:

- `mistral_zero_shot`
- `mistral_few_shot`
- `llama3_zero_shot`
- `llama3_few_shot`
- `gemma2_zero_shot`
- `gemma2_few_shot`

TinyLlama smoke runs are diagnostic only and excluded from final reports, tables, and plots.

## Immediate Next Tasks

- Prepare Phase 5 QLoRA infrastructure.
- Stabilize checkpoint and adapter backup workflow.
- Keep Colab outputs synchronized back to external local storage.
- Continue documentation updates as implementation evolves.

## Operating Rules

- Do not store checkpoints, artifacts, datasets, exports, or model weights in Git.
- Run scripts from the repo root.
- Use external local storage as permanent artifact storage.
- Use Colab Pro for heavy training and large-model inference.
- Use Hugging Face for datasets and models.
- Use `HF_TOKEN` for gated model access.
- Use final-only reporting for Phase 4 benchmarks.
