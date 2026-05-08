# Authorship Attribution in Victorian Periodicals

This repository contains a reproducible NLP research pipeline for paragraph-level authorship attribution and later topic modelling on Victorian periodicals, with the current experiments centered on the PERIAD corpus.

Current status:

- Phase 0 project setup is done.
- Phase 1 PERIAD dataset preparation is done.
- Phase 2 BERT-base baseline reproduction is done with real `outputs/phase2/` artifacts.
- Phase 3 modern encoder experiments are done with real `outputs/phase3/` metrics, predictions, plots, and comparison tables.
- Phase 4+ decoder, ensemble, and topic-modelling work is still future work.

## Project Structure

```text
configs/
  default.yaml                    Base project configuration.
  phase3/                         Modern encoder and LoRA experiment configs.
data/
  raw/                            Original source data. Keep large corpus files out of Git.
  processed/                      Cleaned/transformed datasets when needed.
  cache/                          Local reusable caches.
docs/
  phase3_modern_encoder_experiments.md
notebooks/                        Exploratory notebooks and analysis drafts.
outputs/
  phase1/                         Generated cleaned PERIAD dataset and inspection reports.
  phase2/                         Generated BERT baseline metrics, reports, plots, predictions.
  phase3/                         Generated encoder metrics, reports, plots, predictions, tables.
scripts/
  check_setup.py
  phase1_dataset_pipeline.py
  phase2_train_bert_baseline.py
  phase2_bert_majority_vote.py
  run_phase3_*.ps1
src/
  analysis/                       Phase 3 error-analysis helpers.
  data/                           Dataset loading, inspection, cleaning, tokenization analysis.
  evaluation/                     Metrics, reports, plots, label mapping, error analysis.
  models/                         Model/tokenizer factories.
  topic_modeling/                 Placeholder for later topic-modelling phases.
  training/                       Phase 2 and Phase 3 training workflows.
  utils/                          Config, logging, reproducibility, experiment helpers.
  visualization/                  Phase 3 plots and embedding visualizations.
```

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 0 Setup Check

Phase 0 verifies the config loader, seed setup, logger setup, and experiment directory helpers.

```powershell
python scripts/check_setup.py
```

## Phase 1 Dataset Pipeline

Phase 1 loads PERIAD and Modified-VEAA from Hugging Face, inspects schemas and class distributions, cleans PERIAD text, validates splits, and optionally runs tokenization analysis.

Quick cleaning/inspection run:

```powershell
python scripts/phase1_dataset_pipeline.py --output_dir outputs/phase1 --max_samples_for_token_analysis 50 --skip_tokenization
```

Full Phase 1 run:

```powershell
python scripts/phase1_dataset_pipeline.py --periad_dataset_name celvaigh/periad --modified_veaa_dataset_name NicholasSynovic/Modified-VEAA --output_dir outputs/phase1 --max_samples_for_token_analysis 50
```

Key outputs:

- `outputs/phase1/periad_cleaned/`
- `outputs/phase1/dataset_report.json`
- `outputs/phase1/class_distribution.csv`
- `outputs/phase1/periad_class_distribution.csv`
- `outputs/phase1/label_mapping.json`
- `outputs/phase1/text_quality_report.json`
- `outputs/phase1/tokenization_report.csv`
- `outputs/phase1/plots/`

## Phase 2 BERT Baseline

Phase 2 trains and evaluates a BERT-base authorship classifier on the cleaned PERIAD dataset. Numeric labels are preserved internally for training; author names are used in reports, plots, and prediction exports.

Syntax check:

```powershell
python -m py_compile src/models/bert_classifier.py src/training/train_transformer_classifier.py src/evaluation/label_mapping.py src/evaluation/metrics.py src/evaluation/reports.py src/evaluation/plots.py src/evaluation/error_analysis.py scripts/phase2_train_bert_baseline.py scripts/phase2_bert_majority_vote.py
```

Single-seed baseline:

```powershell
python scripts/phase2_train_bert_baseline.py --model_name bert-base-uncased --seed 42 --epochs 3 --batch_size 8 --output_dir outputs/phase2
```

Majority-vote run:

```powershell
python scripts/phase2_bert_majority_vote.py --num_runs 10 --epochs 5 --batch_size 8 --output_dir outputs/phase2
```

Completed seed-42 BERT result:

| Model | Epochs | Batch size | Test accuracy | Test macro F1 | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bert-base-uncased` | 3 | 8 | 0.9174 | 0.9071 | 0.9275 | 0.9172 |

Key artifacts:

- `outputs/phase2/metrics/bert_base_seed42_metrics.json`
- `outputs/phase2/reports/baseline_comparison.md`
- `outputs/phase2/reports/bert_base_seed42_classification_report.csv`
- `outputs/phase2/plots/bert_base_seed42_confusion_matrix.png`
- `outputs/phase2/predictions/bert_base_seed42_predictions.csv`

## Phase 3 Modern Encoder Experiments

Phase 3 compares stronger encoder models on the same PERIAD authorship task. It includes full fine-tuning for RoBERTa and ModernBERT, plus LoRA rank ablations for DeBERTa v1. DeBERTa-v3 runs are kept under `outputs/phase3/unusable/` because local runs became numerically unstable and are not used for reporting.

Syntax check:

```powershell
python -m py_compile src/training/train_encoder.py src/evaluation/evaluate_encoder_outputs.py src/visualization/plot_phase3_results.py src/visualization/plot_embeddings.py src/analysis/error_analysis_phase3.py
```

Dry-run a config:

```powershell
python src/training/train_encoder.py --config configs/phase3/roberta_base.yaml --dry-run
```

Inspect LoRA target modules:

```powershell
python src/training/train_encoder.py --config configs/phase3/lora/deberta_base_lora_r4_sanity.yaml --list-linear-modules
```

Recommended Phase 3 workflow:

```powershell
.\scripts\run_phase3_core.ps1
.\scripts\run_phase3_lora_ablation.ps1
.\scripts\run_phase3_modernbert_context.ps1
.\scripts\run_phase3_analysis.ps1
```

Completed Phase 3 comparison:

| Experiment | Model | LoRA | Accuracy | Macro F1 | GPU max GB |
| --- | --- | --- | ---: | ---: | ---: |
| `roberta_base` | `roberta-base` | no | 0.9200 | 0.9114 | 4.81 |
| `modernbert_base_512` | `answerdotai/ModernBERT-base` | no | 0.9138 | 0.9065 | 4.40 |
| `deberta_base_lora_r32` | `microsoft/deberta-base` | r32 | 0.9104 | 0.9019 | 3.05 |
| `deberta_base_lora_r16` | `microsoft/deberta-base` | r16 | 0.8983 | 0.8875 | 3.04 |
| `deberta_base_lora_r8` | `microsoft/deberta-base` | r8 | 0.8926 | 0.8822 | 3.04 |
| `deberta_base_lora_r4` | `microsoft/deberta-base` | r4 | 0.8907 | 0.8810 | 3.03 |

Key artifacts:

- `outputs/phase3/tables/encoder_results.md`
- `outputs/phase3/tables/final_encoder_results_table.csv`
- `outputs/phase3/tables/lora_rank_ablation_table.csv`
- `outputs/phase3/tables/modernbert_context_ablation_table.csv`
- `outputs/phase3/plots/`
- per-experiment `metrics.json`, predictions, logits, embeddings, and analysis files under `outputs/phase3/{experiment_name}/`

## Label Mapping

PERIAD labels are mapped as follows:

| Label | Author |
| ---: | --- |
| 0 | Leslie Stephen |
| 1 | John Morley |
| 2 | Eliza Lynn Linton |
| 3 | George Henry Lewes |
| 4 | Anne Mozley |
| 5 | James Fitzjames Stephen |

## Output Conventions

Generated data, model checkpoints, caches, and large experiment outputs should stay out of Git unless a paper-ready lightweight artifact is intentionally selected.

- Store raw corpus files in `data/raw/`.
- Store cleaned datasets and splits in `outputs/phase1/` or `data/processed/`.
- Store model checkpoints and adapter weights under the relevant `outputs/phase*/` experiment directory.
- Store metrics, reports, plots, and predictions under the relevant `outputs/phase*/` directory.
- Do not hand-edit generated output artifacts.

## Roadmap

Done:

- Phase 0: project setup
- Phase 1: dataset pipeline
- Phase 2: BERT-base baseline reproduction
- Phase 3: modern encoder experiments

Next phases:

- Phase 4: decoder prompting
- Phase 5: decoder fine-tuning
- Phase 6: ensemble experiments
- Phase 7: LDA topic modelling
- Phase 8: BERTopic
- Phase 9: topic features for classification
- Phase 10: final packaging
