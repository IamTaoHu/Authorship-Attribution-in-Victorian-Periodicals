# Task Board

This board reflects the current verified repository and artifact state.

## Phase 0 - Infrastructure Refactor

- [x] Define external workspace layout for artifacts, checkpoints, datasets, exports, and backups.
- [x] Add path configuration template and local path config support.
- [x] Add project structure checks.
- [x] Establish Colab/local/GitHub/Hugging Face workflow.
- [ ] Harden all future phase scripts against accidental writes inside the Git repo.

## Phase 1 - Dataset Pipeline

- [x] Configure Hugging Face PERIAD and ModifiedVEAA dataset references.
- [x] Prepare PERIAD train/test data.
- [x] Validate expected PERIAD row counts and canonical six author labels.
- [x] Generate dataset reports, tables, and plots.
- [x] Store generated Phase 1 artifacts outside the repo.

## Phase 2 - Encoder Baselines

- [x] Configure BERT-base, RoBERTa-base, and RoBERTa-large baselines.
- [x] Run final encoder baseline evaluations.
- [x] Generate predictions, metrics, classification reports, and confusion matrices.
- [x] Aggregate Phase 2 tables and report.
- [x] Generate Phase 2 plots.
- [x] Keep diagnostic smoke outputs separate from final reporting.

## Phase 3 - Advanced Encoders

- [x] Configure DeBERTa-v3-base and ModernBERT-base benchmarks.
- [x] Run final Colab benchmark jobs.
- [x] Save final predictions, metrics, reports, and confusion matrices.
- [x] Generate final comparison tables and visualizations.
- [x] Preserve diagnostic/debug artifacts as non-final.

## Phase 4 - Decoder Prompting

- [x] Implement decoder prompting runner, prompts, and parser.
- [x] Configure Mistral, Llama 3, and Gemma 2 zero-shot/few-shot runs.
- [x] Complete six final decoder benchmark runs.
- [x] Keep TinyLlama smoke runs diagnostic only.
- [x] Implement final-only aggregation, reporting, plotting, and checking.
- [x] Generate final-only tables, report, and plots.

## Phase 5 - QLoRA Decoder Fine-Tuning

- [x] Finalize QLoRA training and evaluation design.
- [x] Add Phase 5 configs, scripts, diagnostic runner, validator, and Colab workflow.
- [x] Define ignored `outputs/phase5` and `checkpoints/phase5` checkpoint/adapters policy.
- [x] Add RTX 3050 4GB diagnostic-only mode.
- [x] Tune Llama 3 8B and Gemma 2 9B QLoRA defaults for Colab L4 24GB.
- [x] Run QLoRA training on Colab Pro.
- [x] Evaluate final QLoRA models on the fixed PERIAD test split.
- [x] Aggregate final Phase 5 results.
- [x] Make diagnostic validation optional for full-run Phase 5 checks.
- [x] Mirror lightweight Phase 5 reports, tables, plots, and run CSV/JSON/text files to `artifacts/phase5`.

## Phase 6 - Ensemble

- [x] Define ensemble candidates from Phase 2-5 final encoder, prompting, and QLoRA outputs.
- [x] Support exact and glob allowlists, including multi-seed Phase 2 discovery.
- [x] Exclude diagnostic, smoke, invalid, and incomplete outputs from final reporting.
- [x] Implement hard majority and weighted voting without retraining or decoder inference.
- [x] Normalize invalid decoder predictions to `__INVALID__`, exclude them from votes, and report invalid rates.
- [x] Evaluate ensembles on the fixed test split.
- [x] Generate ensemble predictions, single-model tables, comparison tables, report, confusion matrices, and metric plots.

## Phase 7 - LDA Topic Modelling

- [x] Prepare traditional topic modelling corpus from combined PERIAD train/test paragraphs.
- [x] Implement reproducible sklearn CountVectorizer + LDA pipeline.
- [x] Tune topic counts and preprocessing across k = 10, 20, 30, 40.
- [x] Generate topic tables, reusable document-topic features, models, summary metrics, and visualizations.
- [x] Validate Phase 7 artifact set under external `artifacts/phase7/lda`.
- [x] Interpret topics for paper-ready analysis. `artifacts\phase7\lda\reports\topic_interpretation.md`
- [x] Add visualizations `artifacts\phase7\lda\interactive`

## Phase 8 - BERTopic

- [x] Prepare embedding-based topic modelling workflow.
- [x] Implement BERTopic pipeline, runner, checker, config, plotting module, and compatibility shim.
- [ ] Generate and validate topic clusters, labels, and visualizations under external `artifacts/phase8/bertopic`.
- [x] Add RTX 3050 local-first mode with MiniLM embeddings, batch size 16, conservative UMAP/HDBSCAN, and probabilities disabled by default.
- [x] Export Phase 9-ready `tables/document_topic_features.csv` with one-hot topic columns, including `topic_-1` when outliers exist.
- [ ] Compare BERTopic outputs with LDA.

## Phase 9 - Topic-Aware Classification

- [ ] Design topic feature integration strategy.
- [ ] Build topic-aware classification variants.
- [ ] Evaluate topic-aware models against prior baselines.
- [ ] Analyze whether topic information improves attribution.

## Phase 10 - Final Package

- [ ] Freeze final benchmark tables.
- [ ] Freeze final plots and visualizations.
- [ ] Create final reports and research package.
- [ ] Prepare reproducibility notes.
- [ ] Prepare paper-ready narrative and appendices.
