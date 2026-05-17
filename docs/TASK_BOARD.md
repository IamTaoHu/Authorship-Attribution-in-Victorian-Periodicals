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

## Phase 6 - Ensemble

- [ ] Define ensemble candidates from encoder, prompting, and QLoRA outputs.
- [ ] Implement ensemble voting/probability fusion where output formats allow it.
- [ ] Evaluate ensembles on the fixed test split.
- [ ] Generate ensemble comparison tables and plots.

## Phase 7 - LDA Topic Modelling

- [ ] Prepare traditional topic modelling corpus.
- [ ] Implement LDA pipeline.
- [ ] Tune topic counts and preprocessing.
- [ ] Generate topic tables and visualizations.
- [ ] Interpret topics for paper-ready analysis.

## Phase 8 - BERTopic

- [ ] Prepare embedding-based topic modelling workflow.
- [ ] Implement BERTopic experiments.
- [ ] Generate topic clusters, labels, and visualizations.
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
