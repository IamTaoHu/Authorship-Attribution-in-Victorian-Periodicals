# Roadmap: Phase 0 to Phase 10

This roadmap separates completed work from planned work. Future phases are not complete until validated artifacts exist.

Related docs: [CURRENT_STATE.md](CURRENT_STATE.md), [TASK_BOARD.md](TASK_BOARD.md), [PROJECT_MAP.md](PROJECT_MAP.md).

## Phase 0 - Infrastructure Refactor

Objective: establish a reproducible local, Colab, GitHub, Hugging Face, and external-storage workflow.

Main tasks:

- define workspace and repo layout
- configure path management
- add validation scripts
- establish Colab bootstrap workflow

Deliverables:

- path configs
- structure checks
- Colab bootstrap documentation
- external artifact/checkpoint conventions

Success criteria:

- source remains in Git
- generated outputs remain outside Git
- local and Colab workflows can resolve paths consistently

## Phase 1 - Dataset Pipeline

Objective: prepare and validate the PERIAD authorship attribution dataset.

Main tasks:

- load PERIAD from Hugging Face
- validate row counts and labels
- generate dataset statistics
- generate dataset plots and reports

Deliverables:

- processed train/test data
- validation reports
- dataset summary tables and plots

Success criteria:

- expected train/test row counts are present
- canonical six author labels are present
- Phase 1 artifacts exist under external storage

## Phase 2 - Encoder Baselines

Objective: establish BERT and RoBERTa baseline results.

Main tasks:

- train/evaluate baseline encoders
- save predictions and metrics
- aggregate results
- generate baseline plots

Deliverables:

- BERT-base, RoBERTa-base, and RoBERTa-large outputs
- benchmark tables
- report and plots

Success criteria:

- final baseline runs are complete
- final outputs are stored under `artifacts/phase2`

## Phase 3 - Advanced Encoders

Objective: evaluate stronger modern encoder models.

Main tasks:

- run DeBERTa-v3-base and ModernBERT-base benchmarks
- preserve final Colab outputs
- aggregate valid runs only
- generate comparison plots

Deliverables:

- final advanced encoder predictions and metrics
- final comparison tables
- per-author and confusion matrix visualizations

Success criteria:

- final advanced encoder artifacts exist under `artifacts/phase3`
- diagnostic/debug artifacts remain non-final

## Phase 4 - Decoder Prompting

Objective: evaluate instruction-tuned decoder prompting baselines.

Main tasks:

- implement prompt templates and parser
- run zero-shot and few-shot prompting for Mistral, Llama 3, and Gemma 2
- aggregate only final successful runs
- generate final-only plots and report

Deliverables:

- six final decoder benchmark runs
- final-only aggregate tables
- final-only report and plots

Success criteria:

- final-only outputs contain exactly the six expected runs
- TinyLlama smoke outputs are diagnostic only and excluded

## Phase 5 - QLoRA Decoder Fine-Tuning

Objective: fine-tune decoder models with QLoRA for authorship attribution.

Main tasks:

- finalize QLoRA data formatting
- configure training and adapter storage
- run Colab Pro training
- evaluate final adapters on the fixed test split
- aggregate final QLoRA results

Deliverables:

- planned QLoRA adapters/checkpoints
- planned evaluation outputs
- planned Phase 5 tables and plots

Success criteria:

- validated Phase 5 artifacts exist outside the repo
- no checkpoint or adapter files are committed to Git

## Phase 6 - Ensemble

Objective: combine complementary model outputs.

Main tasks:

- select candidate Phase 2-5 outputs
- implement ensemble rules
- evaluate ensemble predictions
- compare against individual models

Deliverables:

- planned ensemble predictions
- planned ensemble metrics, tables, and plots

Success criteria:

- ensemble outputs improve or clarify tradeoffs against prior phases

## Phase 7 - LDA Topic Modelling

Objective: build traditional topic models for interpretability.

Main tasks:

- prepare text corpus
- train LDA models
- tune topic counts and preprocessing
- generate topic summaries

Deliverables:

- planned LDA topic tables
- planned topic visualizations
- planned interpretation notes

Success criteria:

- coherent topics are produced and documented for analysis

## Phase 8 - BERTopic

Objective: build neural topic models using sentence embeddings.

Main tasks:

- prepare embedding workflow
- run BERTopic experiments
- label and inspect topics
- compare with LDA outputs

Deliverables:

- planned BERTopic clusters
- planned visualizations and topic summaries

Success criteria:

- BERTopic outputs are interpretable and reusable for topic-aware classification

## Phase 9 - Topic-Aware Classification

Objective: evaluate whether topic information improves authorship attribution.

Main tasks:

- integrate topic features with classifier inputs or outputs
- train/evaluate topic-aware variants
- compare with best prior classifiers
- analyze gains and failure cases

Deliverables:

- planned topic-aware model outputs
- planned comparison tables and plots

Success criteria:

- topic-aware results are evaluated on the fixed test split and compared to prior phases

## Phase 10 - Final Research Package

Objective: produce paper-ready final outputs.

Main tasks:

- freeze final benchmark tables
- freeze final visualizations
- write reproducibility notes
- prepare final narrative and appendices

Deliverables:

- final research package
- final exports
- paper-ready figures and tables

Success criteria:

- all reported results are traceable to validated artifacts
- large files remain outside Git
