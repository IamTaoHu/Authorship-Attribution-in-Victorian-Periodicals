# Roadmap: Phase 0 To Phase 10

## Phase 0 - Infrastructure Refactor

Objective: Standardize the repo and workspace for local development, Colab training, Hugging Face resources, and external artifact storage.

Main tasks:

- Finalize repo structure.
- Add path configs and path manager.
- Add dataset/model configs.
- Update `.gitignore`.
- Add local structure validation.
- Add Colab bootstrap documentation.

Deliverables:

- Standard repo folders.
- `configs/paths.example.yaml`
- local-only `configs/paths.local.yaml`
- `configs/datasets.yaml`
- `configs/models.yaml`
- `src/utils/paths.py`
- `scripts/check_project_structure.py`
- Colab bootstrap notebook or markdown.

Success criteria:

- Local structure check passes.
- `configs/paths.local.yaml` is ignored by Git.
- Generated outputs and checkpoints are excluded from Git.

## Phase 1 - Dataset Pipeline

Objective: Load, validate, and summarize the datasets used for authorship attribution and benchmark comparison.

Main tasks:

- Load PERIAD from Hugging Face.
- Validate train/test split counts.
- Validate six canonical authors.
- Load ModifiedVEAA from Hugging Face.
- Generate dataset statistics.
- Generate tokenization analysis.

Deliverables:

- Dataset loading scripts.
- Dataset validation reports.
- Tokenization analysis.
- Dataset summary tables.

Success criteria:

- PERIAD train count is 8279.
- PERIAD test count is 3549.
- Six canonical authors are validated.
- Dataset outputs are stored outside Git.

## Phase 2 - Encoder Baselines

Objective: Establish baseline authorship classification results with standard encoder models.

Main tasks:

- Train/evaluate BERT-base.
- Train/evaluate RoBERTa-base.
- Train/evaluate RoBERTa-large.
- Generate metrics.
- Test majority vote where applicable.

Deliverables:

- Baseline model outputs.
- Metrics tables.
- Initial benchmark plots.

Success criteria:

- Baseline metrics are reproducible from scripts/configs.
- Outputs are stored under external artifacts/checkpoints.

## Phase 3 - Advanced Encoders

Objective: Improve encoder-based classification beyond BERT-base and initial baselines.

Main tasks:

- Train/evaluate DeBERTa-v3-base.
- Train/evaluate ModernBERT-base.
- Run DeBERTa LoRA rank ablation.
- Analyze context length effects.

Deliverables:

- Advanced encoder results.
- LoRA ablation table.
- Context length analysis.

Success criteria:

- Advanced encoder results are comparable with Phase 2.
- Best encoder candidate is identifiable for the ensemble phase.

## Phase 4 - Decoder Prompting

Objective: Evaluate instruction-tuned decoder models through prompting for authorship classification.

Main tasks:

- Build prompt templates.
- Build output parser.
- Run Mistral prompting.
- Run Llama prompting.
- Run Gemma prompting.
- Analyze invalid outputs.

Deliverables:

- Prompt templates.
- Parsed prediction outputs.
- Decoder prompting metrics.
- Invalid-output analysis.

Success criteria:

- Prompted decoder predictions are parseable and measurable.
- Gated model access issues are documented when encountered.

## Phase 5 - QLoRA Decoder Fine-Tuning

Objective: Fine-tune decoder models with QLoRA for authorship classification.

Main tasks:

- Build instruction dataset.
- Fine-tune Mistral with QLoRA.
- Fine-tune Llama with QLoRA.
- Fine-tune Gemma with QLoRA.
- Run LoRA rank ablation.
- Evaluate adapters.
- Aggregate results.

Deliverables:

- Instruction dataset.
- QLoRA adapters/checkpoints.
- Evaluation outputs.
- Aggregated metrics.

Success criteria:

- Evaluation uses the correct adapter paths.
- Checkpoints are preserved outside Git.
- No final results are claimed without validated artifacts.

## Phase 6 - Ensemble

Objective: Combine the best encoder and best decoder predictions to test whether ensemble voting improves attribution.

Main tasks:

- Select best encoder.
- Select best decoder.
- Run majority vote.
- Analyze difficult authors and error overlap.

Deliverables:

- Ensemble predictions.
- Ensemble metrics.
- Hard-author analysis.

Success criteria:

- Ensemble performance is compared against individual models.
- Error analysis identifies where ensemble helps or fails.

## Phase 7 - LDA Topic Modelling

Objective: Model corpus themes with classical topic modelling and relate topics to authorship errors.

Main tasks:

- Run LDA grid search for k=10,20,30,40.
- Compute coherence `c_v`.
- Generate heatmaps.
- Analyze topic entropy versus errors.

Deliverables:

- LDA models.
- Coherence tables.
- Topic heatmaps.
- Topic/error analysis.

Success criteria:

- A defensible topic count is selected.
- Topic features are interpretable enough for analysis.

## Phase 8 - BERTopic

Objective: Run neural topic modelling and export interactive visualizations.

Main tasks:

- Generate embeddings.
- Run UMAP.
- Run HDBSCAN.
- Build BERTopic model.
- Export HTML visualizations.

Deliverables:

- BERTopic outputs.
- Interactive HTML visualizations.
- Topic summaries.

Success criteria:

- HTML visualizations can be opened and shared.
- Neural topics are compared with LDA topics.

## Phase 9 - Topic-Aware Classification

Objective: Test whether topic information improves authorship classification.

Main tasks:

- Extract topic vectors.
- Concatenate topic vectors with `[CLS]` embeddings.
- Train/evaluate topic-aware classifier.
- Compare base versus topic-aware results.

Deliverables:

- Topic vector features.
- Topic-aware classifier outputs.
- Comparative metrics.

Success criteria:

- Topic-aware results are directly comparable to base encoder results.
- Any gain or regression is documented clearly.

## Phase 10 - Final Research Package

Objective: Prepare paper-ready outputs and final reproducible materials.

Main tasks:

- Build unified benchmark tables.
- Generate final plots.
- Export paper-ready files.
- Write `benchmark_summary.md`.

Deliverables:

- Final result tables.
- Final plots.
- Paper-ready exports.
- Scripts/configs needed to reproduce results.
- Interactive topic visualizations.

Success criteria:

- Final package is internally consistent.
- No unsupported results are claimed.
- Large artifacts remain outside Git.
