# Project Map

| Phase | Goal | Main scripts/notebooks | Input | Output | Compute environment | Storage destination |
|---|---|---|---|---|---|---|
| Phase 0 | Infrastructure refactor | `scripts/check_project_structure.py`, `notebooks/colab/00_bootstrap.md` | Repo files, path configs | Validated repo/workspace layout | Local VS Code | Repo for source, workspace folders for artifacts |
| Phase 1 | Dataset pipeline | Dataset loading scripts, local notebooks | `celvaigh/periad`, `NicholasSynovic/ModifiedVEAA` | Dataset validation, stats, tokenization analysis | Local for checks, Colab if needed | `datasets/`, `artifacts/phase1/` |
| Phase 2 | Encoder baselines | Encoder training/evaluation scripts | PERIAD train/test splits | BERT/RoBERTa metrics, baseline outputs | Colab Pro for training, local for aggregation | `checkpoints/`, `artifacts/phase2/`, `exports/` |
| Phase 3 | Advanced encoders | Advanced encoder configs/scripts | PERIAD train/test splits | DeBERTa/ModernBERT metrics, LoRA/context analysis | Colab Pro | `checkpoints/`, `artifacts/phase3/`, `exports/` |
| Phase 4 | Decoder prompting | Prompt notebooks/scripts, parser scripts | Prompt templates, PERIAD test data | Decoder predictions, invalid-output analysis | Colab Pro or local depending model/API access | `artifacts/phase4/`, `exports/` |
| Phase 5 | QLoRA fine-tuning | QLoRA training/evaluation scripts | Instruction dataset, decoder models | Fine-tuned adapters, evaluation metrics | Colab Pro, A100 preferred | `checkpoints/`, `artifacts/phase5/`, `exports/` |
| Phase 6 | Ensemble | Ensemble scripts/notebooks | Best encoder outputs, best decoder outputs | Majority-vote results, hard-author analysis | Local | `artifacts/phase6/`, `exports/` |
| Phase 7 | LDA topic modelling | Topic modelling scripts/notebooks | Cleaned corpus text | LDA models, coherence results, heatmaps | Local or Colab | `artifacts/phase7/`, `exports/` |
| Phase 8 | BERTopic | BERTopic notebooks/scripts | Corpus text, sentence embeddings | Neural topic model, HTML visualizations | Colab Pro recommended | `artifacts/phase8/`, `exports/` |
| Phase 9 | Topic-aware classification | Fusion/classification scripts | Topic vectors, encoder embeddings | Base vs topic-aware comparison | Colab Pro for training, local for analysis | `checkpoints/`, `artifacts/phase9/`, `exports/` |
| Phase 10 | Final research package | Aggregation/report scripts | All validated phase outputs | Tables, plots, paper-ready exports, `benchmark_summary.md` | Local | `exports/`, `artifacts/phase10/` |
