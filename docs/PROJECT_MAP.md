# Project Map

This table maps the planned Phase 0-10 workflow. Planned scripts and notebooks are marked as planned when they do not currently exist.

Related docs: [TASK_BOARD.md](TASK_BOARD.md), [ROADMAP_PHASE0_TO_PHASE10.md](ROADMAP_PHASE0_TO_PHASE10.md), [COLAB_WORKFLOW.md](COLAB_WORKFLOW.md).

| Phase | Goal | Main scripts | Main notebooks | Inputs | Outputs | Compute environment | Storage destination |
|---|---|---|---|---|---|---|---|
| 0 | Refactor infrastructure and paths | `scripts/check_project_structure.py` | `notebooks/colab/00_bootstrap.md` | repo files, path configs | structure checks, path validation | local / Colab setup | repo for source, external workspace for generated checks |
| 1 | Build and validate dataset pipeline | `scripts/prepare_hf_datasets.py`, `scripts/check_phase1_outputs.py`, `src/data/*`, `src/visualization/plot_phase1_dataset.py` | `notebooks/colab/01_phase1_dataset_pipeline.md` | `celvaigh/periad`, `NicholasSynovic/ModifiedVEAA` | processed data, validation reports, dataset tables/plots | local or Colab | `datasets/`, `artifacts/phase1/` |
| 2 | Establish encoder baselines | `src/training/train_encoder_baseline.py`, `src/evaluation/aggregate_phase2_results.py`, `src/visualization/plot_phase2_results.py`, `scripts/check_phase2_outputs.py` | `notebooks/colab/02_phase2_encoder_baselines.ipynb` | processed PERIAD, BERT/RoBERTa configs | predictions, metrics, reports, plots | Colab for training, local for aggregation/plotting | `artifacts/phase2/`, `checkpoints/` |
| 3 | Benchmark advanced encoders | `scripts/train_phase3_encoder.py`, `src/evaluation/evaluate_encoder_outputs.py`, `src/visualization/plot_phase3_results.py`, `scripts/check_phase3_outputs.py` | `notebooks/colab/03_phase3_deberta_modernbert_final_benchmark.*` | processed PERIAD, DeBERTa/ModernBERT configs | final advanced encoder runs, tables, plots | Colab Pro | `artifacts/phase3/`, `checkpoints/` |
| 4 | Evaluate instruction-tuned decoder prompting | `scripts/run_phase4_prompting.py`, `src/evaluation/aggregate_phase4_results.py`, `src/visualization/plot_phase4_results.py`, `scripts/check_phase4_outputs.py` | `notebooks/colab/04_phase4_decoder_prompting_baselines.*` | processed PERIAD, Phase 4 configs, prompts, HF decoder models | six final decoder runs, final-only tables/report/plots | Colab Pro for inference, local for aggregation/plotting | `artifacts/phase4/` |
| 5 | Fine-tune decoder models with QLoRA | planned | planned Colab workflow | processed PERIAD, decoder base models, prompt/instruction format | planned QLoRA adapters, metrics, reports | Colab Pro, A100 preferred | `checkpoints/`, `artifacts/phase5/` |
| 6 | Build ensemble systems | planned | planned | Phase 2-5 predictions and metrics | planned ensemble predictions, metrics, plots | local / Colab as needed | `artifacts/phase6/` |
| 7 | Run LDA topic modelling | planned | planned | PERIAD text corpus | planned LDA topics, tables, visualizations | local or Colab | `artifacts/phase7/` |
| 8 | Run BERTopic topic modelling | planned | planned | PERIAD text corpus, sentence-transformer embeddings | planned BERTopic clusters and visualizations | Colab preferred for heavier jobs | `artifacts/phase8/` |
| 9 | Build topic-aware classifiers | planned | planned | classifier features plus Phase 7/8 topic features | planned topic-aware benchmark outputs | local / Colab as needed | `artifacts/phase9/` |
| 10 | Produce final research package | planned | planned | final outputs from all completed phases | final benchmark tables, plots, reports, reproducibility notes | local | `exports/`, `artifacts/phase10/` |
