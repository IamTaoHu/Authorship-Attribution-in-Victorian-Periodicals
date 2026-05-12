$ErrorActionPreference = "Stop"

python -m py_compile `
  src/training/train_encoder_baseline.py `
  src/evaluation/evaluate_encoder_outputs.py `
  src/evaluation/aggregate_phase2_results.py `
  src/visualization/plot_phase2_results.py `
  scripts/check_phase2_outputs.py

python scripts/check_project_structure.py
python src/training/train_encoder_baseline.py --config configs/phase2/bert_base_smoke.yaml
python src/evaluation/aggregate_phase2_results.py
python src/visualization/plot_phase2_results.py
python scripts/check_phase2_outputs.py
