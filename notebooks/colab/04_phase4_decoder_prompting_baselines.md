# Phase 4 Decoder Prompting Baselines

This Colab workflow runs decoder prompting baselines for PERIAD authorship attribution. Full decoder inference should run in Colab; local VS Code is for code checks, smoke validation, aggregation, and plotting.

## 1. Clone or pull the repository

```bash
cd /content
if [ ! -d Authorship-Attribution ]; then
  mkdir -p Authorship-Attribution/repo
fi
cd /content/Authorship-Attribution/repo
if [ ! -d Authorship-Attribution-in-Victorian-Periodicals ]; then
  git clone <YOUR_REPO_URL> Authorship-Attribution-in-Victorian-Periodicals
fi
cd Authorship-Attribution-in-Victorian-Periodicals
git pull
```

## 2. Install requirements

```bash
pip install -r requirements.txt
```

## 3. Authenticate with Hugging Face

```python
import os
from huggingface_hub import login

login(token=os.environ.get("HF_TOKEN"))
```

If `HF_TOKEN` is not already set in Colab secrets, set it before running gated models.

## 4. Write Colab path config

```bash
cat > configs/paths.local.yaml <<'YAML'
paths:
  workspace_root: /content/Authorship-Attribution
  repo_root: /content/Authorship-Attribution/repo/Authorship-Attribution-in-Victorian-Periodicals
  artifacts_root: /content/Authorship-Attribution/artifacts
  checkpoints_root: /content/Authorship-Attribution/checkpoints
  datasets_root: /content/Authorship-Attribution/datasets
  exports_root: /content/Authorship-Attribution/exports

runtime:
  default_environment: colab
  cloud_training_environment: colab
YAML
```

## 5. Prepare or upload PERIAD processed split

Phase 4 expects these files outside Git:

```text
/content/Authorship-Attribution/datasets/processed/periad/train.csv
/content/Authorship-Attribution/datasets/processed/periad/test.csv
/content/Authorship-Attribution/datasets/processed/periad/label_map.json
```

If they are not already present, run the Phase 1 preparation command from the repo documentation before starting Phase 4.

## 6. Tiny smoke run

```bash
python scripts/run_phase4_prompting.py --config configs/phase4/tinyllama_smoke.yaml --overwrite
```

## 7. Full decoder prompting runs

```bash
python scripts/run_phase4_prompting.py --config configs/phase4/mistral_zero_shot.yaml
python scripts/run_phase4_prompting.py --config configs/phase4/mistral_few_shot.yaml
python scripts/run_phase4_prompting.py --config configs/phase4/llama3_zero_shot.yaml
python scripts/run_phase4_prompting.py --config configs/phase4/llama3_few_shot.yaml
python scripts/run_phase4_prompting.py --config configs/phase4/gemma2_zero_shot.yaml
python scripts/run_phase4_prompting.py --config configs/phase4/gemma2_few_shot.yaml
```

Runs resume from existing `predictions.csv`. To rebuild few-shot examples:

```bash
python scripts/run_phase4_prompting.py --config configs/phase4/mistral_few_shot.yaml --rebuild_few_shot_examples
```

## 8. Aggregate results

```bash
python -m src.evaluation.aggregate_phase4_results
```

## 9. Plot results

```bash
python src/visualization/plot_phase4_results.py
```

## 10. Check outputs

```bash
python scripts/check_phase4_outputs.py --phase4_dir /content/Authorship-Attribution/artifacts/phase4
```

## 11. Zip artifacts

```bash
cd /content/Authorship-Attribution
zip -r phase4_artifacts.zip artifacts/phase4
```

## 12. Download artifacts

```python
from google.colab import files
files.download('/content/Authorship-Attribution/phase4_artifacts.zip')
```
