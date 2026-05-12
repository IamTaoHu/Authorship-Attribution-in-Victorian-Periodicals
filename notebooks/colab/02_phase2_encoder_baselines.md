# Colab Notebook: Phase 2 Encoder Baselines

Use this markdown file as the source for a Colab notebook. Run cells in order.

## 1. Clone repository

```python
from pathlib import Path

WORKSPACE = Path("/content/Authorship-Attribution")
REPO_PARENT = WORKSPACE / "repo"
REPO = REPO_PARENT / "Authorship-Attribution-in-Victorian-Periodicals"
REPO_PARENT.mkdir(parents=True, exist_ok=True)

if not REPO.exists():
    !git clone https://github.com/IamTaoHu/Authorship-Attribution-in-Victorian-Periodicals.git "{REPO}"
%cd "{REPO}"
```

## 2. Install requirements

```python
!pip install -r requirements.txt
```

## 3. Optional Hugging Face login

```python
import os

if os.environ.get("HF_TOKEN"):
    !huggingface-cli login --token "$HF_TOKEN"
else:
    print("HF_TOKEN not set; public models and datasets will still work.")
```

## 4. Configure Colab paths

```python
from pathlib import Path

paths_yaml = f"""paths:
  workspace_root: {WORKSPACE}
  repo_root: {REPO}
  artifacts_root: {WORKSPACE / "artifacts"}
  checkpoints_root: {WORKSPACE / "checkpoints"}
  datasets_root: {WORKSPACE / "datasets"}
  exports_root: {WORKSPACE / "exports"}

runtime:
  default_environment: colab
  cloud_training_environment: colab
"""
Path("configs/paths.local.yaml").write_text(paths_yaml, encoding="utf-8")
print(paths_yaml)
```

## 5. Run structure check

```python
!python scripts/check_project_structure.py
```

## 6. Verify Phase 1 processed PERIAD data or regenerate Phase 1

```python
from pathlib import Path
import yaml

config = yaml.safe_load(Path("configs/paths.local.yaml").read_text())
periad_dir = Path(config["paths"]["datasets_root"]) / "processed" / "periad"
required = ["train.csv", "test.csv", "label_map.json"]
missing = [name for name in required if not (periad_dir / name).exists()]
print("PERIAD directory:", periad_dir)
print("Missing:", missing)
```

```python
if missing:
    !python src/data/prepare_phase1_dataset.py --config configs/phase1/dataset_pipeline.yaml
    !python src/data/dataset_statistics.py --config configs/phase1/dataset_pipeline.yaml
    !python src/data/tokenization_analysis.py --config configs/phase1/dataset_pipeline.yaml
    !python src/visualization/plot_phase1_dataset.py --config configs/phase1/dataset_pipeline.yaml
    !python scripts/check_phase1_outputs.py
else:
    print("Phase 1 processed PERIAD data exists.")
```

## 7. Train BERT-base

```python
!python src/training/train_encoder_baseline.py --config configs/phase2/bert_base.yaml
```

## 8. Train RoBERTa-base

```python
!python src/training/train_encoder_baseline.py --config configs/phase2/roberta_base.yaml
```

## 9. Train RoBERTa-large

```python
!python src/training/train_encoder_baseline.py --config configs/phase2/roberta_large.yaml
```

## 10. Aggregate results

```python
!python src/evaluation/aggregate_phase2_results.py
```

## 11. Plot results

```python
!python src/visualization/plot_phase2_results.py
```

## 12. Check outputs

```python
!python scripts/check_phase2_outputs.py
```

## 13. Zip artifacts and checkpoints

```python
from pathlib import Path

artifacts_zip = Path("/content/phase2_artifacts.zip")
checkpoints_zip = Path("/content/phase2_checkpoints.zip")
!cd "{WORKSPACE}" && zip -r "{artifacts_zip}" artifacts/phase2
!cd "{WORKSPACE}" && zip -r "{checkpoints_zip}" checkpoints/phase2
print(artifacts_zip)
print(checkpoints_zip)
```

## 14. Download or sync

```python
from google.colab import files

files.download("/content/phase2_artifacts.zip")
files.download("/content/phase2_checkpoints.zip")
```

Optional Google Drive sync:

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p "/content/drive/MyDrive/Authorship-Attribution/phase2"
!cp /content/phase2_artifacts.zip "/content/drive/MyDrive/Authorship-Attribution/phase2/"
!cp /content/phase2_checkpoints.zip "/content/drive/MyDrive/Authorship-Attribution/phase2/"
```
