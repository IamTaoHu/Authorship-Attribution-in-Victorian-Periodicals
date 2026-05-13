# Colab Notebook: Phase 3 DeBERTa/ModernBERT Final Benchmark

Use this markdown file as the source for the matching Colab notebook. Run cells in order. The notebook uses existing repository scripts and configs; it does not duplicate training logic.

## 1. Setup / Drive mount

```python
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

DRIVE_WORKSPACE = Path("/content/drive/MyDrive/Authorship-Attribution")
REPO_PARENT = DRIVE_WORKSPACE / "repo"
REPO = REPO_PARENT / "Authorship-Attribution-in-Victorian-Periodicals"

DRIVE_WORKSPACE.mkdir(parents=True, exist_ok=True)
REPO_PARENT.mkdir(parents=True, exist_ok=True)

print("Drive workspace:", DRIVE_WORKSPACE)
```

## 2. Repo path setup

```python
REPO_URL = "https://github.com/IamTaoHu/Authorship-Attribution-in-Victorian-Periodicals.git"

if not REPO.exists():
    !git clone {REPO_URL} "{REPO}"
else:
    %cd "{REPO}"
    !git pull

%cd "{REPO}"
print("Repo:", REPO)
```

## 3. Install dependencies

```python
%cd "{REPO}"
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

## 4. Optional Hugging Face login

```python
try:
    from google.colab import userdata
    from huggingface_hub import login

    hf_token = userdata.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Hugging Face login successful.")
    else:
        print("HF_TOKEN not found in Colab Secrets. Public models may still download.")
except Exception as exc:
    print("Skipping Hugging Face login:", exc)
```

## 5. GPU and version check

```python
!nvidia-smi

import torch
import transformers

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 6. Configure Drive-backed project paths

```python
from pathlib import Path

for folder in ("artifacts", "checkpoints", "datasets", "exports", "backups"):
    (DRIVE_WORKSPACE / folder).mkdir(parents=True, exist_ok=True)

paths_yaml = f"""paths:
  workspace_root: {DRIVE_WORKSPACE}
  repo_root: {REPO}
  artifacts_root: {DRIVE_WORKSPACE / "artifacts"}
  checkpoints_root: {DRIVE_WORKSPACE / "checkpoints"}
  datasets_root: {DRIVE_WORKSPACE / "datasets"}
  exports_root: {DRIVE_WORKSPACE / "exports"}

runtime:
  default_environment: colab
  cloud_training_environment: colab
"""

paths_local = REPO / "configs" / "paths.local.yaml"
paths_local.write_text(paths_yaml, encoding="utf-8")
print(paths_yaml)
```

## 7. Dataset setup from Hugging Face

PERIAD processed files are required before Phase 3 training. If they are not already present in Drive, this notebook prepares them from Hugging Face using the existing project dataset config and Phase 1 standardization logic.

The PERIAD dataset id must be set in `configs/datasets.yaml` if processed PERIAD is not already available. Current expected ids:

- PERIAD: `celvaigh/periad`
- VEAA: `NicholasSynovic/ModifiedVEAA`

Manual command examples:

```bash
python scripts/prepare_hf_datasets.py --dataset periad
python scripts/prepare_hf_datasets.py --dataset veaa
```

```python
import yaml

%cd "{REPO}"

datasets_config = yaml.safe_load((REPO / "configs" / "datasets.yaml").read_text(encoding="utf-8"))
PERIAD_HF_DATASET_ID = datasets_config["datasets"]["periad"]["name"]
VEAA_HF_DATASET_ID = datasets_config["datasets"]["veaa"]["name"]

print("PERIAD_HF_DATASET_ID:", PERIAD_HF_DATASET_ID)
print("VEAA_HF_DATASET_ID:", VEAA_HF_DATASET_ID)

if not PERIAD_HF_DATASET_ID:
    raise ValueError("PERIAD_HF_DATASET_ID must be set in configs/datasets.yaml when processed PERIAD is missing.")

PERIAD_DIR = DRIVE_WORKSPACE / "datasets" / "processed" / "periad"
required_files = ["train.csv", "test.csv", "label_map.json"]
missing = [name for name in required_files if not (PERIAD_DIR / name).exists()]

print("PERIAD directory:", PERIAD_DIR)
if missing:
    print("Missing processed PERIAD files:", missing)
    print("Preparing canonical PERIAD split from Hugging Face.")
    !python scripts/prepare_hf_datasets.py --dataset periad

missing_after_prepare = [name for name in required_files if not (PERIAD_DIR / name).exists()]
if missing_after_prepare:
    raise FileNotFoundError(
        "PERIAD preparation did not create required files: "
        + ", ".join(str(PERIAD_DIR / name) for name in missing_after_prepare)
    )

for name in required_files:
    path = PERIAD_DIR / name
    print(name, path.stat().st_size, "bytes")

print("Optional VEAA preparation command:")
print("python scripts/prepare_hf_datasets.py --dataset veaa")
```

## 8. Validate scripts and final configs

```python
%cd "{REPO}"
!python -m py_compile scripts/prepare_hf_datasets.py
!python -m py_compile scripts/train_phase3_encoder.py
!python -m py_compile scripts/check_phase3_outputs.py
!python scripts/check_phase3_outputs.py --phase3_dir artifacts/phase3 --check_config_only
```

## 9. Run DeBERTa final benchmark

```python
%cd "{REPO}"
DEBERTA_CONFIG = "configs/phase3/deberta_v3_base_colab_final.yaml"
!python scripts/train_phase3_encoder.py --config {DEBERTA_CONFIG}
```

## 10. Run ModernBERT final benchmark

```python
%cd "{REPO}"
MODERNBERT_CONFIG = "configs/phase3/modernbert_base_colab_final.yaml"
!python scripts/train_phase3_encoder.py --config {MODERNBERT_CONFIG}
```

## 11. Validate outputs

```python
%cd "{REPO}"
!python scripts/check_phase3_outputs.py --phase3_dir artifacts/phase3
```

## 12. Display final results table

```python
import pandas as pd

results_path = DRIVE_WORKSPACE / "artifacts" / "phase3" / "tables" / "advanced_encoder_results.csv"
results = pd.read_csv(results_path)
final_results = results[results["final_benchmark"].astype(str).str.lower().eq("true")]

display(
    final_results[
        [
            "run_name",
            "model_name",
            "accuracy",
            "macro_f1",
            "valid_run",
            "invalid_reason",
            "baseline_reference",
            "final_benchmark",
            "local_practical",
        ]
    ].sort_values("macro_f1", ascending=False)
)
```

## 13. Zip artifacts and final checkpoints

```python
%cd "{DRIVE_WORKSPACE}"

ARCHIVE_DIR = DRIVE_WORKSPACE / "exports"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

!zip -r "{ARCHIVE_DIR / 'phase3_final_artifacts.zip'}" artifacts/phase3
!zip -r "{ARCHIVE_DIR / 'phase3_final_checkpoints.zip'}" checkpoints/phase3/deberta_v3_base_seed42_colab_final checkpoints/phase3/modernbert_base_seed42_colab_final

print("Wrote:", ARCHIVE_DIR / "phase3_final_artifacts.zip")
print("Wrote:", ARCHIVE_DIR / "phase3_final_checkpoints.zip")
```
