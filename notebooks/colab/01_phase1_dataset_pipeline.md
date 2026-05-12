# Colab Phase 1 Dataset Pipeline Notebook Cells

Create a new Colab notebook and copy each fenced block into a separate notebook cell.

## 1. Clone Repository

```python
from pathlib import Path

WORKSPACE_ROOT = Path("/content/Authorship-Attribution")
REPO_ROOT = WORKSPACE_ROOT / "repo" / "Authorship-Attribution-in-Victorian-Periodicals"

WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
(WORKSPACE_ROOT / "repo").mkdir(parents=True, exist_ok=True)

if not REPO_ROOT.exists():
    !git clone https://github.com/IamTaoHu/Authorship-Attribution-in-Victorian-Periodicals.git {REPO_ROOT}
else:
    %cd {REPO_ROOT}
    !git pull

%cd {REPO_ROOT}
```

## 2. Install Requirements

```python
%cd {REPO_ROOT}
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

## 3. Optional HF_TOKEN Login

```python
import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN is not set. Public datasets and tokenizers may still work.")
```

## 4. Configure Colab Paths

```python
from pathlib import Path

WORKSPACE_ROOT = Path("/content/Authorship-Attribution")
REPO_ROOT = WORKSPACE_ROOT / "repo" / "Authorship-Attribution-in-Victorian-Periodicals"

for folder in ("artifacts", "checkpoints", "datasets", "exports", "backups"):
    (WORKSPACE_ROOT / folder).mkdir(parents=True, exist_ok=True)

paths_local = REPO_ROOT / "configs" / "paths.local.yaml"
paths_local.write_text(
    f"""paths:
  workspace_root: {WORKSPACE_ROOT}
  repo_root: {REPO_ROOT}
  artifacts_root: {WORKSPACE_ROOT / "artifacts"}
  checkpoints_root: {WORKSPACE_ROOT / "checkpoints"}
  datasets_root: {WORKSPACE_ROOT / "datasets"}
  exports_root: {WORKSPACE_ROOT / "exports"}

runtime:
  default_environment: colab
  cloud_training_environment: colab
""",
    encoding="utf-8",
)
print(paths_local.read_text())
```

## 5. Run Structure Check

```python
%cd {REPO_ROOT}
!python -m py_compile src/utils/paths.py
!python scripts/check_project_structure.py
```

## 6. Run Phase 1 Dataset Pipeline

```python
%cd {REPO_ROOT}
CONFIG = "configs/phase1/dataset_pipeline.yaml"

!python -m py_compile src/data/load_hf_datasets.py src/data/prepare_phase1_dataset.py src/data/validate_periad.py src/data/dataset_statistics.py src/data/tokenization_analysis.py src/visualization/plot_phase1_dataset.py scripts/check_phase1_outputs.py
!python src/data/validate_periad.py --config {CONFIG}
!python src/data/prepare_phase1_dataset.py --config {CONFIG}
!python src/data/dataset_statistics.py --config {CONFIG}
!python src/data/tokenization_analysis.py --config {CONFIG} --skip_missing_tokenizers --max_samples 500
!python src/visualization/plot_phase1_dataset.py --config {CONFIG}
!python scripts/check_phase1_outputs.py --config {CONFIG}
```

## 7. Zip Phase 1 Artifacts And Processed Datasets

```python
%cd {WORKSPACE_ROOT}
!zip -r phase1_artifacts.zip artifacts/phase1 datasets/processed/periad datasets/processed/veaa
```

## 8. Download Or Sync Outputs

```python
from google.colab import files

files.download(str(WORKSPACE_ROOT / "phase1_artifacts.zip"))

print("Optional: mount Google Drive and copy phase1_artifacts.zip there for persistent storage.")
print("Do not commit datasets, artifacts, checkpoints, outputs, or model files to Git.")
print("Phase 1 does not train or fine-tune any model.")
```
