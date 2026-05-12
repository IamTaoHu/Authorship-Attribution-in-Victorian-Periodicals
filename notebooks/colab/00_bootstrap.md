# Colab Bootstrap Notebook Cells

Create a new Colab notebook and copy each fenced block into a separate notebook cell.

## 1. Clone Repository

```python
from pathlib import Path
import os

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

## 3. Optional Google Drive Mount

```python
USE_DRIVE = False

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
```

## 4. Hugging Face Login With HF_TOKEN

```python
import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN is not set. Add it in Colab secrets or set os.environ['HF_TOKEN'].")
```

## 5. GPU Check

```python
!nvidia-smi

import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 6. Path Configuration For Colab

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

## 7. Verify Project Structure

```python
%cd {REPO_ROOT}
!python -m py_compile src/utils/paths.py
!python scripts/check_project_structure.py
```

## 8. Artifact Handling Reminder

```python
print(
    "Do not commit heavy artifacts, checkpoints, safetensors, bin files, or datasets. "
    "Zip and download them, sync them to Drive, or upload appropriate outputs to external storage."
)
```
