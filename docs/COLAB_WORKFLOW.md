# Colab Workflow

Colab Pro is the preferred heavy-training environment for this project. Local VS Code remains the development center; Colab is used when GPU compute is needed.

## Why Colab Pro Instead Of Kaggle

The previous Kaggle QLoRA attempt exposed practical workflow risks:

- GPU quota limits interrupted experiments.
- Checkpoints were not reliable as permanent storage.
- Adapter path mismatch made evaluation brittle.
- Malformed training logs caused plotting failures.
- Gated model access, especially for Gemma, required cleaner Hugging Face authentication.

Colab Pro is preferred because it offers a more flexible interactive workflow, easier Drive integration, simpler secret handling, and better access to stronger GPUs when available.

## Clone The Repository

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

## Install Dependencies

```python
%cd {REPO_ROOT}
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

## Login To Hugging Face

Use a Colab secret or environment variable named `HF_TOKEN`.

```python
import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is not set.")

login(token=hf_token)
```

Llama and Gemma may require gated access approval on Hugging Face before the token can download them.

## Configure Paths In Colab

Create a Colab-local `configs/paths.local.yaml`. This file must not be committed.

```python
from pathlib import Path

WORKSPACE_ROOT = Path("/content/Authorship-Attribution")
REPO_ROOT = WORKSPACE_ROOT / "repo" / "Authorship-Attribution-in-Victorian-Periodicals"

for folder in ("artifacts", "checkpoints", "datasets", "exports", "backups"):
    (WORKSPACE_ROOT / folder).mkdir(parents=True, exist_ok=True)

(REPO_ROOT / "configs" / "paths.local.yaml").write_text(
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
```

## Save Checkpoints

Write checkpoints to the configured `checkpoints_root`, not into the repo:

```text
/content/Authorship-Attribution/checkpoints/
```

Use experiment-specific subfolders, for example:

```text
checkpoints/phase5/mistral_7b_qlora/
```

## Zip And Download Artifacts

```python
!zip -r /content/phase5_artifacts.zip /content/Authorship-Attribution/artifacts/phase5
from google.colab import files
files.download("/content/phase5_artifacts.zip")
```

## Sync Outputs Back To Local Storage

Recommended options:

- Download zipped artifacts from Colab and place them under local `artifacts/` or `exports/`.
- Mount Google Drive and copy artifacts/checkpoints there during long runs.
- Upload selected non-large result tables or summaries to Git only after confirming they are source-controlled deliverables, not generated training output.

## Avoid Committing Large Files

Before committing:

```bash
git status --short
```

Do not commit:

- `configs/paths.local.yaml`
- checkpoints
- downloaded datasets
- generated artifacts
- logs
- `.pt`, `.pth`, `.bin`, `.safetensors`, `.ckpt`, `.onnx`

## Recommended GPU Usage

- T4 is acceptable for encoder baselines and small QLoRA smoke tests.
- A100 is preferred for full QLoRA on 7B-9B decoder models.
