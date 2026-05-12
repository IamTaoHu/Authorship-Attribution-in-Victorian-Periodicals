# Project Structure

This project uses a split workspace: source-controlled files live in the Git repository, while generated data, artifacts, exports, and checkpoints live outside the repository.

## Workspace Tree

```text
Authorship-Attribution/
  artifacts/
  backups/
  checkpoints/
  datasets/
  exports/
  repo/
    Authorship-Attribution-in-Victorian-Periodicals/
```

Workspace root:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution
```

The workspace root can be opened in VS Code so both the repo and non-git storage folders are visible. Source code should still be written inside:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/repo/Authorship-Attribution-in-Victorian-Periodicals
```

## Repository Tree

```text
Authorship-Attribution-in-Victorian-Periodicals/
  configs/
  docs/
  notebooks/
    local/
    colab/
    archive/
  prompts/
  scripts/
  src/
    data/
    evaluation/
    training/
    utils/
    visualization/
  tests/
  README.md
  requirements.txt
  .gitignore
```

Python commands for scripts should normally be run from the repo root:

```powershell
cd C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/repo/Authorship-Attribution-in-Victorian-Periodicals
python scripts/check_project_structure.py
```

## What Belongs In Git

Commit:

- Source code under `src/`
- Lightweight scripts under `scripts/`
- Config templates under `configs/`
- Documentation under `docs/`
- Prompt templates under `prompts/`
- Local and Colab notebooks that are intended to be reused
- Tests
- `README.md`, `requirements.txt`, and `.gitignore`

Do not commit:

- Downloaded datasets
- Processed datasets
- Generated outputs
- Training logs
- Model checkpoints
- Adapter checkpoints
- `.pt`, `.pth`, `.bin`, `.safetensors`, `.ckpt`, or `.onnx` files
- Machine-specific secrets or paths

## Path Config Files

`configs/paths.example.yaml` is the tracked template. It uses environment-agnostic placeholder paths and documents the required keys.

`configs/paths.local.yaml` is the local machine-specific config. It is ignored by Git and should not be committed.

The path manager in `src/utils/paths.py` loads `configs/paths.local.yaml` by default. If the local file is missing, it falls back to `configs/paths.example.yaml` with a warning. Directory-producing helpers must refuse to create directories when placeholder paths such as `/path/to/...` are detected.

## Hugging Face Usage

Hugging Face is the primary source for datasets and pretrained models.

Datasets:

- `celvaigh/periad`
- `NicholasSynovic/ModifiedVEAA`

Models:

- `bert-base-uncased`
- `roberta-base`
- `roberta-large`
- `microsoft/deberta-v3-base`
- `answerdotai/ModernBERT-base`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-2-9b-it`
- `sentence-transformers/all-MiniLM-L6-v2`

Llama and Gemma may require gated access and a valid `HF_TOKEN`.
