# Authorship Attribution in Victorian Periodicals

This repository supports the internship project **Authorship Attribution in Victorian Periodicals: Advancing LLM Classification and Thematic Modelling for the Saturday Review (1855-1875)**.

The workflow is split across local development, Colab training, and external artifact storage:

- **VS Code/local machine** is the development center for source code, configs, notebooks, documentation, lightweight tests, aggregation, and plotting.
- **Colab Pro** is the compute environment for heavy training, including encoder fine-tuning, decoder prompting, QLoRA, BERTopic, and ensemble experiments.
- **Hugging Face** is the primary source for datasets and pretrained models.
- **Local storage outside this repo** is the primary location for artifacts, checkpoints, datasets, exports, and backups.

## Repository Layout

Tracked project files live in this repo:

```text
configs/
docs/
notebooks/
prompts/
scripts/
src/
tests/
README.md
requirements.txt
.gitignore
```

Generated outputs, model checkpoints, downloaded datasets, and large model files should not be committed.

Expected workspace-level folders outside the repo:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/artifacts/
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/checkpoints/
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/datasets/
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/exports/
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/backups/
```

## Path Configuration

Copy or adapt `configs/paths.example.yaml` for each environment. On this local machine, `configs/paths.local.yaml` contains machine-specific paths and is ignored by Git.

Do not commit `configs/paths.local.yaml`.

## Structure Check

From the repo root:

```powershell
python -m py_compile src/utils/paths.py
python scripts/check_project_structure.py
git status --short
```

The structure check verifies required folders/configs, imports `src.utils.paths`, prints resolved paths, and creates a test artifact directory at:

```text
artifacts_root/phase0/test
```

## Colab Bootstrap

Use `notebooks/colab/00_bootstrap.md` as the source for a Colab bootstrap notebook. It includes cells to clone the repo, install requirements, optionally mount Google Drive, authenticate to Hugging Face with `HF_TOKEN`, check GPU availability, and configure Colab paths.

Heavy artifacts and checkpoints created in Colab should be zipped and downloaded or synced to external storage. Do not commit generated outputs, checkpoints, `.safetensors`, `.bin`, or datasets.
