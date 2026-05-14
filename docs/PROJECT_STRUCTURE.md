# Project Structure

This project uses a split workspace: source code lives in Git, while generated data, artifacts, checkpoints, exports, and backups live outside the repository.

Related docs: [CURRENT_STATE.md](CURRENT_STATE.md), [PROJECT_MAP.md](PROJECT_MAP.md), [COLAB_WORKFLOW.md](COLAB_WORKFLOW.md), [GPT_CONTEXT_SUMMARY.md](GPT_CONTEXT_SUMMARY.md).

## Workspace Layout

```text
Authorship-Attribution/
  artifacts/       # generated reports, tables, plots, predictions, metrics
  backups/         # local backup copies and historical external outputs
  checkpoints/     # model checkpoints and adapters
  datasets/        # downloaded and processed datasets
  exports/         # packaged exports for sharing or transfer
  repo/
    Authorship-Attribution-in-Victorian-Periodicals/
```

The current local workspace root is:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution
```

The repository root is:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/repo/Authorship-Attribution-in-Victorian-Periodicals
```

VS Code can open the workspace root, but terminal commands should normally be run from the repository root.

## Repository Layout

```text
Authorship-Attribution-in-Victorian-Periodicals/
  configs/
    phase1/
    phase2/
    phase3/
    phase4/
    datasets.yaml
    models.yaml
    paths.example.yaml
    paths.local.yaml
  docs/
  notebooks/
    local/
    colab/
    archive/
  prompts/
    phase4/
  scripts/
  src/
    data/
    evaluation/
    prompting/
    training/
    utils/
    visualization/
  tests/
  README.md
  requirements.txt
```

## What Belongs in Git

Track:

- source code in `src/`
- scripts in `scripts/`
- configs that are not machine-specific
- prompts
- notebooks intended as reproducible workflow templates
- documentation
- tests
- lightweight metadata needed to reproduce the workflow

Do not track:

- datasets
- generated artifacts
- model checkpoints
- model weights
- adapters
- logs
- exports
- zips
- secrets
- local virtual environments

Large files and generated outputs should remain under workspace-level storage outside `repo/`.

## Path Configuration

`configs/paths.example.yaml` is the portable template. It uses placeholder paths and documents the required path keys:

- `workspace_root`
- `repo_root`
- `artifacts_root`
- `checkpoints_root`
- `datasets_root`
- `exports_root`

`configs/paths.local.yaml` is machine-specific and points to the current local C: workspace. It should not be committed.

Project path helpers in `src/utils/paths.py` resolve artifact, dataset, checkpoint, and export locations from the path config.

## Hugging Face Usage

Hugging Face is used for:

- PERIAD: `celvaigh/periad`
- VEAA comparison data: `NicholasSynovic/ModifiedVEAA`
- pretrained encoder, decoder, and embedding models

Some decoder models require gated access and authentication through `HF_TOKEN`.

## Artifact and Checkpoint Management

Permanent generated outputs should go under:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/artifacts
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/checkpoints
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/datasets
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/exports
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/backups
```

Colab outputs should be downloaded or synced into this external workspace storage, not committed to Git.
