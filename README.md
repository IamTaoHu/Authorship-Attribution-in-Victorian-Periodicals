# Authorship Attribution in Victorian Periodicals

## Overview

This repository contains a supervised authorship attribution project for Victorian periodical texts. It uses the fixed PERIAD train/test split and evaluates classification performance across six canonical authors.

Authorship attribution helps test whether computational models can distinguish stylistic and thematic signals across historically situated writing. The project compares encoder baselines, advanced encoder models, decoder prompting, QLoRA decoder fine-tuning, ensemble voting, topic modeling, and topic-aware classification.

The work culminates in a Phase 10 final research package that gathers validated outputs, reports, tables, figures, and reproducibility records for distribution.

## Key Results

| Model / Configuration | Role | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- |
| `mistral_qlora` | Best single model | `0.9292758523527754` | `0.9286565035360188` | `0.9351746953943576` |
| `weighted_vote_accuracy` | Best ensemble | `0.9478726401803323` | `0.943163809630258` | `0.948160138757328` |
| `roberta_large` | Competitive encoder baseline | `0.9270216962524654` | `0.9190653017918314` | `0.9274597507911` |
| `deberta_text_only_reproduced` | Topic-aware classification | `0.8819385742462665` | `0.8700585212082194` | `0.8829031442452651` |

## Pipeline Overview

- Phase 1: Dataset preparation and validation.
- Phase 2: BERT/RoBERTa encoder baselines.
- Phase 3: Advanced encoder models.
- Phase 4: Decoder prompting baselines.
- Phase 5: QLoRA decoder fine-tuning.
- Phase 6: Ensemble voting.
- Phase 7: LDA topic modeling.
- Phase 8: BERTopic analysis.
- Phase 9: Topic-aware classification.
- Phase 10: Final research package and reproducibility export.

## Repository Structure

```text
configs/                Configuration files and path templates.
scripts/                Phase scripts for checks, training, evaluation, aggregation, and packaging.
tests/                  Lightweight tests and validation checks.
src/                    Shared project code used by scripts and workflows.
docs/                   Project notes and task documentation.
notebooks/              Notebook materials for interactive or hosted runs.
prompts/                Prompt assets for decoder prompting workflows.
requirements.txt        Python dependency list for this repository.
```

Generated artifacts and export folders are expected outside the tracked source tree or in configured output locations:

```text
artifacts/              Expected workspace location for phase outputs and reports.
exports/                Expected workspace location for distributable exports.
exports/final_package/  Expected unpacked Phase 10 final package directory.
```

## Installation

Create and activate a Python virtual environment from the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Usage

Model training, evaluation, topic modeling, ensemble, and validation scripts are organized under `scripts/`. Configuration files and path templates are organized under `configs/`.

To build the Phase 10 final research package from existing validated artifacts, run:

```bash
python scripts/run_phase10_final_package.py
```

## Final Research Package

The final distributable package is located at:

```text
exports/phase10_final_research_package.zip
exports/final_package/
```

The package contains final tables, figures, reports, reproducibility manifests, and a package-level README for navigation.

## Reproducibility

Phase 10 is packaging-only. It reads existing validated Phase 1-9 artifacts, assembles the final research package, and does not retrain models or modify previous artifacts.

Traceability is stored in:

```text
exports/final_package/reproducibility/manifest/source_manifest.csv
```

## Notes on Large Artifacts

Large checkpoints, adapters, binary weights, model caches, and raw logs are intentionally excluded from the final package. These files should remain in external storage or configured local workspace locations.

## Citation

If you use this repository, please cite the associated paper or project report when available.

```bibtex
@misc{authorship_attribution_victorian_periodicals,
  title = {Authorship Attribution in Victorian Periodicals},
  author = {Pawat Isaraporn},
  year = {2026},
  note = {Research project repository}
}
```
