# Colab Workflow

Colab Pro is the preferred environment for heavy GPU work. Local VS Code remains the development and documentation center.

Related docs: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md), [HUGGINGFACE_RESOURCES.md](HUGGINGFACE_RESOURCES.md), [PROJECT_MAP.md](PROJECT_MAP.md).

## Why Colab Pro

Colab Pro is preferred over Kaggle for current heavy compute because earlier attempts exposed:

- GPU quota limits
- fragile checkpoint persistence
- adapter path mismatch risk
- malformed or incomplete logs
- difficulty recovering long-running QLoRA jobs

Colab Pro should be used for decoder inference, QLoRA, and large-model evaluation. T4 is acceptable for encoder or smaller jobs. A100 is preferred for QLoRA on 7B-9B decoder models.

## Clone the Repository

Run commands from the Colab working directory:

```bash
git clone https://github.com/IamTaoHu/Authorship-Attribution-in-Victorian-Periodicals.git
cd Authorship-Attribution-in-Victorian-Periodicals
```

Use GitHub for source control only. Do not commit generated outputs or model weights.

## Install Dependencies

```bash
pip install -r requirements.txt
```

Some GPU workflows may require Colab-specific package versions. Record any temporary Colab-only changes in notes instead of committing environment noise.

## Hugging Face Login

Use a Hugging Face token for datasets and gated models:

```bash
export HF_TOKEN=...
huggingface-cli login --token "$HF_TOKEN"
```

Llama and Gemma require gated access. The token must belong to an account approved for those model repositories.

## Configure Paths

Create or write a Colab path config equivalent to `configs/paths.local.yaml`. It should point artifacts, checkpoints, datasets, and exports outside the cloned repo when possible.

Typical Colab destinations:

```text
/content/Authorship-Attribution/artifacts
/content/Authorship-Attribution/checkpoints
/content/Authorship-Attribution/datasets
/content/Authorship-Attribution/exports
```

If Google Drive is mounted, copy important checkpoints and artifacts there during or immediately after long jobs.

## Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Use Drive for checkpoint backups and zip archives. Do not rely only on temporary `/content` storage.

## Save Checkpoints and Artifacts

Recommended storage policy:

- write active job outputs to configured artifact/checkpoint directories
- back up long-running checkpoints to Google Drive
- download or sync final outputs to local external storage
- never commit checkpoints, adapters, `.safetensors`, `.bin`, `.pt`, `.pth`, zips, or generated datasets

## Zip and Download Outputs

For transfer after a completed phase:

```bash
zip -r phase4_artifacts.zip artifacts/phase4
```

Then download through Colab or copy to Google Drive. After downloading locally, place outputs under:

```text
C:/Users/pawat_2enxw2x/Documents/Authorship-Attribution/artifacts
```

## Sync Back to Local Storage

After a Colab run:

1. Verify expected files exist in Colab.
2. Zip or copy artifacts/checkpoints to Google Drive.
3. Download to local machine.
4. Place files under the matching workspace-level folder outside `repo/`.
5. Run local aggregation, plotting, and checks only when appropriate for that phase.

## Phase 4 Note

Phase 4 final reporting must use final-only aggregation and plotting. TinyLlama smoke runs are diagnostic only and excluded from final tables, reports, and plots.
