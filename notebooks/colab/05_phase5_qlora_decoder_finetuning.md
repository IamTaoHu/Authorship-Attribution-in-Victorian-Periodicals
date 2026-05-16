# Phase 5 - QLoRA Decoder Fine-Tuning

This workflow runs the full Phase 5 decoder QLoRA experiments on Colab GPU hardware. Do not run the 7B-9B configs on the local RTX 3050 4GB machine.

Hardware guidance: Mistral can run on A100 or L4. Llama 3 8B and Gemma 2 9B defaults are tuned for Colab L4 24GB. T4 is not recommended for the Gemma 2 9B full run, and TPU is not supported for this QLoRA pipeline.

## 1. Setup

```bash
!git clone <YOUR_REPO_URL> /content/Authorship-Attribution-in-Victorian-Periodicals
%cd /content/Authorship-Attribution-in-Victorian-Periodicals
!pip install -r requirements.txt
```

If Colab has incompatible preinstalled packages, restart the runtime after installing dependencies.

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Recommended Drive layout:

```text
/content/drive/MyDrive/Authorship-Attribution/
  outputs/phase5/
  checkpoints/phase5/
  datasets/
```

Update `configs/paths.local.yaml` in the Colab runtime if the shared path config is not already pointing at Drive-backed storage.

## 3. Hugging Face Login

```bash
!huggingface-cli login
```

Llama and Gemma may require gated model access. Confirm access before launching long runs.

## 4. Prepare Dataset

Use the existing Phase 1 dataset pipeline if processed PERIAD files are not already present:

```bash
!python scripts/prepare_hf_datasets.py
```

The training scripts expect the processed PERIAD files and canonical six-author `label_map.json`.

## 5. Run Full QLoRA Jobs

The Llama 3 and Gemma 2 configs use L4 24GB practical defaults: `prompt.max_length: 768`, LoRA rank `16`, LoRA alpha `32`, and `training.gradient_accumulation_steps: 8`.

Mistral:

```bash
!python src/training/train_decoder_qlora.py --config configs/phase5/mistral_7b_instruct_qlora.yaml
!python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/mistral_7b_instruct_qlora.yaml
```

Llama 3:

```bash
!python src/training/train_decoder_qlora.py --config configs/phase5/llama3_8b_instruct_qlora.yaml
!python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/llama3_8b_instruct_qlora.yaml
```

Gemma 2:

```bash
!python src/training/train_decoder_qlora.py --config configs/phase5/gemma2_9b_it_qlora.yaml
!python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/gemma2_9b_it_qlora.yaml
```

## 6. Resume From Checkpoint

Pass the trainer checkpoint path when resuming:

```bash
!python src/training/train_decoder_qlora.py \
  --config configs/phase5/mistral_7b_instruct_qlora.yaml \
  --resume_from_checkpoint checkpoints/phase5/mistral_qlora/checkpoint-250
```

Keep adapters and trainer checkpoints in Drive-backed `checkpoints/phase5/`.

## 7. Aggregate, Plot, and Validate

After real full runs exist:

```bash
!python src/visualization/plot_phase5_results.py --phase5_dir outputs/phase5
!python scripts/check_phase5_outputs.py --phase5_dir outputs/phase5 --checkpoint_dir checkpoints/phase5 --require_full_runs
```

Copy final metrics, predictions, reports, tables, and plots back to `outputs/phase5`. Do not commit outputs, adapters, checkpoints, datasets, or model weights.

## 8. OOM Troubleshooting

- For Llama 3 8B or Gemma 2 9B on L4, apply this fallback exactly if OOM occurs: `prompt.max_length: 512`, `lora.r: 8`, `lora.alpha: 16`, and `training.gradient_accumulation_steps: 16`.
- Keep `per_device_train_batch_size: 1`.
- Use A100 or L4 for Mistral; use L4 24GB or A100 for Llama 3 8B and Gemma 2 9B.
- T4 is not recommended for the Gemma 2 9B full run.
- TPU is not supported for this QLoRA pipeline.
- Resume from the latest checkpoint after runtime interruption.
