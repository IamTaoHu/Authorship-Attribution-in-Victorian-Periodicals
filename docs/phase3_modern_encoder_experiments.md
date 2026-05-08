# Phase 3 Modern Encoder Experiments

## Goal
Phase 3 benchmarks stronger encoder models against the reproduced BERT-base PERIAD baseline. The main ranking metric is macro F1, with accuracy as the secondary metric.

## Models
- `roberta-base`
- `roberta-large`
- `microsoft/deberta-v3-base`
- `answerdotai/ModernBERT-base`
- DeBERTa-v3-base LoRA rank ablations: `r=4`, `r=8`, `r=16`, `r=32`
- ModernBERT context ablations: `max_length=256`, `512`, `1024`

## Recommended Run Order
```powershell
.\scripts\run_phase3_core.ps1
.\scripts\run_phase3_lora_ablation.ps1
.\scripts\run_phase3_modernbert_context.ps1
.\scripts\run_phase3_roberta_large.ps1
.\scripts\run_phase3_analysis.ps1
```

For a setup check without training:
```powershell
python src/training/train_encoder.py --config configs/phase3/roberta_base.yaml --dry-run
```

## Windows Torch Loading Notes
On `torch<2.6`, Transformers may refuse to load PyTorch `.bin` weights through `torch.load` because of a CVE safety restriction. Phase 3 defaults to `use_safetensors: true` so compatible checkpoints load safetensors weights instead of `.bin` files.

If a model repository only provides `.bin` weights, upgrade to `torch>=2.6` or choose a safetensors-compatible checkpoint. For an RTX 3050 Laptop GPU with 4GB VRAM, the recommended DeBERTa v1 LoRA settings are:
```yaml
batch_size: 2
eval_batch_size: 4
gradient_accumulation_steps: 8
fp16: false
```

For Trainer `fp16` mixed precision, do not load trainable full-finetuning models with `torch_dtype: auto` or `torch_dtype: float16`. Use `torch_dtype: null` with `fp16: true` so Trainer manages mixed precision safely. `torch_dtype: auto` may be useful for inference or some quantized/adapter workflows, but it can cause `Attempting to unscale FP16 gradients` during full fine-tuning.

For DeBERTa full fine-tuning on an RTX 3050 Laptop GPU with 4GB VRAM and `torch==2.5.1`, disable `fp16` because Accelerate can raise `Attempting to unscale FP16 gradients`. Use `batch_size: 1`, `eval_batch_size: 2`, and `gradient_accumulation_steps: 16` instead.

The previous DeBERTa full fine-tuning run with exploding loss, `nan` gradient norm, `nan` eval loss, macro F1 near `0.04195`, and predictions collapsed to class `0` is invalid and should not be reported as a Phase 3 result.

## DeBERTa on RTX 3050 4GB
`microsoft/deberta-v3-base` and `microsoft/deberta-v3-small` both remain numerically unstable locally in this Phase 3 pipeline. The invalid symptoms were huge loss, `nan` `grad_norm`, `nan` eval loss, macro F1 around `0.04`, and predictions collapsing to one class.

The active DeBERTa branch is now DeBERTa v1, `microsoft/deberta-base`, with the `deberta_base_lora_r4` LoRA experiment. The config file remains `configs/phase3/lora/deberta_base_lora_r4_sanity.yaml` for continuity, but it resolves to the 5-epoch `deberta_base_lora_r4` run. Start with:
```powershell
python src/training/train_encoder.py --config configs/phase3/lora/deberta_base_lora_r4_sanity.yaml
```

Keep the DeBERTa-v3 configs for audit and ablation context, but do not treat them as the recommended next run on this machine. DeBERTa v1 LoRA trains stably on RTX 3050 4GB with no NaN collapse; remaining work is quality/performance tuning.

LoRA target module names differ by DeBERTa family and local Transformers implementation. The DeBERTa v1 fallback `microsoft/deberta-base` uses `attention.self.in_proj`, so its PEFT LoRA target is:
```yaml
lora_target_modules:
  - in_proj
```

Stability was achieved after switching away from failed `query` / `value` and `query_proj` / `value_proj` targets to `in_proj`.

For encoder sequence classification with LoRA, keep the newly initialized classification head trainable through PEFT `modules_to_save`; otherwise the adapter-wrapped model can leave the head frozen and collapse to one predicted class:
```yaml
modules_to_save:
  - classifier
  - pooler
```

DeBERTa-v3 still uses:
```yaml
lora_target_modules:
  - query_proj
  - value_proj
```

If PEFT reports that no modules were targeted for adaptation, inspect the loaded model's Linear layers before training:
```powershell
python src/training/train_encoder.py --config <config> --list-linear-modules
```

Use PEFT suffix names from that listing, not full module paths, unless exact matching is required.

## Output Structure
Each experiment writes to:
```text
outputs/phase3/{experiment_name}/
```

Expected per-run artifacts include:
- `metrics.json`
- `classification_report.json`
- `predictions.csv`
- `logits.npy`
- `embeddings.npy` when enabled
- `tokenizer_stats.json`
- `run_config_resolved.yaml`
- `trainer_state.json` when available
- `confusion_matrix.png`

Aggregate outputs are written under:
```text
outputs/phase3/tables/
outputs/phase3/plots/
```

## Metrics
- `macro_f1`: primary ranking metric; treats all six authors equally.
- `accuracy`: secondary metric; useful but can be affected by class imbalance.
- `precision_macro` and `recall_macro`: diagnose whether macro F1 changes come from precision or recall.
- `weighted_f1`: class-frequency-weighted score.
- Per-class precision, recall, F1, and support are saved in `classification_report.json` and `metrics.json`.

## LoRA Rank Ablation
The DeBERTa LoRA configs compare trainable adapter capacity while keeping the base model fixed. Use `lora_rank_ablation_table.csv` and `deberta_lora_rank_ablation.png` to check whether higher ranks improve macro F1 enough to justify additional trainable parameters.

## ModernBERT Context Ablation
The ModernBERT context configs compare `max_length` values of `256`, `512`, and `1024`. Use `modernbert_context_ablation_table.csv` and `modernbert_context_length_ablation.png` to evaluate whether longer context helps OCR-noisy Victorian paragraphs enough to justify memory cost.

## Comparing Against BERT-base
Use the reproduced Phase 2 BERT-base result as the reference:
- Accuracy: `0.9174`
- Macro F1: `0.9071`

The final Phase 3 comparison table is:
```text
outputs/phase3/tables/final_encoder_results_table.csv
```

Interpret a modern encoder as meaningfully better only if macro F1 improves and the error analysis does not show new failures on James Fitzjames Stephen or Eliza Lynn Linton.

## Final Tables and Plots
Expected table outputs:
- `encoder_results.csv`
- `encoder_results.md`
- `final_encoder_results_table.csv`
- `lora_rank_ablation_table.csv`
- `modernbert_context_ablation_table.csv`

Expected plot outputs:
- `macro_f1_comparison.png`
- `accuracy_comparison.png`
- `deberta_lora_rank_ablation.png`
- `modernbert_context_length_ablation.png`
- per-experiment `confusion_matrix.png`
- per-experiment `training_curves.png` when Trainer logs are available
