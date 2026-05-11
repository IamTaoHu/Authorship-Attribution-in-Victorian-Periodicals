# Phase 5 Kaggle QLoRA Run Guide

This workflow is for Kaggle GPU execution only. Local VS Code should be used for code, configs, dataset conversion, tokenization checks, parser checks, and dry-runs.

## Prerequisites

- Enable a GPU runtime in Kaggle.
- Add `HF_TOKEN` as a Kaggle secret and expose it to the notebook/session.
- Accept Hugging Face access terms before running gated models. `meta-llama/Meta-Llama-3-8B-Instruct` and `google/gemma-2-9b-it` may require access approval and license acceptance.
- Do not hardcode private tokens in scripts, notebooks, configs, or committed files.

## Commands

Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Login to Hugging Face:
```bash
huggingface-cli login --token "$HF_TOKEN"
```

Prepare instruction data:
```bash
python src/data/prepare_decoder_instruction_dataset.py --output_dir outputs/phase5/data
```

Run tokenization analysis:
```bash
python src/data/analyze_decoder_tokenization.py --config configs/phase5/mistral_7b_qlora.yaml
python src/data/analyze_decoder_tokenization.py --config configs/phase5/llama3_8b_qlora.yaml
python src/data/analyze_decoder_tokenization.py --config configs/phase5/gemma2_9b_qlora.yaml
```

Run Mistral smoke, then full Mistral:
```bash
python src/training/train_decoder_qlora.py --config configs/phase5/mistral_7b_qlora_smoke.yaml --train_file outputs/phase5/data/train_instruction.jsonl --eval_file outputs/phase5/data/test_instruction.jsonl --max_train_samples 24 --max_eval_samples 12
python src/training/train_decoder_qlora.py --config configs/phase5/mistral_7b_qlora.yaml --train_file outputs/phase5/data/train_instruction.jsonl --eval_file outputs/phase5/data/test_instruction.jsonl
python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/mistral_7b_qlora.yaml --adapter_dir outputs/phase5/checkpoints/mistral_7b_qlora_r8_seed42/adapter --test_file outputs/phase5/data/test_instruction.jsonl --output_dir outputs/phase5/eval
```

Generate and run Mistral LoRA rank ablation:
```bash
python src/training/run_phase5_lora_ablation.py --base_config configs/phase5/mistral_7b_qlora.yaml
python src/training/run_phase5_lora_ablation.py --base_config configs/phase5/mistral_7b_qlora.yaml --run
```

Run Llama and Gemma:
```bash
python src/training/train_decoder_qlora.py --config configs/phase5/llama3_8b_qlora.yaml --train_file outputs/phase5/data/train_instruction.jsonl --eval_file outputs/phase5/data/test_instruction.jsonl
python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/llama3_8b_qlora.yaml --adapter_dir outputs/phase5/checkpoints/llama3_8b_qlora_r8_seed42/adapter --test_file outputs/phase5/data/test_instruction.jsonl --output_dir outputs/phase5/eval

python src/training/train_decoder_qlora.py --config configs/phase5/gemma2_9b_qlora.yaml --train_file outputs/phase5/data/train_instruction.jsonl --eval_file outputs/phase5/data/test_instruction.jsonl
python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/gemma2_9b_qlora.yaml --adapter_dir outputs/phase5/checkpoints/gemma2_9b_qlora_r8_seed42/adapter --test_file outputs/phase5/data/test_instruction.jsonl --output_dir outputs/phase5/eval
```

Aggregate, analyze, visualize, and export:
```bash
python src/evaluation/aggregate_phase5_results.py
python src/evaluation/analyze_phase5_errors.py
python src/visualization/plot_phase5_results.py
zip -r phase5_outputs.zip outputs/phase5
```
