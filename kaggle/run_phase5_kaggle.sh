#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN"
else
  echo "HF_TOKEN is not set. Llama-3 and Gemma may fail if access approval/license acceptance is required."
fi

python src/data/prepare_decoder_instruction_dataset.py --output_dir outputs/phase5/data

python src/data/analyze_decoder_tokenization.py --config configs/phase5/mistral_7b_qlora.yaml
python src/data/analyze_decoder_tokenization.py --config configs/phase5/llama3_8b_qlora.yaml
python src/data/analyze_decoder_tokenization.py --config configs/phase5/gemma2_9b_qlora.yaml

python src/training/train_decoder_qlora.py --config configs/phase5/mistral_7b_qlora_smoke.yaml --train_file outputs/phase5/data/train_instruction.jsonl --eval_file outputs/phase5/data/test_instruction.jsonl --max_train_samples 24 --max_eval_samples 12
python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/mistral_7b_qlora_smoke.yaml --adapter_dir outputs/phase5/checkpoints/mistral_7b_qlora_smoke_seed42/adapter --test_file outputs/phase5/data/test_instruction.jsonl --output_dir outputs/phase5/eval --max_samples 12

python src/training/run_phase5_lora_ablation.py --base_config configs/phase5/mistral_7b_qlora.yaml
python src/training/run_phase5_lora_ablation.py --base_config configs/phase5/mistral_7b_qlora.yaml --run

python src/training/train_decoder_qlora.py --config configs/phase5/llama3_8b_qlora.yaml --train_file outputs/phase5/data/train_instruction.jsonl --eval_file outputs/phase5/data/test_instruction.jsonl
python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/llama3_8b_qlora.yaml --adapter_dir outputs/phase5/checkpoints/llama3_8b_qlora_r8_seed42/adapter --test_file outputs/phase5/data/test_instruction.jsonl --output_dir outputs/phase5/eval

python src/training/train_decoder_qlora.py --config configs/phase5/gemma2_9b_qlora.yaml --train_file outputs/phase5/data/train_instruction.jsonl --eval_file outputs/phase5/data/test_instruction.jsonl
python src/evaluation/evaluate_decoder_qlora.py --config configs/phase5/gemma2_9b_qlora.yaml --adapter_dir outputs/phase5/checkpoints/gemma2_9b_qlora_r8_seed42/adapter --test_file outputs/phase5/data/test_instruction.jsonl --output_dir outputs/phase5/eval

python src/evaluation/aggregate_phase5_results.py
python src/evaluation/analyze_phase5_errors.py
python src/visualization/plot_phase5_results.py

zip -r phase5_outputs.zip outputs/phase5
