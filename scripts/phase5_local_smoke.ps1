python src/data/prepare_decoder_instruction_dataset.py --output_dir outputs/phase5/data --max_samples_train 24 --max_samples_test 12
python src/training/train_decoder_qlora.py --config configs/phase5/mistral_7b_qlora_smoke.yaml --train_file outputs/phase5/data/train_instruction.jsonl --eval_file outputs/phase5/data/test_instruction.jsonl --dry_run
