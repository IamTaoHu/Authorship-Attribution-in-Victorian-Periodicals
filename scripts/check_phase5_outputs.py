"""Check Phase 5 diagnostic outputs and optional full QLoRA runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CONFIGS = (
    "diagnostic_tinyllama_qlora.yaml",
    "mistral_7b_instruct_qlora.yaml",
    "llama3_8b_instruct_qlora.yaml",
    "gemma2_9b_it_qlora.yaml",
)
FULL_RUNS = ("mistral_qlora", "llama3_qlora", "gemma2_qlora")
PREDICTION_COLUMNS = (
    "sample_id",
    "text",
    "true_author",
    "raw_output",
    "cleaned_output",
    "predicted_author",
    "parsed_status",
    "correct",
)
METRIC_KEYS = (
    "run_name",
    "model_name",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_output_rate",
    "average_generated_length",
    "n_samples",
    "n_valid_predictions",
)
DIAGNOSTIC_FILES = (
    "config.yaml",
    "predictions.csv",
    "metrics.json",
    "classification_report.txt",
    "label_mapping.json",
    "training_log.jsonl",
)
FULL_RUN_FILES = (
    "predictions.csv",
    "metrics.json",
    "classification_report.txt",
    "label_mapping.json",
)
FULL_AGGREGATE_FILES = (
    "tables/decoder_qlora_results.csv",
    "tables/decoder_qlora_results.md",
    "tables/phase5_per_author_f1.csv",
    "reports/phase5_qlora_decoder_finetuning_report.md",
)
WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx")
ALLOWED_ADAPTER_NAMES = {"adapter_model.safetensors", "adapter_model.bin"}


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def check_path(path: Path) -> bool:
    ok = path.exists() and path.stat().st_size >= 0
    print(f"[{'OK' if ok else 'MISSING'}] {path}")
    return ok


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def check_configs() -> bool:
    ok = True
    config_dir = REPO_ROOT / "configs" / "phase5"
    for name in CONFIGS:
        path = config_dir / name
        ok = check_path(path) and ok
        if not path.exists():
            continue
        config = load_yaml(path)
        required = {"phase", "run_name", "model_name", "mode", "dataset", "prompt", "quantization", "lora", "training", "paths"}
        missing = sorted(required.difference(config))
        if config.get("phase") != "phase5":
            print(f"[MISMATCH] {path}: phase={config.get('phase')!r}")
            ok = False
        if missing:
            print(f"[MISSING] {path} keys: {', '.join(missing)}")
            ok = False
    return ok


def check_predictions(path: Path) -> bool:
    frame = pd.read_csv(path, nrows=1)
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        print(f"[MISSING] {path} columns: {', '.join(missing)}")
        return False
    print(f"[OK] {path} contains required prediction columns")
    return True


def check_metrics(path: Path, *, strict: bool = True) -> bool:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        if not strict:
            print(f"[WARN] {path} missing optional metric keys: {', '.join(missing)}")
            return True
        print(f"[MISSING] {path} keys: {', '.join(missing)}")
        return False
    print(f"[OK] {path} contains required metric keys")
    return True


def check_adapter(checkpoint_dir: Path) -> bool:
    ok = check_path(checkpoint_dir / "adapter_config.json")
    adapter_files = [path for path in checkpoint_dir.iterdir()] if checkpoint_dir.exists() else []
    has_adapter = any(path.name in ALLOWED_ADAPTER_NAMES for path in adapter_files)
    if has_adapter:
        print(f"[OK] adapter model file exists in {checkpoint_dir}")
    else:
        print(f"[MISSING] adapter_model.safetensors or adapter_model.bin in {checkpoint_dir}")
    return ok and has_adapter


def check_diagnostic(phase5_dir: Path, checkpoint_dir: Path) -> bool:
    run_dir = phase5_dir / "diagnostic" / "runs" / "tinyllama_qlora_diagnostic"
    ok = True
    for name in DIAGNOSTIC_FILES:
        ok = check_path(run_dir / name) and ok
    if (run_dir / "predictions.csv").exists():
        ok = check_predictions(run_dir / "predictions.csv") and ok
    if (run_dir / "metrics.json").exists():
        ok = check_metrics(run_dir / "metrics.json") and ok
    for path in (
        phase5_dir / "diagnostic" / "tables" / "diagnostic_decoder_qlora_results.csv",
        phase5_dir / "diagnostic" / "tables" / "diagnostic_decoder_qlora_results.md",
        phase5_dir / "diagnostic" / "reports" / "diagnostic_report.md",
    ):
        ok = check_path(path) and ok
    ok = check_adapter(checkpoint_dir / "diagnostic" / "tinyllama_qlora_diagnostic") and ok
    return ok


def diagnostic_present(phase5_dir: Path, checkpoint_dir: Path) -> bool:
    return any(
        path.exists()
        for path in (
            phase5_dir / "diagnostic",
            checkpoint_dir / "diagnostic",
            phase5_dir / "diagnostic" / "runs" / "tinyllama_qlora_diagnostic",
        )
    )


def check_full_runs(phase5_dir: Path, checkpoint_dir: Path) -> bool:
    ok = True
    for run_name in FULL_RUNS:
        run_dir = phase5_dir / "runs" / run_name
        ckpt = checkpoint_dir / run_name
        for name in FULL_RUN_FILES:
            path = run_dir / name
            ok = check_path(path) and ok
        if (run_dir / "predictions.csv").exists():
            ok = check_predictions(run_dir / "predictions.csv") and ok
        if (run_dir / "metrics.json").exists():
            ok = check_metrics(run_dir / "metrics.json", strict=False) and ok
        ok = check_adapter(ckpt) and ok
    for relative in FULL_AGGREGATE_FILES:
        ok = check_path(phase5_dir / relative) and ok
    return ok


def check_repo_hygiene() -> bool:
    ok = True
    bad_inside_outputs = []
    outputs_root = REPO_ROOT / "outputs"
    if outputs_root.exists():
        for path in outputs_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
                bad_inside_outputs.append(path)
    if bad_inside_outputs:
        print("[MISSING] Model/checkpoint files found inside repo outputs:")
        for path in bad_inside_outputs:
            print(f"  {path}")
        ok = False
    else:
        print("[OK] No model/checkpoint files found inside repo outputs")
    try:
        tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
        bad_tracked = [line for line in tracked.splitlines() if Path(line).suffix.lower() in WEIGHT_SUFFIXES]
        if bad_tracked:
            print("[MISSING] Weight files are tracked by Git:")
            for path in bad_tracked:
                print(f"  {path}")
            ok = False
        else:
            print("[OK] No model/checkpoint weight files are tracked by Git")
    except Exception as exc:
        print(f"[WARN] Could not inspect tracked files: {exc}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5_dir", default="outputs/phase5")
    parser.add_argument("--checkpoint_dir", default="checkpoints/phase5")
    parser.add_argument("--require_full_runs", action="store_true")
    parser.add_argument("--require_diagnostic", action="store_true")
    args = parser.parse_args()
    phase5_dir = resolve(args.phase5_dir)
    checkpoint_dir = resolve(args.checkpoint_dir)
    print(f"Resolved phase5_dir: {phase5_dir}")
    print(f"Resolved checkpoint_dir: {checkpoint_dir}")
    ok = check_configs()
    if args.require_diagnostic:
        ok = check_diagnostic(phase5_dir, checkpoint_dir) and ok
    elif args.require_full_runs:
        print("[OK] Diagnostic validation skipped for full-run validation. Pass --require_diagnostic to require it.")
    elif diagnostic_present(phase5_dir, checkpoint_dir):
        ok = check_diagnostic(phase5_dir, checkpoint_dir) and ok
    else:
        print("[OK] Diagnostic artifacts absent; diagnostic validation skipped. Pass --require_diagnostic to require them.")
    if args.require_full_runs:
        ok = check_full_runs(phase5_dir, checkpoint_dir) and ok
    else:
        print("[OK] Full 7B-9B run validation skipped; pass --require_full_runs to require it.")
    ok = check_repo_hygiene() and ok
    if ok:
        print("Phase 5 output check passed.")
        return 0
    print("Phase 5 output check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
