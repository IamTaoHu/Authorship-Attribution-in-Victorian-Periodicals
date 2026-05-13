"""Check required Phase 3 generated outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import checkpoint_dir, get_path_config, phase_artifact_dir  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "phase3" / "deberta_v3_base.yaml"
REQUIRED_RUN_FILES = (
    "predictions.csv",
    "metrics.json",
    "classification_report.csv",
    "confusion_matrix.csv",
    "run_config_resolved.yaml",
    "trainer_state.json",
    "training_log.csv",
)
REQUIRED_PREDICTION_COLUMNS = (
    "sample_id",
    "text",
    "true_label",
    "true_author",
    "predicted_label",
    "predicted_author",
    "correct",
)
REQUIRED_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "precision_macro",
    "recall_macro",
    "n_train",
    "n_test",
    "model_name",
    "seed",
    "smoke_test",
    "run_name",
    "valid_run",
    "invalid_reason",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def check_path(path: Path) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    marker = "OK" if ok else "MISSING"
    print(f"[{marker}] {path}")
    return ok


def resolve_phase3_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    paths = get_path_config()
    normalized = str(path).replace("\\", "/")
    if normalized == "artifacts/phase3":
        return paths["artifacts_root"] / "phase3"
    candidate = paths["artifacts_root"] / path
    if candidate.exists():
        return candidate
    return REPO_ROOT / path


def check_config() -> bool:
    missing = 0
    if not check_path(CONFIG_PATH):
        return False
    config = load_yaml(CONFIG_PATH)
    expected = {
        "phase": "phase3",
        "experiment_name": "deberta_v3_base_seed42_fp32",
        "model_name": "microsoft/deberta-v3-base",
        "baseline_reference": "roberta_large_phase2",
        "seed": 42,
        "max_length": 256,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            print(f"[MISMATCH] {CONFIG_PATH}: expected {key}={value!r}, found {config.get(key)!r}")
            missing += 1
        else:
            print(f"[OK] config {key}={value!r}")
    required_keys = (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "dataloader_num_workers",
        "learning_rate",
        "num_train_epochs",
        "weight_decay",
        "warmup_ratio",
        "fp16",
        "bf16",
        "amp_backend",
        "torch_dtype",
        "force_float32",
        "smoke_test",
    )
    for key in required_keys:
        if key not in config:
            print(f"[MISSING] config key: {key}")
            missing += 1
        else:
            print(f"[OK] config key: {key}")
    return missing == 0


def check_predictions(path: Path) -> bool:
    frame = pd.read_csv(path, nrows=1)
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        print(f"[MISSING] {path} columns: {', '.join(missing)}")
        return False
    print(f"[OK] {path} contains required prediction columns")
    return True


def check_metrics(path: Path) -> bool:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    legacy_validity_keys = {"valid_run", "invalid_reason"}
    if missing and set(missing).issubset(legacy_validity_keys):
        print(f"[OK] {path} contains required metric keys for legacy run; validity is tracked in aggregate table")
        return True
    if missing:
        print(f"[MISSING] {path} keys: {', '.join(missing)}")
        return False
    print(f"[OK] {path} contains required metric keys")
    return True


def check_run(run_dir: Path) -> bool:
    missing = 0
    metrics_path = run_dir / "metrics.json"
    valid_run = True
    if metrics_path.exists():
        try:
            valid_run = bool(json.loads(metrics_path.read_text(encoding="utf-8")).get("valid_run", True))
        except Exception:
            valid_run = True
    required_files = REQUIRED_RUN_FILES
    if not valid_run:
        required_files = (
            "metrics.json",
            "run_config_resolved.yaml",
            "trainer_state.json",
            "training_log.csv",
            "runtime.json",
        )
    for filename in required_files:
        if not check_path(run_dir / filename):
            missing += 1
    predictions_path = run_dir / "predictions.csv"
    if valid_run and predictions_path.exists() and not check_predictions(predictions_path):
        missing += 1
    if metrics_path.exists() and not check_metrics(metrics_path):
        missing += 1
    resolved_path = run_dir / "run_config_resolved.yaml"
    if resolved_path.exists():
        resolved = load_yaml(resolved_path)
        checkpoint_path = Path(str(resolved.get("checkpoint_dir", "")))
        if not checkpoint_path.exists():
            print(f"[MISSING] checkpoint_dir from resolved config: {checkpoint_path}")
            missing += 1
        else:
            print(f"[OK] checkpoint_dir from resolved config: {checkpoint_path}")
    return missing == 0


def check_table(phase3_dir: Path) -> bool:
    table_path = phase3_dir / "tables" / "advanced_encoder_results.csv"
    if not check_path(table_path):
        return False
    frame = pd.read_csv(table_path, nrows=1)
    required = {
        "run_name",
        "model_name",
        "baseline_reference",
        "smoke_test",
        "macro_f1",
        "accuracy",
        "valid_run",
        "invalid_reason",
        "local_practical",
        "final_benchmark",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        print(f"[MISSING] {table_path} columns: {', '.join(missing)}")
        return False
    print(f"[OK] {table_path} contains required aggregate columns")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3_dir", default="artifacts/phase3")
    parser.add_argument("--check_config_only", action="store_true")
    args = parser.parse_args()

    config_ok = check_config()
    phase3_dir = resolve_phase3_dir(args.phase3_dir)
    print(f"Resolved phase3_dir: {phase3_dir}")
    for directory in ("runs", "tables", "plots"):
        path = phase_artifact_dir("phase3", directory) if args.check_config_only else phase3_dir / directory
        marker = "OK" if path.exists() else "MISSING"
        print(f"[{marker}] {path}")
    checkpoint_root = checkpoint_dir("phase3")
    print(f"[OK] checkpoint root available: {checkpoint_root}")

    if args.check_config_only:
        if config_ok:
            print("Phase 3 config-only output check passed.")
            return 0
        print("Phase 3 config-only output check failed.")
        return 1

    runs_dir = phase3_dir / "runs"
    completed = []
    if runs_dir.exists():
        completed = [run_dir for run_dir in sorted(runs_dir.iterdir()) if run_dir.is_dir() and check_run(run_dir)]
    table_ok = check_table(phase3_dir)
    if not config_ok or not completed or not table_ok:
        print("Phase 3 output check failed.")
        return 1
    print(f"Phase 3 output check passed: {len(completed)} completed run(s) found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
