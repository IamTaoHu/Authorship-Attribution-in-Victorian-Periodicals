"""Check required Phase 4 generated outputs and repository hygiene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_path_config  # noqa: E402


CONFIGS = (
    "tinyllama_smoke.yaml",
    "mistral_zero_shot.yaml",
    "mistral_few_shot.yaml",
    "llama3_zero_shot.yaml",
    "llama3_few_shot.yaml",
    "gemma2_zero_shot.yaml",
    "gemma2_few_shot.yaml",
)
PROMPTS = ("zero_shot_direct.txt", "few_shot_direct.txt")
REQUIRED_RUN_FILES = ("predictions.csv", "metrics.json", "classification_report.csv", "confusion_matrix.csv", "run_config_resolved.yaml")
REQUIRED_PREDICTION_COLUMNS = (
    "sample_id",
    "text",
    "true_author",
    "prompt_type",
    "model_name",
    "raw_output",
    "cleaned_output",
    "predicted_author",
    "parsed_status",
    "parse_error",
    "correct",
    "input_tokens",
    "output_tokens",
    "inference_time_sec",
)
REQUIRED_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_output_rate",
    "n_samples",
    "n_valid_predictions",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_inference_time_sec",
    "total_inference_time_sec",
    "run_name",
    "model_name",
    "prompt_type",
    "valid_run",
    "invalid_reason",
)
TABLES = ("decoder_prompting_results.csv", "decoder_prompting_results.md", "phase4_per_author_f1.csv")
REPORTS = ("phase4_decoder_prompting_report.md",)
PLOTS = (
    "decoder_accuracy_bar.png",
    "decoder_macro_f1_bar.png",
    "invalid_output_rate_bar.png",
    "parsed_status_distribution.png",
    "prediction_distribution_by_run.png",
    "per_author_f1_heatmap.png",
)
WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx")


def check_path(path: Path) -> bool:
    ok = path.exists() and path.stat().st_size > 0
    marker = "OK" if ok else "MISSING"
    print(f"[{marker}] {path}")
    return ok


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return payload


def resolve_phase4_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    paths = get_path_config()
    normalized = str(path).replace("\\", "/")
    if normalized == "artifacts/phase4":
        return paths["artifacts_root"] / "phase4"
    return paths["artifacts_root"] / path


def check_configs() -> bool:
    missing = 0
    config_dir = REPO_ROOT / "configs" / "phase4"
    for name in CONFIGS:
        path = config_dir / name
        if not check_path(path):
            missing += 1
            continue
        config = load_yaml(path)
        required = {"phase", "run_name", "model_name", "prompt_type", "split", "seed", "max_new_tokens", "temperature", "do_sample", "torch_dtype", "load_in_4bit", "output_dir", "hf_token_env"}
        absent = sorted(required.difference(config))
        if absent:
            print(f"[MISSING] {path} keys: {', '.join(absent)}")
            missing += 1
        elif config.get("phase") != "phase4":
            print(f"[MISMATCH] {path}: phase={config.get('phase')!r}")
            missing += 1
    return missing == 0


def check_prompts() -> bool:
    missing = 0
    prompt_dir = REPO_ROOT / "prompts" / "phase4"
    for name in PROMPTS:
        path = prompt_dir / name
        if not check_path(path):
            missing += 1
            continue
        text = path.read_text(encoding="utf-8")
        if "{text}" not in text or "Leslie Stephen" not in text or "JSON" not in text:
            print(f"[MISMATCH] prompt template content: {path}")
            missing += 1
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
    if missing:
        print(f"[MISSING] {path} keys: {', '.join(missing)}")
        return False
    print(f"[OK] {path} contains required metric keys")
    return True


def check_run(run_dir: Path) -> bool:
    missing = 0
    for filename in REQUIRED_RUN_FILES:
        if not check_path(run_dir / filename):
            missing += 1
    if (run_dir / "predictions.csv").exists() and not check_predictions(run_dir / "predictions.csv"):
        missing += 1
    if (run_dir / "metrics.json").exists() and not check_metrics(run_dir / "metrics.json"):
        missing += 1
    return missing == 0


def check_tables_reports_plots(phase4_dir: Path, allow_missing_runs: bool) -> bool:
    ok = True
    for name in TABLES:
        path = phase4_dir / "tables" / name
        if not path.exists() and allow_missing_runs:
            print(f"[WARN] aggregate table not generated yet: {path}")
            continue
        ok = check_path(path) and ok
    for name in REPORTS:
        path = phase4_dir / "reports" / name
        if not path.exists() and allow_missing_runs:
            print(f"[WARN] aggregate report not generated yet: {path}")
            continue
        ok = check_path(path) and ok
    for name in PLOTS:
        path = phase4_dir / "plots" / name
        if not path.exists() and allow_missing_runs:
            print(f"[WARN] plot not generated yet: {path}")
            continue
        ok = check_path(path) and ok
    return ok


def check_outputs_outside_repo(phase4_dir: Path) -> bool:
    try:
        phase4_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        print(f"[OK] Phase 4 outputs are outside repo: {phase4_dir}")
        return True
    print(f"[MISSING] Phase 4 outputs must be outside repo: {phase4_dir}")
    return False


def check_tracked_weights() -> bool:
    try:
        output = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    except Exception as exc:
        print(f"[WARN] Could not inspect tracked files: {exc}")
        return True
    bad = [line for line in output.splitlines() if Path(line).suffix.lower() in WEIGHT_SUFFIXES]
    if bad:
        print("[MISSING] Model/checkpoint weight files are tracked by Git:")
        for path in bad:
            print(f"  {path}")
        return False
    print("[OK] No model/checkpoint weight files are tracked by Git")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4_dir", default="artifacts/phase4")
    parser.add_argument("--allow_missing_runs", action="store_true")
    args = parser.parse_args()
    phase4_dir = resolve_phase4_dir(args.phase4_dir)
    print(f"Resolved phase4_dir: {phase4_dir}")
    ok = check_configs()
    ok = check_prompts() and ok
    ok = check_outputs_outside_repo(phase4_dir) and ok
    ok = check_tracked_weights() and ok
    runs_dir = phase4_dir / "runs"
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir()) if runs_dir.exists() else []
    if not run_dirs:
        if args.allow_missing_runs:
            print(f"[WARN] no Phase 4 runs found at {runs_dir}")
        else:
            print(f"[MISSING] no Phase 4 runs found at {runs_dir}")
            ok = False
    for run_dir in run_dirs:
        ok = check_run(run_dir) and ok
    ok = check_tables_reports_plots(phase4_dir, args.allow_missing_runs) and ok
    if ok:
        print("Phase 4 output check passed.")
        return 0
    print("Phase 4 output check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
