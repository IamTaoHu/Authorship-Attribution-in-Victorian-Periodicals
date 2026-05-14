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
FINAL_RUN_NAMES = (
    "mistral_zero_shot",
    "mistral_few_shot",
    "llama3_zero_shot",
    "llama3_few_shot",
    "gemma2_zero_shot",
    "gemma2_few_shot",
)
FINAL_N_SAMPLES = 3549


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


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def check_final_run(run_dir: Path) -> bool:
    ok = check_run(run_dir)
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        return False
    metrics = load_metrics(metrics_path)
    run_name = metrics.get("run_name", run_dir.name)
    if run_name != run_dir.name:
        print(f"[MISMATCH] {metrics_path}: run_name={run_name!r}, directory={run_dir.name!r}")
        ok = False
    if metrics.get("valid_run") is not True:
        print(f"[MISMATCH] {metrics_path}: valid_run={metrics.get('valid_run')!r}")
        ok = False
    if metrics.get("n_samples") != FINAL_N_SAMPLES:
        print(f"[MISMATCH] {metrics_path}: n_samples={metrics.get('n_samples')!r}, expected {FINAL_N_SAMPLES}")
        ok = False
    if str(metrics.get("invalid_reason") or "").strip():
        print(f"[MISMATCH] {metrics_path}: invalid_reason is not empty")
        ok = False
    prediction_rows = len(pd.read_csv(predictions_path))
    if prediction_rows != metrics.get("n_samples"):
        print(f"[MISMATCH] {predictions_path}: rows={prediction_rows}, n_samples={metrics.get('n_samples')!r}")
        ok = False
    return ok


def check_final_tables_and_report(phase4_dir: Path) -> bool:
    ok = True
    expected = set(FINAL_RUN_NAMES)
    results_path = phase4_dir / "tables" / "decoder_prompting_results.csv"
    per_author_path = phase4_dir / "tables" / "phase4_per_author_f1.csv"
    report_path = phase4_dir / "reports" / "phase4_decoder_prompting_report.md"
    if not check_path(results_path):
        ok = False
    else:
        results = pd.read_csv(results_path)
        run_names = set(results.get("run_name", pd.Series(dtype=str)).astype(str))
        extras = sorted(run_names.difference(expected))
        missing = sorted(expected.difference(run_names))
        if len(results) != len(FINAL_RUN_NAMES):
            print(f"[MISMATCH] {results_path}: rows={len(results)}, expected {len(FINAL_RUN_NAMES)}")
            ok = False
        if extras:
            print(f"[MISMATCH] {results_path}: non-final runs present: {', '.join(extras)}")
            ok = False
        if missing:
            print(f"[MISSING] {results_path}: final runs absent: {', '.join(missing)}")
            ok = False
        if results["run_name"].astype(str).str.contains("tinyllama", case=False, na=False).any():
            print(f"[MISMATCH] {results_path}: contains tinyllama_smoke")
            ok = False
        if "valid_run" in results.columns and (results["valid_run"].astype(str).str.lower() != "true").any():
            print(f"[MISMATCH] {results_path}: contains valid_run=false rows")
            ok = False
        if "n_samples" in results.columns and (pd.to_numeric(results["n_samples"], errors="coerce") <= 0).any():
            print(f"[MISMATCH] {results_path}: contains n_samples <= 0 rows")
            ok = False
        if "invalid_reason" in results.columns and results["invalid_reason"].fillna("").astype(str).str.strip().ne("").any():
            print(f"[MISMATCH] {results_path}: contains non-empty invalid_reason rows")
            ok = False
    if not check_path(per_author_path):
        ok = False
    else:
        per_author = pd.read_csv(per_author_path)
        expected_rows = len(FINAL_RUN_NAMES) * 6
        if len(per_author) != expected_rows:
            print(f"[MISMATCH] {per_author_path}: rows={len(per_author)}, expected {expected_rows}")
            ok = False
        if "run_name" in per_author.columns and per_author["run_name"].astype(str).str.contains("tinyllama_smoke", case=False, na=False).any():
            print(f"[MISMATCH] {per_author_path}: contains tinyllama_smoke")
            ok = False
    if not check_path(report_path):
        ok = False
    else:
        report = report_path.read_text(encoding="utf-8")
        if "tinyllama_smoke" in report.lower():
            print(f"[MISMATCH] {report_path}: contains tinyllama_smoke")
            ok = False
        if "Final-only report excludes diagnostic TinyLlama smoke runs." not in report:
            print(f"[MISSING] {report_path}: final-only scope note")
            ok = False
    return ok


def check_tables_reports_plots(phase4_dir: Path, allow_missing_runs: bool, final_only: bool = False) -> bool:
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
    if final_only:
        ok = check_final_tables_and_report(phase4_dir) and ok
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
    parser.add_argument("--final-only", action="store_true", help="Validate only the six final Phase 4 decoder benchmark runs.")
    args = parser.parse_args()
    phase4_dir = resolve_phase4_dir(args.phase4_dir)
    print(f"Resolved phase4_dir: {phase4_dir}")
    ok = check_configs()
    ok = check_prompts() and ok
    ok = check_outputs_outside_repo(phase4_dir) and ok
    ok = check_tracked_weights() and ok
    runs_dir = phase4_dir / "runs"
    if args.final_only:
        run_dirs = [runs_dir / name for name in FINAL_RUN_NAMES]
    else:
        run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir()) if runs_dir.exists() else []
    if not run_dirs:
        if args.allow_missing_runs:
            print(f"[WARN] no Phase 4 runs found at {runs_dir}")
        else:
            print(f"[MISSING] no Phase 4 runs found at {runs_dir}")
            ok = False
    for run_dir in run_dirs:
        if args.final_only:
            if not run_dir.exists():
                print(f"[MISSING] expected final run directory: {run_dir}")
                ok = False
                continue
            ok = check_final_run(run_dir) and ok
        else:
            ok = check_run(run_dir) and ok
    ok = check_tables_reports_plots(phase4_dir, args.allow_missing_runs, args.final_only) and ok
    if ok:
        print("Phase 4 output check passed.")
        return 0
    print("Phase 4 output check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
