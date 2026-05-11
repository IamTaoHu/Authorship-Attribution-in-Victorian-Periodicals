"""Aggregate Phase 5 decoder QLoRA evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULT_COLUMNS = [
    "experiment_name",
    "model_name",
    "lora_r",
    "seed",
    "n_samples",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_output_rate",
    "total_inference_time_sec",
    "avg_inference_time_sec",
    "trainable_params",
    "total_params",
    "trainable_param_pct",
    "peak_gpu_memory_allocated_gb",
    "adapter_dir",
    "predictions_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 5 QLoRA outputs.")
    parser.add_argument("--eval_dir", type=Path, default=Path("outputs/phase5/eval"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("outputs/phase5/checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/phase5/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = resolve_repo_path(args.eval_dir)
    checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = collect_rows(eval_dir, checkpoint_dir)
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not results.empty:
        results = results.sort_values(["macro_f1", "accuracy", "experiment_name"], ascending=[False, False, True], na_position="last")
    results.to_csv(output_dir / "decoder_qlora_results.csv", index=False)
    results.to_csv(output_dir / "lora_ablation_results.csv", index=False)
    write_comparison_tables(results, output_dir)
    markdown = dataframe_to_markdown(results) if not results.empty else "| No Phase 5 QLoRA runs found |\n| --- |"
    (output_dir / "benchmark_summary.md").write_text(markdown + "\n", encoding="utf-8")
    pd.DataFrame(skipped, columns=["experiment_name", "run_dir", "skip_reason"]).to_csv(output_dir / "skipped_runs.csv", index=False)
    print(f"Wrote Phase 5 result tables to {output_dir}")


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def collect_rows(eval_dir: Path, checkpoint_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = []
    skipped = []
    if not eval_dir.exists():
        return rows, skipped
    for run_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
        try:
            metrics_path = run_dir / "metrics.json"
            predictions_path = run_dir / "predictions.csv"
            if not metrics_path.exists() or not predictions_path.exists():
                raise ValueError("missing metrics.json or predictions.csv")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest = read_manifest(checkpoint_dir / run_dir.name / "adapter_manifest.json")
            params = read_manifest(checkpoint_dir / run_dir.name / "trainable_params.json")
            row = {column: metrics.get(column) for column in RESULT_COLUMNS}
            row["experiment_name"] = metrics.get("experiment_name") or run_dir.name
            row["adapter_dir"] = manifest.get("adapter_dir")
            row["predictions_path"] = str(predictions_path)
            for key in ("trainable_params", "total_params", "trainable_param_pct"):
                row[key] = params.get(key, manifest.get(key))
            rows.append(row)
        except Exception as exc:
            skipped.append({"experiment_name": run_dir.name, "run_dir": str(run_dir), "skip_reason": str(exc)})
    return rows, skipped


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_comparison_tables(results: pd.DataFrame, output_dir: Path) -> None:
    prompting = load_optional_csv(PROJECT_ROOT / "outputs" / "phase4" / "tables" / "final_decoder_results_table.csv")
    encoder = load_optional_csv(PROJECT_ROOT / "outputs" / "phase3" / "tables" / "final_encoder_results_table.csv")
    qlora = results.copy()
    qlora["source"] = "phase5_qlora"
    if not prompting.empty:
        prompt_compare = pd.concat([prompting.assign(source="phase4_prompting"), qlora], ignore_index=True, sort=False)
    else:
        prompt_compare = qlora
    prompt_compare.to_csv(output_dir / "prompting_vs_qlora.csv", index=False)
    if not encoder.empty:
        decoder_vs_encoder = pd.concat([encoder.assign(source="phase3_encoder"), qlora], ignore_index=True, sort=False)
    else:
        decoder_vs_encoder = qlora
    decoder_vs_encoder.to_csv(output_dir / "decoder_vs_encoder.csv", index=False)


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
