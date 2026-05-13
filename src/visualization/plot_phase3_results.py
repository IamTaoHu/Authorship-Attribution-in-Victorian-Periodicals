"""Create Phase 3 advanced encoder summary tables and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_path_config, phase_artifact_dir  # noqa: E402


PHASE2_ACCURACY = 0.9270
PHASE2_MACRO_F1 = 0.9191
PHASE2_REFERENCE = "roberta_large_phase2"
AUTHOR_AVG_ROWS = {"accuracy", "macro avg", "weighted avg"}
BOOLEAN_COLUMNS = (
    "smoke_test",
    "valid_run",
    "local_practical",
    "final_benchmark",
    "all_trainable_params_float32",
)
METRIC_COLUMNS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "precision_macro",
    "recall_macro",
)


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


def bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in BOOLEAN_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = False
        normalized[column] = normalized[column].map(bool_or_false)
    for column in METRIC_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in ("run_name", "experiment_name", "model_name", "baseline_reference", "invalid_reason"):
        if column not in normalized.columns:
            normalized[column] = ""
        normalized[column] = normalized[column].fillna("").astype(str)
    normalized["validity_status"] = normalized.apply(validity_status, axis=1)
    normalized["accuracy_delta_vs_phase2"] = normalized["accuracy"] - PHASE2_ACCURACY
    normalized["macro_f1_delta_vs_phase2"] = normalized["macro_f1"] - PHASE2_MACRO_F1
    return normalized


def validity_status(row: pd.Series) -> str:
    if bool_or_false(row.get("smoke_test")):
        return "smoke"
    if not bool_or_false(row.get("valid_run")):
        reason = str(row.get("invalid_reason", "") or "").strip()
        return f"invalid: {reason}" if reason else "invalid"
    if pd.isna(row.get("macro_f1")) or pd.isna(row.get("accuracy")):
        return "missing_metrics"
    return "valid"


def phase2_reference_row(columns: list[str]) -> dict[str, Any]:
    row = {column: pd.NA for column in columns}
    row.update(
        {
            "run_name": PHASE2_REFERENCE,
            "experiment_name": PHASE2_REFERENCE,
            "model_name": PHASE2_REFERENCE,
            "baseline_reference": "phase2",
            "smoke_test": False,
            "valid_run": True,
            "local_practical": False,
            "final_benchmark": True,
            "accuracy": PHASE2_ACCURACY,
            "macro_f1": PHASE2_MACRO_F1,
            "validity_status": "phase2_reference",
            "accuracy_delta_vs_phase2": 0.0,
            "macro_f1_delta_vs_phase2": 0.0,
        }
    )
    return row


def add_phase2_reference(frame: pd.DataFrame) -> pd.DataFrame:
    if (frame.get("run_name", pd.Series(dtype=str)) == PHASE2_REFERENCE).any():
        return frame
    columns = list(frame.columns)
    reference = phase2_reference_row(columns)
    return pd.concat([frame, pd.DataFrame([reference])], ignore_index=True)


def valid_phase3_runs(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame["valid_run"]
        & ~frame["smoke_test"]
        & frame["run_name"].ne(PHASE2_REFERENCE)
        & frame["accuracy"].notna()
        & frame["macro_f1"].notna()
    )
    return frame.loc[mask].copy()


def non_smoke_with_reference(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[~frame["smoke_test"]].copy()


def final_benchmark_with_reference(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame["valid_run"]
        & ~frame["smoke_test"]
        & frame["accuracy"].notna()
        & frame["macro_f1"].notna()
        & (frame["final_benchmark"] | frame["run_name"].eq(PHASE2_REFERENCE))
        & ~frame["local_practical"]
    )
    return frame.loc[mask].copy()


def classify_model_family(row: pd.Series) -> str:
    text = f"{row.get('run_name', '')} {row.get('model_name', '')}".lower()
    if "roberta_large_phase2" in text:
        return "phase2_roberta_large"
    if "modernbert" in text:
        return "modernbert"
    if "deberta" in text:
        return "deberta"
    return "other"


def write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")


def load_metrics_fallback(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"WARNING: Could not parse {path}")
        return {}


def load_results(results_path: Path, runs_dir: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Phase 3 aggregate results not found: {results_path}")
    frame = pd.read_csv(results_path)
    if frame.empty:
        raise ValueError(f"Phase 3 aggregate results are empty: {results_path}")
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        metrics = load_metrics_fallback(run_dir)
        if not metrics:
            continue
        run_name = str(metrics.get("run_name", run_dir.name))
        if run_name in set(frame["run_name"].astype(str)):
            continue
        row = {column: pd.NA for column in frame.columns}
        for key, value in metrics.items():
            if key in row:
                row[key] = value
        row["run_name"] = run_name
        row["experiment_name"] = metrics.get("experiment_name", run_name)
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    return frame


def build_best_runs(clean: pd.DataFrame) -> pd.DataFrame:
    best = valid_phase3_runs(clean).sort_values(["macro_f1", "accuracy"], ascending=False)
    best.insert(0, "rank", range(1, len(best) + 1))
    return best


def read_per_author_reports(clean: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    eligible = valid_phase3_runs(clean)
    for _, result in eligible.iterrows():
        run_name = str(result["run_name"])
        report_path = runs_dir / run_name / "classification_report.csv"
        if not report_path.exists():
            print(f"WARNING: Missing classification report for {run_name}: {report_path}")
            continue
        report = pd.read_csv(report_path)
        if "label" not in report.columns:
            print(f"WARNING: classification_report.csv lacks label column: {report_path}")
            continue
        report = report.loc[~report["label"].astype(str).isin(AUTHOR_AVG_ROWS)].copy()
        report["run_name"] = run_name
        report["model_name"] = result["model_name"]
        report["local_practical"] = result["local_practical"]
        report["final_benchmark"] = result["final_benchmark"]
        report["run_macro_f1"] = result["macro_f1"]
        rows.append(report)
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_name",
                "model_name",
                "label",
                "precision",
                "recall",
                "f1-score",
                "support",
                "local_practical",
                "final_benchmark",
                "run_macro_f1",
            ]
        )
    combined = pd.concat(rows, ignore_index=True)
    for column in ("precision", "recall", "f1-score", "support", "run_macro_f1"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def first_or_empty(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.iloc[0]


def summary_row(label: str, row: pd.Series | None, status: str = "available") -> dict[str, Any]:
    if row is None:
        return {
            "summary_item": label,
            "run_name": "",
            "model_name": "",
            "accuracy": pd.NA,
            "macro_f1": pd.NA,
            "accuracy_delta_vs_phase2": pd.NA,
            "macro_f1_delta_vs_phase2": pd.NA,
            "status": status,
        }
    return {
        "summary_item": label,
        "run_name": row.get("run_name", ""),
        "model_name": row.get("model_name", ""),
        "accuracy": row.get("accuracy", pd.NA),
        "macro_f1": row.get("macro_f1", pd.NA),
        "accuracy_delta_vs_phase2": row.get("accuracy_delta_vs_phase2", pd.NA),
        "macro_f1_delta_vs_phase2": row.get("macro_f1_delta_vs_phase2", pd.NA),
        "status": status,
    }


def build_comparison_summary(clean: pd.DataFrame, best_runs: pd.DataFrame) -> pd.DataFrame:
    valid = valid_phase3_runs(clean)
    valid["model_family"] = valid.apply(classify_model_family, axis=1)
    final_valid = valid.loc[valid["final_benchmark"]]
    local_valid = valid.loc[valid["local_practical"]]
    final_status = "available" if not final_valid.empty else "pending"
    reference = clean.loc[clean["run_name"].eq(PHASE2_REFERENCE)]
    rows = [
        summary_row("phase2_reference", first_or_empty(reference), "available"),
        summary_row("best_valid_phase3", first_or_empty(best_runs), "available" if not best_runs.empty else "pending"),
        summary_row(
            "best_local_deberta",
            first_or_empty(local_valid.loc[local_valid["model_family"].eq("deberta")].sort_values("macro_f1", ascending=False)),
            "available",
        ),
        summary_row(
            "best_modernbert",
            first_or_empty(valid.loc[valid["model_family"].eq("modernbert")].sort_values("macro_f1", ascending=False)),
            "available",
        ),
        summary_row(
            "best_final_deberta",
            first_or_empty(final_valid.loc[final_valid["model_family"].eq("deberta")].sort_values("macro_f1", ascending=False)),
            final_status,
        ),
        summary_row(
            "best_final_modernbert",
            first_or_empty(final_valid.loc[final_valid["model_family"].eq("modernbert")].sort_values("macro_f1", ascending=False)),
            final_status,
        ),
    ]
    summary = pd.DataFrame(rows)
    summary["final_benchmark_status"] = final_status
    return summary


def format_metric(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def write_markdown_summary(summary: pd.DataFrame, best_runs: pd.DataFrame, path: Path) -> None:
    best = first_or_empty(best_runs)
    final_status = summary["final_benchmark_status"].iloc[0] if not summary.empty else "pending"
    lines = [
        "# Phase 3 Visualization Summary",
        "",
        f"Final benchmark status: `{final_status}`",
        "",
        "## Phase 2 Reference",
        "",
        f"- Model: `{PHASE2_REFERENCE}`",
        f"- Accuracy: {PHASE2_ACCURACY:.4f}",
        f"- Macro F1: {PHASE2_MACRO_F1:.4f}",
        "",
        "## Best Valid Phase 3 Run",
        "",
        "Local practical runs are sanity/debug experiments for local hardware. They remain in the clean tables and diagnostic plots, but they are excluded from the main final benchmark comparison plots.",
        "",
    ]
    if best is None:
        lines.append("- No valid non-smoke Phase 3 run found.")
    else:
        lines.extend(
            [
                f"- Run: `{best['run_name']}`",
                f"- Model: `{best['model_name']}`",
                f"- Accuracy: {format_metric(best['accuracy'])}",
                f"- Macro F1: {format_metric(best['macro_f1'])}",
                f"- Accuracy delta vs Phase 2 RoBERTa-large: {format_metric(best['accuracy_delta_vs_phase2'])}",
                f"- Macro F1 delta vs Phase 2 RoBERTa-large: {format_metric(best['macro_f1_delta_vs_phase2'])}",
            ]
        )
    lines.extend(["", "## Comparison Summary", ""])
    if summary.empty:
        lines.append("- No comparison rows available.")
    else:
        lines.append("| Summary Item | Run | Accuracy | Macro F1 | Delta Macro F1 vs Phase 2 | Status |")
        lines.append("|---|---|---:|---:|---:|---|")
        for _, row in summary.iterrows():
            lines.append(
                "| "
                f"{row.get('summary_item', '')} | "
                f"`{row.get('run_name', '')}` | "
                f"{format_metric(row.get('accuracy'))} | "
                f"{format_metric(row.get('macro_f1'))} | "
                f"{format_metric(row.get('macro_f1_delta_vs_phase2'))} | "
                f"{row.get('status', '')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def plot_metric_bar(frame: pd.DataFrame, metric: str, output_path: Path, title: str, valid_only: bool) -> None:
    plot_frame = frame.loc[frame[metric].notna()].sort_values(metric, ascending=False).copy()
    if plot_frame.empty:
        print(f"WARNING: Cannot plot {metric}; no rows available.")
        return
    colors = []
    for _, row in plot_frame.iterrows():
        if row["run_name"] == PHASE2_REFERENCE:
            colors.append("#222222")
        elif not row["valid_run"]:
            colors.append("#BDBDBD")
        elif row["final_benchmark"]:
            colors.append("#4C78A8")
        elif row["local_practical"]:
            colors.append("#F58518")
        else:
            colors.append("#72B7B2")
    plt.figure(figsize=(max(9, 0.65 * len(plot_frame)), 5))
    bars = plt.bar(plot_frame["run_name"].astype(str), plot_frame[metric].astype(float), color=colors)
    if not valid_only:
        for bar, (_, row) in zip(bars, plot_frame.iterrows()):
            if not row["valid_run"]:
                bar.set_hatch("//")
    plt.axhline(PHASE2_MACRO_F1 if metric == "macro_f1" else PHASE2_ACCURACY, color="#222222", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("Run")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def plot_status(clean: pd.DataFrame, output_path: Path) -> None:
    phase3 = clean.loc[clean["run_name"].ne(PHASE2_REFERENCE)].copy()
    if phase3.empty:
        print("WARNING: Cannot plot status; no Phase 3 rows available.")
        return
    def status(row: pd.Series) -> str:
        if row["smoke_test"]:
            return "smoke"
        if not row["valid_run"]:
            return "invalid"
        if row["final_benchmark"]:
            return "final valid"
        if row["local_practical"]:
            return "local valid"
        return "other valid"
    phase3["status"] = phase3.apply(status, axis=1)
    counts = phase3["status"].value_counts().rename_axis("status").reset_index(name="count")
    plt.figure(figsize=(8, 4.5))
    plt.bar(counts["status"].astype(str), counts["count"].astype(int), color="#72B7B2")
    plt.xlabel("Run Status")
    plt.ylabel("Count")
    plt.title("Phase 3 Local, Final, Smoke, and Invalid Run Status")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def plot_per_author_heatmap(per_author: pd.DataFrame, output_path: Path) -> None:
    if per_author.empty:
        print("WARNING: Cannot plot per-author F1 heatmap; no per-author rows available.")
        return
    pivot = per_author.pivot_table(index="label", columns="run_name", values="f1-score", aggfunc="first")
    ordered_columns = per_author[["run_name", "run_macro_f1"]].drop_duplicates().sort_values("run_macro_f1", ascending=False)["run_name"]
    pivot = pivot.reindex(columns=list(ordered_columns))
    plt.figure(figsize=(max(10, 0.8 * len(pivot.columns)), 5.5))
    values = pivot.to_numpy(dtype=float)
    image = plt.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(image, label="F1")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if pd.notna(value):
                plt.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Run")
    plt.ylabel("Author")
    plt.title("Phase 3 Per-Author F1")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def plot_confusion_matrix(run_name: str, runs_dir: Path, output_path: Path, title: str) -> None:
    matrix_path = runs_dir / run_name / "confusion_matrix.csv"
    if not matrix_path.exists():
        print(f"WARNING: Cannot plot confusion matrix; missing {matrix_path}")
        return
    matrix = pd.read_csv(matrix_path, index_col=0)
    plt.figure(figsize=(8, 6.5))
    values = matrix.to_numpy()
    image = plt.imshow(values, cmap="Blues")
    plt.colorbar(image)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            plt.text(column_index, row_index, str(values[row_index, column_index]), ha="center", va="center", fontsize=8)
    plt.xticks(range(len(matrix.columns)), matrix.columns)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.xlabel("Predicted Author")
    plt.ylabel("True Author")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3_dir", default="artifacts/phase3")
    args = parser.parse_args()

    phase3_dir = resolve_phase3_dir(args.phase3_dir)
    tables_dir = phase_artifact_dir("phase3", "tables")
    plots_dir = phase_artifact_dir("phase3", "plots")
    runs_dir = phase3_dir / "runs"
    results_path = phase3_dir / "tables" / "advanced_encoder_results.csv"

    raw = load_results(results_path, runs_dir)
    clean = normalize_results(raw)
    clean = add_phase2_reference(clean)
    clean["model_family"] = clean.apply(classify_model_family, axis=1)
    clean = clean.sort_values(["macro_f1", "accuracy"], ascending=False, na_position="last")
    write_table(clean, tables_dir / "advanced_encoder_results_clean.csv")

    best_runs = build_best_runs(clean)
    write_table(best_runs, tables_dir / "phase3_best_valid_runs.csv")

    per_author = read_per_author_reports(clean, runs_dir)
    write_table(per_author, tables_dir / "phase3_per_author_f1.csv")

    summary = build_comparison_summary(clean, best_runs)
    write_table(summary, tables_dir / "phase3_model_comparison_summary.csv")
    write_markdown_summary(summary, best_runs, tables_dir / "phase3_visualization_summary.md")

    all_non_smoke = non_smoke_with_reference(clean)
    all_valid_with_reference = all_non_smoke.loc[
        all_non_smoke["valid_run"] & all_non_smoke["accuracy"].notna() & all_non_smoke["macro_f1"].notna()
    ].copy()
    final_with_reference = final_benchmark_with_reference(clean)

    plot_metric_bar(all_non_smoke, "accuracy", plots_dir / "phase3_accuracy_bar.png", "Phase 3 Accuracy With Phase 2 Reference", False)
    plot_metric_bar(all_non_smoke, "macro_f1", plots_dir / "phase3_macro_f1_bar.png", "Phase 3 Macro F1 With Phase 2 Reference", False)
    plot_metric_bar(final_with_reference, "accuracy", plots_dir / "phase3_valid_runs_only_accuracy.png", "Final Benchmark Accuracy", True)
    plot_metric_bar(final_with_reference, "macro_f1", plots_dir / "phase3_valid_runs_only_macro_f1.png", "Final Benchmark Macro F1", True)
    plot_metric_bar(
        all_valid_with_reference,
        "accuracy",
        plots_dir / "phase3_all_valid_runs_accuracy_diagnostic.png",
        "All Valid Runs Accuracy Diagnostic",
        True,
    )
    plot_metric_bar(
        all_valid_with_reference,
        "macro_f1",
        plots_dir / "phase3_all_valid_runs_macro_f1_diagnostic.png",
        "All Valid Runs Macro F1 Diagnostic",
        True,
    )
    plot_status(clean, plots_dir / "phase3_local_vs_final_status.png")
    plot_per_author_heatmap(per_author, plots_dir / "phase3_per_author_f1_heatmap.png")

    valid_phase3 = valid_phase3_runs(clean)
    best_deberta = first_or_empty(
        valid_phase3.loc[valid_phase3["model_family"].eq("deberta")].sort_values(["macro_f1", "accuracy"], ascending=False)
    )
    best_modernbert = first_or_empty(
        valid_phase3.loc[valid_phase3["model_family"].eq("modernbert")].sort_values(["macro_f1", "accuracy"], ascending=False)
    )
    if best_deberta is not None:
        plot_confusion_matrix(
            str(best_deberta["run_name"]),
            runs_dir,
            plots_dir / "phase3_confusion_matrix_best_deberta.png",
            f"Best DeBERTa Confusion Matrix: {best_deberta['run_name']}",
        )
    if best_modernbert is not None:
        plot_confusion_matrix(
            str(best_modernbert["run_name"]),
            runs_dir,
            plots_dir / "phase3_confusion_matrix_modernbert.png",
            f"Best ModernBERT Confusion Matrix: {best_modernbert['run_name']}",
        )

    if not best_runs.empty:
        best = best_runs.iloc[0]
        print(
            "Best valid Phase 3 run: "
            f"{best['run_name']} accuracy={best['accuracy']:.4f} macro_f1={best['macro_f1']:.4f} "
            f"delta_macro_f1_vs_phase2={best['macro_f1_delta_vs_phase2']:.4f}"
        )
    final_status = summary["final_benchmark_status"].iloc[0] if not summary.empty else "pending"
    print(f"Final benchmark status: {final_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
