"""Create Phase 4 decoder prompting summary plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANONICAL_AUTHORS = [
    "Leslie Stephen",
    "John Morley",
    "Eliza Lynn Linton",
    "George Henry Lewes",
    "Anne Mozley",
    "James Fitzjames Stephen",
]
INVALID_AUTHOR_LABEL = "INVALID_OR_EMPTY"
TABLE_FILES = {
    "benchmark": "benchmark_summary.csv",
    "parsed_status": "parsed_status_summary.csv",
    "prediction_distribution": "prediction_distribution.csv",
    "final_decoder": "final_decoder_results_table.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 4 decoder prompting results.")
    parser.add_argument("--phase4_dir", type=Path, default=Path("outputs/phase4"))
    parser.add_argument("--tables_dir", type=Path, default=Path("outputs/phase4/tables"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/phase4/plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase4_dir = resolve_repo_path(args.phase4_dir)
    tables_dir = resolve_repo_path(args.tables_dir)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    benchmark = load_required_table(tables_dir / TABLE_FILES["benchmark"])
    parsed_status = load_required_table(tables_dir / TABLE_FILES["parsed_status"])
    prediction_distribution = load_required_table(tables_dir / TABLE_FILES["prediction_distribution"])
    load_optional_table(tables_dir / TABLE_FILES["final_decoder"])

    run_order = benchmark["run_name"].dropna().astype(str).tolist()

    skipped.extend(
        [
            skip
            for skip in [
                plot_metric_bar(
                    benchmark,
                    metric="accuracy",
                    output_path=output_dir / "decoder_accuracy_bar.png",
                    ylabel="Accuracy",
                    sort_descending=True,
                    add_value_labels=True,
                ),
                plot_metric_bar(
                    benchmark,
                    metric="macro_f1",
                    output_path=output_dir / "decoder_macro_f1_bar.png",
                    ylabel="Macro F1",
                    run_order=run_order,
                ),
                plot_metric_bar(
                    benchmark,
                    metric="invalid_output_rate",
                    output_path=output_dir / "invalid_output_rate_bar.png",
                    ylabel="Invalid Output Rate",
                    run_order=run_order,
                ),
                plot_stacked_bar(
                    parsed_status,
                    category_column="parsed_status",
                    output_path=output_dir / "parsed_status_distribution.png",
                    ylabel="Count",
                    run_order=run_order,
                ),
                plot_prediction_distribution(
                    prediction_distribution,
                    output_path=output_dir / "prediction_distribution_by_run.png",
                    run_order=run_order,
                ),
                plot_per_author_f1_heatmap(
                    phase4_dir,
                    output_path=output_dir / "per_author_f1_heatmap.png",
                    run_order=run_order,
                ),
            ]
            if skip is not None
        ]
    )

    print(f"Wrote Phase 4 plots to {output_dir}")
    if skipped:
        print("Skipped plots:")
        for reason in skipped:
            print(f"- {reason}")
    else:
        print("Skipped plots: none")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_required_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 4 table not found: {path}")
    return pd.read_csv(path)


def load_optional_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> bool:
    missing = sorted(columns - set(frame.columns))
    if missing:
        print(f"Skipping {source}: missing columns {missing}")
        return False
    return True


def plot_metric_bar(
    frame: pd.DataFrame,
    metric: str,
    output_path: Path,
    ylabel: str,
    sort_descending: bool = False,
    add_value_labels: bool = False,
    run_order: list[str] | None = None,
) -> str | None:
    if not require_columns(frame, {"run_name", metric}, output_path.name):
        return f"{output_path.name}: missing required columns"

    data = frame[["run_name", metric]].copy()
    data["run_name"] = data["run_name"].astype(str)
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])
    if sort_descending:
        data = data.sort_values(metric, ascending=False)
    elif run_order is not None:
        data = order_by_run(data, run_order)
    if data.empty:
        return f"{output_path.name}: no non-null {metric} values"

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(data["run_name"], data[metric], color="#3a6ea5")
    ax.set_xlabel("Run")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, float(data[metric].max()) * 1.15))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    if add_value_labels:
        for bar, value in zip(bars, data[metric], strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return None


def plot_stacked_bar(
    frame: pd.DataFrame,
    category_column: str,
    output_path: Path,
    ylabel: str,
    run_order: list[str],
) -> str | None:
    required = {"run_name", category_column, "count"}
    if not require_columns(frame, required, output_path.name):
        return f"{output_path.name}: missing required columns"

    data = frame[["run_name", category_column, "count"]].copy()
    data["run_name"] = data["run_name"].astype(str)
    data[category_column] = data[category_column].fillna(INVALID_AUTHOR_LABEL).astype(str)
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)
    if data.empty:
        return f"{output_path.name}: no rows to plot"

    pivot = data.pivot_table(index="run_name", columns=category_column, values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([run for run in run_order if run in pivot.index])
    if pivot.empty:
        return f"{output_path.name}: no matching runs to plot"

    plot_pivot_stacked_bar(pivot, output_path, ylabel)
    return None


def plot_prediction_distribution(frame: pd.DataFrame, output_path: Path, run_order: list[str]) -> str | None:
    required = {"run_name", "predicted_author", "count"}
    if not require_columns(frame, required, output_path.name):
        return f"{output_path.name}: missing required columns"

    data = frame[["run_name", "predicted_author", "count"]].copy()
    data["run_name"] = data["run_name"].astype(str)
    data["predicted_author"] = data["predicted_author"].map(normalize_predicted_author)
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)
    if data.empty:
        return f"{output_path.name}: no rows to plot"

    pivot = data.pivot_table(index="run_name", columns="predicted_author", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([run for run in run_order if run in pivot.index])
    pivot = pivot.reindex(columns=[*CANONICAL_AUTHORS, INVALID_AUTHOR_LABEL], fill_value=0)
    if pivot.empty:
        return f"{output_path.name}: no matching runs to plot"

    plot_pivot_stacked_bar(pivot, output_path, "Count")
    return None


def plot_pivot_stacked_bar(pivot: pd.DataFrame, output_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bottom = np.zeros(len(pivot), dtype=float)
    colors = plt.get_cmap("tab20").colors
    for index, column in enumerate(pivot.columns):
        values = pivot[column].to_numpy(dtype=float)
        ax.bar(pivot.index, values, bottom=bottom, label=str(column), color=colors[index % len(colors)])
        bottom += values
    ax.set_xlabel("Run")
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_per_author_f1_heatmap(phase4_dir: Path, output_path: Path, run_order: list[str]) -> str | None:
    rows: list[dict[str, float | str]] = []
    run_dirs = [phase4_dir / run_name for run_name in run_order]
    for run_dir in run_dirs:
        predictions_path = run_dir / "predictions.csv"
        if not predictions_path.exists():
            continue
        predictions = pd.read_csv(predictions_path)
        if not {"true_author", "predicted_author"}.issubset(predictions.columns):
            continue
        y_true = predictions["true_author"].map(normalize_author_name)
        y_pred = predictions["predicted_author"].map(normalize_author_name)
        y_pred = y_pred.where(y_pred.isin(CANONICAL_AUTHORS), INVALID_AUTHOR_LABEL)
        _, _, f1_scores, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CANONICAL_AUTHORS,
            zero_division=0,
        )
        row: dict[str, float | str] = {"run_name": run_dir.name}
        row.update(dict(zip(CANONICAL_AUTHORS, f1_scores, strict=True)))
        rows.append(row)

    if not rows:
        return f"{output_path.name}: no readable run-level predictions.csv files"

    frame = pd.DataFrame(rows).set_index("run_name")
    frame = frame.reindex([run for run in run_order if run in frame.index])
    values = frame[CANONICAL_AUTHORS].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.6 * len(frame))))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(CANONICAL_AUTHORS)))
    ax.set_yticks(np.arange(len(frame.index)))
    ax.set_xticklabels(CANONICAL_AUTHORS)
    ax.set_yticklabels(frame.index)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_xlabel("Author")
    ax.set_ylabel("Run")
    fig.colorbar(image, ax=ax, label="F1")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text_color = "white" if value >= 0.55 else "black"
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return None


def normalize_author_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_predicted_author(value: object) -> str:
    author = normalize_author_name(value)
    if not author or author.upper() in {"INVALID", "__INVALID__", "NAN", "NONE"}:
        return INVALID_AUTHOR_LABEL
    return author if author in CANONICAL_AUTHORS else INVALID_AUTHOR_LABEL


def order_by_run(frame: pd.DataFrame, run_order: list[str]) -> pd.DataFrame:
    order = {run_name: index for index, run_name in enumerate(run_order)}
    data = frame.copy()
    data["_run_order"] = data["run_name"].map(order)
    return data.sort_values("_run_order", na_position="last").drop(columns=["_run_order"])


if __name__ == "__main__":
    main()
