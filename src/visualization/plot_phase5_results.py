"""Create Phase 5 QLoRA result plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 5 QLoRA outputs.")
    parser.add_argument("--phase5_dir", type=Path, default=Path("outputs/phase5"))
    parser.add_argument("--tables_dir", type=Path, default=Path("outputs/phase5/tables"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/phase5/plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase5_dir = resolve_repo_path(args.phase5_dir)
    tables_dir = resolve_repo_path(args.tables_dir)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    results = load_optional(tables_dir / "decoder_qlora_results.csv")
    prompting = load_optional(tables_dir / "prompting_vs_qlora.csv")
    skipped.extend(
        skip
        for skip in [
            plot_loss_curves(phase5_dir / "checkpoints", output_dir),
            plot_prompting_vs_qlora(prompting, output_dir / "prompting_vs_qlora_gain.png"),
            plot_metric_bar(results, "macro_f1", output_dir / "decoder_qlora_macro_f1_bar.png", "Macro F1"),
            plot_per_author_f1(phase5_dir / "eval", output_dir / "per_author_f1_heatmap.png"),
            plot_scatter(results, "peak_gpu_memory_allocated_gb", "macro_f1", output_dir / "gpu_memory_vs_performance.png"),
            plot_metric_bar(results, "macro_f1", output_dir / "lora_rank_ablation.png", "Macro F1", x_column="lora_r"),
        ]
        if skip
    )
    print(f"Wrote Phase 5 plots to {output_dir}")
    if skipped:
        print("Skipped plots:")
        for item in skipped:
            print(f"- {item}")


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def load_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_loss_curves(checkpoint_dir: Path, output_dir: Path) -> str | None:
    if not checkpoint_dir.exists():
        return "loss curves: checkpoint directory missing"
    plotted = 0
    for log_path in checkpoint_dir.glob("*/training_log.csv"):
        frame = pd.read_csv(log_path)
        if "step" not in frame.columns or not {"loss", "eval_loss"}.intersection(frame.columns):
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for column in ("loss", "eval_loss"):
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                ax.plot(frame["step"], values, marker="o", label=column)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"loss_curves_{log_path.parent.name}.png", dpi=300)
        plt.close(fig)
        plotted += 1
    return None if plotted else "loss curves: no readable training_log.csv files"


def plot_prompting_vs_qlora(frame: pd.DataFrame, output_path: Path) -> str | None:
    if frame.empty or not {"source", "accuracy"}.issubset(frame.columns):
        return "prompting_vs_qlora_gain: missing comparison table"
    data = frame.copy()
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce")
    grouped = data.groupby("source", dropna=False)["accuracy"].max().dropna()
    if grouped.empty:
        return "prompting_vs_qlora_gain: no accuracy values"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(grouped.index.astype(str), grouped.values, color="#4c78a8")
    ax.set_ylabel("Best Accuracy")
    ax.set_ylim(0, max(1.0, float(grouped.max()) * 1.15))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return None


def plot_metric_bar(
    frame: pd.DataFrame,
    metric: str,
    output_path: Path,
    ylabel: str,
    x_column: str = "experiment_name",
) -> str | None:
    if frame.empty or not {x_column, metric}.issubset(frame.columns):
        return f"{output_path.name}: missing {x_column}/{metric}"
    data = frame[[x_column, metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])
    if data.empty:
        return f"{output_path.name}: no metric values"
    data = data.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(data[x_column].astype(str), data[metric], color="#3a6ea5")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_column)
    ax.set_ylim(0, max(1.0, float(data[metric].max()) * 1.15))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return None


def plot_per_author_f1(eval_dir: Path, output_path: Path) -> str | None:
    rows = []
    if not eval_dir.exists():
        return "per_author_f1_heatmap: eval directory missing"
    for predictions_path in eval_dir.glob("*/classification_report.csv"):
        report = pd.read_csv(predictions_path, index_col=0)
        row = {"experiment_name": predictions_path.parent.name}
        for author in CANONICAL_AUTHORS:
            row[author] = float(report.loc[author, "f1-score"]) if author in report.index else 0.0
        rows.append(row)
    if not rows:
        return "per_author_f1_heatmap: no classification reports"
    frame = pd.DataFrame(rows).set_index("experiment_name")
    values = frame[CANONICAL_AUTHORS].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(frame))))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(CANONICAL_AUTHORS)))
    ax.set_xticklabels(CANONICAL_AUTHORS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(frame.index)))
    ax.set_yticklabels(frame.index)
    fig.colorbar(image, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return None


def plot_scatter(frame: pd.DataFrame, x_column: str, y_column: str, output_path: Path) -> str | None:
    if frame.empty or not {x_column, y_column}.issubset(frame.columns):
        return f"{output_path.name}: missing scatter columns"
    data = frame[[x_column, y_column, "experiment_name"]].copy()
    data[x_column] = pd.to_numeric(data[x_column], errors="coerce")
    data[y_column] = pd.to_numeric(data[y_column], errors="coerce")
    data = data.dropna(subset=[x_column, y_column])
    if data.empty:
        return f"{output_path.name}: no scatter values"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data[x_column], data[y_column], color="#4c78a8")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return None


if __name__ == "__main__":
    main()
