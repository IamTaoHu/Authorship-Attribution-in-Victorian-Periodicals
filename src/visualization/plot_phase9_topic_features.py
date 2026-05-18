"""Create Phase 9 topic-aware classification plots from saved outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.topic_features_phase9 import CANONICAL_AUTHORS, default_phase9_dir  # noqa: E402


def resolve_phase9_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path
    return default_phase9_dir()


def _save_bar(frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_frame = frame.copy()
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.loc[plot_frame[metric].notna()].sort_values(metric, ascending=False)
    plt.figure(figsize=(max(8.5, 0.85 * len(plot_frame)), 4.8))
    plt.bar(plot_frame["variant"].astype(str), plot_frame[metric], color="#4C78A8")
    plt.ylim(0, 1)
    plt.xlabel("Variant")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_per_author_f1(per_author: pd.DataFrame, output_path: Path) -> None:
    frame = per_author.copy()
    frame["f1_score"] = pd.to_numeric(frame["f1_score"], errors="coerce")
    pivot = frame.pivot_table(index="author", columns="variant", values="f1_score", aggfunc="first").reindex(index=list(CANONICAL_AUTHORS))
    plt.figure(figsize=(max(10, 1.1 * len(pivot.columns)), 5.8))
    if sns is not None:
        sns.heatmap(pivot, cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".2f", linewidths=0.4, cbar_kws={"label": "F1"})
    else:
        image = plt.imshow(pivot.fillna(0).to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        plt.colorbar(image, label="F1")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.xlabel("Variant")
    plt.ylabel("Author")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_delta(delta: pd.DataFrame, output_path: Path) -> None:
    if delta.empty or "delta_f1_vs_text_only" not in delta.columns:
        return
    frame = delta.copy()
    frame["delta_f1_vs_text_only"] = pd.to_numeric(frame["delta_f1_vs_text_only"], errors="coerce")
    frame = frame.loc[frame["delta_f1_vs_text_only"].notna()]
    pivot = frame.pivot_table(index="author", columns="variant", values="delta_f1_vs_text_only", aggfunc="first").reindex(index=list(CANONICAL_AUTHORS))
    vmax = max(0.05, float(pivot.abs().max().max()) if not pivot.empty else 0.05)
    plt.figure(figsize=(max(10, 1.1 * len(pivot.columns)), 5.8))
    if sns is not None:
        sns.heatmap(pivot, cmap="RdBu", center=0, vmin=-vmax, vmax=vmax, annot=True, fmt=".2f", linewidths=0.4, cbar_kws={"label": "Delta F1"})
    else:
        image = plt.imshow(pivot.fillna(0).to_numpy(dtype=float), aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
        plt.colorbar(image, label="Delta F1")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.xlabel("Variant")
    plt.ylabel("Author")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_confusion_matrix(matrix_path: Path, output_path: Path) -> None:
    if not matrix_path.exists():
        return
    matrix = pd.read_csv(matrix_path, index_col=0).reindex(index=list(CANONICAL_AUTHORS), columns=list(CANONICAL_AUTHORS), fill_value=0)
    plt.figure(figsize=(8, 6.5))
    if sns is not None:
        sns.heatmap(matrix, cmap="Blues", annot=True, fmt="g", linewidths=0.4, cbar_kws={"label": "Count"})
    else:
        image = plt.imshow(matrix.to_numpy(dtype=float), cmap="Blues")
        plt.colorbar(image, label="Count")
        plt.yticks(range(len(matrix.index)), matrix.index)
        plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                plt.text(col, row, str(int(matrix.iloc[row, col])), ha="center", va="center", fontsize=8)
    plt.xlabel("Predicted Author")
    plt.ylabel("True Author")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_topic_only_vs_transformer(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary.copy()
    frame["macro_f1"] = pd.to_numeric(frame["macro_f1"], errors="coerce")
    frame["accuracy"] = pd.to_numeric(frame["accuracy"], errors="coerce")
    frame = frame.loc[frame["macro_f1"].notna() & frame["accuracy"].notna()]
    groups = frame["model_type"].astype(str).map(lambda value: "topic_only" if value == "sklearn_topic_only" else "transformer")
    x = range(len(frame))
    colors = groups.map({"topic_only": "#F58518", "transformer": "#4C78A8"}).fillna("#777777")
    plt.figure(figsize=(max(9, 0.85 * len(frame)), 4.8))
    plt.scatter(x, frame["macro_f1"], s=80, color=colors)
    for idx, name in enumerate(frame["variant"].astype(str)):
        plt.text(idx, frame["macro_f1"].iloc[idx] + 0.015, name.replace("_", "\n"), ha="center", va="bottom", fontsize=7)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.ylabel("Macro F1")
    plt.xlabel("Topic-only and transformer variants")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_phase9_topic_features(phase9_dir: str | Path | None = None) -> dict[str, Path]:
    root = resolve_phase9_dir(str(phase9_dir) if phase9_dir else None)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    summary_path = root / "tables" / "phase9_results_summary.csv"
    per_author_path = root / "tables" / "per_author_f1_comparison.csv"
    delta_path = root / "tables" / "per_author_f1_delta_vs_text_only.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Phase 9 summary table: {summary_path}")
    summary = pd.read_csv(summary_path)
    _save_bar(summary, "macro_f1", plots / "phase9_macro_f1_comparison.png")
    _save_bar(summary, "accuracy", plots / "phase9_accuracy_comparison.png")
    if per_author_path.exists():
        plot_per_author_f1(pd.read_csv(per_author_path), plots / "per_author_f1_comparison.png")
    if delta_path.exists():
        plot_delta(pd.read_csv(delta_path), plots / "per_author_f1_delta_vs_text_only.png")
    plot_confusion_matrix(root / "tables" / "deberta_topic_concat_confusion_matrix.csv", plots / "confusion_matrix_deberta_topic_concat.png")
    plot_topic_only_vs_transformer(summary, plots / "topic_only_vs_transformer_summary.png")
    return {
        "macro_f1": plots / "phase9_macro_f1_comparison.png",
        "accuracy": plots / "phase9_accuracy_comparison.png",
        "per_author_f1": plots / "per_author_f1_comparison.png",
        "delta": plots / "per_author_f1_delta_vs_text_only.png",
        "confusion": plots / "confusion_matrix_deberta_topic_concat.png",
        "summary": plots / "topic_only_vs_transformer_summary.png",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase9_dir", default=None)
    args = parser.parse_args()
    plot_phase9_topic_features(args.phase9_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
