"""Create Phase 7 LDA plots from saved CSV artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from src.utils.paths import get_path_config  # noqa: E402


def default_phase7_dir() -> Path:
    return get_path_config()["artifacts_root"] / "phase7" / "lda"


def topic_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("topic_")]


def save_line_plot(frame: pd.DataFrame, y_column: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(7.5, 4.8))
    if sns is not None:
        sns.lineplot(data=frame, x="k", y=y_column, marker="o", linewidth=2)
    else:
        plt.plot(frame["k"], frame[y_column], marker="o", linewidth=2)
    selected = frame.loc[frame.get("selected", False).astype(bool)]
    if not selected.empty:
        plt.scatter(selected["k"], selected[y_column], color="#D62728", s=70, zorder=3, label="Selected k")
        plt.legend()
    plt.xlabel("Number of Topics (k)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_author_topic_heatmap(author_topics: pd.DataFrame, output_path: Path) -> None:
    columns = topic_columns(author_topics)
    matrix = author_topics.set_index("author")[columns]
    values = matrix.to_numpy(dtype=float)
    vmax = float(values.max()) if values.size else 1.0
    plt.figure(figsize=(max(10, 0.8 * len(columns)), 6.2))
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Mean Topic Probability"},
            vmin=0,
            vmax=vmax,
            annot_kws={"fontsize": 8},
        )
    else:
        image = plt.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        plt.colorbar(image, label="Mean Topic Probability")
        plt.yticks(range(len(matrix.index)), matrix.index)
        plt.xticks(range(len(columns)), columns, rotation=45, ha="right")
        threshold = vmax * 0.6
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                color = "white" if values[row, column] >= threshold else "#222222"
                plt.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", color=color, fontsize=8)
    plt.xlabel("Topic")
    plt.ylabel("Author")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_author_topic_stacked_bar(author_topics: pd.DataFrame, output_path: Path) -> None:
    columns = topic_columns(author_topics)
    plot_frame = author_topics.set_index("author")[columns]
    colors = sns.color_palette("tab20", n_colors=len(columns)) if sns is not None else plt.cm.tab20.colors
    ax = plot_frame.plot(kind="bar", stacked=True, figsize=(10, 5.5), color=colors, width=0.8)
    ax.set_xlabel("Author")
    ax.set_ylabel("Mean Topic Probability")
    ax.legend(title="Topic", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_entropy_by_author(entropy: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5.2))
    if sns is not None:
        sns.boxplot(data=entropy, x="author", y="normalized_topic_entropy", color="#9ECAE1")
        sns.pointplot(data=entropy, x="author", y="normalized_topic_entropy", color="#1F77B4", errorbar=None, markers="D", scale=0.65)
    else:
        authors = list(dict.fromkeys(entropy["author"].astype(str)))
        data = [entropy.loc[entropy["author"].astype(str) == author, "normalized_topic_entropy"].dropna().to_numpy() for author in authors]
        plt.boxplot(data, labels=authors, patch_artist=True, boxprops={"facecolor": "#9ECAE1"})
        means = [values.mean() if len(values) else float("nan") for values in data]
        plt.plot(range(1, len(authors) + 1), means, "D", color="#1F77B4", markersize=5)
    plt.xlabel("Author")
    plt.ylabel("Normalized Topic Entropy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_phase7_lda(phase7_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(phase7_dir).expanduser() if phase7_dir else default_phase7_dir()
    tables_dir = root / "tables"
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(tables_dir / "lda_model_selection.csv")
    author_topics = pd.read_csv(tables_dir / "author_topic_distribution.csv")
    entropy = pd.read_csv(tables_dir / "topic_entropy_by_document.csv")
    summary_path = root / "metrics" / "lda_summary.json"
    if summary_path.exists():
        json.loads(summary_path.read_text(encoding="utf-8"))

    coherence_or_diversity = "approx_coherence" if selection["approx_coherence"].notna().any() else "topic_diversity"
    coherence_label = "Approximate Coherence" if coherence_or_diversity == "approx_coherence" else "Topic Diversity"

    outputs = {
        "perplexity": plots_dir / "model_selection_perplexity.png",
        "coherence_or_diversity": plots_dir / "model_selection_coherence_or_diversity.png",
        "author_heatmap": plots_dir / "author_topic_heatmap.png",
        "author_stacked_bar": plots_dir / "author_topic_stacked_bar.png",
        "entropy_by_author": plots_dir / "topic_entropy_by_author.png",
    }
    save_line_plot(selection, "perplexity", "Perplexity", outputs["perplexity"])
    save_line_plot(selection, coherence_or_diversity, coherence_label, outputs["coherence_or_diversity"])
    plot_author_topic_heatmap(author_topics, outputs["author_heatmap"])
    plot_author_topic_stacked_bar(author_topics, outputs["author_stacked_bar"])
    plot_entropy_by_author(entropy, outputs["entropy_by_author"])
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7_dir", default=None)
    args = parser.parse_args()
    plot_phase7_lda(args.phase7_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
