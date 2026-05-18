"""Create Phase 8 BERTopic plots and interactive exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

from src.topic_modeling.bertopic_phase8 import default_phase8_dir  # noqa: E402


def topic_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("topic_")]


def plot_topic_size_bar(topic_info: pd.DataFrame, output_path: Path) -> None:
    frame = topic_info.copy()
    if "Topic" not in frame.columns or "Count" not in frame.columns:
        raise ValueError("topic_info.csv must contain Topic and Count columns.")
    frame = frame.sort_values("Count", ascending=False).head(30)
    labels = frame["Topic"].astype(str)
    plt.figure(figsize=(max(9, 0.35 * len(frame)), 5.2))
    plt.bar(labels, frame["Count"].astype(float), color="#4C78A8")
    plt.xlabel("Topic")
    plt.ylabel("Documents")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_author_heatmap(author_topics: pd.DataFrame, output_path: Path) -> None:
    columns = topic_columns(author_topics)
    if not columns:
        raise ValueError("per_author_topic_distribution.csv has no topic_* columns.")
    matrix = author_topics.set_index("author")[columns]
    values = matrix.to_numpy(dtype=float)
    vmax = float(values.max()) if values.size else 1.0
    threshold = vmax * 0.55
    annotation_fontsize = 6 if len(columns) > 18 else 7
    plt.figure(figsize=(max(11, 0.85 * len(columns)), max(5.2, 0.55 * len(matrix))))
    if sns is not None:
        ax = sns.heatmap(
            matrix,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": annotation_fontsize},
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": "Share of Documents"},
            vmin=0,
            vmax=vmax,
        )
        for text, value in zip(ax.texts, values.ravel()):
            text.set_color("white" if value >= threshold else "#111111")
    else:
        image = plt.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        plt.colorbar(image, label="Share of Documents")
        plt.yticks(range(len(matrix.index)), matrix.index)
        plt.xticks(range(len(columns)), columns, rotation=45, ha="right")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                color = "white" if value >= threshold else "#111111"
                plt.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=annotation_fontsize)
    plt.title("Per-Author Topic Distribution")
    plt.xlabel("Topic")
    plt.ylabel("Author")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def plot_topic_scatter(root: Path, output_path: Path) -> None:
    coordinates_path = root / "tables" / "document_coordinates.csv"
    if not coordinates_path.exists():
        raise FileNotFoundError(f"Missing reduced coordinates: {coordinates_path}")
    frame = pd.read_csv(coordinates_path)
    required = {"x", "y", "topic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"document_coordinates.csv missing columns: {', '.join(sorted(missing))}")
    plot_frame = frame.copy()
    plot_frame["topic_label"] = plot_frame["topic"].astype(str)
    plt.figure(figsize=(8.8, 6.8))
    if sns is not None:
        sns.scatterplot(data=plot_frame, x="x", y="y", hue="topic_label", s=14, linewidth=0, palette="tab20", legend=False)
    else:
        plt.scatter(plot_frame["x"], plot_frame["y"], c=pd.Categorical(plot_frame["topic_label"]).codes, s=12, cmap="tab20")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Wrote {output_path}")


def load_topic_model(model_dir: Path) -> Any:
    from bertopic import BERTopic

    try:
        return BERTopic.load(str(model_dir))
    except Exception:
        pickle_path = model_dir / "model.pkl"
        if pickle_path.exists():
            return BERTopic.load(str(pickle_path))
        raise


def write_optional_html(fig: Any, output_path: Path) -> None:
    if fig is None:
        raise ValueError("BERTopic returned no figure.")
    fig.write_html(str(output_path))
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise ValueError(f"HTML export is empty: {output_path}")


def export_interactive(root: Path) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    interactive_dir = root / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    log_path = interactive_dir / "interactive_export_log.json"
    try:
        topic_model = load_topic_model(root / "models" / "bertopic_model")
    except Exception as exc:
        failures.append({"visualization": "model_load", "error": str(exc)})
        log_path.write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
        return failures

    exports = (
        ("topics", "topics.html", lambda: topic_model.visualize_topics()),
        ("barchart", "barchart.html", lambda: topic_model.visualize_barchart(top_n_topics=30)),
        ("hierarchy", "hierarchy.html", lambda: topic_model.visualize_hierarchy()),
    )
    for name, filename, builder in exports:
        try:
            write_optional_html(builder(), interactive_dir / filename)
            print(f"Wrote {interactive_dir / filename}")
        except Exception as exc:
            failures.append({"visualization": name, "error": str(exc)})
            print(f"WARNING: Could not export optional BERTopic visualization '{name}': {exc}")
    log_path.write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
    return failures


def plot_phase8_bertopic(phase8_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(phase8_dir).expanduser() if phase8_dir else default_phase8_dir()
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    topic_info = pd.read_csv(root / "tables" / "topic_info.csv")
    author_topics = pd.read_csv(root / "tables" / "per_author_topic_distribution.csv")
    outputs = {
        "topic_size_bar": plots_dir / "topic_size_bar.png",
        "per_author_topic_heatmap": plots_dir / "per_author_topic_heatmap.png",
        "topic_scatter": plots_dir / "topic_scatter.png",
    }
    plot_topic_size_bar(topic_info, outputs["topic_size_bar"])
    plot_author_heatmap(author_topics, outputs["per_author_topic_heatmap"])
    plot_topic_scatter(root, outputs["topic_scatter"])
    export_interactive(root)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8_dir", default=None)
    args = parser.parse_args()
    plot_phase8_bertopic(args.phase8_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
