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
import numpy as np
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


def select_top_topics(topic_info: pd.DataFrame, top_n: int = 15, include_outlier: bool = False) -> pd.DataFrame:
    required = {"Topic", "Count"}
    missing = required.difference(topic_info.columns)
    if missing:
        raise ValueError(f"topic_info.csv missing columns: {', '.join(sorted(missing))}")
    frame = topic_info.copy()
    frame["Topic"] = frame["Topic"].astype(int)
    frame["Count"] = pd.to_numeric(frame["Count"], errors="coerce").fillna(0).astype(int)
    if not include_outlier:
        frame = frame.loc[frame["Topic"] != -1]
    return frame.sort_values(["Count", "Topic"], ascending=[False, True]).head(top_n)


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
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output_path}")


def plot_author_top_topics_heatmap(
    author_topics: pd.DataFrame,
    topic_info: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
    include_outlier: bool = False,
) -> None:
    top_topics = select_top_topics(topic_info, top_n=top_n, include_outlier=include_outlier)
    selected_columns = [f"topic_{int(topic)}" for topic in top_topics["Topic"].tolist()]
    selected_columns = [column for column in selected_columns if column in author_topics.columns]
    if not selected_columns:
        raise ValueError("No selected top topic columns are present in per_author_topic_distribution.csv.")

    matrix = author_topics.set_index("author")[selected_columns]
    values = matrix.to_numpy(dtype=float)
    vmax = float(values.max()) if values.size else 1.0
    threshold = vmax * 0.55
    annotation_fontsize = 7 if len(selected_columns) > 10 else 8
    plt.figure(figsize=(max(11, 0.85 * len(selected_columns)), max(5.2, 0.55 * len(matrix))))
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
        plt.xticks(range(len(selected_columns)), selected_columns, rotation=45, ha="right")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                color = "white" if value >= threshold else "#111111"
                plt.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=annotation_fontsize)
    plt.title("Per-author distribution across top BERTopic topics")
    plt.xlabel("Topic")
    plt.ylabel("Author")
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


def ensure_document_coordinates(root: Path) -> pd.DataFrame:
    coordinates_path = root / "tables" / "document_coordinates.csv"
    if coordinates_path.exists():
        return pd.read_csv(coordinates_path)

    assignments_path = root / "assignments" / "topic_assignments.csv"
    embeddings_path = root / "embeddings" / "document_embeddings.npy"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing assignments for coordinate fallback: {assignments_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing document embeddings for coordinate fallback: {embeddings_path}")

    try:
        from umap import UMAP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("umap-learn is required to regenerate document_coordinates.csv from saved embeddings.") from exc

    assignments = pd.read_csv(assignments_path)
    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(assignments):
        raise ValueError(f"document_embeddings rows {embeddings.shape[0]} != topic_assignments rows {len(assignments)}")

    coords = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=42).fit_transform(embeddings)
    frame = assignments[["sample_id", "split", "author", "topic"]].copy()
    frame["x"] = coords[:, 0]
    frame["y"] = coords[:, 1]
    frame["coordinate_source"] = "fallback_umap_2d"
    coordinates_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(coordinates_path, index=False)
    print(f"Wrote {coordinates_path}")
    return frame


def plot_topic_scatter(root: Path, output_path: Path) -> None:
    frame = ensure_document_coordinates(root)
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


def plot_topic_scatter_by_author(root: Path, output_path: Path) -> None:
    frame = ensure_document_coordinates(root)
    required = {"x", "y", "author"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"document_coordinates.csv missing columns: {', '.join(sorted(missing))}")
    plot_frame = frame.copy()
    authors = list(dict.fromkeys(plot_frame["author"].astype(str)))
    colors = plt.get_cmap("tab10").colors
    plt.figure(figsize=(10.5, 6.8))
    for index, author in enumerate(authors):
        subset = plot_frame.loc[plot_frame["author"].astype(str) == author]
        plt.scatter(
            subset["x"],
            subset["y"],
            s=10,
            alpha=0.5,
            linewidths=0,
            color=colors[index % len(colors)],
            label=author,
        )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("BERTopic document embedding projection by author")
    plt.legend(title="Author", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output_path}")


def resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"topic_words.csv missing {label} column; accepted variants: {', '.join(candidates)}")


def build_top_topic_words(
    topic_info: pd.DataFrame,
    topic_words: pd.DataFrame,
    top_n: int = 15,
    include_outlier: bool = False,
) -> pd.DataFrame:
    top_topics = select_top_topics(topic_info, top_n=top_n, include_outlier=include_outlier)
    topic_column = resolve_column(topic_words, ("topic", "Topic"), "topic")
    rank_column = resolve_column(topic_words, ("rank", "Rank"), "rank")
    term_column = resolve_column(topic_words, ("term", "word", "Term", "Word"), "term/word")

    words = topic_words.copy()
    words["_topic"] = words[topic_column].astype(int)
    words["_rank"] = pd.to_numeric(words[rank_column], errors="coerce")
    rows: list[dict[str, object]] = []
    for _, topic_row in top_topics.iterrows():
        topic = int(topic_row["Topic"])
        topic_terms = (
            words.loc[words["_topic"] == topic]
            .sort_values("_rank")
            .loc[:, term_column]
            .dropna()
            .astype(str)
            .tolist()
        )
        rows.append(
            {
                "topic": topic,
                "topic_name": str(topic_row.get("Name", f"topic_{topic}")),
                "document_count": int(topic_row["Count"]),
                "rank": len(rows) + 1,
                "top_words": "; ".join(topic_terms),
            }
        )
    return pd.DataFrame(rows, columns=["topic", "topic_name", "document_count", "rank", "top_words"])


def write_top_topic_words_report(frame: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Top BERTopic Topic Words",
        "",
        "| Rank | Topic | Topic Name | Documents | Top Words |",
        "|---:|---:|---|---:|---|",
    ]
    for _, row in frame.iterrows():
        topic_name = str(row["topic_name"]).replace("|", "\\|")
        top_words = str(row["top_words"]).replace("|", "\\|")
        lines.append(f"| {row['rank']} | {row['topic']} | {topic_name} | {row['document_count']} | {top_words} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    topic_words = pd.read_csv(root / "tables" / "topic_words.csv")
    author_topics = pd.read_csv(root / "tables" / "per_author_topic_distribution.csv")
    outputs = {
        "topic_size_bar": plots_dir / "topic_size_bar.png",
        "per_author_topic_heatmap": plots_dir / "per_author_topic_heatmap.png",
        "per_author_top_topics_heatmap": plots_dir / "per_author_top_topics_heatmap.png",
        "topic_scatter": plots_dir / "topic_scatter.png",
        "topic_scatter_by_author": plots_dir / "topic_scatter_by_author.png",
        "top_topic_words_csv": root / "tables" / "top_topic_words.csv",
        "top_topic_words_report": root / "reports" / "top_topic_words.md",
    }
    plot_topic_size_bar(topic_info, outputs["topic_size_bar"])
    plot_author_heatmap(author_topics, outputs["per_author_topic_heatmap"])
    plot_author_top_topics_heatmap(author_topics, topic_info, outputs["per_author_top_topics_heatmap"])
    plot_topic_scatter(root, outputs["topic_scatter"])
    plot_topic_scatter_by_author(root, outputs["topic_scatter_by_author"])
    top_topic_words = build_top_topic_words(topic_info, topic_words)
    top_topic_words.to_csv(outputs["top_topic_words_csv"], index=False)
    print(f"Wrote {outputs['top_topic_words_csv']}")
    write_top_topic_words_report(top_topic_words, outputs["top_topic_words_report"])
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
