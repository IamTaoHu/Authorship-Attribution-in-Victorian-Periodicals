"""Plot Phase 1 PERIAD dataset summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import dataset_dir, phase_artifact_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase1" / "dataset_pipeline.yaml"


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"WARNING: Missing input table, skipping: {path}")
        return None
    return pd.read_csv(path)


def plot_author_distribution(table_dir: Path, plot_dir: Path) -> None:
    frame = _read_csv(table_dir / "periad_author_distribution.csv")
    if frame is None or frame.empty:
        return
    pivot = frame.pivot(index="author", columns="split", values="rows").fillna(0)
    ax = pivot.plot(kind="barh", figsize=(11, 6))
    plt.xlabel("Rows")
    plt.ylabel("Author")
    plt.title("PERIAD Author Distribution")
    ax.legend(title="split")
    plt.tight_layout()
    output = plot_dir / "periad_author_distribution.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Wrote {output}")


def plot_text_length_histogram(plot_dir: Path) -> None:
    periad_dir = dataset_dir("processed", "periad")
    paths = [periad_dir / "train.csv", periad_dir / "test.csv"]
    if not all(path.exists() for path in paths):
        print(f"WARNING: Missing processed PERIAD CSV files, skipping text length histogram: {periad_dir}")
        return
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame.empty or "text_length_words" not in frame:
        print("WARNING: Missing text_length_words, skipping text length histogram.")
        return
    plt.figure(figsize=(10, 6))
    for split_name, split_frame in frame.groupby("split"):
        plt.hist(split_frame["text_length_words"], bins=60, alpha=0.45, label=str(split_name))
    plt.xlabel("Text Length (Words)")
    plt.ylabel("Rows")
    plt.title("PERIAD Text Length Distribution")
    plt.legend(title="split")
    plt.tight_layout()
    output = plot_dir / "periad_text_length_histogram.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Wrote {output}")


def plot_split_distribution(table_dir: Path, plot_dir: Path) -> None:
    frame = _read_csv(table_dir / "periad_split_summary.csv")
    if frame is None or frame.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.bar(frame["split"].astype(str), frame["rows"])
    plt.xlabel("Split")
    plt.ylabel("Rows")
    plt.title("PERIAD Split Distribution")
    plt.tight_layout()
    output = plot_dir / "periad_split_distribution.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.parse_args()

    table_dir = phase_artifact_dir("phase1", "tables")
    plot_dir = phase_artifact_dir("phase1", "plots")
    plot_author_distribution(table_dir, plot_dir)
    plot_text_length_histogram(plot_dir)
    plot_split_distribution(table_dir, plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
