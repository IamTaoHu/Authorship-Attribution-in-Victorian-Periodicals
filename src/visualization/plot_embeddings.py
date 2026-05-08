"""Plot Phase 3 embedding projections for experiments with saved embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase 3 embedding projections.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--experiment", default=None, help="Optional single experiment name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = resolve_repo_path(args.outputs_dir)
    experiment_dirs = [outputs_dir / args.experiment] if args.experiment else [
        path for path in sorted(outputs_dir.iterdir()) if path.is_dir() and path.name not in {"tables", "plots"}
    ]
    for experiment_dir in experiment_dirs:
        plot_experiment(experiment_dir)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def plot_experiment(experiment_dir: Path) -> None:
    embeddings_path = experiment_dir / "embeddings.npy"
    predictions_path = experiment_dir / "predictions.csv"
    if not embeddings_path.exists() or not predictions_path.exists():
        return
    embeddings = np.load(embeddings_path)
    predictions = pd.read_csv(predictions_path)
    if embeddings.shape[0] != len(predictions) or embeddings.shape[0] < 3:
        warnings.warn(f"Skipping {experiment_dir.name}: embeddings/predictions are missing or too small.")
        return

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=min(30, len(predictions) - 1))
    coords = tsne.fit_transform(embeddings)
    plot_projection(coords, predictions, "true_label_name", experiment_dir / "embedding_tsne_true.png", "t-SNE by true author")
    plot_projection(coords, predictions, "pred_label_name", experiment_dir / "embedding_tsne_pred.png", "t-SNE by predicted author")

    try:
        import umap
    except ImportError:
        warnings.warn("umap-learn is not installed; skipping UMAP embedding plot.")
        return
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    plot_projection(umap_coords, predictions, "true_label_name", experiment_dir / "embedding_umap_true.png", "UMAP by true author")


def plot_projection(coords: np.ndarray, predictions: pd.DataFrame, hue: str, output_path: Path, title: str) -> None:
    frame = predictions.copy()
    frame["x"] = coords[:, 0]
    frame["y"] = coords[:, 1]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=frame, x="x", y="y", hue=hue, s=18, linewidth=0, alpha=0.85)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
