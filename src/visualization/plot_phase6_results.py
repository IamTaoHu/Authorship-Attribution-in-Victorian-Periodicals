"""Create Phase 6 ensemble plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prompting.phase4_parser import CANONICAL_AUTHORS  # noqa: E402
from src.utils.paths import phase_artifact_dir  # noqa: E402


def plot_confusion_matrix(path: Path, output_path: Path) -> None:
    matrix = pd.read_csv(path, index_col=0).reindex(index=list(CANONICAL_AUTHORS), columns=list(CANONICAL_AUTHORS), fill_value=0)
    plt.figure(figsize=(8, 6.5))
    image = plt.imshow(matrix.to_numpy(dtype=float), cmap="Blues")
    plt.colorbar(image, label="Count")
    plt.xticks(range(len(CANONICAL_AUTHORS)), CANONICAL_AUTHORS, rotation=45, ha="right")
    plt.yticks(range(len(CANONICAL_AUTHORS)), CANONICAL_AUTHORS)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix.iloc[row, column])
            plt.text(column, row, str(value), ha="center", va="center", fontsize=8)
    plt.xlabel("Predicted Author")
    plt.ylabel("True Author")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def comparison_frame(ensemble_results: pd.DataFrame, single_results: pd.DataFrame) -> pd.DataFrame:
    ensembles = ensemble_results.loc[ensemble_results["status"].astype(str) == "ok", ["strategy", "accuracy", "macro_f1"]].copy()
    ensembles = ensembles.rename(columns={"strategy": "name"})
    ensembles["group"] = "ensemble"
    singles = []
    for group_name, frame in (
        ("best_single_encoder", single_results.loc[single_results["architecture_group"] == "encoder"]),
        ("best_single_decoder", single_results.loc[single_results["architecture_group"] == "decoder"]),
        ("best_single_overall", single_results),
    ):
        if frame.empty:
            continue
        row = frame.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
        singles.append({"name": group_name, "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "group": "single_model"})
    return pd.concat([ensembles, pd.DataFrame(singles)], ignore_index=True)


def plot_metric_bar(frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_frame = frame.loc[pd.to_numeric(frame[metric], errors="coerce").notna()].copy()
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.sort_values(metric, ascending=False)
    colors = plot_frame["group"].map({"ensemble": "#4C78A8", "single_model": "#F58518"}).fillna("#777777")
    plt.figure(figsize=(max(9, 0.75 * len(plot_frame)), 4.8))
    plt.bar(plot_frame["name"], plot_frame[metric], color=colors)
    plt.ylim(0, 1)
    plt.xlabel("Strategy / Baseline")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def plot_phase6_results() -> None:
    tables_dir = phase_artifact_dir("phase6", "tables")
    plots_dir = phase_artifact_dir("phase6", "plots")
    for path in sorted(tables_dir.glob("confusion_matrix_*.csv")):
        plot_confusion_matrix(path, plots_dir / f"{path.stem}.png")
    ensemble_results = pd.read_csv(tables_dir / "ensemble_results.csv")
    single_results = pd.read_csv(tables_dir / "single_model_results.csv")
    combined = comparison_frame(ensemble_results, single_results)
    plot_metric_bar(combined, "accuracy", plots_dir / "phase6_accuracy_comparison.png")
    plot_metric_bar(combined, "macro_f1", plots_dir / "phase6_macro_f1_comparison.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    plot_phase6_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
