"""Create Phase 4 decoder prompting plots."""

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


PHASE2_ACCURACY = 0.9270
PHASE2_MACRO_F1 = 0.9191
FINAL_RUN_NAMES = (
    "mistral_zero_shot",
    "mistral_few_shot",
    "llama3_zero_shot",
    "llama3_few_shot",
    "gemma2_zero_shot",
    "gemma2_few_shot",
)


def write_placeholder(output_path: Path, message: str) -> None:
    plt.figure(figsize=(7, 3.5))
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def load_results(tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / "decoder_prompting_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Phase 4 aggregate table not found: {path}")
    return pd.read_csv(path)


def validate_final_results(frame: pd.DataFrame) -> None:
    if "run_name" not in frame.columns:
        raise ValueError("Final-only Phase 4 results table is missing run_name.")
    run_names = set(frame["run_name"].astype(str))
    expected = set(FINAL_RUN_NAMES)
    extra = sorted(run_names.difference(expected))
    missing = sorted(expected.difference(run_names))
    if extra:
        raise ValueError(f"Final-only Phase 4 results contain non-final runs: {', '.join(extra)}")
    if missing:
        raise ValueError(f"Final-only Phase 4 results are missing runs: {', '.join(missing)}")
    if len(frame) != len(FINAL_RUN_NAMES):
        raise ValueError(f"Final-only Phase 4 results must have {len(FINAL_RUN_NAMES)} rows, found {len(frame)}.")
    if frame["run_name"].astype(str).str.contains("tinyllama", case=False, na=False).any():
        raise ValueError("Final-only Phase 4 results contain tinyllama_smoke.")
    if "valid_run" in frame.columns and (frame["valid_run"].astype(str).str.lower() != "true").any():
        raise ValueError("Final-only Phase 4 results contain valid_run=false rows.")
    if "n_samples" in frame.columns and (pd.to_numeric(frame["n_samples"], errors="coerce") <= 0).any():
        raise ValueError("Final-only Phase 4 results contain n_samples <= 0 rows.")
    if "invalid_reason" in frame.columns and frame["invalid_reason"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("Final-only Phase 4 results contain non-empty invalid_reason rows.")


def validate_final_per_author(frame: pd.DataFrame) -> None:
    if "run_name" not in frame.columns:
        raise ValueError("Final-only per-author table is missing run_name.")
    run_names = set(frame["run_name"].astype(str))
    expected = set(FINAL_RUN_NAMES)
    extra = sorted(run_names.difference(expected))
    missing = sorted(expected.difference(run_names))
    if extra:
        raise ValueError(f"Final-only per-author table contains non-final runs: {', '.join(extra)}")
    if missing:
        raise ValueError(f"Final-only per-author table is missing runs: {', '.join(missing)}")
    expected_rows = len(FINAL_RUN_NAMES) * len(CANONICAL_AUTHORS)
    if len(frame) != expected_rows:
        raise ValueError(f"Final-only per-author table must have {expected_rows} rows, found {len(frame)}.")


def prediction_paths(runs_dir: Path, run_names: tuple[str, ...] | None = None) -> list[Path]:
    if run_names is None:
        return sorted(runs_dir.glob("*/predictions.csv"))
    return [runs_dir / run_name / "predictions.csv" for run_name in run_names]


def plot_metric_bar(frame: pd.DataFrame, metric: str, output_path: Path, reference: float | None = None) -> None:
    if frame.empty or metric not in frame.columns:
        write_placeholder(output_path, f"No Phase 4 rows available for {metric}.")
        return
    plot_frame = frame.loc[pd.to_numeric(frame[metric], errors="coerce").notna()].copy()
    if plot_frame.empty:
        write_placeholder(output_path, f"No numeric Phase 4 values available for {metric}.")
        return
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.sort_values(metric, ascending=False)
    plt.figure(figsize=(max(8, 0.7 * len(plot_frame)), 4.8))
    plt.bar(plot_frame["run_name"].astype(str), plot_frame[metric].astype(float), color="#4C78A8")
    if reference is not None:
        plt.axhline(reference, color="#222222", linestyle="--", linewidth=1)
    if metric.endswith("rate") or metric in {"accuracy", "macro_f1"}:
        plt.ylim(0, 1)
    plt.xlabel("Run")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def plot_parsed_status_distribution(runs_dir: Path, output_path: Path, run_names: tuple[str, ...] | None = None) -> None:
    rows = []
    for path in prediction_paths(runs_dir, run_names):
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions for Phase 4 run: {path}")
        frame = pd.read_csv(path, usecols=["parsed_status"])
        if frame.empty:
            continue
        counts = frame["parsed_status"].value_counts().reset_index()
        counts.columns = ["parsed_status", "count"]
        counts["run_name"] = path.parent.name
        rows.append(counts)
    if not rows:
        write_placeholder(output_path, "No parsed output statuses available.")
        return
    combined = pd.concat(rows, ignore_index=True)
    pivot = combined.pivot_table(index="run_name", columns="parsed_status", values="count", aggfunc="sum", fill_value=0)
    pivot.plot(kind="bar", stacked=True, figsize=(max(8, 0.8 * len(pivot)), 5))
    plt.xlabel("Run")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def plot_prediction_distribution(runs_dir: Path, output_path: Path, run_names: tuple[str, ...] | None = None) -> None:
    rows = []
    for path in prediction_paths(runs_dir, run_names):
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions for Phase 4 run: {path}")
        frame = pd.read_csv(path, usecols=["predicted_author"])
        if frame.empty:
            continue
        counts = frame["predicted_author"].fillna("invalid").replace("", "invalid").value_counts().reset_index()
        counts.columns = ["predicted_author", "count"]
        counts["run_name"] = path.parent.name
        rows.append(counts)
    if not rows:
        write_placeholder(output_path, "No prediction distribution available.")
        return
    combined = pd.concat(rows, ignore_index=True)
    pivot = combined.pivot_table(index="run_name", columns="predicted_author", values="count", aggfunc="sum", fill_value=0)
    pivot.plot(kind="bar", stacked=True, figsize=(max(8, 0.8 * len(pivot)), 5))
    plt.xlabel("Run")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def plot_per_author_heatmap(path: Path, output_path: Path, *, final_only: bool = False) -> None:
    if not path.exists():
        write_placeholder(output_path, f"Missing per-author table: {path}")
        return
    frame = pd.read_csv(path)
    if frame.empty:
        write_placeholder(output_path, "No per-author F1 rows available.")
        return
    if final_only:
        validate_final_per_author(frame)
    frame = frame.loc[frame["label"].astype(str).isin(CANONICAL_AUTHORS)].copy()
    frame["f1-score"] = pd.to_numeric(frame["f1-score"], errors="coerce")
    pivot = frame.pivot_table(index="label", columns="run_name", values="f1-score", aggfunc="first")
    pivot = pivot.reindex(index=list(CANONICAL_AUTHORS))
    plt.figure(figsize=(max(9, 0.8 * len(pivot.columns)), 5.5))
    values = pivot.to_numpy(dtype=float)
    image = plt.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(image, label="F1")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if pd.notna(value):
                plt.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Run")
    plt.ylabel("Author")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-only", action="store_true", help="Plot only the six final Phase 4 decoder benchmark runs.")
    args = parser.parse_args()
    tables_dir = phase_artifact_dir("phase4", "tables")
    plots_dir = phase_artifact_dir("phase4", "plots")
    runs_dir = phase_artifact_dir("phase4", "runs")
    results = load_results(tables_dir)
    run_names = FINAL_RUN_NAMES if args.final_only else None
    if args.final_only:
        validate_final_results(results)
    plot_metric_bar(results, "accuracy", plots_dir / "decoder_accuracy_bar.png", PHASE2_ACCURACY)
    plot_metric_bar(results, "macro_f1", plots_dir / "decoder_macro_f1_bar.png", PHASE2_MACRO_F1)
    plot_metric_bar(results, "invalid_output_rate", plots_dir / "invalid_output_rate_bar.png")
    plot_parsed_status_distribution(runs_dir, plots_dir / "parsed_status_distribution.png", run_names)
    plot_prediction_distribution(runs_dir, plots_dir / "prediction_distribution_by_run.png", run_names)
    plot_per_author_heatmap(tables_dir / "phase4_per_author_f1.csv", plots_dir / "per_author_f1_heatmap.png", final_only=args.final_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
