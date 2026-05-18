"""Create Phase 6 ensemble plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prompting.phase4_parser import CANONICAL_AUTHORS  # noqa: E402
from src.utils.paths import get_path_config, phase_artifact_dir  # noqa: E402


ENSEMBLE_REQUIRED_COLUMNS = {"strategy", "status", "accuracy", "macro_f1"}
SINGLE_REQUIRED_COLUMNS = {"run_name", "architecture_group", "accuracy", "macro_f1"}
PER_AUTHOR_REQUIRED_COLUMNS = {"source", "source_type", "author", "f1"}
DEFAULT_HARD_AUTHORS = "James Fitzjames Stephen,Eliza Lynn Linton"
VOTING_PLOT_FILENAMES = {
    "macro_f1_gain": "ensemble_macro_f1_gain_over_best_single.png",
    "accuracy_gain": "ensemble_accuracy_gain_over_best_single.png",
    "hard_author": "hard_author_f1_improvement.png",
    "per_author_delta": "per_author_best_ensemble_vs_best_single_f1_delta.png",
}


def resolve_phase6_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else REPO_ROOT / path
    return get_path_config()["artifacts_root"] / "phase6"


def validate_columns(path: Path, frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        actual = ", ".join(frame.columns.astype(str))
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}. Actual columns found: {actual}")


def coerce_numeric_required(path: Path, frame: pd.DataFrame, columns: tuple[str, ...], *, required_mask: pd.Series | None = None) -> None:
    if required_mask is None:
        required_mask = pd.Series(True, index=frame.index)
    for column in columns:
        original_required = frame.loc[required_mask, column]
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        bad_mask = required_mask & frame[column].isna()
        if bad_mask.any():
            bad_values = sorted({str(value) for value in original_required.loc[frame.loc[required_mask, column].isna()].tolist()})
            preview = ", ".join(bad_values[:8])
            suffix = "" if len(bad_values) <= 8 else ", ..."
            raise ValueError(f"{path} column '{column}' has non-numeric or missing values in required rows after coercion: {preview}{suffix}")


def load_required_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 6 input table not found: {path}")
    frame = pd.read_csv(path)
    validate_columns(path, frame, required_columns)
    return frame


def best_row(frame: pd.DataFrame, name_column: str, label: str) -> pd.Series:
    valid = frame.loc[frame["macro_f1"].notna() & frame["accuracy"].notna()].copy()
    if valid.empty:
        raise ValueError(f"No rows with numeric macro_f1 and accuracy are available for {label}.")
    return valid.sort_values(["macro_f1", "accuracy", name_column], ascending=[False, False, True]).iloc[0]


def display_name(value: object) -> str:
    return str(value).replace("_", " ")


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


def plot_gain_bar(frame: pd.DataFrame, metric: str, baseline: float, output_path: Path) -> None:
    delta_column = f"{metric}_gain"
    plot_frame = frame.loc[frame["status"].astype(str) == "ok", ["strategy", metric]].copy()
    plot_frame[delta_column] = plot_frame[metric] - baseline
    plot_frame = plot_frame.sort_values(delta_column, ascending=False)
    colors = ["#4C78A8" if value >= 0 else "#E45756" for value in plot_frame[delta_column]]
    plt.figure(figsize=(max(8.5, 0.75 * len(plot_frame)), 4.8))
    plt.bar(plot_frame["strategy"].astype(str), plot_frame[delta_column], color=colors)
    plt.axhline(0, color="#222222", linewidth=1)
    plt.xlabel("Ensemble Strategy")
    plt.ylabel(f"Delta {metric.replace('_', ' ').title()}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Wrote {output_path}")


def plot_hard_author_f1(per_author: pd.DataFrame, categories: list[dict[str, str]], hard_authors: list[str], output_path: Path) -> list[dict[str, object]]:
    rows = []
    values = []
    for category in categories:
        subset = per_author.loc[(per_author["source"].astype(str) == category["source"]) & (per_author["author"].astype(str).isin(hard_authors))]
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            rows.append({"category": category["label"], "author": row["author"], "f1": row["f1"]})
            values.append({"category": category["label"], "source": category["source"], "author": row["author"], "f1": float(row["f1"])})
    if not rows:
        raise ValueError("No hard-author F1 rows are available for the selected comparison categories.")
    plot_frame = pd.DataFrame(rows)
    pivot = plot_frame.pivot_table(index="category", columns="author", values="f1", aggfunc="first").reindex(columns=hard_authors)
    ax = pivot.plot(kind="bar", figsize=(max(8.5, 1.2 * len(pivot)), 4.8), color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(pivot.columns)])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model / Ensemble")
    ax.set_ylabel("Per-Author F1")
    ax.legend(title="Author")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Wrote {output_path}")
    return values


def plot_per_author_delta(per_author: pd.DataFrame, best_ensemble_name: str, output_path: Path) -> pd.DataFrame:
    single = per_author.loc[per_author["source_type"].astype(str) == "single_model", ["author", "f1"]].copy()
    ensemble = per_author.loc[per_author["source"].astype(str) == best_ensemble_name, ["author", "f1"]].copy()
    if single.empty:
        raise ValueError("per_author_f1.csv has no rows with source_type == 'single_model'.")
    if ensemble.empty:
        raise ValueError(f"per_author_f1.csv has no rows for best ensemble source: {best_ensemble_name}")
    best_single = single.groupby("author", as_index=False)["f1"].max().rename(columns={"f1": "best_single_f1"})
    best_ensemble = ensemble.rename(columns={"f1": "best_ensemble_f1"})
    merged = best_single.merge(best_ensemble, on="author", how="inner")
    if merged.empty:
        raise ValueError("No overlapping authors exist between best single per-author F1 rows and best ensemble rows.")
    merged["delta_f1"] = merged["best_ensemble_f1"] - merged["best_single_f1"]
    merged = merged.sort_values("delta_f1", ascending=True)
    colors = ["#4C78A8" if value >= 0 else "#E45756" for value in merged["delta_f1"]]
    plt.figure(figsize=(max(8.5, 0.9 * len(merged)), 4.8))
    plt.bar(merged["author"].astype(str), merged["delta_f1"], color=colors)
    plt.axhline(0, color="#222222", linewidth=1)
    plt.xlabel("Author")
    plt.ylabel("Delta F1")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Wrote {output_path}")
    return merged


def build_hard_author_categories(single_results: pd.DataFrame, ensemble_results: pd.DataFrame, warnings: list[str]) -> list[dict[str, str]]:
    categories = []
    for group, label in (("encoder", "Best single encoder"), ("decoder", "Best single decoder / QLoRA")):
        frame = single_results.loc[single_results["architecture_group"].astype(str) == group]
        if frame.empty:
            message = f"No {group} rows found in single_model_results.csv; skipping {label} in hard-author plot."
            print(f"WARNING: {message}")
            warnings.append(message)
            continue
        row = best_row(frame, "run_name", label)
        categories.append({"label": f"{label}: {row['run_name']}", "source": str(row["run_name"])})

    ensemble_ok = ensemble_results.loc[ensemble_results["status"].astype(str) == "ok"]
    best_ensemble = best_row(ensemble_ok, "strategy", "best ensemble")
    categories.append({"label": f"Best ensemble: {best_ensemble['strategy']}", "source": str(best_ensemble["strategy"])})

    for strategy, label in (("cross_architecture_majority_vote", "Cross-architecture majority vote"), ("weighted_vote_macro_f1", "Weighted vote by macro F1")):
        if (ensemble_ok["strategy"].astype(str) == strategy).any():
            categories.append({"label": label, "source": strategy})
        else:
            message = f"{strategy} not found as an ok ensemble strategy; skipping it in hard-author plot."
            print(f"WARNING: {message}")
            warnings.append(message)
    return categories


def write_voting_improvement_outputs(phase6_dir: Path, output_dir: Path, hard_authors: list[str]) -> dict[str, object]:
    tables_dir = phase6_dir / "tables"
    metrics_dir = phase6_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ensemble_path = tables_dir / "ensemble_results.csv"
    single_path = tables_dir / "single_model_results.csv"
    per_author_path = tables_dir / "per_author_f1.csv"
    input_files = [ensemble_path, single_path, per_author_path]
    print("Phase 6 voting improvement input files:")
    for path in input_files:
        print(f"  {path}")

    ensemble_results = load_required_csv(ensemble_path, ENSEMBLE_REQUIRED_COLUMNS)
    single_results = load_required_csv(single_path, SINGLE_REQUIRED_COLUMNS)
    per_author = load_required_csv(per_author_path, PER_AUTHOR_REQUIRED_COLUMNS)

    coerce_numeric_required(ensemble_path, ensemble_results, ("accuracy", "macro_f1"), required_mask=ensemble_results["status"].astype(str) == "ok")
    coerce_numeric_required(single_path, single_results, ("accuracy", "macro_f1"))
    coerce_numeric_required(per_author_path, per_author, ("f1",))

    ensemble_ok = ensemble_results.loc[ensemble_results["status"].astype(str) == "ok"].copy()
    if ensemble_ok.empty:
        raise ValueError("ensemble_results.csv has no rows with status == 'ok'.")
    best_single = best_row(single_results, "run_name", "best single model")
    best_ensemble = best_row(ensemble_ok, "strategy", "best ensemble")

    generated = {
        "macro_f1_gain": output_dir / VOTING_PLOT_FILENAMES["macro_f1_gain"],
        "accuracy_gain": output_dir / VOTING_PLOT_FILENAMES["accuracy_gain"],
        "hard_author": output_dir / VOTING_PLOT_FILENAMES["hard_author"],
        "per_author_delta": output_dir / VOTING_PLOT_FILENAMES["per_author_delta"],
    }
    plot_gain_bar(ensemble_results, "macro_f1", float(best_single["macro_f1"]), generated["macro_f1_gain"])
    plot_gain_bar(ensemble_results, "accuracy", float(best_single["accuracy"]), generated["accuracy_gain"])

    warnings: list[str] = []
    categories = build_hard_author_categories(single_results, ensemble_results, warnings)
    available_sources = set(per_author["source"].astype(str))
    filtered_categories = []
    for category in categories:
        if category["source"] in available_sources:
            filtered_categories.append(category)
        else:
            message = f"per_author_f1.csv has no rows for source '{category['source']}'; skipping it in hard-author plot."
            print(f"WARNING: {message}")
            warnings.append(message)
    hard_author_values = plot_hard_author_f1(per_author, filtered_categories, hard_authors, generated["hard_author"])
    per_author_delta = plot_per_author_delta(per_author, str(best_ensemble["strategy"]), generated["per_author_delta"])

    summary_path = metrics_dir / "voting_improvement_summary.json"
    summary = {
        "best_single_model_name": str(best_single["run_name"]),
        "best_single_accuracy": float(best_single["accuracy"]),
        "best_single_macro_f1": float(best_single["macro_f1"]),
        "best_ensemble_name": str(best_ensemble["strategy"]),
        "best_ensemble_accuracy": float(best_ensemble["accuracy"]),
        "best_ensemble_macro_f1": float(best_ensemble["macro_f1"]),
        "macro_f1_gain": float(best_ensemble["macro_f1"] - best_single["macro_f1"]),
        "accuracy_gain": float(best_ensemble["accuracy"] - best_single["accuracy"]),
        "hard_author_f1_values": hard_author_values,
        "per_author_best_ensemble_vs_best_single_f1_delta": per_author_delta.to_dict(orient="records"),
        "generated_plot_paths": [str(path) for path in generated.values()],
        "warnings": warnings,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return {"input_files": [str(path) for path in input_files], "generated_plot_paths": summary["generated_plot_paths"], "summary_path": str(summary_path)}


def plot_phase6_results(phase6_dir: Path | None = None, output_dir: Path | None = None, hard_authors: list[str] | None = None) -> dict[str, object]:
    if phase6_dir is None:
        tables_dir = phase_artifact_dir("phase6", "tables")
        plots_dir = phase_artifact_dir("phase6", "plots")
        phase6_dir = tables_dir.parent
    else:
        tables_dir = phase6_dir / "tables"
        plots_dir = output_dir or phase6_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

    confusion_inputs = sorted(tables_dir.glob("confusion_matrix_*.csv"))
    if confusion_inputs:
        print("Phase 6 confusion matrix input files:")
        for path in confusion_inputs:
            print(f"  {path}")
    for path in sorted(tables_dir.glob("confusion_matrix_*.csv")):
        plot_confusion_matrix(path, plots_dir / f"{path.stem}.png")

    ensemble_results = load_required_csv(tables_dir / "ensemble_results.csv", ENSEMBLE_REQUIRED_COLUMNS)
    single_results = load_required_csv(tables_dir / "single_model_results.csv", SINGLE_REQUIRED_COLUMNS)
    coerce_numeric_required(tables_dir / "ensemble_results.csv", ensemble_results, ("accuracy", "macro_f1"), required_mask=ensemble_results["status"].astype(str) == "ok")
    coerce_numeric_required(tables_dir / "single_model_results.csv", single_results, ("accuracy", "macro_f1"))
    combined = comparison_frame(ensemble_results, single_results)
    plot_metric_bar(combined, "accuracy", plots_dir / "phase6_accuracy_comparison.png")
    plot_metric_bar(combined, "macro_f1", plots_dir / "phase6_macro_f1_comparison.png")
    return write_voting_improvement_outputs(phase6_dir, plots_dir, hard_authors or [name.strip() for name in DEFAULT_HARD_AUTHORS.split(",") if name.strip()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase6_dir", default=None, help="Phase 6 artifact directory. Defaults to configured artifacts_root/phase6.")
    parser.add_argument("--output_dir", default=None, help="Plot output directory. Defaults to phase6_dir/plots.")
    parser.add_argument("--hard_authors", default=DEFAULT_HARD_AUTHORS, help="Comma-separated authors for the hard-author comparison plot.")
    args = parser.parse_args()

    phase6_dir = resolve_phase6_dir(args.phase6_dir)
    output_dir = Path(args.output_dir) if args.output_dir else phase6_dir / "plots"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    hard_authors = [name.strip() for name in args.hard_authors.split(",") if name.strip()]
    plot_phase6_results(phase6_dir, output_dir, hard_authors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
