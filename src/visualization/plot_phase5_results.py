"""Create Phase 5 QLoRA plots from real result tables only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_phase5_results import FULL_RUNS, aggregate_phase5_results  # noqa: E402


FULL_TABLE = "decoder_qlora_results.csv"
DIAGNOSTIC_TABLE = "diagnostic_decoder_qlora_results.csv"
PHASE2_ACCURACY = 0.9270
PHASE2_MACRO_F1 = 0.9191


def resolve_phase5_dir(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def write_warning(report_path: Path, message: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Phase 5 Plot Warnings\n\n"
    if message not in existing:
        report_path.write_text(existing.rstrip() + f"\n\n- {message}\n", encoding="utf-8")
    print(f"WARNING: {message}")


def plot_bar(frame: pd.DataFrame, metric: str, output_path: Path, title: str, reference: float | None = None) -> None:
    plot_frame = frame.loc[pd.to_numeric(frame.get(metric), errors="coerce").notna()].copy()
    if plot_frame.empty:
        raise ValueError(f"No numeric {metric} values available.")
    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.sort_values(metric, ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, 0.75 * len(plot_frame)), 4))
    plt.bar(plot_frame["run_name"].astype(str), plot_frame[metric], color="#4C78A8")
    if reference is not None:
        plt.axhline(reference, color="#222222", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Run")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def has_completed_full_runs(phase5_dir: Path) -> bool:
    return all((phase5_dir / "runs" / run_name / "predictions.csv").exists() and (phase5_dir / "runs" / run_name / "metrics.json").exists() for run_name in FULL_RUNS)


def ensure_full_tables(phase5_dir: Path, full_table: Path, warning_report: Path) -> None:
    required = (
        full_table,
        phase5_dir / "tables" / "decoder_qlora_results.md",
        phase5_dir / "tables" / "phase5_per_author_f1.csv",
        phase5_dir / "reports" / "phase5_qlora_decoder_finetuning_report.md",
    )
    if all(path.exists() for path in required):
        return
    if has_completed_full_runs(phase5_dir):
        aggregate_phase5_results(phase5_dir)
    else:
        write_warning(warning_report, f"Full run directories incomplete; aggregation skipped under {phase5_dir / 'runs'}")


def find_phase4_table() -> Path | None:
    candidates = (
        REPO_ROOT / "artifacts" / "phase4" / "tables" / "decoder_prompting_results.csv",
        REPO_ROOT / "artifacts" / "phase4" / "tables" / "phase4_decoder_prompting_results.csv",
        REPO_ROOT / "outputs" / "phase4" / "tables" / "decoder_prompting_results.csv",
        REPO_ROOT / "outputs" / "phase4" / "tables" / "phase4_decoder_prompting_results.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def build_comparison_frame(full: pd.DataFrame, warning_report: Path) -> pd.DataFrame | None:
    qlora = full.loc[pd.to_numeric(full.get("macro_f1"), errors="coerce").notna(), ["run_name", "macro_f1"]].copy()
    if qlora.empty:
        write_warning(warning_report, "Comparison plot skipped because no valid Phase 5 macro_f1 values exist.")
        return None
    phase4_table = find_phase4_table()
    if phase4_table is None:
        write_warning(warning_report, "Comparison plot skipped because Phase 4 prompting comparison data is unavailable.")
        return None
    phase4 = pd.read_csv(phase4_table)
    if "macro_f1" not in phase4.columns or "run_name" not in phase4.columns:
        write_warning(warning_report, f"Comparison plot skipped because {phase4_table} lacks run_name or macro_f1.")
        return None
    phase4 = phase4.loc[pd.to_numeric(phase4["macro_f1"], errors="coerce").notna(), ["run_name", "macro_f1"]].copy()
    if phase4.empty:
        write_warning(warning_report, f"Comparison plot skipped because {phase4_table} has no numeric macro_f1 values.")
        return None
    best_prompting = phase4.sort_values("macro_f1", ascending=False).head(1).assign(run_name=lambda frame: "phase4_" + frame["run_name"].astype(str))
    return pd.concat(
        [
            qlora.assign(group="phase5_qlora"),
            best_prompting.assign(group="phase4_prompting"),
            pd.DataFrame([{"run_name": "phase2_roberta_large", "macro_f1": PHASE2_MACRO_F1, "group": "encoder_reference"}]),
        ],
        ignore_index=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5_dir", default="outputs/phase5")
    args = parser.parse_args()

    phase5_dir = resolve_phase5_dir(args.phase5_dir)
    plots_dir = phase5_dir / "plots"
    warning_report = phase5_dir / "reports" / "phase5_plot_warnings.md"
    diagnostic_table = phase5_dir / "diagnostic" / "tables" / DIAGNOSTIC_TABLE
    full_table = phase5_dir / "tables" / FULL_TABLE
    ensure_full_tables(phase5_dir, full_table, warning_report)

    if diagnostic_table.exists():
        diagnostic = pd.read_csv(diagnostic_table)
        plot_bar(diagnostic, "accuracy", plots_dir / "diagnostic_qlora_accuracy_bar.png", "Phase 5 Diagnostic Accuracy")
        plot_bar(diagnostic, "macro_f1", plots_dir / "diagnostic_qlora_macro_f1_bar.png", "Phase 5 Diagnostic Macro F1")
    else:
        write_warning(warning_report, f"Diagnostic table missing; diagnostic plots skipped: {diagnostic_table}")

    if full_table.exists():
        full = pd.read_csv(full_table)
        if full.empty:
            write_warning(warning_report, f"Full result table is empty; full plots skipped: {full_table}")
        else:
            plot_bar(full, "accuracy", plots_dir / "qlora_accuracy_bar.png", "Phase 5 QLoRA Accuracy", PHASE2_ACCURACY)
            plot_bar(full, "macro_f1", plots_dir / "qlora_macro_f1_bar.png", "Phase 5 QLoRA Macro F1", PHASE2_MACRO_F1)
            comparison = build_comparison_frame(full, warning_report)
            if comparison is not None:
                plot_bar(comparison, "macro_f1", plots_dir / "qlora_vs_prompting_vs_encoder.png", "QLoRA vs Prompting vs Encoder", PHASE2_MACRO_F1)
    else:
        write_warning(warning_report, f"Full result table missing; full plots skipped until Colab runs complete: {full_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
