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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5_dir", default="outputs/phase5")
    args = parser.parse_args()

    phase5_dir = resolve_phase5_dir(args.phase5_dir)
    plots_dir = phase5_dir / "plots"
    warning_report = phase5_dir / "reports" / "phase5_plot_warnings.md"
    diagnostic_table = phase5_dir / "diagnostic" / "tables" / DIAGNOSTIC_TABLE
    full_table = phase5_dir / "tables" / FULL_TABLE

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
            comparison = full.copy()
            comparison = comparison.loc[pd.to_numeric(comparison["macro_f1"], errors="coerce").notna()].copy()
            if comparison.empty:
                write_warning(warning_report, "Full comparison plot skipped because no valid full macro_f1 values exist.")
            else:
                comparison = pd.concat(
                    [
                        comparison[["run_name", "macro_f1"]].assign(group="phase5_qlora"),
                        pd.DataFrame(
                            [
                                {"run_name": "phase2_roberta_large", "macro_f1": PHASE2_MACRO_F1, "group": "encoder_reference"},
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                plot_bar(comparison, "macro_f1", plots_dir / "qlora_vs_prompting_vs_encoder.png", "QLoRA vs Encoder Reference", PHASE2_MACRO_F1)
    else:
        write_warning(warning_report, f"Full result table missing; full plots skipped until Colab runs complete: {full_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
