"""Build the Phase 10 final research package without retraining models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fnmatch
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_path_config  # noqa: E402


PHASE_ROOTS = {
    "phase1": ("phase1",),
    "phase2": ("phase2",),
    "phase3": ("phase3",),
    "phase4": ("phase4",),
    "phase5": ("phase5",),
    "phase6": ("phase6",),
    "phase7": ("phase7", "lda"),
    "phase8": ("phase8", "bertopic"),
    "phase9": ("phase9", "topic_features"),
}

SUMMARY_PATTERNS = (
    "metrics.csv",
    "results.csv",
    "summary.csv",
    "*_summary.csv",
    "*_results.csv",
    "leaderboard.csv",
    "report.md",
    "*_report.md",
    "summary.md",
    "*_summary.md",
    "summary.json",
    "*_summary.json",
    "metrics.json",
)

FORBIDDEN_NAMES = {"training_log.csv", "training_log.jsonl"}
FORBIDDEN_DIRS = {
    "adapters",
    "checkpoints",
    "model_cache",
    "models",
    ".cache",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}

REPORT_FORBIDDEN_TERMS = (
    "dropout prediction",
    "student retention",
    "oulad",
    "uci",
    "shap",
    "education",
)


@dataclass
class BuildState:
    artifacts_dir: Path
    output_dir: Path
    export_dir: Path
    zip_path: Path
    strict: bool
    phase_dirs: dict[str, Path]
    validated_phases: list[str] = field(default_factory=list)
    missing_phases: list[str] = field(default_factory=list)
    selected_summaries: dict[str, str] = field(default_factory=dict)
    generated_tables: list[str] = field(default_factory=list)
    generated_figures: list[str] = field(default_factory=list)
    generated_reports: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    missing_required: list[dict[str, str]] = field(default_factory=list)
    missing_optional: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_rows: list[dict[str, str]] = field(default_factory=list)
    zip_validation: dict[str, Any] = field(default_factory=dict)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_defaults(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    paths = get_path_config()

    def resolve(raw: str, default_root: Path, default_relative: str) -> Path:
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        if path.as_posix() == Path(default_relative).as_posix():
            return default_root if len(path.parts) == 1 else default_root / Path(*path.parts[1:])
        return (Path.cwd() / path).resolve()

    artifacts_dir = resolve(args.artifacts_dir, paths["artifacts_root"], "artifacts")
    output_dir = resolve(args.output_dir, paths["artifacts_root"], "artifacts/phase10")
    export_dir = resolve(args.export_dir, paths["exports_root"], "exports/final_package")
    zip_path = resolve(args.zip_path, paths["exports_root"], "exports/phase10_final_research_package.zip")
    return artifacts_dir, output_dir, export_dir, zip_path


def relative_to_workspace(path: Path, state: BuildState) -> str:
    for root in (state.artifacts_dir.parent, REPO_ROOT):
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return path.resolve().as_posix()


def record_generated(state: BuildState, path: Path, output_type: str) -> None:
    rel = relative_to_workspace(path, state)
    if rel not in state.generated_files:
        state.generated_files.append(rel)
    if output_type == "table" and rel not in state.generated_tables:
        state.generated_tables.append(rel)
    elif output_type == "figure" and rel not in state.generated_figures:
        state.generated_figures.append(rel)
    elif output_type == "report" and rel not in state.generated_reports:
        state.generated_reports.append(rel)


def record_source(
    state: BuildState,
    output: Path,
    output_type: str,
    action: str,
    sources: list[Path],
    notes: str = "",
) -> None:
    source_text = ";".join(relative_to_workspace(path, state) for path in sources)
    source_phase = infer_source_phase(sources)
    state.source_rows.append(
        {
            "generated_output": relative_to_workspace(output, state),
            "output_type": output_type,
            "action": action,
            "source_phase": source_phase,
            "source_files": source_text,
            "notes": notes,
        }
    )


def infer_source_phase(paths: list[Path]) -> str:
    phases: list[str] = []
    for path in paths:
        text = path.as_posix()
        for phase in PHASE_ROOTS:
            if f"/{phase}/" in text or text.endswith(f"/{phase}"):
                phases.append(phase)
    return ";".join(sorted(set(phases))) if phases else "phase10"


def add_missing(state: BuildState, required: bool, item: str, impact: str) -> None:
    target = state.missing_required if required else state.missing_optional
    target.append({"artifact": item, "impact": impact})
    level = "required" if required else "optional"
    state.warnings.append(f"Missing {level} artifact: {item} ({impact})")


def ensure_safe_target(path: Path, expected_name: str, kind: str) -> None:
    if kind == "dir" and path.name != expected_name:
        raise ValueError(f"Refusing to recreate unexpected directory target: {path}")
    if kind == "zip" and path.name != expected_name:
        raise ValueError(f"Refusing to overwrite unexpected zip target: {path}")


def recreate_dir(path: Path, expected_name: str) -> None:
    ensure_safe_target(path, expected_name, "dir")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def remove_zip(path: Path) -> None:
    ensure_safe_target(path, "phase10_final_research_package.zip", "zip")
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def is_forbidden(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if parts.intersection(FORBIDDEN_DIRS):
        return True
    if path.name.lower() in FORBIDDEN_NAMES:
        return True
    return path.suffix.lower() in FORBIDDEN_SUFFIXES


def copy_tree_filtered(src: Path, dst: Path, state: BuildState) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        if path.is_dir() or is_forbidden(path):
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def discover_phase_dirs(state: BuildState) -> None:
    for phase, parts in PHASE_ROOTS.items():
        path = state.artifacts_dir.joinpath(*parts)
        state.phase_dirs[phase] = path
        if path.exists() and path.is_dir():
            state.validated_phases.append(phase)
        else:
            state.missing_phases.append(phase)
            add_missing(state, True, path.as_posix(), f"{phase} cannot be fully summarized")


def phase_specificity_score(path: Path, phase: str) -> tuple[int, int, str]:
    name = path.name.lower()
    parent = path.parent.name.lower()
    phase_number = phase.replace("phase", "")
    score = 0
    if phase in name or f"phase{phase_number}" in name:
        score += 100
    if phase in parent:
        score += 25
    if "diagnostic" in path.as_posix().lower() or "smoke" in path.as_posix().lower():
        score -= 1000
    return (score, -len(path.parts), name)


def discover_summary_file(root: Path, phase: str) -> Path | None:
    if not root.exists():
        return None
    for pattern in SUMMARY_PATTERNS:
        matches = [path for path in root.rglob(pattern) if path.is_file() and not is_forbidden(path)]
        if matches:
            return sorted(matches, key=lambda item: phase_specificity_score(item, phase), reverse=True)[0]
    return None


def discover_core_summaries(state: BuildState) -> None:
    for phase, root in state.phase_dirs.items():
        if phase in state.missing_phases:
            continue
        selected = discover_summary_file(root, phase)
        if selected:
            state.selected_summaries[phase] = relative_to_workspace(selected, state)
        else:
            add_missing(state, True, root.as_posix(), f"no core summary file found for {phase}")


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_csv(state: BuildState, frame: pd.DataFrame, path: Path, sources: list[Path], notes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    record_generated(state, path, "table")
    record_source(state, path, "table", "generated", sources, notes)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def build_model_comparison(state: BuildState) -> None:
    sources: list[Path] = []
    preferred = state.phase_dirs["phase6"] / "tables" / "single_model_results.csv"
    frame = read_csv(preferred)
    if frame is not None:
        sources.append(preferred)
        out = frame.copy()
    else:
        frames: list[pd.DataFrame] = []
        candidates = [
            ("phase2", state.phase_dirs["phase2"] / "tables" / "encoder_results.csv"),
            ("phase3", state.phase_dirs["phase3"] / "tables" / "advanced_encoder_results_clean.csv"),
            ("phase4", state.phase_dirs["phase4"] / "tables" / "decoder_prompting_results.csv"),
            ("phase5", state.phase_dirs["phase5"] / "tables" / "decoder_qlora_results.csv"),
        ]
        for phase, path in candidates:
            item = read_csv(path)
            if item is None:
                add_missing(state, False, path.as_posix(), f"{phase} rows omitted from final_model_comparison.csv")
                continue
            item = item.copy()
            item.insert(0, "phase", phase)
            frames.append(item)
            sources.append(path)
        if not frames:
            add_missing(state, True, "model comparison source CSVs", "final_model_comparison.csv not generated")
            return
        out = pd.concat(frames, ignore_index=True, sort=False)

    preferred_columns = [
        "phase",
        "run_name",
        "experiment_name",
        "model_name",
        "architecture_group",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "invalid_output_rate",
        "n_samples",
        "predictions_path",
    ]
    columns = [column for column in preferred_columns if column in out.columns]
    columns += [column for column in out.columns if column not in columns]
    write_csv(
        state,
        out.loc[:, columns],
        state.output_dir / "tables" / "final_model_comparison.csv",
        sources,
        "final single-model benchmark comparison",
    )


def build_phase_summary(state: BuildState) -> None:
    rows = []
    for phase, root in state.phase_dirs.items():
        rows.append(
            {
                "phase": phase,
                "artifact_dir": relative_to_workspace(root, state),
                "validated": phase in state.validated_phases,
                "selected_core_summary": state.selected_summaries.get(phase, ""),
                "file_count": len([path for path in root.rglob("*") if path.is_file()]) if root.exists() else 0,
            }
        )
    write_csv(
        state,
        pd.DataFrame(rows),
        state.output_dir / "tables" / "final_phase_summary.csv",
        [state.artifacts_dir.parent / rel for rel in state.selected_summaries.values()],
        "phase validation and selected core summary inventory",
    )


def normalize_per_author(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = frame.copy()
    if "source" not in out.columns and "run_name" in out.columns:
        out.insert(0, "source", out["run_name"])
    elif "source" not in out.columns:
        out.insert(0, "source", source_name)
    if "source_type" not in out.columns:
        out.insert(1, "source_type", source_name)
    return out


def build_per_author_metrics(state: BuildState) -> None:
    candidates = [
        ("phase3", state.phase_dirs["phase3"] / "tables" / "phase3_per_author_f1.csv"),
        ("phase4", state.phase_dirs["phase4"] / "tables" / "phase4_per_author_f1.csv"),
        ("phase5", state.phase_dirs["phase5"] / "tables" / "phase5_per_author_f1.csv"),
        ("phase6", state.phase_dirs["phase6"] / "tables" / "per_author_f1.csv"),
        ("phase9", state.phase_dirs["phase9"] / "tables" / "per_author_f1_comparison.csv"),
    ]
    frames: list[pd.DataFrame] = []
    sources: list[Path] = []
    for phase, path in candidates:
        frame = read_csv(path)
        if frame is None:
            add_missing(state, False, path.as_posix(), f"{phase} omitted from final_per_author_metrics.csv")
            continue
        item = normalize_per_author(frame, phase)
        item.insert(0, "phase", phase)
        frames.append(item)
        sources.append(path)
    if not frames:
        add_missing(state, False, "per-author metric CSVs", "final_per_author_metrics.csv not generated")
        return
    write_csv(
        state,
        pd.concat(frames, ignore_index=True, sort=False),
        state.output_dir / "tables" / "final_per_author_metrics.csv",
        sources,
        "per-author precision/recall/F1 metrics where available",
    )


def copy_table_if_available(state: BuildState, source: Path, output_name: str, required: bool, impact: str) -> None:
    frame = read_csv(source)
    if frame is None:
        add_missing(state, required, source.as_posix(), impact)
        return
    write_csv(state, frame, state.output_dir / "tables" / output_name, [source], f"copied from {source.name}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_topic_modeling_summary(state: BuildState) -> None:
    lda_summary = state.phase_dirs["phase7"] / "metrics" / "lda_summary.json"
    bertopic_summary = state.phase_dirs["phase8"] / "reports" / "bertopic_summary.json"
    sources: list[Path] = []
    rows: list[dict[str, Any]] = []
    lda = load_json(lda_summary)
    if lda:
        sources.append(lda_summary)
        rows.append(
            {
                "phase": "phase7",
                "method": "LDA",
                "selected_topics": lda.get("selected_k", ""),
                "n_documents": lda.get("n_documents", ""),
                "input_total": (lda.get("input_row_counts") or {}).get("total", ""),
                "summary_path": relative_to_workspace(lda_summary, state),
            }
        )
    else:
        add_missing(state, False, lda_summary.as_posix(), "LDA summary fields omitted")
    bertopic = load_json(bertopic_summary)
    if bertopic:
        sources.append(bertopic_summary)
        rows.append(
            {
                "phase": "phase8",
                "method": "BERTopic",
                "selected_topics": bertopic.get("n_topics_including_outlier", ""),
                "n_documents": bertopic.get("n_documents", ""),
                "input_total": (bertopic.get("input_row_counts") or {}).get("total", ""),
                "outlier_documents": bertopic.get("n_outlier_documents", ""),
                "embedding_model": bertopic.get("embedding_model", ""),
                "summary_path": relative_to_workspace(bertopic_summary, state),
            }
        )
    else:
        add_missing(state, False, bertopic_summary.as_posix(), "BERTopic summary fields omitted")
    if not rows:
        add_missing(state, False, "topic modeling summaries", "final_topic_modeling_summary.csv not generated")
        return
    write_csv(
        state,
        pd.DataFrame(rows),
        state.output_dir / "tables" / "final_topic_modeling_summary.csv",
        sources,
        "LDA and BERTopic summary metrics",
    )


def build_tables(state: BuildState) -> None:
    build_model_comparison(state)
    build_phase_summary(state)
    build_per_author_metrics(state)
    copy_table_if_available(
        state,
        state.phase_dirs["phase6"] / "tables" / "ensemble_results.csv",
        "final_ensemble_comparison.csv",
        False,
        "ensemble comparison table not generated",
    )
    build_topic_modeling_summary(state)
    copy_table_if_available(
        state,
        state.phase_dirs["phase9"] / "tables" / "phase9_results_summary.csv",
        "final_topic_aware_comparison.csv",
        False,
        "topic-aware comparison table not generated",
    )


def copy_figure(state: BuildState, output_name: str, candidates: list[Path]) -> None:
    source = first_existing(candidates)
    output = state.output_dir / "figures" / output_name
    if source is None:
        add_missing(state, False, output_name, "figure not generated because no validated source figure was found")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    record_generated(state, output, "figure")
    record_source(state, output, "figure", "copied", [source], "closest validated existing figure")


def build_figures(state: BuildState) -> None:
    copy_figure(
        state,
        "final_model_macro_f1.png",
        [
            state.phase_dirs["phase6"] / "plots" / "phase6_macro_f1_comparison.png",
            state.phase_dirs["phase3"] / "plots" / "phase3_valid_runs_only_macro_f1.png",
            state.phase_dirs["phase2"] / "plots" / "encoder_macro_f1_bar.png",
        ],
    )
    copy_figure(
        state,
        "final_accuracy_macro_f1_tradeoff.png",
        [
            state.phase_dirs["phase6"] / "plots" / "phase6_accuracy_comparison.png",
            state.phase_dirs["phase3"] / "plots" / "phase3_valid_runs_only_accuracy.png",
            state.phase_dirs["phase2"] / "plots" / "encoder_accuracy_bar.png",
        ],
    )
    copy_figure(
        state,
        "final_confusion_matrix_best_model.png",
        [
            state.phase_dirs["phase6"] / "plots" / "confusion_matrix_best_encoder_best_decoder_vote.png",
            state.phase_dirs["phase3"] / "plots" / "phase3_confusion_matrix_best_deberta.png",
            state.phase_dirs["phase2"] / "plots" / "roberta_large_confusion_matrix.png",
        ],
    )
    copy_figure(
        state,
        "final_per_author_f1.png",
        [
            state.phase_dirs["phase6"] / "plots" / "per_author_best_ensemble_vs_best_single_f1_delta.png",
            state.phase_dirs["phase4"] / "plots" / "per_author_f1_heatmap.png",
            state.phase_dirs["phase9"] / "plots" / "per_author_f1_comparison.png",
        ],
    )
    copy_figure(
        state,
        "final_ensemble_vs_single_model.png",
        [
            state.phase_dirs["phase6"] / "plots" / "ensemble_macro_f1_gain_over_best_single.png",
            state.phase_dirs["phase6"] / "plots" / "ensemble_accuracy_gain_over_best_single.png",
        ],
    )
    copy_figure(
        state,
        "final_topic_distribution.png",
        [
            state.phase_dirs["phase8"] / "plots" / "per_author_top_topics_heatmap.png",
            state.phase_dirs["phase8"] / "plots" / "per_author_topic_heatmap.png",
            state.phase_dirs["phase7"] / "plots" / "author_topic_heatmap.png",
        ],
    )
    copy_figure(
        state,
        "final_topic_aware_comparison.png",
        [
            state.phase_dirs["phase9"] / "plots" / "phase9_macro_f1_comparison.png",
            state.phase_dirs["phase9"] / "plots" / "topic_only_vs_transformer_summary.png",
        ],
    )


def best_row(path: Path) -> dict[str, Any]:
    frame = read_csv(path)
    if frame is None or frame.empty:
        return {}
    metric = "macro_f1" if "macro_f1" in frame.columns else "accuracy" if "accuracy" in frame.columns else None
    if metric is None:
        return frame.iloc[0].to_dict()
    values = pd.to_numeric(frame[metric], errors="coerce")
    if values.notna().any():
        return frame.iloc[int(values.idxmax())].to_dict()
    return frame.iloc[0].to_dict()


def format_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "not available"


def report_guard(text: str) -> str:
    lower = text.lower()
    for term in REPORT_FORBIDDEN_TERMS:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, lower):
            raise ValueError(f"Generated report contains unrelated term: {term}")
    return text


def write_report(state: BuildState, name: str, lines: list[str], sources: list[Path] | None = None) -> None:
    path = state.output_dir / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = report_guard("\n".join(lines).rstrip() + "\n")
    path.write_text(text, encoding="utf-8")
    record_generated(state, path, "report")
    record_source(state, path, "report", "generated", sources or [], "markdown report")


def build_missing_report(state: BuildState) -> None:
    lines = [
        "# Missing Artifacts",
        "",
        "This report lists expected Phase 1-9 inputs that were unavailable during Phase 10 packaging.",
        "",
        "## Missing Required Artifacts",
        "",
    ]
    if state.missing_required:
        for item in state.missing_required:
            lines.append(f"- `{item['artifact']}`: {item['impact']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Missing Optional Artifacts", ""])
    if state.missing_optional:
        for item in state.missing_optional:
            lines.append(f"- `{item['artifact']}`: {item['impact']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Impact",
            "",
            "Missing required artifacts can make strict Phase 10 packaging fail.",
            "Missing optional artifacts reduce the completeness of selected final tables, figures, or narrative sections, but available validated sources are still reported.",
        ]
    )
    write_report(state, "missing_artifacts.md", lines)


def build_reports(state: BuildState) -> None:
    model_path = state.output_dir / "tables" / "final_model_comparison.csv"
    ensemble_path = state.output_dir / "tables" / "final_ensemble_comparison.csv"
    topic_path = state.output_dir / "tables" / "final_topic_modeling_summary.csv"
    topic_aware_path = state.output_dir / "tables" / "final_topic_aware_comparison.csv"
    best_model = best_row(ensemble_path if ensemble_path.exists() else model_path)
    best_name = best_model.get("strategy") or best_model.get("run_name") or best_model.get("model_name") or "not available"

    write_report(
        state,
        "final_research_report.md",
        [
            "# Final Research Report",
            "",
            "## Project",
            "",
            "Authorship Attribution in Victorian Periodicals evaluates supervised authorship attribution for six canonical authors using the fixed PERIAD train/test split.",
            "",
            "## Experimental Pipeline",
            "",
            "- Phase 1 prepared and validated the dataset artifacts.",
            "- Phase 2 evaluated encoder baselines.",
            "- Phase 3 evaluated advanced encoders.",
            "- Phase 4 evaluated decoder prompting baselines.",
            "- Phase 5 evaluated QLoRA decoder fine-tuning outputs.",
            "- Phase 6 evaluated ensemble voting over existing predictions.",
            "- Phase 7 generated LDA topic modeling artifacts.",
            "- Phase 8 generated BERTopic artifacts.",
            "- Phase 9 evaluated topic-aware classification variants.",
            "- Phase 10 packages the validated outputs without retraining.",
            "",
            "## Key Result",
            "",
            f"The best available final row is `{best_name}` with accuracy {format_metric(best_model.get('accuracy'))} and macro F1 {format_metric(best_model.get('macro_f1'))}.",
            "",
            "## Traceability",
            "",
            "Every generated Phase 10 output is traced in `manifest/source_manifest.csv`.",
        ],
        [path for path in (model_path, ensemble_path) if path.exists()],
    )

    write_report(
        state,
        "final_results_summary.md",
        [
            "# Final Results Summary",
            "",
            f"- Validated phases: {', '.join(state.validated_phases) if state.validated_phases else 'none'}",
            f"- Missing phases: {', '.join(state.missing_phases) if state.missing_phases else 'none'}",
            f"- Best available row: `{best_name}`",
            f"- Accuracy: {format_metric(best_model.get('accuracy'))}",
            f"- Macro F1: {format_metric(best_model.get('macro_f1'))}",
            "",
            "Final comparison tables are stored in `tables/`; final figures are stored in `figures/`.",
        ],
        [path for path in (model_path, ensemble_path, topic_aware_path) if path.exists()],
    )

    write_report(
        state,
        "final_reproducibility_notes.md",
        [
            "# Final Reproducibility Notes",
            "",
            "Run Phase 10 from the project repository with:",
            "",
            "```bash",
            "python scripts/run_phase10_final_package.py",
            "```",
            "",
            "Phase 10 reads existing artifacts only. It does not retrain models, run inference, or modify Phase 1-9 artifacts.",
            "",
            "The local path configuration is read from `configs/paths.local.yaml` when available.",
        ],
    )

    manifest_path = state.output_dir / "manifest" / "source_manifest.csv"
    write_report(
        state,
        "final_artifact_manifest.md",
        [
            "# Final Artifact Manifest",
            "",
            f"- Source manifest: `{relative_to_workspace(manifest_path, state)}`",
            f"- Generated files: {len(state.generated_files)}",
            f"- Generated tables: {len(state.generated_tables)}",
            f"- Generated figures: {len(state.generated_figures)}",
            f"- Generated reports: {len(state.generated_reports)}",
            "",
            "The CSV manifest contains source paths for each generated final output.",
        ],
    )

    write_report(
        state,
        "final_limitations.md",
        [
            "# Final Limitations",
            "",
            "- Phase 10 reports only artifacts available in the local validated artifact mirror.",
            "- Missing optional artifacts are documented rather than reconstructed.",
            "- Topic-aware classification is summarized from existing Phase 9 outputs only.",
            "- Large model files, adapters, checkpoints, and raw training logs are excluded from the final export.",
        ],
    )

    write_report(
        state,
        "final_appendix.md",
        [
            "# Final Appendix",
            "",
            "## Selected Core Summary Files",
            "",
            *(f"- {phase}: `{path}`" for phase, path in sorted(state.selected_summaries.items())),
            "",
            "## Topic Modeling Summary",
            "",
            f"See `{relative_to_workspace(topic_path, state)}` for LDA and BERTopic package-level fields.",
        ],
        [topic_path] if topic_path.exists() else [],
    )
    build_missing_report(state)


def write_source_manifest(state: BuildState) -> None:
    path = state.output_dir / "manifest" / "source_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["generated_output", "output_type", "action", "source_phase", "source_files", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(state.source_rows)
    record_generated(state, path, "table")


def write_run_metadata(state: BuildState) -> None:
    path = state.output_dir / "manifest" / "run_metadata.json"
    metadata = {
        "timestamp": utc_timestamp(),
        "strict": state.strict,
        "paths": {
            "artifacts_dir": state.artifacts_dir.resolve().as_posix(),
            "output_dir": state.output_dir.resolve().as_posix(),
            "export_dir": state.export_dir.resolve().as_posix(),
            "zip_path": state.zip_path.resolve().as_posix(),
        },
        "validated_phases": state.validated_phases,
        "missing_phases": state.missing_phases,
        "selected_core_summary_files": state.selected_summaries,
        "generated_files": sorted(set(state.generated_files)),
        "generated_tables": sorted(set(state.generated_tables)),
        "generated_figures": sorted(set(state.generated_figures)),
        "generated_reports": sorted(set(state.generated_reports)),
        "missing_required_files": state.missing_required,
        "missing_optional_files": state.missing_optional,
        "warnings": state.warnings,
        "zip_validation": state.zip_validation,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record_generated(state, path, "table")


def export_package(state: BuildState) -> None:
    recreate_dir(state.export_dir, "final_package")
    for name in ("tables", "figures", "reports"):
        copy_tree_filtered(state.output_dir / name, state.export_dir / name, state)
    copy_tree_filtered(REPO_ROOT / "configs", state.export_dir / "configs_snapshot", state)
    copy_tree_filtered(REPO_ROOT / "prompts", state.export_dir / "prompts_snapshot", state)
    copy_tree_filtered(state.output_dir / "manifest", state.export_dir / "reproducibility" / "manifest", state)
    readme = state.export_dir / "README_FINAL_PACKAGE.md"
    readme.write_text(
        report_guard(
            "\n".join(
                [
                    "# Final Research Package",
                    "",
                    "This export contains the Phase 10 final package for Authorship Attribution in Victorian Periodicals.",
                    "",
                    "- `tables/`: final CSV tables",
                    "- `figures/`: final paper-ready figures copied from validated sources",
                    "- `reports/`: final Markdown reports",
                    "- `configs_snapshot/`: lightweight configuration snapshot",
                    "- `prompts_snapshot/`: prompt templates used by prior phases",
                    "- `reproducibility/`: manifests and run metadata",
                    "",
                    "Large model files, adapters, checkpoints, and raw training logs are intentionally excluded.",
                ]
            )
            + "\n"
        ),
        encoding="utf-8",
    )


def create_zip(state: BuildState) -> None:
    remove_zip(state.zip_path)
    with zipfile.ZipFile(state.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in state.export_dir.rglob("*"):
            if path.is_dir() or is_forbidden(path):
                continue
            archive.write(path, path.relative_to(state.export_dir).as_posix())


def validate_zip(state: BuildState) -> None:
    forbidden_entries: list[str] = []
    entries: list[str] = []
    if state.zip_path.exists():
        with zipfile.ZipFile(state.zip_path) as archive:
            entries = archive.namelist()
    for entry in entries:
        entry_path = Path(entry)
        if is_forbidden(entry_path) or any(fnmatch.fnmatch(entry, f"*{suffix}") for suffix in FORBIDDEN_SUFFIXES):
            forbidden_entries.append(entry)
    state.zip_validation = {
        "zip_exists": state.zip_path.exists(),
        "zip_size_bytes": state.zip_path.stat().st_size if state.zip_path.exists() else 0,
        "zip_size_gt_zero": state.zip_path.exists() and state.zip_path.stat().st_size > 0,
        "readme_exists": (state.export_dir / "README_FINAL_PACKAGE.md").exists(),
        "forbidden_entries": forbidden_entries,
        "forbidden_entries_absent": not forbidden_entries,
        "valid": state.zip_path.exists()
        and state.zip_path.stat().st_size > 0
        and (state.export_dir / "README_FINAL_PACKAGE.md").exists()
        and not forbidden_entries,
    }


def print_summary(state: BuildState) -> None:
    print("\nPHASE 10 FINAL PACKAGE REPORT")
    print(f"- validated_phases: {', '.join(state.validated_phases) if state.validated_phases else 'none'}")
    print(f"- missing_phases: {', '.join(state.missing_phases) if state.missing_phases else 'none'}")
    print(f"- generated_tables: {len(state.generated_tables)}")
    for path in sorted(set(state.generated_tables)):
        print(f"  - {path}")
    print(f"- generated_figures: {len(state.generated_figures)}")
    for path in sorted(set(state.generated_figures)):
        print(f"  - {path}")
    print(f"- generated_reports: {len(state.generated_reports)}")
    for path in sorted(set(state.generated_reports)):
        print(f"  - {path}")
    print(f"- export_dir: {state.export_dir.resolve().as_posix()}")
    print(f"- zip_path: {state.zip_path.resolve().as_posix()}")
    print("- zip_validation:")
    for key, value in state.zip_validation.items():
        print(f"  - {key}: {value}")
    print(f"- warnings: {len(state.warnings)}")
    for warning in state.warnings:
        print(f"  - {warning}")


def strict_check(state: BuildState) -> None:
    if not state.strict:
        return
    errors = []
    if state.missing_phases:
        errors.append(f"missing phases: {', '.join(state.missing_phases)}")
    if state.missing_required:
        errors.append(f"missing required artifacts: {len(state.missing_required)}")
    if errors:
        raise FileNotFoundError("; ".join(errors))


def run(args: argparse.Namespace) -> BuildState:
    artifacts_dir, output_dir, export_dir, zip_path = resolve_defaults(args)
    state = BuildState(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        export_dir=export_dir,
        zip_path=zip_path,
        strict=args.strict,
        phase_dirs={},
    )
    discover_phase_dirs(state)
    discover_core_summaries(state)
    strict_check(state)
    recreate_dir(state.output_dir, "phase10")
    for subdir in ("tables", "figures", "reports", "manifest"):
        (state.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    build_tables(state)
    build_figures(state)
    build_reports(state)
    write_source_manifest(state)
    export_package(state)
    create_zip(state)
    validate_zip(state)
    write_run_metadata(state)
    copy_tree_filtered(state.output_dir / "manifest", state.export_dir / "reproducibility" / "manifest", state)
    create_zip(state)
    validate_zip(state)
    write_run_metadata(state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--output_dir", default="artifacts/phase10")
    parser.add_argument("--export_dir", default="exports/final_package")
    parser.add_argument("--zip_path", default="exports/phase10_final_research_package.zip")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = run(args)
    print_summary(state)
    if not state.zip_validation.get("valid", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
