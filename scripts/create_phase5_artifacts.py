"""Create a lightweight browsable Phase 5 artifact mirror."""

from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
SOURCE_ROOT = REPO_ROOT / "outputs" / "phase5"
ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts" / "phase5"

FULL_RUNS = ("mistral_qlora", "llama3_qlora", "gemma2_qlora")
RUN_ALLOWED_FILES = {
    "predictions.csv",
    "metrics.json",
    "classification_report.txt",
    "label_mapping.json",
    "config.yaml",
    "training_log.jsonl",
}
ROOT_PLOTS = (
    "qlora_accuracy_bar.png",
    "qlora_macro_f1_bar.png",
    "qlora_vs_prompting_vs_encoder.png",
)
ROOT_REPORTS = ("phase5_qlora_decoder_finetuning_report.md",)
ROOT_TABLES = (
    "decoder_qlora_results.csv",
    "decoder_qlora_results.md",
    "phase5_per_author_f1.csv",
)
FORBIDDEN_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
FORBIDDEN_NAMES = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
DIAGNOSTIC_DIRS = (
    ARTIFACT_ROOT / "diagnostic",
    ARTIFACT_ROOT / "plots" / "diagnostic",
    ARTIFACT_ROOT / "reports" / "diagnostic",
    ARTIFACT_ROOT / "tables" / "diagnostic",
    ARTIFACT_ROOT / "runs" / "diagnostic",
)


def is_forbidden(path: Path) -> bool:
    return (
        path.name in FORBIDDEN_NAMES
        or path.suffix.lower() in FORBIDDEN_SUFFIXES
        or any(part.startswith("checkpoint-") for part in path.parts)
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def touch_gitkeep_if_empty(path: Path) -> None:
    ensure_dir(path)
    if not any(path.iterdir()):
        (path / ".gitkeep").write_text("", encoding="utf-8")


def copy_file(source: Path, destination: Path, copied: list[Path], skipped: list[str]) -> None:
    if not source.exists():
        skipped.append(f"missing optional: {source}")
        return
    if source.is_dir():
        skipped.append(f"not a file: {source}")
        return
    if is_forbidden(source) or is_forbidden(destination):
        raise RuntimeError(f"Refusing to copy forbidden model/checkpoint file: {source}")
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)
    copied.append(destination)


def copy_named_files(source_dir: Path, destination_dir: Path, names: tuple[str, ...], copied: list[Path], skipped: list[str]) -> None:
    for name in names:
        copy_file(source_dir / name, destination_dir / name, copied, skipped)


def copy_run(run_name: str, copied: list[Path], skipped: list[str]) -> None:
    source_dir = SOURCE_ROOT / "runs" / run_name
    destination_dir = ARTIFACT_ROOT / "runs" / run_name
    ensure_dir(destination_dir)
    if not source_dir.exists():
        skipped.append(f"missing run directory: {source_dir}")
        return
    for source in sorted(source_dir.iterdir()):
        if source.is_dir():
            if source.name.startswith("checkpoint-"):
                skipped.append(f"skipped checkpoint directory: {source}")
            continue
        if source.name not in RUN_ALLOWED_FILES:
            skipped.append(f"skipped run file outside allowlist: {source}")
            continue
        copy_file(source, destination_dir / source.name, copied, skipped)


def copy_diagnostic(copied: list[Path], skipped: list[str]) -> None:
    for path in DIAGNOSTIC_DIRS:
        ensure_dir(path)
    diagnostic_root = SOURCE_ROOT / "diagnostic"
    if not diagnostic_root.exists():
        for path in DIAGNOSTIC_DIRS:
            touch_gitkeep_if_empty(path)
        skipped.append(f"missing optional diagnostic directory: {diagnostic_root}")
        return

    for source in sorted((diagnostic_root / "tables").glob("*")) if (diagnostic_root / "tables").exists() else []:
        if source.is_file():
            copy_file(source, ARTIFACT_ROOT / "tables" / "diagnostic" / source.name, copied, skipped)
    for source in sorted((diagnostic_root / "reports").glob("*")) if (diagnostic_root / "reports").exists() else []:
        if source.is_file():
            copy_file(source, ARTIFACT_ROOT / "reports" / "diagnostic" / source.name, copied, skipped)
    for run_dir in sorted((diagnostic_root / "runs").iterdir()) if (diagnostic_root / "runs").exists() else []:
        if not run_dir.is_dir():
            continue
        destination_dir = ARTIFACT_ROOT / "runs" / "diagnostic" / run_dir.name
        ensure_dir(destination_dir)
        for source in sorted(run_dir.iterdir()):
            if source.is_file() and source.name in RUN_ALLOWED_FILES:
                copy_file(source, destination_dir / source.name, copied, skipped)
            elif source.is_file():
                skipped.append(f"skipped diagnostic run file outside allowlist: {source}")

    plots_dir = SOURCE_ROOT / "plots"
    for source in sorted(plots_dir.glob("diagnostic*.png")) if plots_dir.exists() else []:
        copy_file(source, ARTIFACT_ROOT / "plots" / "diagnostic" / source.name, copied, skipped)

    for path in DIAGNOSTIC_DIRS:
        touch_gitkeep_if_empty(path)


def assert_no_forbidden_artifacts() -> None:
    if not ARTIFACT_ROOT.exists():
        return
    forbidden = [path for path in ARTIFACT_ROOT.rglob("*") if is_forbidden(path)]
    if forbidden:
        formatted = "\n".join(str(path) for path in forbidden)
        raise RuntimeError(f"Forbidden model/checkpoint artifacts found under {ARTIFACT_ROOT}:\n{formatted}")


def create_phase5_artifacts() -> tuple[list[Path], list[str]]:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Phase 5 source outputs not found: {SOURCE_ROOT}")
    copied: list[Path] = []
    skipped: list[str] = []
    for directory in (
        ARTIFACT_ROOT / "plots",
        ARTIFACT_ROOT / "plots" / "diagnostic",
        ARTIFACT_ROOT / "reports",
        ARTIFACT_ROOT / "reports" / "diagnostic",
        ARTIFACT_ROOT / "runs",
        ARTIFACT_ROOT / "runs" / "diagnostic",
        ARTIFACT_ROOT / "tables",
        ARTIFACT_ROOT / "tables" / "diagnostic",
    ):
        ensure_dir(directory)
    copy_named_files(SOURCE_ROOT / "plots", ARTIFACT_ROOT / "plots", ROOT_PLOTS, copied, skipped)
    copy_named_files(SOURCE_ROOT / "reports", ARTIFACT_ROOT / "reports", ROOT_REPORTS, copied, skipped)
    copy_named_files(SOURCE_ROOT / "tables", ARTIFACT_ROOT / "tables", ROOT_TABLES, copied, skipped)
    for run_name in FULL_RUNS:
        copy_run(run_name, copied, skipped)
    copy_diagnostic(copied, skipped)
    assert_no_forbidden_artifacts()
    return copied, skipped


def main() -> int:
    copied, skipped = create_phase5_artifacts()
    print(f"Created/updated {ARTIFACT_ROOT}")
    print(f"Copied files: {len(copied)}")
    for path in copied:
        print(f"  copied {path.relative_to(WORKSPACE_ROOT)}")
    print(f"Skipped optional files/items: {len(skipped)}")
    for item in skipped:
        print(f"  warning: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
