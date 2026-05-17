"""Validate the lightweight Phase 5 artifact mirror."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts" / "phase5"

REQUIRED_FILES = (
    "plots/qlora_accuracy_bar.png",
    "plots/qlora_macro_f1_bar.png",
    "reports/phase5_qlora_decoder_finetuning_report.md",
    "tables/decoder_qlora_results.csv",
    "tables/decoder_qlora_results.md",
    "tables/phase5_per_author_f1.csv",
    "runs/mistral_qlora/predictions.csv",
    "runs/llama3_qlora/predictions.csv",
    "runs/gemma2_qlora/predictions.csv",
)
FORBIDDEN_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def is_forbidden(path: Path) -> bool:
    return path.suffix.lower() in FORBIDDEN_SUFFIXES or any(part.startswith("checkpoint-") for part in path.parts)


def main() -> int:
    ok = True
    print(f"Checking artifact root: {ARTIFACT_ROOT}")
    for relative in REQUIRED_FILES:
        path = ARTIFACT_ROOT / relative
        exists = path.exists() and path.is_file() and path.stat().st_size > 0
        print(f"[{'OK' if exists else 'MISSING'}] {path}")
        ok = exists and ok
    forbidden = [path for path in ARTIFACT_ROOT.rglob("*") if is_forbidden(path)] if ARTIFACT_ROOT.exists() else []
    if forbidden:
        print("[MISSING] Forbidden model/checkpoint files found under artifacts/phase5:")
        for path in forbidden:
            print(f"  {path}")
        ok = False
    else:
        print("[OK] No model/checkpoint weight files found under artifacts/phase5")
    if ok:
        print("Phase 5 artifact check passed.")
        return 0
    print("Phase 5 artifact check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
