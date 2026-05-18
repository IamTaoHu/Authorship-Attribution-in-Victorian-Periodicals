"""Run Phase 8 BERTopic topic modelling and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic_modeling.bertopic_phase8 import default_phase8_dir, run_phase8  # noqa: E402
from src.visualization.plot_phase8_bertopic import plot_phase8_bertopic  # noqa: E402


PROTECTED_PLOTS_ONLY_PATHS = (
    "assignments/topic_assignments.csv",
    "assignments/topic_probabilities.npy",
    "embeddings/document_embeddings.npy",
    "models/bertopic_model",
)


def resolve_phase8_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path
    return default_phase8_dir()


def snapshot_protected_mtimes(root: Path) -> dict[str, int | None]:
    mtimes: dict[str, int | None] = {}
    for relative in PROTECTED_PLOTS_ONLY_PATHS:
        path = root / relative
        mtimes[relative] = path.stat().st_mtime_ns if path.exists() else None
    return mtimes


def assert_protected_mtimes_unchanged(root: Path, before: dict[str, int | None]) -> None:
    changed: list[str] = []
    for relative, before_mtime in before.items():
        path = root / relative
        after_mtime = path.stat().st_mtime_ns if path.exists() else None
        if after_mtime != before_mtime:
            changed.append(relative)
    if changed:
        formatted = ", ".join(changed)
        raise RuntimeError(f"--plots_only modified protected Phase 8 core artifact(s): {formatted}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase8/bertopic.yaml")
    parser.add_argument("--output_dir", default=None, help="Optional override for artifacts/phase8/bertopic.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    probability_group = parser.add_mutually_exclusive_group()
    probability_group.add_argument("--calculate_probabilities", action="store_true", help="Enable BERTopic probability matrix export.")
    probability_group.add_argument("--no_probabilities", action="store_true", help="Disable BERTopic probability matrix export.")
    parser.add_argument("--limit", type=int, default=None, help="Optional diagnostic document limit.")
    parser.add_argument("--plots_only", action="store_true", help="Regenerate Phase 8 plots/report tables from existing artifacts without fitting BERTopic.")
    args = parser.parse_args()

    if args.plots_only:
        phase8_dir = resolve_phase8_dir(args.output_dir)
        protected_mtimes = snapshot_protected_mtimes(phase8_dir)
        outputs = plot_phase8_bertopic(phase8_dir)
        assert_protected_mtimes_unchanged(phase8_dir, protected_mtimes)
        print(f"Regenerated Phase 8 plots/report tables only: {phase8_dir}")
        print(f"Polished heatmap: {outputs['per_author_top_topics_heatmap']}")
        print(f"Author scatter: {outputs['topic_scatter_by_author']}")
        print(f"Top topic words: {outputs['top_topic_words_csv']}")
        return 0

    probability_override = None
    if args.calculate_probabilities:
        probability_override = True
    elif args.no_probabilities:
        probability_override = False

    outputs = run_phase8(
        args.config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        calculate_probabilities=probability_override,
        limit=args.limit,
    )
    try:
        plot_phase8_bertopic(outputs["phase8_dir"])
    except Exception as exc:
        raise RuntimeError(f"Phase 8 core outputs were written, but required static plotting failed: {exc}") from exc

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    print(f"Wrote Phase 8 BERTopic artifacts: {outputs['phase8_dir']}")
    print(f"Summary: {outputs['summary']}")
    print(f"Assignments: {outputs['topic_assignments']}")
    print(f"Phase 9 features: {outputs['document_topic_features']}")
    print(f"Documents: {summary['n_documents']}")
    print(f"Topics including outlier: {summary['n_topics_including_outlier']}")
    print(f"Probabilities saved: {summary['probabilities_saved']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
