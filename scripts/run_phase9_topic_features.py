"""Run Phase 9 topic-aware classification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.topic_features_phase9 import run_phase9  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase9/topic_aware_classification.yaml")
    parser.add_argument("--dataset_train", default=None)
    parser.add_argument("--dataset_test", default=None)
    parser.add_argument("--phase8_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--skip_transformers", action="store_true")
    parser.add_argument("--plots_only", action="store_true")
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--allow_repo_outputs", action="store_true")
    args = parser.parse_args()
    outputs = run_phase9(
        args.config,
        dataset_train=args.dataset_train,
        dataset_test=args.dataset_test,
        phase8_dir=args.phase8_dir,
        output_dir=args.output_dir,
        variants_raw=args.variants,
        skip_transformers=args.skip_transformers,
        plots_only=args.plots_only,
        report_only=args.report_only,
        allow_repo_outputs=args.allow_repo_outputs,
    )
    print(f"Phase 9 output directory: {outputs['phase9_dir']}")
    if "summary" in outputs:
        print(f"Summary table: {outputs['summary']}")
    if "report" in outputs:
        print(f"Report: {outputs['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
