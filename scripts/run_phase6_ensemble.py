"""Run Phase 6 ensemble evaluation and plotting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ensemble_phase6 import run_phase6  # noqa: E402
from src.visualization.plot_phase6_results import plot_phase6_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6/ensemble.yaml")
    args = parser.parse_args()
    outputs = run_phase6(args.config)
    plot_phase6_results()
    print(f"Wrote Phase 6 report: {outputs['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
