"""Run Phase 7 LDA topic modelling and plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic_modeling.lda_phase7 import run_phase7  # noqa: E402
from src.visualization.plot_phase7_lda import plot_phase7_lda  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase7/lda.yaml")
    parser.add_argument("--phase7_dir", default=None, help="Optional override for artifacts/phase7/lda.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Phase 7 LDA outputs.")
    args = parser.parse_args()
    outputs = run_phase7(args.config, phase7_dir=args.phase7_dir, force=args.force)
    plot_phase7_lda(outputs["phase7_dir"])
    print(f"Wrote Phase 7 LDA artifacts: {outputs['phase7_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
