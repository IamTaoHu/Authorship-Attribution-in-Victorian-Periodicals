from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_phase10_final_package.py"
SPEC = importlib.util.spec_from_file_location("run_phase10_final_package", SCRIPT_PATH)
phase10 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["run_phase10_final_package"] = phase10
SPEC.loader.exec_module(phase10)


class Phase10FinalPackageTests(unittest.TestCase):
    def test_discover_summary_prefers_priority_before_phase_specificity(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir) / "phase6"
            nested = root / "tables"
            nested.mkdir(parents=True)
            (nested / "summary.csv").write_text("name,value\nsummary,1\n", encoding="utf-8")
            (nested / "phase6_results.csv").write_text("name,value\nphase6,1\n", encoding="utf-8")

            selected = phase10.discover_summary_file(root, "phase6")

            self.assertEqual(selected, nested / "summary.csv")

    def test_forbidden_file_filtering_is_pathlib_safe(self) -> None:
        self.assertTrue(phase10.is_forbidden(Path("checkpoints") / "phase5" / "model.bin"))
        self.assertTrue(phase10.is_forbidden(Path("runs") / "training_log.jsonl"))
        self.assertTrue(phase10.is_forbidden(Path("models") / "adapter.safetensors"))
        self.assertFalse(phase10.is_forbidden(Path("tables") / "final_model_comparison.csv"))


if __name__ == "__main__":
    unittest.main()
