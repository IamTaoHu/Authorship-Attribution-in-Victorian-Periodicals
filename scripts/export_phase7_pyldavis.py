"""Export Phase 7 LDA pyLDAvis HTML from saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pyLDAvis


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic_modeling.lda_phase7 import load_config, load_corpus, normalize_text, resolve_phase7_paths  # noqa: E402


DEFAULT_OUTPUT_NAME = "phase7_lda_pyldavis.html"


def require_path(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing required {label}: {path}")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_outside_repo(output_path: Path) -> None:
    resolved_output = output_path.resolve()
    resolved_repo = REPO_ROOT.resolve()
    if is_relative_to(resolved_output, resolved_repo):
        raise ValueError(f"Refusing to write generated pyLDAvis HTML inside the repository: {resolved_output}")


def prepare_panel(best_lda: Any, dtm: Any, vectorizer: Any) -> Any:
    try:
        import pyLDAvis.sklearn as pyldavis_sklearn

        return pyldavis_sklearn.prepare(best_lda, dtm, vectorizer, sort_topics=False)
    except ModuleNotFoundError:
        import pyLDAvis.lda_model as pyldavis_lda_model

        return pyldavis_lda_model.prepare(best_lda, dtm, vectorizer, sort_topics=False)
    except AttributeError:
        import pyLDAvis.lda_model as pyldavis_lda_model

        return pyldavis_lda_model.prepare(best_lda, dtm, vectorizer, sort_topics=False)


def resolve_output_path(phase7_root: Path, output: str | Path | None) -> Path:
    if output is None:
        return phase7_root / "interactive" / DEFAULT_OUTPUT_NAME
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = phase7_root / output_path
    return output_path


def export_pyldavis(config_path: str | Path, phase7_dir: str | Path | None, output: str | Path | None, force: bool) -> Path:
    config = load_config(config_path)
    paths = resolve_phase7_paths(config, phase7_dir)
    output_path = resolve_output_path(paths.root, output)
    validate_output_outside_repo(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output HTML already exists: {output_path}. Pass --force to overwrite.")

    model_path = paths.models / "best_lda.joblib"
    vectorizer_path = paths.models / "vectorizer.joblib"
    summary_path = paths.metrics / "lda_summary.json"
    require_path(model_path, "Phase 7 best LDA model")
    require_path(vectorizer_path, "Phase 7 vectorizer")
    require_path(summary_path, "Phase 7 summary JSON")
    json.loads(summary_path.read_text(encoding="utf-8"))

    best_lda = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    corpus, _, _ = load_corpus(config)
    texts = corpus["text"].map(normalize_text)
    dtm = vectorizer.transform(texts)
    panel = prepare_panel(best_lda, dtm, vectorizer)
    pyLDAvis.save_html(panel, str(output_path))
    size = output_path.stat().st_size
    print(f"Wrote pyLDAvis HTML: {output_path}")
    print(f"HTML file size: {size} bytes")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase7/lda.yaml")
    parser.add_argument("--phase7_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    export_pyldavis(args.config, args.phase7_dir, args.output, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
