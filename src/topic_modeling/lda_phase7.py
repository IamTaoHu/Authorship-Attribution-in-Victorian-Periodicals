"""Phase 7 LDA topic modelling pipeline."""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import dataset_dir, get_path_config, phase_artifact_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase7" / "lda.yaml"
REQUIRED_INPUT_COLUMNS = ("text",)
META_COLUMNS = ("sample_id", "split", "author", "label", "text")


@dataclass(frozen=True)
class Phase7Paths:
    root: Path
    tables: Path
    metrics: Path
    models: Path
    plots: Path


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Phase 7 config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Phase 7 config must be a YAML mapping: {path}")
    config["_config_path"] = str(path)
    return config


def resolve_phase7_paths(config: dict[str, Any], phase7_dir: str | Path | None = None) -> Phase7Paths:
    output_config = config.get("output", {}) if isinstance(config.get("output", {}), dict) else {}
    configured = phase7_dir or output_config.get("phase7_dir")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = phase_artifact_dir("phase7", str(output_config.get("subdir", "lda")))

    tables = root / "tables"
    metrics = root / "metrics"
    models = root / "models"
    plots = root / "plots"
    for path in (tables, metrics, models, plots):
        path.mkdir(parents=True, exist_ok=True)
    return Phase7Paths(root=root, tables=tables, metrics=metrics, models=models, plots=plots)


def prepare_output_dir(paths: Phase7Paths, force: bool) -> None:
    required_children = ("tables", "metrics", "models", "plots")
    has_outputs = paths.root.exists() and any((paths.root / child).exists() and any((paths.root / child).iterdir()) for child in required_children)
    if has_outputs and not force:
        raise FileExistsError(f"Phase 7 output directory already contains files: {paths.root}. Pass --force to overwrite.")
    if has_outputs and force:
        for child in required_children:
            child_path = paths.root / child
            if child_path.exists():
                shutil.rmtree(child_path)
            child_path.mkdir(parents=True, exist_ok=True)


def resolve_input_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    input_config = config.get("inputs", {}) if isinstance(config.get("inputs", {}), dict) else {}
    parts = input_config.get("processed_periad_parts", ["processed", "periad"])
    if not isinstance(parts, list):
        raise ValueError("inputs.processed_periad_parts must be a list.")
    processed_dir = dataset_dir(*parts)
    train_path = processed_dir / str(input_config.get("train_file", "train.csv"))
    test_path = processed_dir / str(input_config.get("test_file", "test.csv"))
    missing = [path for path in (train_path, test_path) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing Phase 1 processed PERIAD dataset files. Run Phase 1 first or check configs/paths.local.yaml.\n"
            f"Expected files:\n{formatted}"
        )
    return train_path, test_path


def _ensure_sample_id(frame: pd.DataFrame, split: str) -> pd.Series:
    if "sample_id" in frame.columns:
        sample_ids = frame["sample_id"].fillna("").astype(str).str.strip()
        missing = sample_ids == ""
        if not missing.any():
            return sample_ids
        sample_ids.loc[missing] = [f"{split}_{index:06d}" for index in frame.index[missing]]
        return sample_ids
    return pd.Series([f"{split}_{index:06d}" for index in range(len(frame))], index=frame.index, dtype="string")


def load_split(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    output = pd.DataFrame(
        {
            "sample_id": _ensure_sample_id(frame, split),
            "split": frame["split"].fillna(split).astype(str) if "split" in frame.columns else split,
            "author": frame["author"].fillna("").astype(str) if "author" in frame.columns else "",
            "label": frame["label"] if "label" in frame.columns else "",
            "text": frame["text"].fillna("").astype(str),
        }
    )
    output["split"] = output["split"].replace("", split)
    return output


def load_corpus(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    train_path, test_path = resolve_input_paths(config)
    train = load_split(train_path, "train")
    test = load_split(test_path, "test")
    corpus = pd.concat([train, test], ignore_index=True)
    if corpus["sample_id"].duplicated().any():
        duplicates = corpus.loc[corpus["sample_id"].duplicated(), "sample_id"].head(10).tolist()
        raise ValueError(f"Duplicate sample_id values found in combined corpus, examples: {duplicates}")
    row_counts = {"train": int(len(train)), "test": int(len(test)), "total": int(len(corpus))}
    return corpus, row_counts, {"train": str(train_path), "test": str(test_path)}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def build_vectorizer(config: dict[str, Any]) -> CountVectorizer:
    vectorizer_config = config.get("vectorizer", {}) if isinstance(config.get("vectorizer", {}), dict) else {}
    ngram_range = vectorizer_config.get("ngram_range", [1, 1])
    if not isinstance(ngram_range, list) or len(ngram_range) != 2:
        raise ValueError("vectorizer.ngram_range must be a two-item list.")
    return CountVectorizer(
        lowercase=bool(vectorizer_config.get("lowercase", True)),
        stop_words=vectorizer_config.get("stop_words", "english"),
        min_df=vectorizer_config.get("min_df", 5),
        max_df=vectorizer_config.get("max_df", 0.8),
        max_features=vectorizer_config.get("max_features", 20000),
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        token_pattern=str(vectorizer_config.get("token_pattern", r"(?u)\b[a-zA-Z][a-zA-Z][a-zA-Z]+\b")),
    )


def build_lda(config: dict[str, Any], n_topics: int) -> LatentDirichletAllocation:
    lda_config = config.get("lda", {}) if isinstance(config.get("lda", {}), dict) else {}
    return LatentDirichletAllocation(
        n_components=int(n_topics),
        random_state=int(config.get("random_state", 42)),
        max_iter=int(lda_config.get("max_iter", 20)),
        learning_method=str(lda_config.get("learning_method", "batch")),
        evaluate_every=int(lda_config.get("evaluate_every", -1)),
        n_jobs=lda_config.get("n_jobs", -1),
    )


def top_word_indices(model: LatentDirichletAllocation, top_n: int) -> np.ndarray:
    return np.argsort(model.components_, axis=1)[:, ::-1][:, :top_n]


def topic_diversity(model: LatentDirichletAllocation, top_n: int) -> float:
    indices = top_word_indices(model, top_n)
    flattened = indices.ravel().tolist()
    return float(len(set(flattened)) / len(flattened)) if flattened else 0.0


def approximate_umass_coherence(model: LatentDirichletAllocation, matrix: sparse.spmatrix, top_n: int) -> float | None:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return None
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    doc_freq = np.asarray(binary.sum(axis=0)).ravel()
    indices = top_word_indices(model, top_n)
    scores: list[float] = []
    for topic_indices in indices:
        topic_scores = []
        for i in range(1, len(topic_indices)):
            wi = int(topic_indices[i])
            wi_docs = binary[:, wi]
            for j in range(0, i):
                wj = int(topic_indices[j])
                co_docs = wi_docs.multiply(binary[:, wj]).sum()
                denominator = doc_freq[wj]
                if denominator > 0:
                    topic_scores.append(math.log((float(co_docs) + 1.0) / float(denominator)))
        if topic_scores:
            scores.append(float(np.mean(topic_scores)))
    return float(np.mean(scores)) if scores else None


def topic_word_frame(model: LatentDirichletAllocation, feature_names: np.ndarray, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic_index, component in enumerate(model.components_):
        order = np.argsort(component)[::-1][:top_n]
        total = float(component.sum())
        for rank, word_index in enumerate(order, start=1):
            weight = float(component[word_index])
            rows.append(
                {
                    "topic_id": topic_index,
                    "topic": f"topic_{topic_index:02d}",
                    "rank": rank,
                    "term": str(feature_names[word_index]),
                    "weight": weight,
                    "normalized_weight": weight / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def fit_topic_grid(
    config: dict[str, Any],
    texts: pd.Series,
) -> tuple[pd.DataFrame, LatentDirichletAllocation, CountVectorizer, sparse.spmatrix, pd.DataFrame]:
    normalized_texts = texts.map(normalize_text)
    vectorizer = build_vectorizer(config)
    matrix = vectorizer.fit_transform(normalized_texts)
    if matrix.shape[1] == 0:
        raise ValueError("Vectorizer produced an empty vocabulary. Relax min_df/max_df or token_pattern.")

    lda_config = config.get("lda", {}) if isinstance(config.get("lda", {}), dict) else {}
    topic_counts = [int(value) for value in lda_config.get("topic_counts", [10, 20, 30, 40])]
    if not topic_counts:
        raise ValueError("lda.topic_counts must contain at least one topic count.")
    top_n = int(config.get("topics", {}).get("top_n_words", 20))
    coherence_top_n = int(config.get("topics", {}).get("coherence_top_n_words", 10))

    model_rows: list[dict[str, Any]] = []
    fitted: list[LatentDirichletAllocation] = []
    for n_topics in topic_counts:
        model = build_lda(config, n_topics)
        started = time.perf_counter()
        model.fit(matrix)
        runtime = time.perf_counter() - started
        coherence = approximate_umass_coherence(model, matrix, coherence_top_n)
        model_rows.append(
            {
                "k": n_topics,
                "perplexity": float(model.perplexity(matrix)),
                "approx_coherence": coherence,
                "topic_diversity": topic_diversity(model, top_n),
                "runtime_seconds": runtime,
            }
        )
        fitted.append(model)

    selection = pd.DataFrame(model_rows)
    if selection["approx_coherence"].notna().any():
        best_index = selection.sort_values(["approx_coherence", "topic_diversity", "perplexity"], ascending=[False, False, True]).index[0]
        selection_method = "highest_approx_coherence"
    else:
        best_index = selection.sort_values(["perplexity", "topic_diversity"], ascending=[True, False]).index[0]
        selection_method = "lowest_perplexity"
    selection["selected"] = False
    selection.loc[best_index, "selected"] = True
    selection["selection_method"] = selection_method
    best_model = fitted[list(selection.index).index(best_index)]
    topics = topic_word_frame(best_model, vectorizer.get_feature_names_out(), top_n)
    return selection, best_model, vectorizer, matrix, topics


def document_topic_distribution(model: LatentDirichletAllocation, matrix: sparse.spmatrix, corpus: pd.DataFrame) -> pd.DataFrame:
    distributions = model.transform(matrix)
    topic_columns = [f"topic_{index:02d}" for index in range(distributions.shape[1])]
    topic_frame = pd.DataFrame(distributions, columns=topic_columns)
    return pd.concat([corpus.loc[:, list(META_COLUMNS)].reset_index(drop=True), topic_frame], axis=1)


def author_topic_distribution(doc_topics: pd.DataFrame) -> pd.DataFrame:
    topic_columns = [column for column in doc_topics.columns if column.startswith("topic_")]
    grouped = doc_topics.groupby("author", dropna=False)[topic_columns].mean().reset_index()
    counts = doc_topics.groupby("author", dropna=False).size().rename("n_documents").reset_index()
    return counts.merge(grouped, on="author", how="left")


def topic_entropy_by_document(doc_topics: pd.DataFrame) -> pd.DataFrame:
    topic_columns = [column for column in doc_topics.columns if column.startswith("topic_")]
    values = doc_topics[topic_columns].to_numpy(dtype=float)
    entropy = -np.sum(np.where(values > 0, values * np.log(values), 0.0), axis=1)
    normalized = entropy / math.log(len(topic_columns)) if len(topic_columns) > 1 else entropy
    return doc_topics[["sample_id", "split", "author"]].assign(topic_entropy=entropy, normalized_topic_entropy=normalized)


def config_for_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def run_phase7(config_path: str | Path = DEFAULT_CONFIG, phase7_dir: str | Path | None = None, force: bool = False) -> dict[str, Path]:
    config = load_config(config_path)
    paths = resolve_phase7_paths(config, phase7_dir)
    prepare_output_dir(paths, force)
    paths = resolve_phase7_paths(config, paths.root)

    corpus, row_counts, input_paths = load_corpus(config)
    selection, best_model, vectorizer, matrix, topics = fit_topic_grid(config, corpus["text"])
    doc_topics = document_topic_distribution(best_model, matrix, corpus)
    author_topics = author_topic_distribution(doc_topics)
    entropy = topic_entropy_by_document(doc_topics)

    selection_path = paths.tables / "lda_model_selection.csv"
    topics_path = paths.tables / "best_topics.csv"
    doc_topics_path = paths.tables / "document_topic_distribution.csv"
    author_topics_path = paths.tables / "author_topic_distribution.csv"
    entropy_path = paths.tables / "topic_entropy_by_document.csv"
    model_path = paths.models / "best_lda.joblib"
    vectorizer_path = paths.models / "vectorizer.joblib"
    summary_path = paths.metrics / "lda_summary.json"

    selection.to_csv(selection_path, index=False)
    topics.to_csv(topics_path, index=False)
    doc_topics.to_csv(doc_topics_path, index=False)
    author_topics.to_csv(author_topics_path, index=False)
    entropy.to_csv(entropy_path, index=False)
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    selected_row = selection.loc[selection["selected"]].iloc[0]
    summary = {
        "phase": "phase7",
        "method": "lda",
        "selected_k": int(selected_row["k"]),
        "selection_method": str(selected_row["selection_method"]),
        "random_state": int(config.get("random_state", 42)),
        "input_row_counts": row_counts,
        "input_paths": input_paths,
        "output_paths": {
            "phase7_dir": str(paths.root),
            "model_selection": str(selection_path),
            "best_topics": str(topics_path),
            "document_topic_distribution": str(doc_topics_path),
            "author_topic_distribution": str(author_topics_path),
            "topic_entropy_by_document": str(entropy_path),
            "best_lda": str(model_path),
            "vectorizer": str(vectorizer_path),
        },
        "config_path": str(config["_config_path"]),
        "config_values": config_for_summary(config),
        "vocabulary_size": int(matrix.shape[1]),
        "n_documents": int(matrix.shape[0]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "phase7_dir": paths.root,
        "model_selection": selection_path,
        "best_topics": topics_path,
        "document_topic_distribution": doc_topics_path,
        "author_topic_distribution": author_topics_path,
        "topic_entropy_by_document": entropy_path,
        "summary": summary_path,
        "best_lda": model_path,
        "vectorizer": vectorizer_path,
    }


def default_phase7_dir() -> Path:
    return get_path_config()["artifacts_root"] / "phase7" / "lda"
