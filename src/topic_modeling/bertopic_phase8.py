"""Phase 8 BERTopic topic modelling pipeline."""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import dataset_dir, get_path_config, phase_artifact_dir  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase8" / "bertopic.yaml"
REQUIRED_INPUT_COLUMNS = ("sample_id", "split", "author", "label", "text")
META_COLUMNS = ("sample_id", "split", "author", "label", "text", "source_row_index", "source_row_id")


@dataclass(frozen=True)
class Phase8Paths:
    root: Path
    assignments: Path
    tables: Path
    embeddings: Path
    models: Path
    plots: Path
    interactive: Path
    reports: Path


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Phase 8 config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Phase 8 config must be a YAML mapping: {path}")
    config["_config_path"] = str(path)
    return config


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_outside_repo(path: Path) -> None:
    if _is_relative_to(path, REPO_ROOT):
        raise ValueError(f"Refusing to write Phase 8 artifacts inside repo root: {path}")


def resolve_phase8_paths(config: dict[str, Any], output_dir: str | Path | None = None) -> Phase8Paths:
    output_config = config.get("output", {}) if isinstance(config.get("output", {}), dict) else {}
    configured = output_dir or output_config.get("phase8_dir")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        ensure_outside_repo(root)
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = phase_artifact_dir("phase8", str(output_config.get("subdir", "bertopic")))
        ensure_outside_repo(root)

    children = {
        "assignments": root / "assignments",
        "tables": root / "tables",
        "embeddings": root / "embeddings",
        "models": root / "models",
        "plots": root / "plots",
        "interactive": root / "interactive",
        "reports": root / "reports",
    }
    for path in children.values():
        path.mkdir(parents=True, exist_ok=True)
    return Phase8Paths(root=root, **children)


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
        raise FileNotFoundError(f"Missing Phase 1 PERIAD files:\n{formatted}")
    return train_path, test_path


def load_split(path: Path, expected_split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    output = frame.loc[:, list(REQUIRED_INPUT_COLUMNS)].copy()
    output["sample_id"] = output["sample_id"].fillna("").astype(str)
    output["split"] = output["split"].fillna(expected_split).astype(str).replace("", expected_split)
    output["author"] = output["author"].fillna("").astype(str)
    output["text"] = output["text"].fillna("").astype(str)
    output["source_row_index"] = frame.index.astype(int)
    if "source_row_id" in frame.columns:
        output["source_row_id"] = frame["source_row_id"].fillna("").astype(str)
        missing_source = output["source_row_id"].str.strip() == ""
        output.loc[missing_source, "source_row_id"] = [f"{expected_split}:{idx}" for idx in output.loc[missing_source, "source_row_index"]]
    else:
        output["source_row_id"] = [f"{expected_split}:{idx}" for idx in output["source_row_index"]]
    return output


def load_corpus(config: dict[str, Any], limit: int | None = None) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    train_path, test_path = resolve_input_paths(config)
    train = load_split(train_path, "train")
    test = load_split(test_path, "test")
    corpus = pd.concat([train, test], ignore_index=True)
    if corpus["sample_id"].duplicated().any():
        duplicates = corpus.loc[corpus["sample_id"].duplicated(), "sample_id"].head(10).tolist()
        raise ValueError(f"Duplicate sample_id values found in combined corpus, examples: {duplicates}")
    full_counts = {"train": int(len(train)), "test": int(len(test)), "total": int(len(corpus))}
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        corpus = corpus.head(limit).reset_index(drop=True)
    return corpus, full_counts, {"train": str(train_path), "test": str(test_path)}


def resolve_device(raw_device: str) -> str | None:
    if raw_device == "auto":
        return None
    return raw_device


def encode_documents(config: dict[str, Any], texts: list[str], batch_size: int | None = None, device: str | None = None) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("sentence-transformers is required for Phase 8 BERTopic embeddings.") from exc

    embedding_config = config.get("embeddings", {}) if isinstance(config.get("embeddings", {}), dict) else {}
    model_name = str(embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    resolved_batch_size = int(batch_size or embedding_config.get("batch_size", 16))
    resolved_device = resolve_device(str(device or embedding_config.get("device", "auto")))
    model = SentenceTransformer(model_name, device=resolved_device)
    embeddings = model.encode(
        texts,
        batch_size=resolved_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(embedding_config.get("normalize_embeddings", True)),
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_models(config: dict[str, Any], calculate_probabilities: bool):
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("bertopic, umap-learn, and hdbscan are required for Phase 8.") from exc

    random_state = int(config.get("random_state", 42))
    umap_config = config.get("umap", {}) if isinstance(config.get("umap", {}), dict) else {}
    hdbscan_config = config.get("hdbscan", {}) if isinstance(config.get("hdbscan", {}), dict) else {}
    bertopic_config = config.get("bertopic", {}) if isinstance(config.get("bertopic", {}), dict) else {}

    umap_model = UMAP(
        n_neighbors=int(umap_config.get("n_neighbors", 15)),
        n_components=int(umap_config.get("n_components", 5)),
        min_dist=float(umap_config.get("min_dist", 0.0)),
        metric=str(umap_config.get("metric", "cosine")),
        low_memory=bool(umap_config.get("low_memory", True)),
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=int(hdbscan_config.get("min_cluster_size", 25)),
        min_samples=int(hdbscan_config.get("min_samples", 5)),
        metric=str(hdbscan_config.get("metric", "euclidean")),
        prediction_data=bool(hdbscan_config.get("prediction_data", True)),
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=calculate_probabilities,
        top_n_words=int(bertopic_config.get("top_n_words", 20)),
        verbose=bool(bertopic_config.get("verbose", True)),
    )
    return topic_model


def one_hot_topic_features(assignments: pd.DataFrame) -> pd.DataFrame:
    topics = sorted(assignments["topic"].astype(int).unique().tolist())
    rows = assignments[["sample_id", "split", "author", "topic"]].copy()
    for topic in topics:
        rows[f"topic_{topic}"] = (assignments["topic"].astype(int) == topic).astype(int)
    return rows


def topic_distribution(assignments: pd.DataFrame, group_column: str) -> pd.DataFrame:
    features = one_hot_topic_features(assignments)
    topic_columns = [column for column in features.columns if column.startswith("topic_")]
    counts = assignments.groupby(group_column, dropna=False).size().rename("n_documents").reset_index()
    means = features.groupby(group_column, dropna=False)[topic_columns].mean().reset_index()
    return counts.merge(means, on=group_column, how="left")


def topic_words_frame(topic_model: Any, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    info = topic_model.get_topic_info()
    for topic in info["Topic"].astype(int).tolist():
        words = topic_model.get_topic(topic) or []
        for rank, pair in enumerate(words[:top_n], start=1):
            term, weight = pair
            rows.append({"topic": topic, "rank": rank, "term": str(term), "weight": float(weight)})
    return pd.DataFrame(rows, columns=["topic", "rank", "term", "weight"])


def save_topic_model(topic_model: Any, model_dir: Path) -> str:
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    try:
        topic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
        return "safetensors"
    except Exception as exc:
        errors.append(f"safetensors: {exc}")
    try:
        topic_model.save(str(model_dir), serialization="pytorch", save_ctfidf=True, save_embedding_model=False)
        return "pytorch"
    except Exception as exc:
        errors.append(f"pytorch: {exc}")
    pickle_path = model_dir / "model.pkl"
    try:
        topic_model.save(str(pickle_path), serialization="pickle")
        return "pickle"
    except TypeError:
        topic_model.save(str(pickle_path))
        return "default_pickle"
    except Exception as exc:
        errors.append(f"pickle: {exc}")
        raise RuntimeError(f"Could not save BERTopic model under {model_dir}: {'; '.join(errors)}") from exc


def save_probabilities(probabilities: Any, assignments: pd.DataFrame, output_path: Path) -> tuple[bool, tuple[int, ...] | None]:
    if probabilities is None:
        return False, None
    array = np.asarray(probabilities)
    if array.ndim == 0 or array.shape[0] != len(assignments):
        raise ValueError(
            "BERTopic probabilities are not row-aligned with topic_assignments.csv: "
            f"probability shape={array.shape}, assignments rows={len(assignments)}"
        )
    np.save(output_path, array)
    return True, tuple(int(value) for value in array.shape)


def save_topic_embeddings(topic_model: Any, output_path: Path) -> bool:
    embeddings = getattr(topic_model, "topic_embeddings_", None)
    if embeddings is None:
        return False
    np.save(output_path, np.asarray(embeddings, dtype=np.float32))
    return True


def reduced_coordinates(topic_model: Any, embeddings: np.ndarray) -> pd.DataFrame:
    reducer = getattr(topic_model, "umap_model", None)
    if reducer is not None and hasattr(reducer, "embedding_"):
        coords = np.asarray(reducer.embedding_)
        if coords.ndim == 2 and coords.shape[0] == embeddings.shape[0] and coords.shape[1] >= 2:
            return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "coordinate_source": "bertopic_umap"})
    try:
        from umap import UMAP

        coords = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine", random_state=42).fit_transform(embeddings)
        return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "coordinate_source": "fallback_umap_2d"})
    except Exception:
        return pd.DataFrame(columns=["x", "y", "coordinate_source"])


def write_report(summary: dict[str, Any], topic_info: pd.DataFrame, output_path: Path) -> None:
    non_outlier = topic_info.loc[topic_info["Topic"].astype(int) != -1] if "Topic" in topic_info.columns else topic_info
    lines = [
        "# Phase 8 BERTopic Report",
        "",
        f"- Documents: {summary['n_documents']}",
        f"- Topics excluding outliers: {len(non_outlier)}",
        f"- Outlier documents: {summary.get('n_outlier_documents', 0)}",
        f"- Embedding model: {summary['embedding_model']}",
        f"- Probabilities requested: {summary['calculate_probabilities']}",
        f"- Probabilities saved: {summary['probabilities_saved']}",
        "",
        "## Largest Topics",
        "",
    ]
    if {"Topic", "Count", "Name"}.issubset(topic_info.columns):
        for _, row in topic_info.head(15).iterrows():
            lines.append(f"- Topic {row['Topic']}: {row['Name']} ({row['Count']} documents)")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_for_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def run_phase8(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    calculate_probabilities: bool | None = None,
    limit: int | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    paths = resolve_phase8_paths(config, output_dir)
    corpus, full_counts, input_paths = load_corpus(config, limit=limit)

    bertopic_config = config.get("bertopic", {}) if isinstance(config.get("bertopic", {}), dict) else {}
    resolved_probabilities = bool(bertopic_config.get("calculate_probabilities", False)) if calculate_probabilities is None else bool(calculate_probabilities)
    embedding_config = config.get("embeddings", {}) if isinstance(config.get("embeddings", {}), dict) else {}
    embedding_model_name = str(embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))

    started = time.perf_counter()
    embeddings = encode_documents(config, corpus["text"].astype(str).tolist(), batch_size=batch_size, device=device)
    embeddings_path = paths.embeddings / "document_embeddings.npy"
    np.save(embeddings_path, embeddings)

    topic_model = build_models(config, resolved_probabilities)
    topics, probabilities = topic_model.fit_transform(corpus["text"].astype(str).tolist(), embeddings=embeddings)

    assignments = corpus.loc[:, list(META_COLUMNS)].copy()
    assignments["topic"] = np.asarray(topics, dtype=int)
    assignments_path = paths.assignments / "topic_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    probabilities_path = paths.assignments / "topic_probabilities.npy"
    probabilities_saved, probability_shape = save_probabilities(probabilities, assignments, probabilities_path)

    topic_info = topic_model.get_topic_info()
    topic_info_path = paths.tables / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)

    topic_words_path = paths.tables / "topic_words.csv"
    topic_words = topic_words_frame(topic_model, int(bertopic_config.get("top_n_words", 20)))
    topic_words.to_csv(topic_words_path, index=False)

    author_distribution_path = paths.tables / "per_author_topic_distribution.csv"
    split_distribution_path = paths.tables / "per_split_topic_distribution.csv"
    topic_distribution(assignments, "author").to_csv(author_distribution_path, index=False)
    topic_distribution(assignments, "split").to_csv(split_distribution_path, index=False)

    document_features_path = paths.tables / "document_topic_features.csv"
    one_hot_topic_features(assignments).to_csv(document_features_path, index=False)

    coordinates = reduced_coordinates(topic_model, embeddings)
    coordinates_path = paths.tables / "document_coordinates.csv"
    if not coordinates.empty:
        pd.concat([assignments[["sample_id", "split", "author", "topic"]].reset_index(drop=True), coordinates.reset_index(drop=True)], axis=1).to_csv(
            coordinates_path,
            index=False,
        )

    topic_embeddings_path = paths.embeddings / "topic_embeddings.npy"
    topic_embeddings_saved = save_topic_embeddings(topic_model, topic_embeddings_path)

    model_dir = paths.models / "bertopic_model"
    model_serialization = save_topic_model(topic_model, model_dir)

    output_paths = {
        "phase8_dir": str(paths.root),
        "document_embeddings": str(embeddings_path),
        "topic_assignments": str(assignments_path),
        "topic_probabilities": str(probabilities_path) if probabilities_saved else None,
        "topic_info": str(topic_info_path),
        "topic_words": str(topic_words_path),
        "per_author_topic_distribution": str(author_distribution_path),
        "per_split_topic_distribution": str(split_distribution_path),
        "document_topic_features": str(document_features_path),
        "document_coordinates": str(coordinates_path) if coordinates_path.exists() else None,
        "topic_embeddings": str(topic_embeddings_path) if topic_embeddings_saved else None,
        "bertopic_model": str(model_dir),
    }
    summary = {
        "phase": "phase8",
        "method": "bertopic",
        "embedding_model": embedding_model_name,
        "calculate_probabilities": resolved_probabilities,
        "probabilities_saved": probabilities_saved,
        "probability_shape": probability_shape,
        "probability_alignment": "row order matches assignments/topic_assignments.csv",
        "random_state": int(config.get("random_state", 42)),
        "input_row_counts": full_counts,
        "n_documents": int(len(assignments)),
        "limit": int(limit) if limit is not None else None,
        "n_topics_including_outlier": int(assignments["topic"].nunique()),
        "n_outlier_documents": int((assignments["topic"] == -1).sum()),
        "input_paths": input_paths,
        "output_paths": output_paths,
        "config_path": str(config["_config_path"]),
        "config_values": config_for_summary(config),
        "model_serialization": model_serialization,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    summary_path = paths.reports / "bertopic_summary.json"
    summary["output_paths"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = paths.reports / "bertopic_report.md"
    write_report(summary, topic_info, report_path)

    return {
        "phase8_dir": paths.root,
        "summary": summary_path,
        "report": report_path,
        "topic_assignments": assignments_path,
        "topic_info": topic_info_path,
        "document_topic_features": document_features_path,
        "document_embeddings": embeddings_path,
    }


def default_phase8_dir() -> Path:
    return get_path_config()["artifacts_root"] / "phase8" / "bertopic"
