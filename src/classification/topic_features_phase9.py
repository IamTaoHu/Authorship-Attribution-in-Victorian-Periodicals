"""Phase 9 topic-aware authorship classification."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_path_config  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase9" / "topic_aware_classification.yaml"
CANONICAL_AUTHORS = (
    "Leslie Stephen",
    "John Morley",
    "Eliza Lynn Linton",
    "George Henry Lewes",
    "Anne Mozley",
    "James Fitzjames Stephen",
)
OUTPUT_DIRS = ("configs", "tables", "metrics", "predictions", "plots", "reports", "models", "logs")
METADATA_COLUMNS = {
    "sample_id",
    "source_row_id",
    "source_row_index",
    "split",
    "author",
    "label",
    "true_author",
    "predicted_author",
    "text",
    "paragraph",
    "content",
    "correct",
}


@dataclass(frozen=True)
class Phase9Paths:
    root: Path
    configs: Path
    tables: Path
    metrics: Path
    predictions: Path
    plots: Path
    reports: Path
    models: Path
    logs: Path


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Phase 9 config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Phase 9 config must be a YAML mapping: {path}")
    config["_config_path"] = str(path)
    return config


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_output_allowed(path: Path, allow_repo_outputs: bool = False) -> None:
    if is_relative_to(path, REPO_ROOT) and not allow_repo_outputs:
        raise ValueError(
            f"Refusing to write Phase 9 outputs inside Git repo root: {path}. "
            "Pass --allow_repo_outputs only for intentional diagnostics."
        )


def default_phase9_dir() -> Path:
    return get_path_config()["artifacts_root"] / "phase9" / "topic_features"


def resolve_phase9_paths(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    allow_repo_outputs: bool = False,
) -> Phase9Paths:
    configured = output_dir or (config.get("paths", {}) or {}).get("output_dir")
    root = Path(configured).expanduser() if configured else default_phase9_dir()
    if not root.is_absolute():
        root = REPO_ROOT / root
    ensure_output_allowed(root, allow_repo_outputs=allow_repo_outputs)
    root.mkdir(parents=True, exist_ok=True)
    children = {name: root / name for name in OUTPUT_DIRS}
    for path in children.values():
        path.mkdir(parents=True, exist_ok=True)
    return Phase9Paths(root=root, **children)


def resolve_path(raw: str | Path | None, fallback: Path | None = None) -> Path:
    if raw is None:
        if fallback is None:
            raise ValueError("Missing required path and no fallback is available.")
        return fallback
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_text_column(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> str:
    return _resolve_column(df, candidates, "text")


def resolve_label_column(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> str:
    return _resolve_column(df, candidates, "label")


def resolve_id_column(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> str:
    return _resolve_column(df, candidates, "identifier")


def _resolve_column(df: pd.DataFrame, candidates: list[str] | tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not resolve {label} column. Candidates={list(candidates)} available={list(df.columns)}")


def load_phase1_dataset(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = train.copy()
    test = test.copy()
    train["split"] = train.get("split", "train")
    test["split"] = test.get("split", "test")
    train["source_row_index"] = np.arange(len(train), dtype=int)
    test["source_row_index"] = np.arange(len(test), dtype=int)
    return train, test


def load_phase8_topic_features(phase8_dir: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    feature_file = str((config.get("topic_features", {}) or {}).get("feature_file", "tables/document_topic_features.csv"))
    root = Path(phase8_dir).expanduser()
    path = root / feature_file
    if not path.exists():
        raise FileNotFoundError(f"Phase 8 BERTopic-derived topic feature file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Phase 8 topic feature table is empty: {path}")
    frame = _augment_probability_features(root, frame)
    return frame


def _augment_probability_features(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if any(column.startswith("prob_topic_") for column in frame.columns):
        return frame
    probability_candidates = (
        root / "topic_probabilities.npy",
        root / "assignments" / "topic_probabilities.npy",
    )
    probability_path = next((path for path in probability_candidates if path.exists()), None)
    if probability_path is None:
        return frame
    probabilities = np.load(probability_path)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(frame):
        raise ValueError(
            f"topic_probabilities.npy is not row-aligned with document_topic_features.csv: "
            f"shape={probabilities.shape}, feature_rows={len(frame)}"
        )
    assignment_candidates = (
        root / "topic_assignments.csv",
        root / "assignments" / "topic_assignments.csv",
    )
    assignment_path = next((path for path in assignment_candidates if path.exists()), None)
    if assignment_path is not None:
        assignments = pd.read_csv(assignment_path)
        if len(assignments) != len(frame):
            raise ValueError(f"{assignment_path} rows do not align with {probability_path}")
        for key in (("sample_id", "split"), ("source_row_id", "split"), ("source_row_index", "split")):
            if _key_available(frame, key) and _key_available(assignments, key):
                if not frame.loc[:, list(key)].astype(str).reset_index(drop=True).equals(assignments.loc[:, list(key)].astype(str).reset_index(drop=True)):
                    raise ValueError(f"{probability_path} row order does not align with document features on {' + '.join(key)}")
                break
    topic_info_path = root / "tables" / "topic_info.csv"
    names = [f"prob_topic_{idx}" for idx in range(probabilities.shape[1])]
    if topic_info_path.exists():
        topic_info = pd.read_csv(topic_info_path)
        if {"Topic", "Count"}.issubset(topic_info.columns):
            topics = sorted(topic_info.loc[topic_info["Topic"].astype(int) != -1, "Topic"].astype(int).tolist())
            if len(topics) == probabilities.shape[1]:
                names = [f"prob_topic_{topic}" for topic in topics]
    output = frame.copy()
    for index, name in enumerate(names):
        output[name] = probabilities[:, index].astype(float)
    clipped = np.clip(probabilities.astype(float), 1e-12, 1.0)
    output["topic_entropy"] = -(clipped * np.log(clipped)).sum(axis=1)
    output["top_topic"] = probabilities.argmax(axis=1).astype(int)
    return output


def _key_available(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return all(column in frame.columns for column in columns)


def _validate_key(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    if frame.loc[:, list(columns)].isna().any().any():
        raise ValueError(f"{name} merge key contains missing values: {columns}")
    duplicates = frame.duplicated(list(columns), keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, list(columns)].head(5).to_dict("records")
        raise ValueError(f"{name} contains duplicated merge keys for {columns}; examples={examples}")


def _choose_merge_key(dataset: pd.DataFrame, topic_df: pd.DataFrame) -> tuple[str, ...]:
    candidates = (("sample_id", "split"), ("source_row_id", "split"), ("source_row_index", "split"))
    for columns in candidates:
        if _key_available(dataset, columns) and _key_available(topic_df, columns):
            return columns
    raise ValueError(
        "Cannot merge Phase 1 rows with Phase 8 topic features. Tried keys in order: "
        "sample_id+split, source_row_id+split, source_row_index+split. "
        f"Phase 1 columns={list(dataset.columns)} Phase 8 columns={list(topic_df.columns)}"
    )


def merge_dataset_with_topic_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    topic_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    train = train_df.copy()
    test = test_df.copy()
    dataset = pd.concat([train, test], ignore_index=True)
    key = _choose_merge_key(dataset, topic_df)
    _validate_key(dataset, key, "Phase 1 dataset")
    _validate_key(topic_df, key, "Phase 8 topic feature table")
    topic_columns = [column for column in topic_df.columns if column not in {"author", "label", "text", "paragraph", "content"}]
    merged = dataset.merge(topic_df.loc[:, topic_columns], on=list(key), how="left", validate="one_to_one", indicator=True)
    missing = merged["_merge"].ne("both")
    if missing.any():
        examples = merged.loc[missing, list(key)].head(10).to_dict("records")
        raise ValueError(f"Missing BERTopic-derived topic feature vector for {int(missing.sum())} rows; examples={examples}")
    merged = merged.drop(columns=["_merge"])
    train_merged = merged.loc[merged["split"].astype(str) == "train"].copy()
    test_merged = merged.loc[merged["split"].astype(str) == "test"].copy()
    if len(train_merged) != len(train_df):
        raise ValueError(f"Merged train row count {len(train_merged)} != Phase 1 train row count {len(train_df)}")
    if len(test_merged) != len(test_df):
        raise ValueError(f"Merged test row count {len(test_merged)} != Phase 1 test row count {len(test_df)}")
    return train_merged.reset_index(drop=True), test_merged.reset_index(drop=True), key


def validate_authors(train_df: pd.DataFrame, test_df: pd.DataFrame, label_column: str) -> None:
    observed = set(pd.concat([train_df[label_column], test_df[label_column]], ignore_index=True).dropna().astype(str))
    missing = [author for author in CANONICAL_AUTHORS if author not in observed]
    extra = sorted(observed.difference(CANONICAL_AUTHORS))
    if missing or extra:
        raise ValueError(f"Canonical author validation failed. Missing={missing} extra={extra}")


def select_topic_feature_columns(df: pd.DataFrame, config: dict[str, Any], include_outlier_topic: bool = True) -> list[str]:
    topic_config = config.get("topic_features", {}) if isinstance(config.get("topic_features", {}), dict) else {}
    columns: list[str] = []
    if bool(topic_config.get("include_topic_onehot", True)):
        columns.extend(column for column in df.columns if column.startswith("topic_"))
    if bool(topic_config.get("include_topic_probabilities", True)):
        columns.extend(column for column in df.columns if column.startswith("prob_topic_"))
    if bool(topic_config.get("include_topic_entropy", True)):
        columns.extend(column for column in ("topic_entropy",) if column in df.columns)
    if bool(topic_config.get("include_top_topic_numeric", True)):
        columns.extend(column for column in ("top_topic", "topic") if column in df.columns)
    columns = [column for column in dict.fromkeys(columns) if column not in METADATA_COLUMNS]
    if not include_outlier_topic:
        outlier_column = str(topic_config.get("outlier_column", "topic_-1"))
        columns = [column for column in columns if column != outlier_column]
    numeric_columns = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            numeric_columns.append(column)
    if not numeric_columns:
        raise ValueError("No numeric BERTopic-derived topic feature columns were selected.")
    return numeric_columns


def build_topic_feature_matrix(
    df: pd.DataFrame,
    config: dict[str, Any],
    include_outlier_topic: bool = True,
    *,
    feature_columns: list[str] | None = None,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, list[str], StandardScaler | None]:
    topic_config = config.get("topic_features", {}) if isinstance(config.get("topic_features", {}), dict) else {}
    selected = feature_columns or select_topic_feature_columns(df, config, include_outlier_topic=include_outlier_topic)
    matrix = df.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    strategy = str(topic_config.get("missing_value_strategy", "zero"))
    if strategy != "zero":
        raise ValueError(f"Unsupported topic feature missing_value_strategy: {strategy}")
    matrix = matrix.fillna(0.0).to_numpy(dtype=np.float32)
    if bool(topic_config.get("standardize_features", True)):
        if fit_scaler:
            scaler = StandardScaler()
            matrix = scaler.fit_transform(matrix).astype(np.float32)
        elif scaler is not None:
            matrix = scaler.transform(matrix).astype(np.float32)
    return matrix, selected, scaler


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def labels_for(df: pd.DataFrame, label_column: str) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    label_to_id = {author: index for index, author in enumerate(CANONICAL_AUTHORS)}
    id_to_label = {index: author for author, index in label_to_id.items()}
    labels = df[label_column].astype(str).map(label_to_id)
    if labels.isna().any():
        bad = sorted(df.loc[labels.isna(), label_column].astype(str).unique().tolist())
        raise ValueError(f"Unexpected labels: {bad}")
    return labels.to_numpy(dtype=np.int64), label_to_id, id_to_label


def metric_dict(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CANONICAL_AUTHORS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CANONICAL_AUTHORS), average="weighted", zero_division=0)),
    }


def prediction_base(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    columns = [column for column in ("sample_id", "source_row_id", "source_row_index", "split") if column in df.columns]
    out = df.loc[:, columns].copy()
    out["true_author"] = df[label_column].astype(str).to_numpy()
    return out


def save_eval_outputs(
    variant: dict[str, Any],
    paths: Phase9Paths,
    test_df: pd.DataFrame,
    label_column: str,
    y_pred: list[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    name = str(variant["name"])
    y_true = test_df[label_column].astype(str).tolist()
    predictions = prediction_base(test_df, label_column)
    predictions["predicted_author"] = y_pred
    predictions["correct"] = predictions["true_author"].astype(str) == predictions["predicted_author"].astype(str)
    if probabilities is not None:
        for index, author in enumerate(CANONICAL_AUTHORS):
            predictions[f"prob_{index}_{author.replace(' ', '_')}"] = probabilities[:, index]
    predictions.to_csv(paths.predictions / f"{name}_predictions.csv", index=False)
    metrics = metric_dict(y_true, y_pred)
    metrics.update({"variant": name, "model_type": str(variant.get("type", "")), "n_test": int(len(test_df))})
    save_json(paths.metrics / f"{name}_metrics.json", metrics)
    report = classification_report(y_true, y_pred, labels=list(CANONICAL_AUTHORS), output_dict=True, zero_division=0)
    report_frame = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "author"})
    report_frame.to_csv(paths.tables / f"{name}_classification_report.csv", index=False)
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(CANONICAL_AUTHORS)),
        index=list(CANONICAL_AUTHORS),
        columns=list(CANONICAL_AUTHORS),
    )
    matrix.to_csv(paths.tables / f"{name}_confusion_matrix.csv")
    return metrics


def run_sklearn_variant(
    variant: dict[str, Any],
    paths: Phase9Paths,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_x: np.ndarray,
    test_x: np.ndarray,
    label_column: str,
) -> dict[str, Any]:
    train_y, _, id_to_label = labels_for(train_df, label_column)
    if str(variant.get("model")) == "linear_svc":
        model = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    else:
        model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    model.fit(train_x, train_y)
    pred_ids = model.predict(test_x)
    y_pred = [id_to_label[int(value)] for value in pred_ids]
    probabilities = model.predict_proba(test_x) if hasattr(model, "predict_proba") else None
    metrics = save_eval_outputs(variant, paths, test_df, label_column, y_pred, probabilities)
    metrics["best_epoch"] = None
    return metrics


class TopicFeatureDataset:
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray | None,
        tokenizer: Any,
        max_length: int,
        topic_features: np.ndarray | None = None,
    ) -> None:
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
        self.topic_features = topic_features

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        if self.topic_features is not None:
            item["topic_features"] = torch.tensor(self.topic_features[index], dtype=torch.float)
        return item


def build_topic_aware_model_class() -> type:
    import torch
    from torch import nn
    from transformers import AutoModel, PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class TopicAwareTransformerForSequenceClassification(PreTrainedModel):
        def __init__(self, hf_config: Any, model_name: str, topic_dim: int, topic_projection_dim: int, dropout: float = 0.1) -> None:
            super().__init__(hf_config)
            self.num_labels = hf_config.num_labels
            self.encoder = AutoModel.from_pretrained(model_name, config=hf_config)
            self.topic_projection = nn.Sequential(nn.Linear(topic_dim, topic_projection_dim), nn.ReLU(), nn.Dropout(dropout))
            self.classifier = nn.Linear(hf_config.hidden_size + topic_projection_dim, hf_config.num_labels)
            self.dropout = nn.Dropout(dropout)
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(
            self,
            input_ids: Any = None,
            attention_mask: Any = None,
            labels: Any = None,
            topic_features: Any = None,
            **kwargs: Any,
        ) -> Any:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            pooled = outputs.last_hidden_state[:, 0, :]
            if topic_features is None:
                topic_features = torch.zeros((pooled.shape[0], self.topic_projection[0].in_features), device=pooled.device, dtype=pooled.dtype)
            projected = self.topic_projection(topic_features.to(pooled.dtype))
            logits = self.classifier(torch.cat([self.dropout(pooled), projected], dim=-1))
            loss = self.loss_fn(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    return TopicAwareTransformerForSequenceClassification


def run_transformer_variant(
    variant: dict[str, Any],
    config: dict[str, Any],
    paths: Phase9Paths,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_x: np.ndarray | None,
    test_x: np.ndarray | None,
    text_column: str,
    label_column: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

    seed = int(config.get("seed", 42))
    set_seed(seed)
    model_config = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    train_config = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    model_name = str(model_config.get("model_name", "microsoft/deberta-v3-base"))
    max_length = int(model_config.get("max_length", 256))
    train_labels, label_to_id, id_to_label = labels_for(train_df, label_column)
    test_labels, _, _ = labels_for(test_df, label_column)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    use_topics = str(variant.get("type")) == "transformer_topic_concat"
    train_dataset = TopicFeatureDataset(train_df[text_column].fillna("").astype(str).tolist(), train_labels, tokenizer, max_length, train_x if use_topics else None)
    test_dataset = TopicFeatureDataset(test_df[text_column].fillna("").astype(str).tolist(), test_labels, tokenizer, max_length, test_x if use_topics else None)
    hf_config = AutoConfig.from_pretrained(model_name, num_labels=len(CANONICAL_AUTHORS), label2id=label_to_id, id2label=id_to_label)
    if use_topics:
        model_cls = build_topic_aware_model_class()
        model = model_cls(
            hf_config,
            model_name=model_name,
            topic_dim=int(train_x.shape[1]),
            topic_projection_dim=int(model_config.get("topic_projection_dim", 64)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=hf_config)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        y_true = [id_to_label[int(value)] for value in labels]
        y_pred = [id_to_label[int(value)] for value in pred]
        return metric_dict(y_true, y_pred)

    variant_model_dir = paths.models / str(variant["name"])
    args_kwargs = {
        "output_dir": str(variant_model_dir),
        "num_train_epochs": float(train_config.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(train_config.get("per_device_train_batch_size", 8)),
        "per_device_eval_batch_size": int(train_config.get("per_device_eval_batch_size", 16)),
        "gradient_accumulation_steps": int(train_config.get("gradient_accumulation_steps", 2)),
        "learning_rate": float(train_config.get("learning_rate", 2e-5)),
        "weight_decay": float(train_config.get("weight_decay", 0.01)),
        "warmup_ratio": float(train_config.get("warmup_ratio", 0.06)),
        "fp16": bool(train_config.get("fp16", True)) and torch.cuda.is_available(),
        "logging_steps": int(train_config.get("logging_steps", 50)),
        "save_strategy": str(train_config.get("save_strategy", "epoch")),
        "load_best_model_at_end": bool(train_config.get("load_best_model_at_end", True)),
        "metric_for_best_model": str(train_config.get("metric_for_best_model", "macro_f1")),
        "greater_is_better": bool(train_config.get("greater_is_better", True)),
        "save_total_limit": int(train_config.get("save_total_limit", 2)),
        "report_to": [],
        "push_to_hub": False,
    }
    eval_strategy = str(train_config.get("evaluation_strategy", "epoch"))
    try:
        training_args = TrainingArguments(evaluation_strategy=eval_strategy, **args_kwargs)
    except TypeError:
        training_args = TrainingArguments(eval_strategy=eval_strategy, **args_kwargs)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(variant_model_dir))
    predictions = trainer.predict(test_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    y_pred = [id_to_label[int(value)] for value in pred_ids]
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    metrics = save_eval_outputs(variant, paths, test_df, label_column, y_pred, probs)
    metrics["best_epoch"] = getattr(trainer.state, "epoch", None)
    save_json(paths.metrics / f"{variant['name']}_trainer_metrics.json", {key: _jsonable(value) for key, value in predictions.metrics.items()})
    return metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def aggregate_results(paths: Phase9Paths, rows: list[dict[str, Any]], topic_feature_dim_by_variant: dict[str, int]) -> None:
    summary_rows = []
    per_author_rows = []
    for row in rows:
        name = str(row["variant"])
        report_path = paths.tables / f"{name}_classification_report.csv"
        report = pd.read_csv(report_path)
        for _, item in report.iterrows():
            author = str(item["author"])
            if author in CANONICAL_AUTHORS:
                per_author_rows.append(
                    {
                        "author": author,
                        "variant": name,
                        "precision": float(item.get("precision", 0.0)),
                        "recall": float(item.get("recall", 0.0)),
                        "f1_score": float(item.get("f1-score", 0.0)),
                        "support": int(float(item.get("support", 0))),
                    }
                )
        summary_rows.append(
            {
                "variant": name,
                "model_type": row.get("model_type", ""),
                "accuracy": row.get("accuracy"),
                "macro_f1": row.get("macro_f1"),
                "weighted_f1": row.get("weighted_f1"),
                "best_epoch": row.get("best_epoch"),
                "topic_feature_dim": topic_feature_dim_by_variant.get(name),
                "include_outlier_topic": row.get("include_outlier_topic", True),
                "output_dir": str(paths.root),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths.tables / "phase9_results_summary.csv", index=False)
    per_author = pd.DataFrame(per_author_rows)
    per_author.to_csv(paths.tables / "per_author_f1_comparison.csv", index=False)
    text = per_author.loc[per_author["variant"] == "deberta_text_only_reproduced", ["author", "f1_score"]].rename(columns={"f1_score": "text_only_f1"})
    delta = per_author.merge(text, on="author", how="left")
    delta = delta.loc[delta["variant"] != "deberta_text_only_reproduced"].copy()
    delta["delta_f1_vs_text_only"] = delta["f1_score"] - delta["text_only_f1"]
    delta.to_csv(paths.tables / "per_author_f1_delta_vs_text_only.csv", index=False)


def write_report(paths: Phase9Paths, config: dict[str, Any], merge_key: tuple[str, ...], feature_columns: list[str]) -> None:
    summary_path = paths.tables / "phase9_results_summary.csv"
    per_author_path = paths.tables / "per_author_f1_comparison.csv"
    delta_path = paths.tables / "per_author_f1_delta_vs_text_only.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    per_author = pd.read_csv(per_author_path) if per_author_path.exists() else pd.DataFrame()
    delta = pd.read_csv(delta_path) if delta_path.exists() else pd.DataFrame()
    has_distribution = any(column.startswith("prob_topic_") for column in feature_columns)
    vector_name = "topic distribution vector" if has_distribution else "BERTopic-derived topic feature vector"
    lines = [
        "# Phase 9 Topic-Aware Classification",
        "",
        "## Inputs",
        "",
        f"- Dataset train: {(config.get('paths', {}) or {}).get('dataset_train')}",
        f"- Dataset test: {(config.get('paths', {}) or {}).get('dataset_test')}",
        f"- Phase 8 directory: {(config.get('paths', {}) or {}).get('phase8_dir')}",
        f"- Merge key: {' + '.join(merge_key)}",
        "",
        "## Topic Features",
        "",
        f"- Feature type: {vector_name}",
        f"- Topic feature dimension: {len(feature_columns)}",
        f"- Phase 9 uses Phase 8 BERTopic artifacts and does not retrain topic models.",
        "",
        "## Overall Metrics",
        "",
        markdown_table(summary) if not summary.empty else "No summary rows available.",
        "",
        "## Per-Author F1",
        "",
        markdown_table(per_author) if not per_author.empty else "No per-author rows available.",
        "",
        "## Delta vs Text-Only Baseline",
        "",
        markdown_table(delta) if not delta.empty else "Delta table requires the text-only baseline.",
        "",
        "## Interpretation",
        "",
        "- Topic-only rows indicate whether BERTopic-derived features contain authorship signal.",
        "- Topic-aware DeBERTa rows indicate whether concatenating topic features improves macro F1 over text-only DeBERTa.",
        "- Per-author deltas identify which authors improved or degraded.",
        "- Comparing `deberta_topic_concat` with `deberta_topic_concat_no_outlier` indicates whether `topic_-1` helped or hurt.",
    ]
    (paths.reports / "phase9_topic_aware_classification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    limited = frame.copy()
    lines = [
        "| " + " | ".join(str(column) for column in limited.columns) + " |",
        "| " + " | ".join("---" for _ in limited.columns) + " |",
    ]
    for _, row in limited.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in limited.columns) + " |")
    return "\n".join(lines)


def validate_prediction_row_counts(paths: Phase9Paths, rows: list[dict[str, Any]], expected_test_rows: int) -> None:
    for row in rows:
        path = paths.predictions / f"{row['variant']}_predictions.csv"
        count = len(pd.read_csv(path))
        if count != expected_test_rows:
            raise ValueError(f"{path} row count {count} != Phase 1 test.csv row count {expected_test_rows}")


def selected_variants(config: dict[str, Any], raw: str, skip_transformers: bool) -> list[dict[str, Any]]:
    variants = config.get("variants", [])
    if raw != "all":
        wanted = {item.strip() for item in raw.split(",") if item.strip()}
        variants = [variant for variant in variants if str(variant.get("name")) in wanted]
    if skip_transformers:
        variants = [variant for variant in variants if not str(variant.get("type", "")).startswith("transformer")]
    if not variants:
        raise ValueError("No Phase 9 variants selected.")
    return variants


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        from transformers import set_seed

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        set_seed(seed)
    except Exception:
        pass


def run_phase9(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    dataset_train: str | Path | None = None,
    dataset_test: str | Path | None = None,
    phase8_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    variants_raw: str = "all",
    skip_transformers: bool = False,
    plots_only: bool = False,
    report_only: bool = False,
    allow_repo_outputs: bool = False,
) -> dict[str, Path]:
    config = load_config(config_path)
    paths_config = config.setdefault("paths", {})
    if dataset_train is not None:
        paths_config["dataset_train"] = str(dataset_train)
    if dataset_test is not None:
        paths_config["dataset_test"] = str(dataset_test)
    if phase8_dir is not None:
        paths_config["phase8_dir"] = str(phase8_dir)
    if output_dir is not None:
        paths_config["output_dir"] = str(output_dir)
    paths = resolve_phase9_paths(config, output_dir=output_dir, allow_repo_outputs=allow_repo_outputs)
    if plots_only:
        from src.visualization.plot_phase9_topic_features import plot_phase9_topic_features

        plot_phase9_topic_features(paths.root)
        return {"phase9_dir": paths.root}
    if report_only:
        feature_path = paths.tables / "topic_feature_columns.json"
        feature_columns = json.loads(feature_path.read_text(encoding="utf-8")) if feature_path.exists() else []
        write_report(paths, config, tuple(), feature_columns)
        return {"phase9_dir": paths.root}

    set_all_seeds(int(config.get("seed", 42)))
    resolved_config_path = paths.configs / "resolved_topic_aware_classification.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    path_roots = get_path_config()
    train_path = resolve_path(paths_config.get("dataset_train"), path_roots["datasets_root"] / "processed" / "periad" / "train.csv")
    test_path = resolve_path(paths_config.get("dataset_test"), path_roots["datasets_root"] / "processed" / "periad" / "test.csv")
    phase8_path = resolve_path(paths_config.get("phase8_dir"), path_roots["artifacts_root"] / "phase8" / "bertopic")
    paths_config["dataset_train"] = str(train_path)
    paths_config["dataset_test"] = str(test_path)
    paths_config["phase8_dir"] = str(phase8_path)
    paths_config["output_dir"] = str(paths.root)
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_df, test_df = load_phase1_dataset(train_path, test_path)
    topic_df = load_phase8_topic_features(phase8_path, config)
    train_df, test_df, merge_key = merge_dataset_with_topic_features(train_df, test_df, topic_df)
    data_config = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    text_column = resolve_text_column(train_df, data_config.get("text_column_candidates", ["text", "paragraph", "content"]))
    label_column = resolve_label_column(train_df, data_config.get("label_column_candidates", ["author", "label", "true_author"]))
    validate_authors(train_df, test_df, label_column)

    rows: list[dict[str, Any]] = []
    topic_dims: dict[str, int] = {}
    feature_columns_written = False
    for variant in selected_variants(config, variants_raw, skip_transformers):
        include_outlier = bool(variant.get("include_outlier_topic", (config.get("topic_features", {}) or {}).get("include_outlier_topic", True)))
        use_topics = str(variant.get("type")) in {"sklearn_topic_only", "transformer_topic_concat"}
        train_x = test_x = None
        feature_columns: list[str] = []
        if use_topics:
            train_x, feature_columns, scaler = build_topic_feature_matrix(train_df, config, include_outlier_topic=include_outlier, fit_scaler=True)
            test_x, _, _ = build_topic_feature_matrix(test_df, config, include_outlier_topic=include_outlier, feature_columns=feature_columns, scaler=scaler)
            topic_dims[str(variant["name"])] = int(train_x.shape[1])
            if not feature_columns_written:
                save_json(paths.tables / "topic_feature_columns.json", feature_columns)
                feature_columns_written = True
        else:
            topic_dims[str(variant["name"])] = 0

        if str(variant.get("type")) == "sklearn_topic_only":
            metrics = run_sklearn_variant(variant, paths, train_df, test_df, train_x, test_x, label_column)
        elif str(variant.get("type")).startswith("transformer"):
            metrics = run_transformer_variant(variant, config, paths, train_df, test_df, train_x, test_x, text_column, label_column)
        else:
            raise ValueError(f"Unsupported Phase 9 variant type: {variant}")
        metrics["include_outlier_topic"] = include_outlier
        rows.append(metrics)

    if not feature_columns_written:
        _, feature_columns, _ = build_topic_feature_matrix(train_df, config)
        save_json(paths.tables / "topic_feature_columns.json", feature_columns)
    aggregate_results(paths, rows, topic_dims)
    from src.visualization.plot_phase9_topic_features import plot_phase9_topic_features

    plot_phase9_topic_features(paths.root)
    feature_columns = json.loads((paths.tables / "topic_feature_columns.json").read_text(encoding="utf-8"))
    write_report(paths, config, merge_key, feature_columns)
    validate_prediction_row_counts(paths, rows, len(pd.read_csv(test_path)))
    return {
        "phase9_dir": paths.root,
        "summary": paths.tables / "phase9_results_summary.csv",
        "report": paths.reports / "phase9_topic_aware_classification.md",
    }
