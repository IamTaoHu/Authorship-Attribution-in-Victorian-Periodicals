"""Build Phase 6 ensembles from existing Phase 2-5 predictions."""

from __future__ import annotations

import fnmatch
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prompting.phase4_parser import CANONICAL_AUTHORS  # noqa: E402
from src.utils.paths import get_path_config, phase_artifact_dir  # noqa: E402


ENSEMBLE_STRATEGIES = (
    "encoder_only_majority_vote",
    "decoder_only_majority_vote",
    "cross_architecture_majority_vote",
    "best_encoder_best_decoder_vote",
    "weighted_vote_macro_f1",
    "weighted_vote_accuracy",
)
REQUIRED_OUTPUT_COLUMNS = ("sample_id", "true_author", "predicted_author")
INVALID_LABEL = "__INVALID__"
METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1")


@dataclass
class Candidate:
    phase: str
    run_name: str
    model_name: str
    architecture_group: str
    predictions_path: Path
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    invalid_count: int
    invalid_rate: float
    ensemble_candidate: bool


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key, metrics.get(f"eval_{key}"))
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def is_allowed(run_name: str, phase_config: dict[str, Any]) -> bool:
    exact = {str(item) for item in phase_config.get("exact_allowlist", [])}
    patterns = [str(item) for item in phase_config.get("glob_allowlist", [])]
    return run_name in exact or any(fnmatch.fnmatch(run_name, pattern) for pattern in patterns)


def contains_excluded_token(path: Path, excluded_tokens: list[str]) -> bool:
    lowered = [part.lower() for part in path.parts]
    return any(token.lower() in part for token in excluded_tokens for part in lowered)


def is_majority_baseline(run_name: str) -> bool:
    lowered = run_name.lower()
    return "majority" in lowered or "ensemble" in lowered or "vote" in lowered


def resolve_metric_path(run_dir: Path) -> Path | None:
    for name in ("metrics.json", "eval_metrics.json"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def discover_prediction_paths(phase: str, phase_config: dict[str, Any], excluded_tokens: list[str]) -> list[Path]:
    paths = get_path_config()
    runs_dir = paths["artifacts_root"] / phase / "runs"
    if not runs_dir.exists():
        return []
    discovered = []
    for path in sorted(runs_dir.rglob("predictions.csv")):
        run_name = path.parent.name
        if contains_excluded_token(path.parent.relative_to(runs_dir), excluded_tokens):
            continue
        if is_allowed(run_name, phase_config):
            discovered.append(path)
    return discovered


def normalize_predictions(path: Path, candidate: dict[str, str], invalid_label: str) -> tuple[pd.DataFrame, int, float]:
    frame = pd.read_csv(path)
    true_column = "true_author" if "true_author" in frame.columns else "author" if "author" in frame.columns else None
    pred_column = "predicted_author" if "predicted_author" in frame.columns else "pred_author" if "pred_author" in frame.columns else None
    missing = [column for column in ("sample_id",) if column not in frame.columns]
    if true_column is None:
        missing.append("true_author/author")
    if pred_column is None:
        missing.append("predicted_author/pred_author")
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")

    normalized = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "true_author": frame[true_column].astype(str),
            "predicted_author": frame[pred_column].fillna("").astype(str).str.strip(),
            "model_name": candidate["model_name"],
            "phase": candidate["phase"],
            "run_name": candidate["run_name"],
        }
    )
    valid = set(CANONICAL_AUTHORS)
    invalid_mask = ~normalized["predicted_author"].isin(valid)
    normalized.loc[invalid_mask, "predicted_author"] = invalid_label
    invalid_count = int(invalid_mask.sum())
    invalid_rate = invalid_count / len(normalized) if len(normalized) else 0.0
    return normalized, invalid_count, invalid_rate


def recompute_metrics(predictions: pd.DataFrame, invalid_label: str) -> dict[str, float]:
    y_true = predictions["true_author"].astype(str)
    y_pred = predictions["predicted_author"].astype(str)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(predictions) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CANONICAL_AUTHORS), average="macro", zero_division=0)) if len(predictions) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CANONICAL_AUTHORS), average="weighted", zero_division=0)) if len(predictions) else 0.0,
        "invalid_output_rate": float((y_pred == invalid_label).sum() / len(predictions)) if len(predictions) else 0.0,
    }


def discover_candidates(config: dict[str, Any]) -> tuple[list[Candidate], list[str]]:
    excluded_tokens = [str(item) for item in config.get("exclude_tokens", [])]
    invalid_label = str(config.get("invalid_label", INVALID_LABEL))
    include_majority = bool(config.get("include_majority_vote_candidates", False))
    warnings: list[str] = []
    candidates: list[Candidate] = []
    phase_configs = config.get("phases", {})
    if not isinstance(phase_configs, dict):
        raise ValueError("Config must contain a 'phases' mapping.")
    for phase, phase_config in phase_configs.items():
        if not isinstance(phase_config, dict) or not phase_config.get("enabled", True):
            continue
        architecture_group = str(phase_config.get("architecture_group", "encoder"))
        paths = discover_prediction_paths(str(phase), phase_config, excluded_tokens)
        if not paths:
            warnings.append(f"No allowed prediction files found for {phase}.")
            continue
        for path in paths:
            run_dir = path.parent
            run_name = run_dir.name
            metrics_path = resolve_metric_path(run_dir)
            metrics = read_json(metrics_path) if metrics_path else {}
            model_name = str(metrics.get("model_name") or run_name)
            normalized, invalid_count, invalid_rate = normalize_predictions(
                path,
                {"phase": str(phase), "run_name": run_name, "model_name": model_name},
                invalid_label,
            )
            computed = recompute_metrics(normalized, invalid_label)
            for key in METRIC_KEYS:
                metrics.setdefault(key, computed[key])
            metrics.setdefault("invalid_output_rate", computed["invalid_output_rate"])
            ensemble_candidate = True
            if str(phase) == "phase2" and is_majority_baseline(run_name) and not include_majority:
                ensemble_candidate = False
                warnings.append(f"Included Phase 2 majority-vote baseline as a single-model baseline only: {run_name}.")
            candidates.append(
                Candidate(
                    phase=str(phase),
                    run_name=run_name,
                    model_name=model_name,
                    architecture_group=architecture_group,
                    predictions_path=path,
                    metrics=metrics,
                    predictions=normalized,
                    invalid_count=invalid_count,
                    invalid_rate=invalid_rate,
                    ensemble_candidate=ensemble_candidate,
                )
            )
    return candidates, warnings


def validate_candidate_alignment(candidates: list[Candidate]) -> pd.DataFrame:
    if not candidates:
        raise ValueError("No Phase 6 candidates were discovered.")
    reference = candidates[0].predictions[["sample_id", "true_author"]].copy()
    reference_pairs = set(map(tuple, reference.to_numpy()))
    for candidate in candidates[1:]:
        current = candidate.predictions[["sample_id", "true_author"]]
        current_pairs = set(map(tuple, current.to_numpy()))
        if current_pairs != reference_pairs:
            raise ValueError(f"Candidate sample_id/true_author mismatch: {candidate.run_name}")
    return reference


def candidate_metric(candidate: Candidate, key: str) -> float:
    value = metric(candidate.metrics, key)
    return 0.0 if value is None else value


def sort_by_macro_f1(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda item: (candidate_metric(item, "macro_f1"), candidate_metric(item, "accuracy")), reverse=True)


def prediction_lookup(candidate: Candidate) -> dict[str, str]:
    return dict(zip(candidate.predictions["sample_id"], candidate.predictions["predicted_author"]))


def fallback_prediction(sample_id: str, sorted_candidates: list[Candidate], lookups: dict[str, dict[str, str]], invalid_label: str) -> str:
    for candidate in sorted_candidates:
        prediction = lookups[candidate.run_name].get(sample_id, invalid_label)
        if prediction in CANONICAL_AUTHORS:
            return prediction
    raise ValueError(f"No valid fallback prediction available for sample_id={sample_id}")


def vote_predictions(
    name: str,
    candidates: list[Candidate],
    reference: pd.DataFrame,
    *,
    invalid_label: str,
    weight_metric: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not candidates:
        raise ValueError(f"No candidates supplied for strategy {name}")
    sorted_candidates = sort_by_macro_f1(candidates)
    lookups = {candidate.run_name: prediction_lookup(candidate) for candidate in candidates}
    rows = []
    tie_count = 0
    invalid_fallback_count = 0
    for _, row in reference.iterrows():
        sample_id = str(row["sample_id"])
        scores: Counter[str] = Counter()
        for candidate in candidates:
            prediction = lookups[candidate.run_name].get(sample_id, invalid_label)
            if prediction not in CANONICAL_AUTHORS:
                continue
            weight = 1.0 if weight_metric is None else candidate_metric(candidate, weight_metric)
            if weight <= 0:
                continue
            scores[prediction] += weight
        if not scores:
            invalid_fallback_count += 1
            prediction = fallback_prediction(sample_id, sorted_candidates, lookups, invalid_label)
        else:
            best_score = max(scores.values())
            tied = sorted(author for author, score in scores.items() if score == best_score)
            if len(tied) > 1:
                tie_count += 1
                prediction = fallback_prediction(sample_id, sorted_candidates, lookups, invalid_label)
            else:
                prediction = tied[0]
        rows.append({"sample_id": sample_id, "true_author": row["true_author"], "predicted_author": prediction, "strategy": name})
    return pd.DataFrame(rows), {"tie_count": tie_count, "invalid_fallback_count": invalid_fallback_count}


def metrics_row(name: str, predictions: pd.DataFrame, candidate_count: int, details: dict[str, int], best_single_macro_f1: float, status: str = "ok") -> dict[str, Any]:
    metrics = recompute_metrics(predictions, INVALID_LABEL) if not predictions.empty else {"accuracy": None, "macro_f1": None, "weighted_f1": None}
    macro_f1 = metrics["macro_f1"]
    return {
        "strategy": name,
        "status": status,
        "candidate_count": candidate_count,
        "accuracy": metrics["accuracy"],
        "macro_f1": macro_f1,
        "weighted_f1": metrics["weighted_f1"],
        "tie_count": details.get("tie_count", 0),
        "invalid_fallback_count": details.get("invalid_fallback_count", 0),
        "macro_f1_delta_vs_best_single": None if macro_f1 is None else macro_f1 - best_single_macro_f1,
    }


def per_author_metrics(name: str, predictions: pd.DataFrame, group: str) -> pd.DataFrame:
    report = classification_report(
        predictions["true_author"],
        predictions["predicted_author"],
        labels=list(CANONICAL_AUTHORS),
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for author in CANONICAL_AUTHORS:
        rows.append(
            {
                "source": name,
                "source_type": group,
                "author": author,
                "precision": report[author]["precision"],
                "recall": report[author]["recall"],
                "f1": report[author]["f1-score"],
                "support": report[author]["support"],
            }
        )
    return pd.DataFrame(rows)


def write_confusion_matrix(name: str, predictions: pd.DataFrame, tables_dir: Path) -> None:
    matrix = confusion_matrix(predictions["true_author"], predictions["predicted_author"], labels=list(CANONICAL_AUTHORS))
    frame = pd.DataFrame(matrix, index=list(CANONICAL_AUTHORS), columns=list(CANONICAL_AUTHORS))
    frame.index.name = "true_author"
    frame.to_csv(tables_dir / f"confusion_matrix_{name}.csv")


def single_model_results(candidates: list[Candidate]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "phase": candidate.phase,
                "run_name": candidate.run_name,
                "model_name": candidate.model_name,
                "architecture_group": candidate.architecture_group,
                "accuracy": candidate_metric(candidate, "accuracy"),
                "macro_f1": candidate_metric(candidate, "macro_f1"),
                "weighted_f1": candidate_metric(candidate, "weighted_f1"),
                "invalid_output_rate": candidate.invalid_rate,
                "invalid_count": candidate.invalid_count,
                "n_samples": len(candidate.predictions),
                "predictions_path": str(candidate.predictions_path),
                "ensemble_candidate": candidate.ensemble_candidate,
            }
        )
    return pd.DataFrame(rows)


def select_strategy_candidates(name: str, candidates: list[Candidate]) -> list[Candidate]:
    eligible = [candidate for candidate in candidates if candidate.ensemble_candidate]
    encoders = [candidate for candidate in eligible if candidate.architecture_group == "encoder"]
    decoders = [candidate for candidate in eligible if candidate.architecture_group == "decoder"]
    if name == "encoder_only_majority_vote":
        return encoders
    if name == "decoder_only_majority_vote":
        return decoders
    if name == "cross_architecture_majority_vote":
        return eligible
    if name == "best_encoder_best_decoder_vote":
        selected = []
        if encoders:
            selected.append(sort_by_macro_f1(encoders)[0])
        if decoders:
            selected.append(sort_by_macro_f1(decoders)[0])
        return selected
    if name in {"weighted_vote_macro_f1", "weighted_vote_accuracy"}:
        return eligible
    raise ValueError(f"Unknown strategy: {name}")


def skipped_strategy_reason(name: str, candidates: list[Candidate]) -> str | None:
    eligible = [candidate for candidate in candidates if candidate.ensemble_candidate]
    encoders = [candidate for candidate in eligible if candidate.architecture_group == "encoder"]
    decoders = [candidate for candidate in eligible if candidate.architecture_group == "decoder"]
    if name == "encoder_only_majority_vote" and not encoders:
        return "No valid encoder candidates were found."
    if name == "decoder_only_majority_vote" and not decoders:
        return "No valid decoder candidates were found."
    if name == "best_encoder_best_decoder_vote":
        if not encoders and not decoders:
            return "No valid encoder or decoder candidates were found."
        if not decoders:
            return "No valid decoder candidates were found."
        if not encoders:
            return "No valid encoder candidates were found."
    if name in {"cross_architecture_majority_vote", "weighted_vote_macro_f1", "weighted_vote_accuracy"} and not eligible:
        return "No valid ensemble candidates were found."
    return None


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    numeric = display.select_dtypes(include="number").columns
    display[numeric] = display[numeric].round(4)
    return display.to_markdown(index=False)


def write_report(
    path: Path,
    ensemble_results: pd.DataFrame,
    single_results: pd.DataFrame,
    focus: pd.DataFrame,
    warnings: list[str],
) -> None:
    ok_results = ensemble_results.loc[ensemble_results["status"] == "ok"].copy()
    best_single = single_results.sort_values(["macro_f1", "accuracy"], ascending=False).head(1)
    best_ensemble = ok_results.sort_values(["macro_f1", "accuracy"], ascending=False).head(1)
    best_single_macro = float(best_single.iloc[0]["macro_f1"]) if not best_single.empty else 0.0
    best_ensemble_macro = float(best_ensemble.iloc[0]["macro_f1"]) if not best_ensemble.empty else 0.0
    improved = best_ensemble_macro > best_single_macro
    cross = ensemble_results.loc[ensemble_results["strategy"] == "cross_architecture_majority_vote"]
    encoder = ensemble_results.loc[ensemble_results["strategy"] == "encoder_only_majority_vote"]
    decoder = ensemble_results.loc[ensemble_results["strategy"] == "decoder_only_majority_vote"]
    cross_help = "not available"
    if not cross.empty and cross.iloc[0]["status"] == "ok":
        cross_macro = float(cross.iloc[0]["macro_f1"])
        peers = [
            float(row["macro_f1"])
            for _, row in pd.concat([encoder, decoder]).iterrows()
            if row.get("status") == "ok" and pd.notna(row.get("macro_f1"))
        ]
        cross_help = "yes" if peers and cross_macro > max(peers) else "no"

    lines = [
        "# Phase 6 Ensemble Report",
        "",
        "Phase 6 evaluates ensembles from existing final Phase 2-5 predictions. It does not retrain models, run decoder inference, or modify checkpoints/adapters.",
        "",
        "Diagnostic, smoke, invalid, and incomplete outputs are excluded by discovery rules. Invalid decoder predictions are normalized to `__INVALID__` and excluded from votes.",
        "",
        "## Required Answers",
        "",
        f"- Did any ensemble improve over the best single overall model? {'Yes' if improved else 'No'}.",
        f"- Did cross-architecture voting help? {cross_help}.",
        f"- Which strategy performed best? `{best_ensemble.iloc[0]['strategy']}`." if not best_ensemble.empty else "- Which strategy performed best? No completed ensemble strategy.",
    ]
    for author in ("James Fitzjames Stephen", "Eliza Lynn Linton"):
        author_focus = focus.loc[focus["author"] == author].copy()
        ensemble_focus = author_focus.loc[author_focus["source_type"] == "ensemble"].sort_values("f1", ascending=False).head(1)
        single_focus = author_focus.loc[author_focus["source_type"] == "single_model"].sort_values("f1", ascending=False).head(1)
        if ensemble_focus.empty or single_focus.empty:
            lines.append(f"- Did performance improve for {author}? Not available.")
        else:
            lines.append(
                f"- Did performance improve for {author}? "
                f"{'Yes' if float(ensemble_focus.iloc[0]['f1']) > float(single_focus.iloc[0]['f1']) else 'No'}."
            )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None."])
    lines.extend(["", "## Ensemble Results", "", markdown_table(ensemble_results), "", "## Single Model Results", "", markdown_table(single_results)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase6(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    invalid_label = str(config.get("invalid_label", INVALID_LABEL))
    predictions_dir = phase_artifact_dir("phase6", "predictions")
    tables_dir = phase_artifact_dir("phase6", "tables")
    reports_dir = phase_artifact_dir("phase6", "reports")
    phase_artifact_dir("phase6", "plots")

    candidates, warnings = discover_candidates(config)
    reference = validate_candidate_alignment(candidates)
    single_results = single_model_results(candidates)
    single_results.to_csv(tables_dir / "single_model_results.csv", index=False)
    best_single_macro = float(single_results["macro_f1"].max()) if not single_results.empty else 0.0

    ensemble_rows: list[dict[str, Any]] = []
    per_author_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        per_author_frames.append(per_author_metrics(candidate.run_name, candidate.predictions, "single_model"))

    for strategy in ENSEMBLE_STRATEGIES:
        reason = skipped_strategy_reason(strategy, candidates)
        selected = select_strategy_candidates(strategy, candidates)
        if reason is not None:
            warnings.append(f"Skipped {strategy}: {reason}")
            ensemble_rows.append(metrics_row(strategy, pd.DataFrame(), len(selected), {}, best_single_macro, status="skipped"))
            continue
        weight_metric = "macro_f1" if strategy == "weighted_vote_macro_f1" else "accuracy" if strategy == "weighted_vote_accuracy" else None
        predictions, details = vote_predictions(strategy, selected, reference, invalid_label=invalid_label, weight_metric=weight_metric)
        predictions.to_csv(predictions_dir / f"{strategy}.csv", index=False)
        write_confusion_matrix(strategy, predictions, tables_dir)
        ensemble_rows.append(metrics_row(strategy, predictions, len(selected), details, best_single_macro))
        per_author_frames.append(per_author_metrics(strategy, predictions, "ensemble"))

    ensemble_results = pd.DataFrame(ensemble_rows)
    ensemble_results.to_csv(tables_dir / "ensemble_results.csv", index=False)
    per_author = pd.concat(per_author_frames, ignore_index=True)
    per_author.to_csv(tables_dir / "per_author_f1.csv", index=False)
    focus_authors = [str(item) for item in config.get("focus_authors", [])]
    focus = per_author.loc[per_author["author"].isin(focus_authors)].copy()
    focus.to_csv(tables_dir / "hard_author_focus.csv", index=False)
    write_report(reports_dir / "phase6_ensemble_report.md", ensemble_results, single_results, focus, warnings)
    return {
        "predictions_dir": predictions_dir,
        "tables_dir": tables_dir,
        "reports_dir": reports_dir,
        "report": reports_dir / "phase6_ensemble_report.md",
    }
