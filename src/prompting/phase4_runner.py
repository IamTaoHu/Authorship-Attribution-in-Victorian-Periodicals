"""Run Phase 4 decoder prompting baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.prompting.phase4_parser import CANONICAL_AUTHORS, is_valid_parse, parse_phase4_output
from src.prompting.phase4_prompts import build_prompt, load_or_build_few_shot_examples
from src.utils.paths import dataset_dir, get_path_config, phase_artifact_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase4" / "tinyllama_smoke.yaml"
REQUIRED_SPLIT_COLUMNS = ("sample_id", "text", "author")
PREDICTION_COLUMNS = (
    "sample_id",
    "text",
    "true_author",
    "prompt_type",
    "model_name",
    "raw_output",
    "cleaned_output",
    "predicted_author",
    "parsed_status",
    "parse_error",
    "correct",
    "input_tokens",
    "output_tokens",
    "inference_time_sec",
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")


def read_split(path: Path, split_name: str, max_samples: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required PERIAD {split_name} split not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_SPLIT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, list(REQUIRED_SPLIT_COLUMNS)].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["author"] = frame["author"].astype(str)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).copy()
    return frame


def validate_periad(periad_dir: Path) -> dict[str, int]:
    label_map_path = periad_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Required PERIAD label map not found: {label_map_path}")
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    if list(label_map.keys()) != list(CANONICAL_AUTHORS):
        raise ValueError(f"Unexpected PERIAD label map author order: {label_map_path}")
    full_train = pd.read_csv(periad_dir / "train.csv", usecols=["author"])
    full_test = pd.read_csv(periad_dir / "test.csv", usecols=["author"])
    if len(full_train) != 8279 or len(full_test) != 3549:
        raise ValueError(f"Unexpected PERIAD split sizes: train={len(full_train)}, test={len(full_test)}")
    observed = sorted(set(full_train["author"].astype(str)).union(set(full_test["author"].astype(str))))
    if observed != sorted(CANONICAL_AUTHORS):
        raise ValueError(f"Unexpected PERIAD authors: {observed}")
    return {"train": int(len(full_train)), "test": int(len(full_test))}


def resolve_run_dir(config: dict[str, Any]) -> Path:
    return phase_artifact_dir("phase4", "runs", str(config["run_name"]))


def resolve_dtype(value: str) -> Any:
    normalized = str(value or "auto").lower()
    if normalized in {"auto", ""}:
        return "auto"
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {value}")


def load_generation_model(config: dict[str, Any]) -> tuple[Any, Any]:
    model_name = str(config["model_name"])
    token_env = str(config.get("hf_token_env", "HF_TOKEN"))
    token = os.environ.get(token_env)
    trust_remote_code = bool(config.get("trust_remote_code", False))
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "device_map": "auto"}
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token
    torch_dtype = resolve_dtype(str(config.get("torch_dtype", "auto")))
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if bool(config.get("load_in_4bit", False)):
        model_kwargs["load_in_4bit"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


def render_for_model(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_one(tokenizer: Any, model: Any, prompt: str, config: dict[str, Any]) -> tuple[str, int, int, float]:
    rendered = render_for_model(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt", truncation=False)
    input_tokens = int(inputs["input_ids"].shape[-1])
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": int(config.get("max_new_tokens", 32)),
        "do_sample": bool(config.get("do_sample", False)),
        "temperature": float(config.get("temperature", 0.0)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not generation_kwargs["do_sample"]:
        generation_kwargs.pop("temperature", None)
    started = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    elapsed = time.time() - started
    generated_ids = output_ids[0, input_tokens:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return raw_output, input_tokens, int(generated_ids.shape[-1]), elapsed


def load_existing_predictions(path: Path, overwrite: bool) -> pd.DataFrame:
    if overwrite or not path.exists():
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    frame = pd.read_csv(path)
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Existing predictions.csv is missing required columns: {', '.join(missing)}")
    return frame.loc[:, list(PREDICTION_COLUMNS)].copy()


def write_predictions(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(PREDICTION_COLUMNS)].to_csv(path, index=False)


def _empty_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def compute_and_write_reports(predictions: pd.DataFrame, run_dir: Path, config: dict[str, Any], invalid_reason: str = "") -> dict[str, Any]:
    if predictions.empty:
        predictions = _empty_prediction_frame()
    valid_mask = predictions["parsed_status"].map(is_valid_parse) if not predictions.empty else pd.Series(dtype=bool)
    valid = predictions.loc[valid_mask].copy() if not predictions.empty else predictions.copy()
    n_samples = int(len(predictions))
    n_valid = int(len(valid))
    y_true = predictions["true_author"].astype(str) if n_samples else []
    y_pred = predictions["predicted_author"].fillna("").astype(str) if n_samples else []
    metrics = {
        "run_name": str(config.get("run_name", "")),
        "model_name": str(config.get("model_name", "")),
        "prompt_type": str(config.get("prompt_type", "")),
        "accuracy": float(accuracy_score(y_true, y_pred)) if n_samples else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CANONICAL_AUTHORS), average="macro", zero_division=0)) if n_samples else None,
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CANONICAL_AUTHORS), average="weighted", zero_division=0)) if n_samples else None,
        "invalid_output_rate": float(1 - (n_valid / n_samples)) if n_samples else None,
        "n_samples": n_samples,
        "n_valid_predictions": n_valid,
        "avg_input_tokens": float(pd.to_numeric(predictions["input_tokens"], errors="coerce").mean()) if n_samples else None,
        "avg_output_tokens": float(pd.to_numeric(predictions["output_tokens"], errors="coerce").mean()) if n_samples else None,
        "avg_inference_time_sec": float(pd.to_numeric(predictions["inference_time_sec"], errors="coerce").mean()) if n_samples else None,
        "total_inference_time_sec": float(pd.to_numeric(predictions["inference_time_sec"], errors="coerce").sum()) if n_samples else 0.0,
        "valid_run": not bool(invalid_reason),
        "invalid_reason": invalid_reason,
    }
    write_json(run_dir / "metrics.json", metrics)
    invalid = predictions.loc[~valid_mask].copy() if not predictions.empty else _empty_prediction_frame()
    invalid.to_csv(run_dir / "invalid_outputs.csv", index=False)
    print(f"Wrote {run_dir / 'invalid_outputs.csv'}")
    if n_samples:
        report = classification_report(y_true, y_pred, labels=list(CANONICAL_AUTHORS), output_dict=True, zero_division=0)
        report_frame = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
        matrix = confusion_matrix(y_true, y_pred, labels=list(CANONICAL_AUTHORS))
    else:
        report_frame = pd.DataFrame(columns=["label", "precision", "recall", "f1-score", "support"])
        matrix = np.zeros((len(CANONICAL_AUTHORS), len(CANONICAL_AUTHORS)), dtype=int)
    report_frame.to_csv(run_dir / "classification_report.csv", index=False)
    print(f"Wrote {run_dir / 'classification_report.csv'}")
    matrix_frame = pd.DataFrame(matrix, index=list(CANONICAL_AUTHORS), columns=list(CANONICAL_AUTHORS))
    matrix_frame.index.name = "true_author"
    matrix_frame.to_csv(run_dir / "confusion_matrix.csv")
    print(f"Wrote {run_dir / 'confusion_matrix.csv'}")
    return metrics


def write_resolved_config(path: Path, config_path: Path, config: dict[str, Any], run_dir: Path, periad_dir: Path) -> None:
    paths = get_path_config()
    resolved = {
        "config_path": str(config_path.resolve()),
        "run_name": str(config.get("run_name", "")),
        "model_name": str(config.get("model_name", "")),
        "prompt_type": str(config.get("prompt_type", "")),
        "dataset_dir": str(periad_dir),
        "run_artifact_dir": str(run_dir),
        "phase4_dir": str(paths["artifacts_root"] / "phase4"),
        "config": config,
    }
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    print(f"Wrote {path}")


def run_phase4(config_path: str | Path, *, overwrite: bool = False, rebuild_few_shot_examples: bool = False) -> int:
    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    if config.get("phase") != "phase4":
        raise ValueError(f"Expected phase4 config, found: {config.get('phase')}")
    set_seed(int(config.get("seed", 42)))
    run_dir = resolve_run_dir(config)
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    periad_dir = dataset_dir("processed", "periad")
    write_resolved_config(run_dir / "run_config_resolved.yaml", config_path, config, run_dir, periad_dir)
    split_counts = validate_periad(periad_dir)
    max_samples = config.get("max_samples")
    train_frame = read_split(periad_dir / "train.csv", "train")
    eval_split = str(config.get("split", "test"))
    eval_frame = read_split(periad_dir / f"{eval_split}.csv", eval_split, max_samples)
    few_shot_examples = None
    if config["prompt_type"] == "few_shot_direct":
        few_shot_path = phase_artifact_dir("phase4") / "few_shot_examples.json"
        few_shot_examples = load_or_build_few_shot_examples(
            train_frame,
            few_shot_path,
            rebuild=rebuild_few_shot_examples,
            min_text_chars=int(config.get("few_shot_min_text_chars", 200)),
            max_example_chars=config.get("few_shot_max_example_chars"),
        )
    predictions_path = run_dir / "predictions.csv"
    predictions = load_existing_predictions(predictions_path, overwrite=False)
    completed_ids = set(predictions["sample_id"].astype(str)) if not predictions.empty else set()
    remaining = eval_frame.loc[~eval_frame["sample_id"].astype(str).isin(completed_ids)].copy()
    if predictions_path.exists() and not remaining.empty:
        print(f"Resuming {config['run_name']}: {len(completed_ids)} completed, {len(remaining)} remaining.")
    try:
        tokenizer, model = load_generation_model(config)
    except Exception as exc:
        reason = f"model_load_failed: {type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"
        print(f"WARNING: {reason}")
        write_predictions(predictions_path, predictions)
        metrics = compute_and_write_reports(predictions, run_dir, config, invalid_reason=reason)
        metrics.update({"runtime_seconds": round(time.time() - started, 4), **{f"official_split_{k}_rows": v for k, v in split_counts.items()}})
        write_json(run_dir / "metrics.json", metrics)
        return 0
    for _, row in remaining.iterrows():
        prompt = build_prompt(
            text=row["text"],
            prompt_type=str(config["prompt_type"]),
            max_input_chars=config.get("max_input_chars"),
            few_shot_examples=few_shot_examples,
        )
        try:
            raw_output, input_tokens, output_tokens, inference_time = generate_one(tokenizer, model, prompt, config)
            parsed = parse_phase4_output(raw_output)
            predicted_author = parsed.predicted_author
            parsed_status = parsed.parsed_status
            parse_error = parsed.parse_error
            cleaned_output = parsed.cleaned_output
        except Exception as exc:
            raw_output = ""
            cleaned_output = ""
            predicted_author = ""
            parsed_status = "invalid_empty"
            parse_error = f"inference_failed: {type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"
            input_tokens = 0
            output_tokens = 0
            inference_time = 0.0
        record = {
            "sample_id": str(row["sample_id"]),
            "text": str(row["text"]),
            "true_author": str(row["author"]),
            "prompt_type": str(config["prompt_type"]),
            "model_name": str(config["model_name"]),
            "raw_output": raw_output,
            "cleaned_output": cleaned_output,
            "predicted_author": predicted_author,
            "parsed_status": parsed_status,
            "parse_error": parse_error,
            "correct": bool(predicted_author == str(row["author"])) if predicted_author else False,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "inference_time_sec": float(inference_time),
        }
        predictions = pd.concat([predictions, pd.DataFrame([record])], ignore_index=True)
        write_predictions(predictions_path, predictions)
        print(f"Wrote {predictions_path} ({len(predictions)}/{len(eval_frame)})")
    metrics = compute_and_write_reports(predictions, run_dir, config)
    metrics.update({"runtime_seconds": round(time.time() - started, 4), **{f"official_split_{k}_rows": v for k, v in split_counts.items()}})
    write_json(run_dir / "metrics.json", metrics)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild_few_shot_examples", action="store_true")
    args = parser.parse_args()
    return run_phase4(args.config, overwrite=args.overwrite, rebuild_few_shot_examples=args.rebuild_few_shot_examples)


if __name__ == "__main__":
    raise SystemExit(main())



