"""BERT sequence-classification model factory."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a fast tokenizer for a sequence classifier."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_sequence_classifier(
    model_name: str,
    label_id_to_name: dict[int, str],
) -> PreTrainedModel:
    """Load a sequence-classification model with stable label metadata."""
    id2label = {int(label_id): name for label_id, name in label_id_to_name.items()}
    label2id = {name: int(label_id) for label_id, name in id2label.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
