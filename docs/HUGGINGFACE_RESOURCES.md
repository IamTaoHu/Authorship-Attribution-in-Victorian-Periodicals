# Hugging Face Resources

Hugging Face is the primary source for datasets and pretrained models in this project.

## Datasets

| Resource | Purpose |
|---|---|
| `celvaigh/periad` | Main PERIAD corpus for authorship attribution on Victorian periodicals |
| `NicholasSynovic/ModifiedVEAA` | Benchmark comparison dataset |

Expected PERIAD counts:

- Train rows: 8279
- Test rows: 3549
- Total rows: 11828

Canonical PERIAD authors:

- Leslie Stephen
- John Morley
- Eliza Lynn Linton
- George Henry Lewes
- Anne Mozley
- James Fitzjames Stephen

## Encoder Models

| Model | Intended use |
|---|---|
| `bert-base-uncased` | Baseline encoder |
| `roberta-base` | Modern encoder baseline |
| `roberta-large` | Larger encoder baseline |
| `microsoft/deberta-v3-base` | Advanced encoder baseline |
| `answerdotai/ModernBERT-base` | Long-context modern encoder baseline |

## Decoder Models

| Model | Intended use | Access note |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | Instruction prompting and QLoRA | May require token-authenticated download |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Instruction prompting and QLoRA | Gated; requires access approval and `HF_TOKEN` |
| `google/gemma-2-9b-it` | Instruction prompting and QLoRA | Gated; requires access approval and `HF_TOKEN` |

## Embedding Model

| Model | Intended use |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence embeddings for BERTopic and related analysis |

## Authentication

Use `HF_TOKEN` for Hugging Face authentication in local and Colab environments. Gated models such as Llama and Gemma require both a valid token and approved access on the Hugging Face account associated with that token.
