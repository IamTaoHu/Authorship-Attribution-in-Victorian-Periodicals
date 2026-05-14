# Hugging Face Resources

Hugging Face provides the project datasets, pretrained models, and tokenizers.

Related docs: [COLAB_WORKFLOW.md](COLAB_WORKFLOW.md), [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md), [GPT_CONTEXT_SUMMARY.md](GPT_CONTEXT_SUMMARY.md).

## Datasets

| Resource | Use |
|---|---|
| `celvaigh/periad` | Main PERIAD corpus for authorship attribution experiments |
| `NicholasSynovic/ModifiedVEAA` | Comparison dataset resource for broader authorship attribution context |

The PERIAD pipeline expects six canonical authors:

- Leslie Stephen
- John Morley
- Eliza Lynn Linton
- George Henry Lewes
- Anne Mozley
- James Fitzjames Stephen

## Encoder Models

| Model | Use |
|---|---|
| `bert-base-uncased` | Phase 2 baseline encoder |
| `roberta-base` | Phase 2 baseline encoder |
| `roberta-large` | Phase 2 stronger encoder baseline |
| `microsoft/deberta-v3-base` | Phase 3 advanced encoder |
| `answerdotai/ModernBERT-base` | Phase 3 advanced encoder |

## Decoder Models

| Model | Use | Access |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | Phase 4 decoder prompting; candidate for future fine-tuning | may require token depending on environment |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Phase 4 decoder prompting; candidate for future fine-tuning | gated |
| `google/gemma-2-9b-it` | Phase 4 decoder prompting; candidate for future fine-tuning | gated |

Llama and Gemma require gated Hugging Face access. Colab runs should authenticate with `HF_TOKEN` before loading these models.

## Embedding Model

| Model | Use |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Planned BERTopic and neural topic modelling embeddings |

## Authentication Notes

Use `HF_TOKEN` in local or Colab environments. For gated models:

1. Request access on the Hugging Face model page.
2. Confirm access has been granted.
3. Login inside Colab or export `HF_TOKEN`.
4. Avoid hard-coding tokens in notebooks, scripts, or committed files.
