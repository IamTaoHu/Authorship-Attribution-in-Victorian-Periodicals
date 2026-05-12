# Task Board

## Phase 0 - Infrastructure Refactor

- [ ] finalize repo structure
- [ ] create configs/paths.example.yaml
- [ ] create configs/paths.local.yaml locally
- [ ] add src/utils/paths.py
- [ ] add configs/datasets.yaml
- [ ] add configs/models.yaml
- [ ] update .gitignore
- [ ] add scripts/check_project_structure.py
- [ ] add Colab bootstrap notebook
- [ ] validate local structure

## Phase 1 - Dataset Pipeline

- [ ] load PERIAD from Hugging Face
- [ ] validate expected split counts train=8279, test=3549
- [ ] validate six canonical authors
- [ ] load ModifiedVEAA from Hugging Face
- [ ] generate dataset statistics
- [ ] generate tokenization analysis

## Phase 2 - Encoder Baselines

- [ ] BERT-base
- [ ] RoBERTa-base
- [ ] RoBERTa-large
- [ ] metrics
- [ ] majority vote

## Phase 3 - Advanced Encoders

- [ ] DeBERTa-v3-base
- [ ] ModernBERT-base
- [ ] DeBERTa LoRA rank ablation
- [ ] context length analysis

## Phase 4 - Decoder Prompting

- [ ] prompt templates
- [ ] parser
- [ ] Mistral prompting
- [ ] Llama prompting
- [ ] Gemma prompting
- [ ] invalid output analysis

## Phase 5 - QLoRA Decoder Fine-Tuning

- [ ] instruction dataset
- [ ] Mistral QLoRA
- [ ] Llama QLoRA
- [ ] Gemma QLoRA
- [ ] LoRA rank ablation
- [ ] evaluation
- [ ] aggregation

## Phase 6 - Ensemble

- [ ] select best encoder
- [ ] select best decoder
- [ ] majority vote
- [ ] hard author analysis

## Phase 7 - LDA

- [ ] grid search k=10,20,30,40
- [ ] coherence c_v
- [ ] heatmaps
- [ ] topic entropy vs errors

## Phase 8 - BERTopic

- [ ] embeddings
- [ ] UMAP
- [ ] HDBSCAN
- [ ] HTML export

## Phase 9 - Topic-Aware Classification

- [ ] topic vector extraction
- [ ] [CLS] + topic vector fusion
- [ ] compare base vs topic-aware

## Phase 10 - Final Package

- [ ] unified benchmark tables
- [ ] final plots
- [ ] paper-ready exports
- [ ] benchmark_summary.md
