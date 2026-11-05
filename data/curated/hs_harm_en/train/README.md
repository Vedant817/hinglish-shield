---
tags:
- curatorkit
configs:
- config_name: ppo
  default: true
  description: curatorkit ppo export, sha256 fc1de8092436c15d7bf7aee0aba20385eea2cd83ece5231cafcba8d735fec31d
  data_files:
  - split: train
    path: ppo.jsonl
---

# Dataset Card: hs_harm_en_train

## Summary

- **Run timestamp**: 2026-11-05T19:59:21.117741Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `5f298b07dd6e228a`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 465 | 465 | — | 0 | 465 |
| SchemaGate | 465 | 465 | 0 | 0 | 465 |
| ExactDeduplicator | 465 | 465 | — | — | 465 |
| TextCleaner | 465 | 465 | — | — | 465 |
| PPOExporter | — | 465 | — | — | 465 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 465/0 passed (N/A), 0 rejected
- **SchemaGate**: 465/465 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 0

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.21
- **python**: 3.13.5

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `ppo` | train (465) | `prompt`: string |
