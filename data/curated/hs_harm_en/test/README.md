---
tags:
- curatorkit
configs:
- config_name: ppo
  default: true
  description: curatorkit ppo export, sha256 3c8b7529d62c103293fe2c3e3337d63471fb00d07950437c9f7dcd2b061b7f4d
  data_files:
  - split: train
    path: ppo.jsonl
---

# Dataset Card: hs_harm_en_test

## Summary

- **Run timestamp**: 2026-11-05T19:59:21.140561Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `5d994f2e16d46c44`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 335 | 335 | — | 0 | 335 |
| SchemaGate | 335 | 335 | 0 | 0 | 335 |
| ExactDeduplicator | 335 | 335 | — | — | 335 |
| TextCleaner | 335 | 335 | — | — | 335 |
| PPOExporter | — | 335 | — | — | 335 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 335/0 passed (N/A), 0 rejected
- **SchemaGate**: 335/335 passed (100.0%), 0 rejected


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
| `ppo` | train (335) | `prompt`: string |
