---
tags:
- curatorkit
configs:
- config_name: ppo
  default: true
  description: curatorkit ppo export, sha256 c696fbb99aec87649c28f4d581d22d2e02dd706b6a1e4620a21ff2af15b11f75
  data_files:
  - split: train
    path: ppo.jsonl
---

# Dataset Card: hs_benign_en_test

## Summary

- **Run timestamp**: 2026-11-06T07:21:11.714369Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `b95b36bd57e089ee`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 369 | 369 | - | 0 | 369 |
| SchemaGate | 369 | 369 | 0 | 0 | 369 |
| ExactDeduplicator | 369 | 369 | - | - | 369 |
| PayloadPreservingTextCleaner | 369 | 369 | - | - | 369 |
| PPOExporter | - | 369 | - | - | 369 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 369/0 passed (N/A), 0 rejected
- **SchemaGate**: 369/369 passed (100.0%), 0 rejected


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
| `ppo` | train (369) | `prompt`: string |

## Hinglish Shield split note (family_splits_v1)

- Split: test (family-level calibration/dev/test 0.20/0.30/0.50, seed 42, stratified by hs_label x category), per preregistration research/baseline-prereg@8b6af17.
- Every row inherits its split from its semantic family in data/splits/family_splits_v1.json (sha256 803790ea867c3c7b58c23aa180e47e0eaabb0f7d5583853bee73c0cd34a065b9). No family crosses splits; verify with: python -m src.curate.check_family_splits data/curated/hs_benign_en/test/hs_eval.jsonl
- frozen: true. test_access: nov11_final_pass_only. This split is read ONLY by the Nov 11 final frozen-arm pass; never used for locate, repair, selection or tuning.
- The CuratorKIT-generated HF config label "train" above is the exporter's default config name, not the Hinglish Shield split.
