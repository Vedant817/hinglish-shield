---
tags:
- curatorkit
configs:
- config_name: ppo
  default: true
  description: curatorkit ppo export, sha256 86d7e4325b183261a22bbbe182b2d54f58d656df28a9940a1e287e37f29a91ad
  data_files:
  - split: train
    path: ppo.jsonl
---

# Dataset Card: hs_benign_en_dev

## Summary

- **Run timestamp**: 2026-11-06T07:24:40.909086Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `e63a9e2a1ee888cc`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 245 | 245 | - | 0 | 245 |
| SchemaGate | 245 | 245 | 0 | 0 | 245 |
| ExactDeduplicator | 245 | 245 | - | - | 245 |
| PayloadPreservingTextCleaner | 245 | 245 | - | - | 245 |
| PPOExporter | - | 245 | - | - | 245 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 245/0 passed (N/A), 0 rejected
- **SchemaGate**: 245/245 passed (100.0%), 0 rejected


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
| `ppo` | train (245) | `prompt`: string |

## Hinglish Shield split note (family_splits_v2)

- Split: dev. Family-level, per preregistration research/baseline-prereg@8b6af17. Non-XSTest families calibration/dev/test 0.20/0.30/0.50; XSTest families dev:test 30:50 only (excluded from calibration). Seed 42, stratified by (is_xstest, hs_label, source_dataset, category).
- Every row inherits its split from its semantic family in data/splits/family_splits_v2.json (sha256 334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531). No family crosses splits; verify with: python -m src.curate.check_family_splits data/curated/hs_benign_en/dev/hs_eval.jsonl
- Access: locate and repair work may use this split (prereg access.dev).
- The CuratorKIT-generated HF config label "train" above is the exporter's default config name, not the Hinglish Shield split.
