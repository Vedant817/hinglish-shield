---
tags:
- curatorkit
configs:
- config_name: ppo
  default: true
  description: curatorkit ppo export, sha256 90259b97e31158972d36169eaacd2e65446aed529625ffe2db3ccfcb34302787
  data_files:
  - split: train
    path: ppo.jsonl
---

# Dataset Card: hs_harm_en_dev

## Summary

- **Run timestamp**: 2026-11-06T07:21:11.652129Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `25bda56d7cd7553b`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 283 | 283 | - | 0 | 283 |
| SchemaGate | 283 | 283 | 0 | 0 | 283 |
| ExactDeduplicator | 283 | 283 | - | - | 283 |
| PayloadPreservingTextCleaner | 283 | 283 | - | - | 283 |
| PPOExporter | - | 283 | - | - | 283 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 283/0 passed (N/A), 0 rejected
- **SchemaGate**: 283/283 passed (100.0%), 0 rejected


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
| `ppo` | train (283) | `prompt`: string |

## Hinglish Shield split note (family_splits_v1)

- Split: dev (family-level calibration/dev/test 0.20/0.30/0.50, seed 42, stratified by hs_label x category), per preregistration research/baseline-prereg@8b6af17.
- Every row inherits its split from its semantic family in data/splits/family_splits_v1.json (sha256 803790ea867c3c7b58c23aa180e47e0eaabb0f7d5583853bee73c0cd34a065b9). No family crosses splits; verify with: python -m src.curate.check_family_splits data/curated/hs_harm_en/dev/hs_eval.jsonl
- Access: locate and repair work may use this split (prereg access.dev).
- The CuratorKIT-generated HF config label "train" above is the exporter's default config name, not the Hinglish Shield split.
