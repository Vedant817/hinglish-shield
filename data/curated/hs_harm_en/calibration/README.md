---
tags:
- curatorkit
configs:
- config_name: ppo
  default: true
  description: curatorkit ppo export, sha256 5cdda3847a73039862fd969bbd0acf89a4d7320b7ec96cafba31dae97e5f0ae0
  data_files:
  - split: train
    path: ppo.jsonl
---

# Dataset Card: hs_harm_en_calibration

## Summary

- **Run timestamp**: 2026-11-06T07:21:11.639413Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `1fe16cda07e63113`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 185 | 185 | - | 0 | 185 |
| SchemaGate | 185 | 185 | 0 | 0 | 185 |
| ExactDeduplicator | 185 | 185 | - | - | 185 |
| PayloadPreservingTextCleaner | 185 | 185 | - | - | 185 |
| PPOExporter | - | 185 | - | - | 185 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 185/0 passed (N/A), 0 rejected
- **SchemaGate**: 185/185 passed (100.0%), 0 rejected


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
| `ppo` | train (185) | `prompt`: string |

## Hinglish Shield split note (family_splits_v1)

- Split: calibration (family-level calibration/dev/test 0.20/0.30/0.50, seed 42, stratified by hs_label x category), per preregistration research/baseline-prereg@8b6af17.
- Every row inherits its split from its semantic family in data/splits/family_splits_v1.json (sha256 803790ea867c3c7b58c23aa180e47e0eaabb0f7d5583853bee73c0cd34a065b9). No family crosses splits; verify with: python -m src.curate.check_family_splits data/curated/hs_harm_en/calibration/hs_eval.jsonl
- Access: locate and repair work may use this split (prereg access.calibration).
- The CuratorKIT-generated HF config label "train" above is the exporter's default config name, not the Hinglish Shield split.
