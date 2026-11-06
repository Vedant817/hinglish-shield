# Dataset Card: hs_harm_en_val

## Summary

- **Run timestamp**: 2026-11-05T19:59:21.127189Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `bd0429340b9e28f4`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 143 | 143 | — | 0 | 143 |
| SchemaGate | 143 | 143 | 0 | 0 | 143 |
| ExactDeduplicator | 143 | 143 | — | — | 143 |
| TextCleaner | 143 | 143 | — | — | 143 |
| PPOExporter | — | 143 | — | — | 143 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 143/0 passed (N/A), 0 rejected
- **SchemaGate**: 143/143 passed (100.0%), 0 rejected


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
