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
