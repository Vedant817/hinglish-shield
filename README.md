# Hinglish Shield

Multilingual and code-mixed (Hindi, Hinglish, English) safety audit and repair pipeline for [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).

Lexsi AI Safety and Alignment Hackathon, Track 2 (AI Red Teaming, Safety and Alignment). Submission due Nov 14, 2026 AoE.

## Status

Measured attack success rate, over-refusal, capability, and repair acceptance: **Not run**.

The tracked `metrics.json` files are `artifact_kind: fixture` with null benchmark fields: `runs/_example/metrics.json` and `runs/wildguard-fixture-20261107T141335276834Z/metrics.json`. Fixture generations are not ASR.

Target chat-template pin: **Not run**. The latest evidence is the 2026-11-08 Kaggle editor archive at `runs/gpu-20261108-234246-dd685561/`, ingested 2026-11-09. It declares a CUDA T4 x2 session, but it is not a saved Kaggle version and has no run log, source notebook or identity asset. It records no token-presence boolean and no chat-template hash. Its D8 status is an unscored blocker; the 401 in that projection remains historical v11 evidence. Version 12 remains the earlier CPU-only `secret_missing` failure under `runs/kaggle-parity-v12-2026-11-08/`. In the notebook editor, attach `HF_TOKEN`, select the existing account's free GPU, then Save Version and Save & Run All.

Dev baseline: **Not run**. D8 in `docs/research/PREREG_baseline.md` says a future dev score is automated-judge only (WildGuard primary, pinned Qwen3Guard-Gen-0.6B secondary) and must be labeled "automated-judge only, human calibration pending". Unreviewed rows are not human labels. The 2026-11-08 editor archive contains only an unscored D8 blocker with all metrics null.

Original fp32 parity: **Not run**. The 2026-11-08 editor archive records `incomplete` after a 29.05/31.347 GiB `MemoryError` (short KL empty, long KL null). Its bf16 variant is provisional only, as are the CPU bf16 files under `runs/kaggle-parity-v8-2026-11-07/`; none is original parity credit or ASR. The repair arm is not frozen. The locked test stays unrun before Nov 11 IST.

## Pipeline (ADR summary)

Seeds (Aya red-team HI, HarmBench, XSTest, OR-Bench) -> translate/transliterate -> CuratorKIT -> AuditKIT baseline -> CircuitKIT / refusal directions -> SafeTune steering then optional LoRA -> AuditKIT after eval -> HF Hub + Space.

Key ADRs: Tiny Aya Global first; steering before fine-tune; WildGuard primary judge; Lexsi interop formats; pinned reproducible configs; Arditi difference-of-means fallback if CircuitKIT cannot load Tiny Aya.

## Layout

```
data/          seeds and curated exports
configs/       YAML configs
src/           curate, eval, locate, repair, demo
runs/          metrics.json per run (example under runs/_example)
schemas/       metrics JSON Schema
scripts/       smoke and helpers
notebooks/     exploration
space/         Gradio HF Space
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make smoke
```

`make smoke` checks layout, validates `runs/_example/metrics.json` against `schemas/metrics.schema.json`, and imports local packages. It does not download models or spend GPU.

On Windows, activate an external Python 3.12 environment, install `requirements.txt`,
then run `make smoke PYTHON=python`. The same override works for
`make check-family-splits-translations PYTHON=python`. Frozen manifests and YAML
configs use LF checkout bytes so their committed SHA-256 values survive Windows checkout.

## Conventions

- Branches: `role/short-task`. Commits imperative.
- Never invent numbers; every claim links to a run.
- Do not use em dashes in docs.
