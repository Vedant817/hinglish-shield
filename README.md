# Hinglish Shield

Multilingual and code-mixed (Hindi, Hinglish, English) safety audit and repair pipeline for [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).

Lexsi AI Safety and Alignment Hackathon, Track 2 (AI Red Teaming, Safety and Alignment). Submission due Nov 14, 2026 AoE.

## Status

Measured attack success rate, over-refusal, capability, and repair acceptance: **Not run**.

The tracked `metrics.json` files are `artifact_kind: fixture` with null benchmark fields: `runs/_example/metrics.json` and `runs/wildguard-fixture-20261107T141335276834Z/metrics.json`. Fixture generations are not ASR.

Target chat-template pin: **Not run**. Private Kaggle version 11 (`runs/tokenizer-metadata-v11-2026-11-08/`) finished with `failure_phase` `secret_missing` and an unauthenticated HTTP 401. No tokenizer asset was downloaded. The Kaggle CLI cannot attach `HF_TOKEN`. The existing secret has to be attached in the notebook editor before another metadata or GPU run.

Dev baseline: **Not run**. D8 in `docs/research/PREREG_baseline.md` says a future dev score is automated-judge only (WildGuard primary, pinned Qwen3Guard-Gen-0.6B secondary) and must be labeled "automated-judge only, human calibration pending". Unreviewed rows are not human labels.

Original fp32 parity: **Not run**. The CPU bf16 files under `runs/kaggle-parity-v8-2026-11-07/` are a provisional variant, not original parity credit and not ASR. The repair arm is not frozen. The locked test stays unrun before Nov 11 IST.

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
