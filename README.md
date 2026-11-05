# Hinglish Shield

Multilingual and code-mixed (Hindi, Hinglish, English) safety audit and repair pipeline for [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).

Lexsi AI Safety and Alignment Hackathon, Track 2 (AI Red Teaming, Safety and Alignment). Submission due Nov 14, 2026 AoE.

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
redteam/       attack templates and wrappers (research eval)
space/         Gradio HF Space
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make smoke
```

`make smoke` checks layout, validates `runs/_example/metrics.json` against `schemas/metrics.schema.json`, and imports local packages. It does not download models or spend GPU.

## Conventions

- Branches: `role/short-task`. Commits imperative.
- Never invent numbers; every claim links to a run.
- Do not use em dashes in docs.
