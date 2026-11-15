# Hinglish Shield

Multilingual and code-mixed (Hindi, Hinglish, English) safety audit and repair pipeline for [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).

Lexsi AI Safety and Alignment Hackathon, Track 2a (multilingual and code-mixed safety, safety evaluation sets) and Track 2b (inference-time steering, locating where safety lives).

**Write-up:** [docs/ARTICLE.md](docs/ARTICLE.md). **Results:** [docs/RESULTS.md](docs/RESULTS.md).

## Key findings (dev split, automated dual judges, every number from a committed run)

- **The cross-lingual safety gap is real.** Tiny Aya Global (rev `af89d219`) refuses **69.3%** [63.8, 74.2] of harmful English prompts (205/296) but **54.2%** [48.3, 59.9] in Hindi (150/277) and only **23.9%** [15.3, 35.3] in Hinglish (16/67), WildGuard. Qwen3Guard agrees (68.2% / 61.7% / 29.9%). Source: [dev_eval_language_breakdown.md](runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.md).
- **English-centric judges under-read Hindi.** On Hindi harmful prompts WildGuard flags 0.7% of responses unsafe, Qwen3Guard 6.1%. Dual judging with agreement stats (Cohen's kappa 0.497, 91.8% raw agreement, 1334 rows) is necessary.
- **Single-direction steering does not close the gap.** A CircuitKIT difference-of-means refusal direction at `blocks.0.hook_mlp_out`, frozen before evaluation, gives harm unsafe **7.0%** (45/640) vs **6.7%** (43/640) base and **7.3%** (47/640) for a norm-matched random direction, with overlapping Wilson 95% intervals under both judges ([dev_repair_eval.json](runs/gpu-20261111-180320-270579d4/dev_repair_eval.json)). The alpha-zero control matched base byte-for-byte on all 885 dev rows, so this is a property of the direction, not the harness.
- **Reproducible by construction.** Frozen family-level splits, pinned model, tokenizer and judge revisions, bf16 parity and fp16 guard gates, pre-registration (`docs/research/PREREG_baseline.md`), and a single-shot locked-test gate (`configs/eval/test_locked.yaml`: 1364 EN+HI rows x `base`, `refuse_everything`, `frozen_repair`).

Scope: dev-split results with automated judges and machine-translated Hindi/Hinglish prompts; the frozen locked test is packaged as a single-shot gated notebook and its scoring is the next milestone.

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
