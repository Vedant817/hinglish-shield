# Hinglish Shield

Multilingual and code-mixed (Hindi, Hinglish, English) safety audit and repair pipeline for [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).

Lexsi AI Safety and Alignment Hackathon, Track 2 (AI Red Teaming, Safety and Alignment). Submission due Nov 14, 2026 AoE.

## Status

Measured attack success rate, over-refusal, capability, and repair acceptance: **Not run** (no human-calibrated, scored baseline). See [docs/RESULTS.md](docs/RESULTS.md) for current descriptive results and locked-test placeholders.

The tracked `metrics.json` files are `artifact_kind: fixture` or descriptive/non-scored evidence with null primary benchmark fields where applicable (for example `runs/_example/metrics.json` and `runs/wildguard-fixture-20261107T141335276834Z/metrics.json`). Fixture generations are not ASR. Automated-judge outputs are labeled "automated-judge only, human calibration pending" with `metrics_eligible: false` and `primary_asr: null`.

**Dev repair (v21, run 20261111-180320-270579d4, dev split only): clean null, no safety claim.** The frozen provisional bf16 steering candidate (`candidate_alpha_1`, alpha 1.0, MLP 0 / `blocks.0.hook_mlp_out` equivalent, v18 calibration direction `d5ec351a...`, freeze `6fc30b1...`) did not separate from `random_norm_matched_seed_7` or `base`: Wilson 95% intervals overlap under both WildGuard (primary) and Qwen3Guard (secondary) on harm unsafe rate and benign over-refusal. `alpha_zero` matched `base` byte-for-byte on all 885 dev rows (0 mismatches). Inter-judge agreement was modest (Cohen's kappa around 0.55, all-dev scored rows). This is automated-judge only, dev-only, hi/hinglish rows are unreviewed machine translations, and the arm is a provisional bf16 research arm chosen on calibration logit effect only. No safety-effect or circuit-superiority claim. Details: [docs/RESULTS.md](docs/RESULTS.md), [runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.md](runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.md).

**Locked test (frozen EN+HI primary, single shot): PLACEHOLDER, not run.** Hinglish test extension (descriptive secondary, approved amendment): **PLACEHOLDER, not run**. No locked-test or extension numbers exist. A prepared notebook version (v22, `--no-run`) is saved at <https://www.kaggle.com/code/vedant817/hinglish-shield-gpu-parity-and-baseline> for one guarded Save & Run All on GPU T4 x2; it has not been executed.

Original fp32 parity remains incomplete/not credited; provisional bf16 Locate parity is not original fp32 credit. The locked test remains single-shot and gated (never run locally).

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
