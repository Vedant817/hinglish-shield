# Hinglish Shield pitch (Lexsi Track 2)

## One line
Tiny Aya Global refuses 69% of harmful requests in English but only 24% in Hinglish; Hinglish Shield is the pre-registered, dual-judge pipeline that measures this gap reproducibly and shows that a single steering direction does not close it.

## Problem
Hundreds of millions of Indian users type Hindi in Latin script mixed with English. Small open models are what gets deployed for them, and their safety is measured almost only in English. If refusal weakens in Hinglish, English-only evals overstate deployment safety.

## Approach
- CuratorKIT: clean, dedup, gate and export EN harm and benign sets (Aya Red-Teaming, HarmBench, XSTest, OR-Bench); native Hindi from Aya Red-Teaming; Hinglish rewrites filtered by script checks (473 to 118).
- Frozen family-level splits (`data/splits/family_splits_v2.json`) so translations never leak across splits; single-shot locked-test gate.
- AuditKIT: two pinned judges (WildGuard primary, Qwen3Guard secondary), Wilson 95% intervals, Cohen's kappa.
- CircuitKIT: difference-of-means refusal direction at `blocks.0.hook_mlp_out`, applied as inference-time steering, frozen by sha256 before evaluation, with alpha-zero and norm-matched random controls.

## Evidence (dev split, `runs/gpu-20261111-180320-270579d4/`)
- Harmful-prompt refusal, base, WildGuard: EN 69.3% (205/296), HI 54.2% (150/277), Hinglish 23.9% (16/67); Qwen3Guard 68.2% / 61.7% / 29.9%.
- Judges diverge on Hindi: WildGuard 0.7% unsafe vs Qwen3Guard 6.1%.
- Steering: harm unsafe 7.0% (45/640) vs base 6.7% (43/640) vs random 7.3% (47/640), overlapping CIs under both judges; alpha-zero byte-identical to base on 885/885 rows.

## 3 minute demo script outline
1. 0:00-0:30 The gap: EN vs HI vs Hinglish refusal table.
2. 0:30-1:00 Data: seeds, CuratorKIT, Hinglish rewrite filter, frozen family splits.
3. 1:00-1:40 Measurement: two judges, kappa, why WildGuard alone would call Hindi safe.
4. 1:40-2:20 Repair: CircuitKIT refusal direction, freeze, controls, the controlled negative result.
5. 2:20-3:00 What it means and next: language-conditioned directions, SafeTune train-time hardening on Hinglish, locked-test scoring.

## Call to action
Use the harness to test any Indic safety fix: it will tell you, with controls, whether the fix actually moves Hinglish refusal.
