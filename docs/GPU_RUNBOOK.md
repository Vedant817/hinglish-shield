# GPU runbook (free tier only)

## Platform: Kaggle Notebooks (GPU T4 x2)

**Why not Colab free:** CircuitKIT original Tiny Aya parity
(`tests/backends/tl_compat/test_cohere_parity.py`) loads two float32 copies
(~12.5 GiB each, ~25 GiB weights). The HF copy stays on CPU. Colab free system
RAM is typically ~12 to 13 GB, which cannot hold that CPU copy. Kaggle free GPU
sessions typically expose ~29 to 30 GB system RAM plus T4 GPUs (16 GB VRAM each),
so the original gate is at least plausible there. Quota is about 30 GPU hours per
week (Kaggle efficient-GPU docs).

**Still a risk on Kaggle:** TL float32 on one T4 (~12.5 to 13.4 GiB weights) plus
the 4196-token sliding-window forward can OOM. The notebook records an exact skip
reason if preflight RAM/VRAM are insufficient, or captures the pytest failure if
it runs and OOMs. Do not treat a skip or a reduced variant as the original gate.

## What Vedant must do (under 6 steps)

1. Sign in to an existing Kaggle account (phone-verified if required for GPU). No new accounts, no paid upgrades.
2. Create a notebook (or upload `notebooks/gpu_parity_and_baseline.ipynb`) and set **Settings → Accelerator → GPU T4 x2**.
3. Add secret `HF_TOKEN` (Add-ons → Secrets) for the Hugging Face account that already accepted Tiny Aya terms. Never paste the token into a cell.
4. Optional: attach a curated prompts JSONL and set `CURATED_PROMPTS_PATH` in the config cell. Otherwise the notebook uses a labeled FIXTURE set only (not metrics).
5. Click **Run All**. Expected wall time: roughly 45 to 120 minutes (install + downloads + optional original parity + fp16 guard + short baseline). Stop early only if a cell hard-fails.
6. Download `hs_gpu_artifacts_<run_id>.zip` from `/kaggle/working/` and copy it into the repo as `runs/<run_id>/` (unzip so manifests and `predictions.jsonl` are visible). Commit on a data/results branch if desired; never commit tokens or model weights.

## Cells overview

| Section | Purpose |
|---|---|
| Config | Pins, flags, proposed fp16 thresholds |
| Env report | nvidia-smi, RAM, versions |
| Secrets | Load `HF_TOKEN` without printing |
| Install | CircuitKIT `@e71d5e1` (0.2.0), SafeTune `@3901a9f` (0.1.12), transformers 5.22.1 |
| Original parity | Runs upstream pytest only if RAM/VRAM pass preflight |
| Proposed variant | `RUN_PARITY_VARIANT=False` by default; bf16 sequential KL, logged as gate change |
| fp16 guard | Required before scored T4 baseline; fail → fp32 baseline (deviation D2) |
| Baseline | Tiny Aya Global revision `af89d219...`; fixture or curated JSONL |
| Zip | Pack artifacts for download |

## Proposed fp16 guard thresholds (not yet Lead-approved)

- `FP16_MAX_ABS_LOGIT_DIFF = 0.05`
- `FP16_MIN_GREEDY_AGREEMENT = 0.95`
- Abort on any NaN/Inf

## Proposed reduced parity variant (for Lead review)

- **Change:** load HF and TL one at a time in bf16, cache next-token log-probs on CPU, compute the same KL(HF||TL) formula.
- **Memory:** peak ≈ one 6.24 GiB copy (+ tiny logit cache) vs ≈ 25 GiB for two fp32 copies.
- **Interpretation:** diagnostic evidence only. Never replace the original KL < 1e-4 float32 dual-copy gate until the Lead promotes it.

## Bringing artifacts home

```bash
# after download
mkdir -p runs/gpu-<run_id>
unzip ~/Downloads/hs_gpu_artifacts_<run_id>.zip -d runs/gpu-<run_id>/
# expect parity_original.json, fp16_guard.json, predictions.jsonl, baseline_manifest.json, versions.json
```

Append a short IST note to `LOG.md` pointing at that directory. Do not invent metric numbers from fixtures.
