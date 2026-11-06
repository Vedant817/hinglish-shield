# GPU runbook (free tier only, prereg D4)

## Platform: Kaggle Notebooks (GPU T4 x2)

**Why not Colab free:** Sequential original parity needs one Tiny Aya float32
copy on CPU (~12.5 GiB). Colab free system RAM is typically ~12 to 13 GB.
Kaggle free GPU sessions typically expose ~29 to 30 GB RAM plus T4 GPUs.
Quota about 30 GPU hours per week (Kaggle efficient-GPU docs).

## What Vedant must do (under 6 steps)

1. Sign in to an existing Kaggle account (phone-verified if required for GPU). No new accounts, no paid upgrades.
2. Create a notebook (or upload `notebooks/gpu_parity_and_baseline.ipynb`) and set **Settings → Accelerator → GPU T4 x2**.
3. Add secret `HF_TOKEN` (Add-ons → Secrets) for the Hugging Face account that already accepted Tiny Aya terms. Never paste the token into a cell.
4. Optional: attach a curated prompts JSONL and set `CURATED_PROMPTS_PATH` in the config cell. Otherwise the notebook uses a labeled FIXTURE set only (not metrics).
5. Click **Run All**. Expected wall time: roughly 45 to 150 minutes (install + sequential fp32 parity + fp16 guard + short baseline). Stop early only if a cell hard-fails.
6. Download `hs_gpu_artifacts_<run_id>.zip` from `/kaggle/working/` and copy it into the repo as `runs/<run_id>/` (unzip so manifests are visible). Never commit tokens or model weights.

## Gate details (prereg D4, commit 99a3829)

Frozen in the notebook config cell marked **prereg D4, frozen before run**.

### Parity (Locate credit)

1. **DEFAULT / ORIGINAL (load-order change only):** sequential fp32 on CPU.
   Load HF fp32, cache next-token log-probs for the four prompts from
   `test_cohere_parity.py`, unload (`del` / `gc` / `empty_cache`), load TL fp32,
   compute the same KL(HF||TL) formula, threshold **KL < 1e-4**.
   Record `load_order: ["hf_fp32","tl_fp32"]`, dtype, hardware, RAM peak.
   Peak weights about **13 GiB**.
2. **Optional dual-resident pytest** (`RUN_ORIGINAL_PYTEST=False`): holds two
   fp32 copies (~25 GiB). Off by default; will not fit free-tier RAM.
3. **bf16 variant** (`gate_change`): auto-enabled only if sequential fp32
   OOMs or cannot run. Accept when KL(HF bf16, TL bf16) <= 2 * floor with
   floor = KL(HF fp32, HF bf16) on the same prompts, and top-1 >= 0.99.
   If floor cannot be measured: provisional mean KL < 1e-2 and top-1 >= 0.98.

### fp16 guard (before scored T4 baseline; supersedes D1 thresholds)

- Reference: fp32 on CPU if RAM allows (~29 GB on Kaggle => yes), else bf16. Record which.
- Candidate: fp16 on CUDA.
- Abort on NaN/Inf in logits **or** residual stream (forward hooks on each decoder layer; removed in `finally`).
- Log `max_abs_logit_diff` but **do not** gate on it.
- PASS requires all of: mean next-token KL <= 1e-2, top-1 agreement >= 0.98,
  exact match on the first 32 greedy tokens for at least 17 of 20 prompts.
- FAIL => baseline uses fp32 on the T4 and logs deviation **D2**.

Every result JSON records dtype, load_order and hardware.

## Bringing artifacts home

```bash
mkdir -p runs/gpu-<run_id>
unzip ~/Downloads/hs_gpu_artifacts_<run_id>.zip -d runs/gpu-<run_id>/
```

Append a short IST note to `LOG.md`. Do not invent metric numbers from fixtures.
