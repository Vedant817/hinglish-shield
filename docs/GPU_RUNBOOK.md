# GPU runbook (free tier only, prereg D4)

## Platform: Kaggle Notebooks (GPU T4 x2)

**Why not Colab free:** One Tiny Aya float32 copy is about 13 GiB.
Colab free system RAM is typically about 12 to 13 GB, so it cannot hold that copy.
Kaggle free GPU sessions typically expose about 29 to 30 GB RAM plus T4 GPUs.
Quota is about 30 GPU hours per week (Kaggle efficient-GPU docs).

Conversion is a separate, higher peak. `HookedTransformer.from_pretrained(hf_model=...)`
holds the HF donor and the TL model at the same time: about 25 GiB plus allocator
overhead. A sequential forward is one resident model. The long HF forward must not
run while TL is still resident. See prereg D4 in `docs/research/PREREG_baseline.md`.

## What Vedant must do (under 6 steps)

1. Sign in to an existing Kaggle account (phone-verified if required for GPU). No new accounts, no paid upgrades.
2. Create a notebook (or upload `notebooks/gpu_parity_and_baseline.ipynb`) and set **Settings → Accelerator → GPU T4 x2**.
3. Add secret `HF_TOKEN` (Add-ons → Secrets) for the Hugging Face account that already accepted Tiny Aya terms. Never paste the token into a cell.
4. Leave `CURATED_PROMPTS_PATH` unset. This notebook refuses a curated JSONL. Generations here are fixtures (`artifact_kind: fixture`, `split: example`). A scored curated run belongs in a separate pipeline. `max_new_tokens` 64 is fixture length only. Scored generation stays at 256.
5. Click **Run All**. Expected wall time: roughly 60 to 240 minutes (install + sequential fp32 short + fp32 CPU long-prompt 4196-token forward + fp16 guard + short fixture baseline). The long-prompt fp32 CPU forward alone is a rough estimate of about 20 to 90 minutes (planning guess only, not a measurement). Conversion can memory-fail near 25 GiB plus overhead even when one copy would fit. Stop early only if a cell hard-fails.
6. Download `hs_gpu_artifacts_<run_id>.zip` from `/kaggle/working/` and copy it into the repo as `runs/<run_id>/` (unzip so manifests are visible). Never commit tokens or model weights.

## Gate details (prereg D4)

Frozen in the notebook config cell marked **prereg D4, frozen before run**.
D4 was written before any GPU run. Provenance is `PREREG_SOURCE_SHA256` (LF bytes of
`docs/research/PREREG_baseline.md`) and the protocol dict hash the notebook computes.
There is no git commit pin. An earlier draft cited a commit id that is not in this repository.

### Parity (Locate credit)

1. **DEFAULT / ORIGINAL (load-order change only):** sequential fp32 on CPU.
   Load HF fp32, cache detached clones of the next-token log-probs for the four
   prompts from `test_cohere_parity.py`, release the HF global by name, then load
   TL. Same KL(HF||TL) formula, threshold **KL < 1e-4**.
   Record `load_order: ["hf_fp32","tl_fp32"]`, dtype, hardware, RAM peak.
   One forward-resident copy is about **13 GiB**. Conversion peaks at about **25 GiB plus overhead**.
   The long cell releases any resident TL, runs the HF 4196 forward alone, releases HF, then loads TL again.
   Short and long KL are reported separately. The gate passes only if **both** are < 1e-4.
   If the long case does not run, status is `incomplete`. A missing long case is never a pass.
   `t_short0` is set before the short attempt, so a failure still records `wall_time_short_s`.
   Release uses `release_named` on the kernel globals. `del` of a function-local alias does not drop those globals.
   Provenance of the copied prompts, KL source, and long spec: hashes checked by `scripts/verify_parity_copy.py` against `vendor/CircuitKIT` at the recorded CircuitKIT SHA. Those three hashes are unchanged.
2. **Optional dual-resident pytest** (`RUN_ORIGINAL_PYTEST=False`): holds two
   fp32 copies for the whole test (about 25 GiB). Off by default.
3. **bf16 variant** (`gate_change`): auto-enabled only after a genuine memory failure
   (or `FORCE_PARITY_VARIANT`). Auth errors and other logic errors do not enable it.
   CPU bf16 only, and only when the accelerator name contains T4. CUDA bf16 is refused.
   Short accept rule: KL(HF bf16, TL bf16) <= 2 * floor, floor = KL(HF fp32, HF bf16)
   on the same prompt, and top-1 >= 0.99.
   If that floor cannot be measured: provisional mean KL < 1e-2 and top-1 >= 0.98.
   Long bf16 does **not** use the fp32 1e-4 rule. A measured long floor would use
   KL <= 2 * that floor. This notebook does not manufacture a long floor with another
   fp32 4196 forward, and it does not reuse the short floor. Predeclared provisional
   long rule when the floor is unavailable: that long KL < 1e-2. Missing long stays `incomplete`.

### fp16 guard (before any later scored T4 baseline; supersedes D1 numeric thresholds)

- Reference: fp32 on CPU if available RAM is at least 14 GiB. Otherwise bf16 on CPU, and only on a T4. Record which. If RAM is short and the GPU is not a T4, fail closed and use fp32 for the baseline.
- Candidate: fp16 on CUDA.
- Residual-stream hooks on every decoder layer, removed in `finally`. Absent or empty hooks fail closed.
- All 20 fixture prompts must have finite logits.
- Log `max_abs_logit_diff`. It is not a gate.
- PASS requires all of: hooks attached, 20/20 finite, mean next-token KL <= 1e-2, top-1 agreement >= 0.98, exact match on the first 32 greedy tokens for at least 17 of 20 prompts.
- FAIL => baseline uses fp32 on the T4 and logs deviation **D2**.
- Guard JSON is `artifact_kind: setup-smoke`, `split: example`. These 20 strings are hand-written fixtures, not the frozen calibration split.

Every result JSON records dtype, load_order, hardware, and artifact_kind.

## Bringing artifacts home

```bash
python scripts/ingest_gpu_artifacts.py ~/Downloads/hs_gpu_artifacts_<run_id>.zip --run-id gpu-<run_id>
```

Append a short IST note to `LOG.md`. Do not invent metric numbers from fixtures or from this setup-smoke guard.
