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

### Launch validation follow-up (2026-11-07, before model execution)

The copied prompts, KL formula and acceptance thresholds remain unchanged.
The notebook now rejects negative, boolean, string and non-finite KL evidence,
including measured bf16 floors. A supplied floor must cover the same prompt keys;
invalid measured floors fail instead of falling back to the provisional rule.
Zero is valid. Values are never clipped. The importer already rejects negative KL.

All parity tokenizer, HF donor and TL configuration loads use
`CohereLabs/tiny-aya-base@d62ac17e241c871fc52e9a95d7b0e2615189d7d5`.
TL receives the same pinned tokenizer used for the HF inputs, rather than fetching
another tokenizer from a moving branch. Original and variant JSON record the pin.
This is the parity checkpoint; the fixture generation checkpoint remains Global.

The install cell resolves Transformers, the shared HF Hub pin and both pinned
Lexsi libraries in one pip transaction. HF Hub is pinned to `1.33.0`, within
CircuitKIT's declared `>=1.5,<2` range and Transformers `5.22.1`'s declared
`>=1.31.0,<3.0` range. Published dependency metadata supports that intersection;
it does not prove the complete environment installs or imports on Kaggle.
The cell records the resolved Python, Torch, Transformers, HF Hub, TransformerLens
and Accelerate versions. It preserves the global pip check in
`dependency_check_global.json` and checks every active transitive requirement
reachable from the explicit notebook package roots in `dependency_check.json`.
Requested extras are included; missing packages, incompatible versions and
unverified direct-URL dependencies fail closed. Installed CircuitKIT and SafeTune
VCS commits must match their pins. Unrelated preinstalled packages are outside
this scoped check. Do not use `--no-deps` to bypass dependency resolution.

The first executed dependency preflight is preserved in
`runs/kaggle-dependencies-2026-11-07-failed-1/`. Installation succeeded, then
global pip check failed on the base environment and imports were skipped. It
does not verify imports or model execution. That observed failure motivated the
explicit dependency scope above; the original failure is retained rather than
relabeled a pass.

The successor is retained in `runs/kaggle-dependencies-2026-11-07-imports/`.
Its scoped dependency check, installed Lexsi VCS revision checks and imports
completed successfully on Kaggle. Global diagnostics still failed and are saved.
The import subprocess detected one Tesla T4; do not infer two GPUs from the
requested accelerator or the older hardware preflight. No CUDA tensor operation,
model weights, inference, parity, dtype guard or judged baseline was executed.

The model-free regression suite and metadata checks do not establish a successful
Kaggle install, a short or long parity pass, a dtype guard pass, or a scored run.

For a dependency-only check before attaching a token, upload
`scripts/kaggle_dependency_preflight.py` as a private script kernel on the existing
free account. It fetches the immutable notebook version recorded in its constants,
verifies the notebook bytes, then runs the same install list, dependency check and
compatibility import order. It receives no HF token, opens no prompt dataset,
loads no weights and runs no inference. Download its setup-smoke ZIP, including
failure logs. A dependency preflight pass is only environment evidence and cannot
grant parity, Locate or judge credit. If the notebook install pins change, prepare
a new preflight against the new committed notebook and preserve the older run.

## Bringing artifacts home

```bash
python scripts/ingest_gpu_artifacts.py ~/Downloads/hs_gpu_artifacts_<run_id>.zip --run-id gpu-<run_id>
```

Append a short IST note to `LOG.md`. Do not invent metric numbers from fixtures or from this setup-smoke guard.
