# Kaggle GPU run: instructions for an agent

Use this when an agent (with a browser and/or a shell on Vedant's machine) runs the GPU parity and fixture-generation notebook for Vedant. Free tier only, on Vedant's existing Kaggle account. No new accounts, no paid upgrades. Source of truth: `docs/GPU_RUNBOOK.md`, prereg D4 in `docs/research/PREREG_baseline.md`, and `notebooks/gpu_parity_and_baseline.ipynb` on `main`. Verify the notebook's prereg source hash before launch. This is not the Nov 8 scored dev baseline. Leave `CURATED_PROMPTS_PATH` unset.

## Current evidence and development boundary (Nov 7)

- Development review is limited to `data/staging/review/vedant_review_nov7_calibration_dev.csv` (26 rows). The mixed historical review queue contains 33 test rows; do not open or print its prompt content. The family-split checker reports identity/count invariants only.
- Public HinGE quality ratings and IndicGuard source labels are not accepts for our finance translations and are not human labels of our generated responses. Keep translation admission and response calibration separate.
- The exposure of frozen family `fam_983e49e2566f` is unresolved. Preserve the frozen manifest and required exposure ledger/guard. A separately reviewed prospective resolution is required before the single Nov 11 frozen-arm test pass. No earlier test generation, scoring or tuning. Nov 8 H1 is dev-only.
- PR21 preserves v8 raw reports and hashes. CPU bf16 meets the provisional D4 variant rule, while short mean KL exceeds the original `1e-4` gate. Original fp32 short-plus-4196 credit is incomplete, so weight-backed Locate/Repair acceptance remains blocked. D5 and D6 are explicit operational deviations; no threshold changes after observing v8.
- PR22's pinned WildGuard fixture proves English load/parsing only. Both responses are safe. Judge accuracy, an unsafe-response positive control and multilingual human calibration remain unverified. No fixture result is ASR.
- Review the existing evidence before launching another job. Any new launch needs a concrete unresolved runtime question and the pre-run identity pins, not an unchanged council loop.

## Before you start

- Kaggle account must be Vedant's existing one and GPU-eligible (Settings shows phone verification). If it is not verified, stop and ask Vedant; verification is his step.
- The Hugging Face account behind HF_TOKEN must have accepted the CohereLabs/tiny-aya-global and tiny-aya-base terms (done on Nov 6).
- Never print, log, commit, or paste HF_TOKEN or the Kaggle API key into a cell, file, PR, or chat.
- Quota: about 30 GPU hours per week; a single session is capped at about 12 hours. Expected run: 60 to 240 minutes.

## Path A: Kaggle website (simplest, needed at least once for the secret)

1. Sign in at https://www.kaggle.com with Vedant's account (Vedant does the sign-in or 2FA if asked).
2. Create > New Notebook. Then File > Import Notebook, and import from GitHub
   `Vedant817/hinglish-shield`, path `notebooks/gpu_parity_and_baseline.ipynb` on `main`
   (or upload the .ipynb file from a local clone).
3. Right panel, Session options: Accelerator = GPU T4 x2. Internet = On (needed for pip and HF downloads).
4. Add-ons > Secrets > Add secret: label `HF_TOKEN`, value = Vedant's HF token. Tick the checkbox to attach it to this notebook. Vedant should type the value himself if the agent does not already hold it securely.
5. Run All (or Save Version > Save & Run All to run headless and survive a closed tab; recommended for long runs).
6. When it finishes, open the Output tab (or `/kaggle/working/`) and download `hs_gpu_artifacts_<run_id>.zip`.

## Path B: Kaggle CLI (for re-runs from a shell)

Verify flags with `kaggle kernels --help` against the installed version; docs at https://github.com/Kaggle/kaggle-api.

1. Install Kaggle in an external environment. On this Windows machine, CLI 2.2.5 is installed at `../hinglish-shield-scratch/.venv-ops/Scripts/kaggle.exe`. Its first import needed `six` installed separately. `kaggle --version` and `kaggle kernels --help` work.
2. Test existing authentication with `kaggle kernels list --mine`. Do not inspect credential files or print tokens. This machine authenticated successfully on Nov 6. If authentication is missing, Vedant performs `kaggle auth login` on his existing account, or supplies a credential through the tool's secure secret input.
3. Make a push folder outside the repo, for example `/tmp/hs-kaggle/`, and copy the notebook in.
4. `kaggle kernels init -p /tmp/hs-kaggle` and edit `kernel-metadata.json`:
   ```json
   {
     "id": "<kaggle-username>/hinglish-shield-gpu-parity",
     "title": "hinglish-shield-gpu-parity",
     "code_file": "gpu_parity_and_baseline.ipynb",
     "language": "python",
     "kernel_type": "notebook",
     "is_private": true,
     "enable_gpu": true,
     "enable_internet": true,
     "dataset_sources": [],
     "competition_sources": [],
     "kernel_sources": []
   }
   ```
   Keep `is_private: true` (publishing anything public needs Vedant's OK).
5. First save without execution: `kaggle kernels push -p /tmp/hs-kaggle --no-run`. CLI 2.2.5 supports this flag. Keep the notebook private.
6. Secrets cannot be set from the CLI. Open the notebook on kaggle.com, attach `HF_TOKEN` (Path A step 4), verify existing-account phone verification and free GPU availability, and select GPU T4 x2. Then use Save & Run All. For a later CLI run after these checks, `kaggle kernels push -p /tmp/hs-kaggle --accelerator NvidiaTeslaT4` requests GPU T4 x2. Confirm the secret remains attached. Do not launch merely to discover a missing secret.
7. Poll: `kaggle kernels status <kaggle-username>/hinglish-shield-gpu-parity` (wait for complete or error; do not poll faster than every few minutes).
8. Fetch outputs: `kaggle kernels output <kaggle-username>/hinglish-shield-gpu-parity -p /tmp/hs-kaggle/out`

## Filing the results in the repo

```bash
cd hinglish-shield && git checkout main && git pull
git checkout -b runs/gpu-<run_id>
mkdir -p runs/gpu-<run_id>
python scripts/ingest_gpu_artifacts.py /path/to/hs_gpu_artifacts_<run_id>.zip --run-id gpu-<run_id>
```

- The importer rejects unexpected files, path traversal, symlinks, duplicate filenames, oversized files, nonfinite JSON, model weights and known token patterns before writing. It verifies ZIP CRC, hashes copied files, requires fixture labels for predictions/metric stubs, and recomputes the original short+long parity gate. Missing long output is incomplete even when a source JSON says pass. This is artifact inspection, not authentication of the producing machine or proof of a scored baseline. Review `ingestion.json` and the raw artifacts before committing. Imports never overwrite an existing directory.
- If the notebook wrote a manifest with hashes, verify it (for parity copies use `python scripts/verify_parity_copy.py` as described in the runbook).
- Read the results honestly and write them in `LOG.md`:
  - Parity: short and long KL must each be < 1e-4 to pass. If the 4196-token long case did not run, the result is `incomplete`, not a pass. Record fp32 vs bf16 fallback.
  - fp16 guard: any NaN or inf aborts. Record max diff and agreement rate; if it fails, scored T4 runs use fp32 (deviation D2).
  - Guard outputs are `artifact_kind: setup-smoke`; generation outputs are `artifact_kind: fixture`, all `split: example`. Never report them as ASR or dev results. The bf16 gate is a separate, possibly provisional variant; use D4's floor/provisional rules, not the original fp32 threshold.
- Open a PR titled `Runs: Kaggle GPU parity <run_id>` for independent artifact and protocol review. Green CI or Kilo alone is not acceptance. Do not merge merely to complete an audit. Update the Task Board with the actual gate status, including incomplete/provisional evidence.

## If something fails

- Out of memory: genuine allocation failures enable the CPU bf16 variant on a T4. Auth and logic errors do not. D5 refuses the dual-resident fp32 conversion when available RAM or total RAM is under 32 GiB, so a Kaggle host restart does not kill the process before that variant starts. If the host still kills the kernel, report incomplete and do not invent a KL. Version 6 died during the bf16 `from_pretrained` call while the donor was resident, after D5 had already skipped fp32. D6 loads that TL copy with `from_pretrained_no_processing` and no live donor. A later host kill is still incomplete. Do not cut a long case that actually ran. A missing long case stays incomplete.
- NaN or inf guard abort: do not retry in fp16; use fp32 and log deviation D2.
- 401 or 403 from Hugging Face: the secret is missing, unattached, or the HF account lost model access. Check Add-ons > Secrets; ask Vedant if access is the problem.
- GPU quota exhausted or no GPU offered: stop and tell Vedant the reset time Kaggle shows. Do not move to a paid provider or create accounts.
- Session timeout near 12 hours: use Save & Run All (headless) and re-run; keep partial outputs labeled incomplete.
