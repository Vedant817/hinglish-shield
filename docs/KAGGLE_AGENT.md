# Kaggle GPU run: instructions for an agent

Use this when an agent (with a browser and/or a shell on Vedant's machine) runs the GPU parity and baseline notebook for Vedant. Free tier only, on Vedant's existing Kaggle account. No new accounts, no paid upgrades. Source of truth for what the notebook does: `docs/GPU_RUNBOOK.md` and `notebooks/gpu_parity_and_baseline.ipynb` on `main` (final notebook commit 909b84f or later).

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

1. `pip install --upgrade kaggle`
2. API credentials: Vedant creates a token at https://www.kaggle.com/settings (API section). Save it to `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME` and `KAGGLE_KEY` in the shell. Never commit it.
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
5. `kaggle kernels push -p /tmp/hs-kaggle`
6. Secrets cannot be set from the CLI. After the first push, open the notebook on kaggle.com once, attach the `HF_TOKEN` secret (Path A step 4), and set the accelerator to GPU T4 x2 if it did not pick it up, then save a version. Later CLI pushes to the same id reuse the notebook; confirm the secret is still attached if the run fails with a 401 or 403.
7. Poll: `kaggle kernels status <kaggle-username>/hinglish-shield-gpu-parity` (wait for complete or error; do not poll faster than every few minutes).
8. Fetch outputs: `kaggle kernels output <kaggle-username>/hinglish-shield-gpu-parity -p /tmp/hs-kaggle/out`

## Filing the results in the repo

```bash
cd hinglish-shield && git checkout main && git pull
git checkout -b runs/gpu-<run_id>
mkdir -p runs/gpu-<run_id>
unzip /path/to/hs_gpu_artifacts_<run_id>.zip -d runs/gpu-<run_id>/
```

- Check that no model weights, caches, or tokens landed in the folder (`rg -i "hf_[A-Za-z0-9]{20,}" runs/gpu-<run_id>` must find nothing; remove any `*.safetensors`, `*.bin`).
- If the notebook wrote a manifest with hashes, verify it (for parity copies use `python scripts/verify_parity_copy.py` as described in the runbook).
- Read the results honestly and write them in `LOG.md`:
  - Parity: short and long KL must each be < 1e-4 to pass. If the 4196-token long case did not run, the result is `incomplete`, not a pass. Record fp32 vs bf16 fallback.
  - fp16 guard: any NaN or inf aborts. Record max diff and agreement rate; if it fails, scored T4 runs use fp32 (deviation D2).
  - Baseline cells on fixtures are `artifact_kind: fixture`, never ASR results.
- Open a PR titled `Runs: Kaggle GPU parity <run_id>` and merge on green CI (standing permission). Update the Notion Task Board item for the Nov 9 Locate gate.

## If something fails

- Out of memory: the notebook falls back to bf16 for parity automatically; record it. If it still dies, restart the session and run again; do not cut the long case silently.
- NaN or inf guard abort: do not retry in fp16; use fp32 and log deviation D2.
- 401 or 403 from Hugging Face: the secret is missing, unattached, or the HF account lost model access. Check Add-ons > Secrets; ask Vedant if access is the problem.
- GPU quota exhausted or no GPU offered: stop and tell Vedant the reset time Kaggle shows. Do not move to a paid provider or create accounts.
- Session timeout near 12 hours: use Save & Run All (headless) and re-run; keep partial outputs labeled incomplete.
