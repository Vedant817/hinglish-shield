# Submission checklist (Lexsi Stage 1, due Nov 15, 2026, 11:59 PM AoE)

Form: <https://hackathon.lexsi.ai/submit> (Tally form `o9E0Ve`). Paste-ready answers: `SUBMISSION_PACK.md` (kept outside the repo).

## Ready
- [x] Public GitHub repo with README reproducibility notes: <https://github.com/Vedant817/hinglish-shield>
- [x] Write-up following the Lexsi layout: `docs/ARTICLE.md`
- [x] Results with Wilson CIs and kappa: `docs/RESULTS.md`, `runs/gpu-20261111-180320-270579d4/`
- [x] Notebook in repo: `notebooks/gpu_parity_and_baseline.ipynb`
- [x] Frozen splits, pre-registration, locked-test config: `data/splits/family_splits_v2.json`, `docs/research/PREREG_baseline.md`, `configs/eval/test_locked.yaml`

## Needs Vedant (required form fields)
- [ ] Public HF model collection (steering direction + freeze manifest, plus links to base and judge models): `scripts/publish_hf_collections.py`
- [ ] Public HF dataset collection (dev splits, generations, judge scores, run logs): same script
- [ ] HF article (or use the GitHub write-up link `docs/ARTICLE.md`, which the FAQ allows)
- [ ] Submit the form with the registered email
