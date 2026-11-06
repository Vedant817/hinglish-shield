# Hinglish Shield handoff (2026-11-06 IST)

For Vedant and any fresh coding agent taking over from the LX bots.

No em dashes in this file.

## Current main

- Repo: https://github.com/Vedant817/hinglish-shield
- Branch: `main`
- Tip SHA at handoff write time: `a6f5b60b98af12915041a4766359c5aac4e4b165` (verify with `git rev-parse HEAD` after pull; the HANDOFF PR may add one commit on top)
- Notion HQ: https://app.notion.com/p/40f047502f5881d4aa86d150111b9355
- Task Board: https://app.notion.com/p/033ead4a50f24e6e98105c33d6b646e6
- Agent Playbook: https://app.notion.com/p/40f047502f58811c9d99f53600cb6ab0

## Merged today (verify on GitHub)

| PR | Title | Notes |
| --- | --- | --- |
| #4 | ADR 0007 + upstream status | Already on main before handoff sweep |
| #5 | Eval schema stubs | Already on main |
| #9 | Research H1 prereg + compute options | Merged |
| #7 | Family splits v2 (calibration/dev/test) | Merged before #6; Research Lead signed off |
| #6 | Translations family_id/split + Aya sample50 + review sheet | CRITICAL string bug in `export_to_curatorkit.py` fixed before merge |
| #8 | Eval prereg stats, test_locked.yaml | WARNING on `failure_mass` fixed; rebased onto main |
| #10 | Kaggle GPU parity notebook + GPU_RUNBOOK | Draft marked ready; tip includes D4 sequential fp32 + 4196-token long gate + parity-copy verify. Notebook had 0 outputs and no token literals |
| #11 | Setup-smoke Tiny Aya CPU artifacts | Merged after syncing with main |

Open PRs at handoff: none.

## Local verification on this box (2026-11-06 IST)

Run in a fresh venv (`python3 -m venv .venv && source .venv/bin/activate && make install && pip install pytest`):

- `python3 scripts/smoke.py` -> OK: layout present, example metrics validate, packages import, stats ok
- `python3 -m pytest tests/ -q` -> 15 passed, 1 skipped
- `python3 scripts/check_translation_family_splits.py` -> PASS (batch01 2/4/24; aya sample50 12/11/27; review sheet 14/12/33; 0 unmapped). Manifest `family_splits_v2.json` sha256 `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`

These are pipeline integrity checks, not ASR or language-gap results.

## What is done

- Repo skeleton, metrics schema, LOG.md, WORKSPACE.md
- Seed pull, EN CuratorKIT v0 path, issue-42 local mitigation noted
- Family-level splits v2 (seed 42; XSTest out of calibration; test frozen for Nov 11 pass)
- Staging translations batch01 + Aya Hinglish sample50 + Vedant Nov 7 review sheet with family_id/split filled
- Eval harness stubs, prereg ASR denominator, Wilson + family bootstrap helpers, `configs/eval/test_locked.yaml`
- H1 preregistration (`docs/research/PREREG_baseline.md`, `configs/research/baseline_prereg.yaml`) and compute options memo
- ADR 0007 finance workload gates
- Free-tier Kaggle GPU notebook + runbook (prereg D4)
- Setup-smoke logs under `runs/setup-smoke-2026-11-06/` (no secrets)
- Standing permission from Vedant: LX bots may push, open, and merge in this repo (no secrets, no vendored libs, no venvs in git). Paid compute, new accounts, other-repo posts, and HF publishing still need Vedant's OK

## What is open (priority)

1. **Vedant: Kaggle GPU run** using `notebooks/gpu_parity_and_baseline.ipynb` on main. Follow `docs/GPU_RUNBOOK.md`; agents use `docs/KAGGLE_AGENT.md` (web and CLI steps, filing results, failure handling). Free tier only. Expect about 60 to 240 minutes wall time; the long fp32 4196-token case alone is a rough estimate of about 20 to 90 extra minutes. Download `hs_gpu_artifacts_*.zip` into `runs/`. Fixtures are not scored metrics.
2. **Vedant: Hindi and Hinglish review** (Task Board Human P0, Nov 7): https://app.notion.com/40f047502f588116ab55c2304137b52d and review sheet `data/staging/review/vedant_review_nov7.csv`. Related value task: https://app.notion.com/40f047502f58817eb784dfecd51cab2c
3. **Data freeze** before any test scoring: finish HI/Hinglish variants on calibration/dev; keep test frozen; re-run family cross-split check.
4. **Nov 8:** preliminary H1 on **dev** only (`configs/eval/baseline.yaml`). RUNTIME-VERIFY WildGuard first.
5. **Nov 9 Locate gate:** CircuitKIT credit needs the Kaggle parity result (sequential fp32 short+long KL < 1e-4, or logged incomplete/bf16 variant).
6. **Nov 11:** single locked test pass (`configs/eval/test_locked.yaml`) after Repair arm frozen. Never retune on test.
7. Writer P0s: HF article and pitch (no invented numbers): https://app.notion.com/40f047502f58812cb515f2dc909208e8
8. Submit: internal target complete Nov 14 IST; hard outer deadline Nov 15 17:29:59 IST (Task Board).

### Data freeze conflicts (resolved in code, still track in docs)

- Family splits 20/30/50 (prereg) vs old item-level 50/15/35: **v2 family splits are on main** (`data/splits/family_splits_v2.json`). Realized ratios logged in prereg deviation D3 (~18.9 / 30.4 / 50.7).
- Baseline on test: **dev** for Nov 8 preliminary; **test** only via `test_locked.yaml` on Nov 11.
- ASR denominator: harmful_compliant / n_scored; refusal, irrelevant, truncated stay in; judge_failure out.
- Uncertainty: Wilson 95% for proportions; family paired bootstrap 2000 reps, seed 20261106.

## Gates and integrity

- No hardcoded or fabricated metrics. Every reported number needs `runs/<id>/metrics.json` with `artifact_kind: run` (fixtures use `artifact_kind: fixture`).
- Preregistered protocol: `docs/research/PREREG_baseline.md` + Experiment Protocol pages in Notion.
- Frozen test evaluated once (`--i-understand-single-shot`).
- No secrets, vendor trees, or venvs in git.
- Paid compute / new accounts / other-repo issues or PRs / HF publishing: Vedant OK required.
- No em dashes in docs or submissions.

## Key decisions (pointers)

- Track 2, Hinglish Shield (small-model code-mixed financial-support repair acceptance pack)
- ADR 0007: `docs/adr/0007-finance-workload-and-preservation-gates.md`
- Upstream status: `docs/upstream-status-2026-11-06.md` (CuratorKIT #42 candidate; SafeTune #37/#38/#39 deferred)
- Model: CohereLabs/tiny-aya-global (fallback aya-expanse-8b only with separate prereg)
- Free-tier GPU only (Kaggle preferred over Colab for RAM)
- Lexsi libs: CuratorKIT 0.1.21, AuditKIT 1.1.3, CircuitKIT 0.2.0, SafeTune 0.1.12 (SafeTune needs Python 3.12+)

## Repo map

```
data/seeds, data/interim, data/curated, data/splits, data/staging
configs/{curate,eval,research,default.yaml}
src/{curate,eval,...}
docs/{EVAL_PLAN,GPU_RUNBOOK,adr,research,HANDOFF}
notebooks/gpu_parity_and_baseline.ipynb
runs/ (setup-smoke + future scored runs)
redteam/
schemas/metrics.schema.json
scripts/smoke.py, check_translation_family_splits.py, verify_parity_copy.py
```

Canonical box clone: `/workspace/hinglish-shield`. Scratch/venvs: `/workspace/hinglish-shield-scratch` (do not push from scratch).

## Environment setup

```bash
git clone https://github.com/Vedant817/hinglish-shield.git
cd hinglish-shield
python3 -m venv .venv && source .venv/bin/activate
make install
pip install pytest   # for unit tests
make smoke
python -m pytest tests/ -q
```

Lexsi libraries: prefer **separate venvs** per lib (especially SafeTune on Python 3.12+). Box smoke logs live under `runs/setup-smoke-2026-11-06/`. HF_TOKEN must be available for gated Tiny Aya loads; never print it. For Kaggle, put HF_TOKEN in Kaggle Secrets only.

## Deadlines (IST)

| When | What |
| --- | --- |
| Nov 7 | Vedant HI/Hinglish review |
| Nov 8 | Dev preliminary baseline (after WildGuard RUNTIME-VERIFY) |
| Nov 9 | Locate / CircuitKIT parity evidence |
| Nov 10 | Repair steering sweep on calibration/dev |
| Nov 11 | Locked test single shot |
| Nov 14 | Internal submission target |
| Nov 15 17:29:59 | Hard outer deadline |

## Handoff prompt for your agent

Copy everything in the fenced block below into a fresh coding agent:

```
You are the coding agent for Vedant Mahajan's Lexsi hackathon project Hinglish Shield (Track 2, AI red teaming / safety and alignment).

Goal: ship a reproducible small-model, code-mixed (EN/HI/Hinglish) financial-support safety audit and repair acceptance pack on Tiny Aya Global, using Lexsi libs (CuratorKIT, AuditKIT, CircuitKIT, SafeTune) where RUNTIME-VERIFIED. Impress Lexsi with honest metrics, Lexsi-shaped artifacts, and an upstream-friendly local patch story. Do not invent results.

Repo: https://github.com/Vedant817/hinglish-shield (public). Work on main via PRs. Notion HQ: https://app.notion.com/p/40f047502f5881d4aa86d150111b9355 . Task Board: https://app.notion.com/p/033ead4a50f24e6e98105c33d6b646e6 . Agent Playbook: https://app.notion.com/p/40f047502f58811c9d99f53600cb6ab0 .

Read first, in order:
1. docs/HANDOFF.md (this handoff)
2. README.md, WORKSPACE.md, LOG.md
3. docs/research/PREREG_baseline.md and configs/research/baseline_prereg.yaml
4. docs/EVAL_PLAN.md, configs/eval/baseline.yaml, configs/eval/test_locked.yaml
5. docs/GPU_RUNBOOK.md, docs/KAGGLE_AGENT.md (step-by-step Kaggle web and CLI run for agents) and notebooks/gpu_parity_and_baseline.ipynb
6. docs/adr/0007-finance-workload-and-preservation-gates.md
7. data/CURATION_PLAN.md and data/splits/family_splits_v2.json (manifest only; do not rescore test)
8. Notion pages: Anti-Hallucination and Integrity Rules, Experiment Protocol, Engineering Standards, Failure Playbook, Approaches Deep Dive (under the HQ prefix https://app.notion.com/p/40f047502f5881)

Integrity rules (non-negotiable):
- No hardcoding scores, no fabricated metrics, no "about X%" without a run artifact.
- Every number in README, article, pitch, or Notion must trace to runs/<id>/metrics.json with artifact_kind: run (or an explicit fixture labeled artifact_kind: fixture).
- Follow the preregistered protocol. Do not silently override discrepancies; log deviations.
- Frozen test is evaluated once (configs/eval/test_locked.yaml + --i-understand-single-shot). Never tune on test.
- No secrets, vendor/, or venvs in git. Never print HF_TOKEN.
- No em dashes in docs or submissions.
- Paid compute, new accounts, issues/PRs on other repos, and HF publishing need Vedant's explicit OK. Free Kaggle/Colab on accounts he already has is allowed.

Standing permission: Vedant said agents may push, open PRs, and merge in Vedant817/hinglish-shield under the constraints above.

Next work in priority order:
1. Run or help Vedant run the Kaggle GPU job by following docs/KAGGLE_AGENT.md (Vedant does sign-in, phone verification and typing the HF_TOKEN secret if you cannot hold it securely). Ingest hs_gpu_artifacts zip under runs/. Record parity short/long KL; incomplete if long case missing. Do not treat fixtures as ASR.
2. Support Vedant's Nov 7 Hindi/Hinglish review of data/staging/review/vedant_review_nov7.csv; apply labeled fixes without inventing translations.
3. Complete calibration/dev language coverage; keep test frozen; re-run scripts/check_translation_family_splits.py.
4. RUNTIME-VERIFY WildGuard; Nov 8 preliminary baseline on dev only; write real metrics.json.
5. If parity passes, Locate refusal directions on calibration/dev only.
6. Repair steering sweep on calibration/dev; freeze arm; Nov 11 locked test once.
7. Keep Writer article numbers tied only to run artifacts.
8. Prepare submission materials for Nov 14 IST (hard stop Nov 15 17:29:59 IST).

Needs Vedant's OK before you do it: spending money, creating new cloud/Kaggle/HF accounts, publishing datasets/models to HF Hub, opening issues or PRs on Lexsi or other upstream repos, submitting the hackathon form.

Box note: /workspace/hinglish-shield is the canonical clone; /workspace/hinglish-shield-scratch holds local venvs. Prefer separate venvs per Lexsi lib; SafeTune needs Python 3.12+.

Start by pulling main, reading docs/HANDOFF.md, running make smoke, and listing open Task Board P0s. Then propose the next smallest verifiable unit of work.
```
