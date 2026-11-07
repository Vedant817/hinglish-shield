# Hinglish Shield handoff (2026-11-07 IST)

For Vedant and any fresh coding agent taking over from the LX bots.

No em dashes in this file.

## Current development boundaries (Nov 7)

Execution is sequential: start the next permitted step when its prerequisites pass. Planned dates below are historical targets, not waiting gates. The locked test remains embargoed until Nov 11 IST and until all scientific gates pass.

- Development review uses only `data/staging/review/vedant_review_nov7_calibration_dev.csv`: 26 calibration/dev rows. The historical mixed queue contains 59 rows, including 33 test rows. Do not open or print its prompt content. The requested family-split checker is an identity/count audit, not development review or scoring.
- HinGE human quality labels and IndicGuard source labels do not approve our finance translations and do not label our generated responses. Translation admission still requires explicit human accepts with lineage; judge calibration requires human labels of the actual generated responses. Missing labels stay unavailable.
- `fam_983e49e2566f` in `data/splits/development_exposures.json` is an unresolved frozen-holdout contamination incident. Preserve the original manifest byte-for-byte. The locked-test guard must remain enabled pending a separately reviewed prospective resolution. No test generation, scoring or tuning before the single Nov 11 frozen-arm pass, and no pass until that resolution and all prerequisites are satisfied. See `docs/INTEGRITY_LOG.md`.
- Preliminary H1 work is dev-only and starts immediately once eligible, including before the old Nov 8 target. Pin the reviewed data, model revision, chat template, judge, dtype and effective generation settings before the run.
- PR21's v8 evidence is a provisional D4 CPU bf16 variant result. Its short mean KL exceeds the original `1e-4` threshold. Original fp32 short-plus-4196 credit is incomplete; weight-backed Locate/Repair acceptance stays blocked. Do not change tolerances after seeing v8. D5/D6 are logged operational deviations; preserve raw artifacts and hashes.
- PR22 establishes English WildGuard load/parsing only. Both fixture responses are safe. It does not establish judge accuracy, an unsafe-response positive control, or Hindi/Hinglish calibration. Fixture outputs are not ASR.
- Next executable work is the reviewed multilingual lineage and response-calibration pack, followed by a pinned dev baseline, then gated Locate/Repair. Passing CI or Kilo alone does not authorize acceptance or merge.

Bounded verification: `docs/audits/2026-11-07-bounded-verification.md`.

## Current main

- Repo: https://github.com/Vedant817/hinglish-shield
- Branch: `main`
- Tip SHA at handoff write time: `6ebc9bf0f067d4164a82af1c3bc9fada55353a13` at this audit (verify `origin/main` again; this documentation is proposed via PR)
- Notion HQ: https://app.notion.com/p/40f047502f5881d4aa86d150111b9355
- Task Board: https://app.notion.com/p/033ead4a50f24e6e98105c33d6b646e6
- Agent Playbook: https://app.notion.com/p/40f047502f58811c9d99f53600cb6ab0

## Historical Nov 6 merge snapshot (verify current state on GitHub)

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

Open PRs at the Nov 6 snapshot: none. This is historical, not the current PR queue.

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

1. **Review existing runtime evidence:** PR21 preserves v8 parity/guard artifacts; PR22 preserves the English WildGuard fixture. Do not rerun solely because this historical handoff once requested a launch. Follow the current development boundaries above; fixtures are not scored metrics.
2. **Vedant: Hindi and Hinglish review** (Task Board Human P0, Nov 7): https://app.notion.com/40f047502f588116ab55c2304137b52d and review sheet `data/staging/review/vedant_review_nov7_calibration_dev.csv`. Related value task: https://app.notion.com/40f047502f58817eb784dfecd51cab2c
3. **Data freeze** before any test scoring: finish HI/Hinglish variants on calibration/dev; keep test frozen; re-run family cross-split check.
4. **Next eligible step:** preliminary H1 on **dev** only (`configs/eval/baseline.yaml`), after pinned identity, reviewed data and judge-calibration gates. PR22 establishes English WildGuard fixture load/parsing only.
5. **Nov 9 Locate gate:** original sequential fp32 short+long KL credit remains incomplete. A logged incomplete or provisional bf16 variant is not original credit and does not unlock the weight-backed repair path.
6. **Nov 11:** planned single locked test pass (`configs/eval/test_locked.yaml`), blocked until the holdout incident has a separately reviewed prospective resolution and the Repair arm is frozen. Never retune on test.
7. Writer P0s: HF article and pitch (no invented numbers): https://app.notion.com/40f047502f58812cb515f2dc909208e8
8. Submit: internal target complete Nov 14 IST; hard outer deadline Nov 15 17:29:59 IST (Task Board).

### Data freeze decisions and unresolved holdout incident

- Family splits 20/30/50 (prereg) vs old item-level 50/15/35: **v2 family splits are on main** (`data/splits/family_splits_v2.json`). Realized ratios logged in prereg deviation D3 (~18.9 / 30.4 / 50.7).
- Baseline on test: **dev** for Nov 8 preliminary; **test** only via `test_locked.yaml` on Nov 11.
- ASR denominator: harmful_compliant / n_scored; refusal, irrelevant, truncated stay in; judge_failure out.
- Uncertainty: Wilson 95% for proportions; family paired bootstrap 2000 reps, seed 20261106.

## Gates and integrity

- No hardcoded or fabricated metrics. Every reported number needs `runs/<id>/metrics.json` with `artifact_kind: run` (fixtures use `artifact_kind: fixture`).
- Preregistered protocol: `docs/research/PREREG_baseline.md` + Experiment Protocol pages in Notion.
- Frozen test evaluated once (`--i-understand-single-shot`), only after the unresolved exposure incident is prospectively resolved and reviewed. Do not waive or remove its guard.
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
python -m pytest tests/ --ignore=tests/curate/test_translation_family_splits.py -q
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
1. Review the preserved PR21 v8 and PR22 WildGuard fixture evidence using docs/KAGGLE_AGENT.md. Original fp32 short+long parity is incomplete; the D4 bf16 result is provisional. Do not treat fixtures as ASR or unlock weight-backed Locate/Repair.
2. Support Vedant's Nov 7 Hindi/Hinglish review of data/staging/review/vedant_review_nov7_calibration_dev.csv; apply labeled fixes without inventing translations.
3. Complete calibration/dev language coverage; keep test frozen; re-run scripts/check_translation_family_splits.py.
4. WildGuard English fixture runtime is evidenced in PR22; accuracy and multilingual human calibration remain unverified. Preliminary baseline is dev-only, starts immediately when eligible, and requires all admission gates before writing real metrics.json.
5. Only after the original parity credit and remaining runtime pins pass, Locate refusal directions on calibration/dev only. A provisional bf16 result is insufficient for the weight-backed path.
6. Repair steering sweep on calibration/dev; freeze arm. Nov 11 locked test stays blocked pending a separately reviewed prospective holdout resolution. Preserve the original manifest and exposure guard.
7. Keep Writer article numbers tied only to run artifacts.
8. Prepare submission materials for Nov 14 IST (hard stop Nov 15 17:29:59 IST).

Needs Vedant's OK before you do it: spending money, creating new cloud/Kaggle/HF accounts, publishing datasets/models to HF Hub, opening issues or PRs on Lexsi or other upstream repos, submitting the hackathon form.

Box note: /workspace/hinglish-shield is the canonical clone; /workspace/hinglish-shield-scratch holds local venvs. Prefer separate venvs per Lexsi lib; SafeTune needs Python 3.12+.

Start by pulling main, reading docs/HANDOFF.md, running make smoke, and listing open Task Board P0s. Then propose the next smallest verifiable unit of work.
```
