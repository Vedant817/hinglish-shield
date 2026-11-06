# Upstream defect and PR status (Hinglish Shield)

**Date:** 2026-11-06 ~01:30 IST (Asia/Calcutta)  
**Task:** [Value] Refresh upstream defect and PR status before choosing a local patch  
**Sources:** live GitHub MCP `issue_read`, `list_commits`, `list_pull_requests`, `search_pull_requests` (user-Github). No issues or PRs opened by this refresh.

## Pinned default-branch HEAD SHAs

| Repo | Default branch HEAD SHA | Tip commit (UTC) | Tip subject |
|---|---|---|---|
| Lexsi-Labs/CuratorKIT | `dd9e835301fcddc74a37501d9e207a5a060d209f` | 2026-10-12T06:41:07Z (12:11 IST) | Merge PR #32 discord_add |
| Lexsi-Labs/SafeTune | `3901a9f48a88984a181c8c9493be2d3a20ab160f` | 2026-11-03T06:29:42Z (11:59 IST) | Merge PR #33 docs fix SafeLoRATrainer |

Issue bodies for SafeTune 37/38/39 cite code `@ 3901a9f`, matching the pinned SafeTune HEAD.

## Per-issue status

| Issue | State | Title | Labels | Assignees | Reporter | Last update (UTC / IST) | Closing / linked PRs |
|---|---|---|---|---|---|---|---|
| [CuratorKIT #42](https://github.com/Lexsi-Labs/CuratorKIT/issues/42) | **open** | strip_html (on by default) silently removes tag-based prompt injections from safety datasets | none | none | **aquilu** (NONE) | 2026-11-05T12:01:26Z / 17:31 IST | `closed_by_pull_requests.total_count=0`; no aquilu PRs; search for strip_html / #42 returned 0 PRs |
| [SafeTune #37](https://github.com/Lexsi-Labs/SafeTune/issues/37) | **open** | [BUG] SafeGrad under fp16 AMP: .grad is loss-scaled but g_align is not | none | none | **aquilu** (NONE) | 2026-11-05T12:01:20Z / 17:31 IST | `closed_by_pull_requests.total_count=0`; no aquilu PRs |
| [SafeTune #38](https://github.com/Lexsi-Labs/SafeTune/issues/38) | **open** | [BUG] build_safety_dataset supervises the prompt (the harmful request) and template, not only the refusal | none | none | **aquilu** (NONE) | 2026-11-05T12:01:22Z / 17:31 IST | `closed_by_pull_requests.total_count=0`; no aquilu PRs |
| [SafeTune #39](https://github.com/Lexsi-Labs/SafeTune/issues/39) | **open** | [BUG] _tokenize_qa_rows assumes right padding; with a left-padding tokenizer the prompt is supervised | none | none | **aquilu** (NONE) | 2026-11-05T12:01:24Z / 17:31 IST | `closed_by_pull_requests.total_count=0`; no aquilu PRs |

**Attribution:** all four issues filed by GitHub user `aquilu`, with context links to https://github.com/aquilu/guardian-colombiano (Lexsi AI Safety Hackathon). Not Lexsi-Labs maintainers (`author_association: NONE`).

## Related open PRs (do not confuse with these issues)

These open PRs exist on the same repos but **do not** close or reference CuratorKIT #42 or SafeTune #37/#38/#39:

- **CuratorKIT** open PRs include #34–#41 (GRPO, EvolInstruct, PII spans, judge gates, batch of #34–#40). PR numbers 37/38/39 on CuratorKIT are **different defects** (PII / empty judge batch / NaN scores), not SafeTune issues.
- **SafeTune** open PRs include #34–#36 and #40–#44 (reward metric word-boundary / capping fixes by ANDROIDHASSAN). None reference SafeGrad AMP, `build_safety_dataset`, or `_tokenize_qa_rows`.

`search_pull_requests` for `author:aquilu` on both repos: **0 results**.

## Chosen bounded local candidate (Value Alignment Rank 1)

**Candidate: CuratorKIT #42, safety-payload preservation under `strip_html`.**

**Bounded local scope (proposed, not yet implemented in this refresh):**
- Keep vendored CuratorKIT pinned at `dd9e835…`.
- In Hinglish Shield curation config / pipeline wrappers (for example `configs/curate/hs_harm_en.yaml`, which currently sets `strip_html: true`), turn `strip_html` **off** for safety / red-team sets, or gate it by task type, and log when cleaning would have altered tag-shaped payloads.
- Optionally add a small local regression check: input containing `<system>…</system>` (or similar) must retain tags after the clean stage.
- Do **not** open an upstream issue or PR; do **not** wait on Lexsi-Labs merge.

**Rationale vs Rank 2 (SafeTune #38 / #39):**
- Rank 1 preference is data-path integrity before training. HS curation (HarmBench / Aya / OR-Bench / XSTest style injections) runs CuratorKIT now; silent tag stripping corrupts labels and kills the attack signal before any SafeTune step.
- SafeTune #38 (supervise refusal only) and #39 (left-padding mask) matter for Harden / safety-dataset training correctness, but training is not required yet under the current pipeline order (seeds → CuratorKIT → AuditKIT baseline → CircuitKIT → optional SafeTune). Hold Rank 2 until a training run is authorized.
- SafeTune #37 (fp16 SafeGrad scale mismatch) is also training-only and T4/fp16 specific; same deferral.

## Explicit non-actions

- **No issues opened** on Lexsi-Labs or any other repo.
- **No PRs opened** on Lexsi-Labs or any other repo.
- This document is a status pin only; local patch work is a separate Task Board item after claim.

## Tool evidence snapshot

- CuratorKIT HEAD via `list_commits` perPage=1 → `dd9e835301fcddc74a37501d9e207a5a060d209f`
- SafeTune HEAD via `list_commits` perPage=1 → `3901a9f48a88984a181c8c9493be2d3a20ab160f`
- Each target issue via `issue_read` method=`get` (+ `get_labels`): state open, empty labels/assignees, `closed_by_pull_requests.total_count=0`
- PR linkage searches: 0 matching PRs for CuratorKIT strip_html/#42; 0 aquilu PRs; SafeTune open list and SafeGrad/AMP search show no fix PRs for #37/#38/#39
