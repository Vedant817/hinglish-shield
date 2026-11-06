# API verification ledger

## AuditKIT WildGuard contract, Nov 7 IST

- Package: auditkit 1.1.3, PyPI wheel SHA-256 `1efa0500c2da2cbd1bebe35b158d895ea4c1bd2b87ffc2935d8807845a752388`.
- Source: `auditkit/metrics/guard.py`, `GuardJudge(profile="wildguard", judge_model=..., judge_model_args=...)`; `score(Sample(input=prompt), output)` returns `Score` with classifier fields and truncated raw text in metadata.
- Source: `auditkit/model/hf_gen.py`, `HFGenModel` forwards revision, dtype and device-map options at model load. The WildGuard profile sends the raw classifier template with `apply_chat_template=False` and a generation cap of 32.
- EXECUTED: import and signature inspection in an isolated Python 3.12 environment. A malformed-output parser probe returned value 0.0 with missing classifier fields. Local wrapper rejects incomplete or duplicate fields instead of counting them as safe.
- Contract fixtures exercise the real installed GuardJudge with a fake transport. They are not WildGuard inference, judge calibration, or ASR evidence.
- MODEL INFERENCE UNVERIFIED: authenticated config access for Tiny Aya global/base succeeded. WildGuard initially returned HTTP 403; a Nov 7 recheck returned HTTP 200 after the owner enabled access. This is configuration access only, with no judge weights loaded. A separate free Kaggle hardware preflight proved T4 access, without loading models. Platform secret attachment is still required for the model fixture.
- No downstream scoring is enabled by this ledger. Human calibration is required for validated Hinglish judgments.

Council round 2 correction: AuditKIT truncates its stored `raw` diagnostic text.
The local custom profile now validates the complete output in its parse hook,
before AuditKIT truncates diagnostics. A valid-looking prefix followed by a late
duplicate field is a judge failure, not a safe score. Contract tests cover that
case and a valid answer preceded by long whitespace.

## WildGuard reference-format adaptation

- Official source pinned at `allenai/wildguard@495b9dc837be594c6c5665b0f7f56169baf0d5be`; `wildguard/utils.py` defines the raw template without a literal BOS token. Template SHA-256: `5e74b62445cca8e91d51deba51848059267bd2b8c279ac1c7a94adef445d8bdf`.
- Local AuditKIT custom profile `hs_wildguard_v1` removes AuditKIT's leading literal BOS and sets max_tokens to 128, matching the reference HF generation cap. Missing/non-yes-no fields remain unscored, including n/a for response assessment. This is stricter than the reference parser's optional n/a fields.
- `python -m src.eval.smoke_wildguard --revision FULL_HF_SHA --out /kaggle/working/wg-smoke` is the real-model fixture entrypoint. It requires platform-secret HF_TOKEN and a GPU session. It unloads the judge before a single CPU retry after allocation failure, if RAM permits. Output metrics remain null and explicitly fixture-labelled.
- Source-checked AuditKIT tag commit: `71316da345784ff220d5ed59337441a59ab31526`. No official hackathon weighted rubric or named judges were found in the connected Company and Judges page. Council readiness ratings are engineering opinions, not experimental metrics.
