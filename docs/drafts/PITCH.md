# Local pitch draft (skeleton)

Not approved for publication or submission. Measured benchmark results:
**Not run**. Every number comes from `runs/<run-id>/metrics.json` after the
ingest command runs. Placeholders show the field each figure will take;
they stay empty now.

## The ask

We are building a reproducible audit and repair acceptance pack for a small
multilingual financial-support assistant. Hinglish Shield asks whether Tiny
Aya Global can become safer without losing useful support behavior, using
English, Hindi and Hinglish.

## What exists today (no numbers)

- Source-family identity and translation review keep related prompts
  together and record lineage.
- English WildGuard and secondary-adapter fixture loading/parsing are
  evidenced. Hindi and Hinglish accuracy are **Not run**.
- Original fp32 parity is **Not run**. The CPU bf16 parity files are a
  provisional variant, not ASR and not original fp32 credit.
- The dev ASR and a measured repair acceptance are **Not run**.

## What the next run will put here

- [[METRICS runs/<run-id>/metrics.json -> target_identity.status]] for the
  target chat-template pin.
- [[METRICS -> d8_descriptive.by_slice.<slice>.wildguard_unsafe_rate]] for
  descriptive judge rates, labeled `automated-judge only, human calibration
  pending`, with `metrics_eligible` false.
- [[METRICS -> provisional_locate.l2_norm]] and [[METRICS ->
  provisional_locate.checks.alpha_zero_matches_base]] for provisional
  locate evidence (no fp32 credit).
- Repair freeze [[METRICS -> runs/<id>/repair_freeze.json verification]]:
  research arm only, never a deployable claim.

## The standard

The submission will show the reproducible artifacts and honest unavailable
results. Fixtures are never presented as proof of safety. Final evaluation
waits on a frozen repair, preservation controls, and the Nov 11 IST locked
test. Do not fill any placeholder by hand.
