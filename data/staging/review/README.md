# Vedant review queue (Nov 7 target)

For development review, use `vedant_review_nov7_calibration_dev.csv` only.
It is derived with `python scripts/prepare_dev_review.py`, checked against the
frozen membership manifest, and excludes every test family. The generator refuses
to overwrite an existing sheet, preserving human edits.

Review intent equivalence, naturalness, and the harmful/benign distinction. Fill
`vedant_verdict` as accept, revise, or reject. For revisions, fill only the corrected
language fields that need changes and explain the reason in `vedant_notes`.
These are human review inputs, not model-judge calibration results. Do not edit
the mixed-split queue during development or use test examples to select repairs.

- Sheet: `vedant_review_nov7.csv` (59 rows)
- From batch01 FLAG: 9
- From aya_hinglish_sample50: 50
- Fill `vedant_verdict` with accept | revise | reject.
- If revise, put fixes in `corrected_hindi` / `corrected_hinglish`.

Filled from DE B family_lookup.csv (v2). batch01 subset: 24/30 test-split.
