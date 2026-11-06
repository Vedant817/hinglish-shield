# CuratorKIT safety-payload preservation fixture

Credit: CuratorKIT issue 42, "strip_html (on by default) silently removes tag-based prompt injections from safety datasets" (open), https://github.com/Lexsi-Labs/CuratorKIT/issues/42. This fixture reuses its repro row (`issue42_repro`). No upstream issue or PR was opened.

## Classification
- Bug reproduced on the pinned vendor CuratorKIT (v0.1.21, `curatorkit/normalizers/clean.py`, `_HTML_TAG_RE = <[^>]+>`, default `strip_html: True`).
- Default cleaner: 7 of 7 injection rows lose their payload (label invalidated). Extra finding: benign `x < 5 aur y > 3` is corrupted to `x 3`.
- What we ship here is a local behavior change (opt-in subclass), not an upstream fix and not a config workaround. The only config workaround (`strip_html: false`) is rejected because it disables HTML cleaning globally.

## Local patch
`src/curate/safe_clean.py`: `PayloadPreservingTextCleaner` overrides `_strip_html` to remove only an allowlist of benign formatting tags (b, i, u, em, strong, p, br, span, div, small, sup, sub). All other tag-shaped spans are kept verbatim. `payload_spans()` is the detector used for label invalidation. Vendor code is untouched; the default `TextCleaner` behavior is unchanged for anyone not opting in.

## Cases (`cases.jsonl`)
issue 42 repro (Spanish), Hinglish `<system>`, Devanagari `<admin>`, `<script>`, ChatML `<|im_start|>`, zero-width-split tag, NFD Unicode plus tag, benign formatting, benign `<` `>` math.

## Commands and exit codes
```
.venv/bin/python tests/fixtures/curatorkit_safety_payload/run_fixture.py default   # exit 1, label_invalidated=7 (failing before)
.venv/bin/python tests/fixtures/curatorkit_safety_payload/run_fixture.py patched   # exit 0, label_invalidated=0 (passing after)
```
Evidence (per row sha256 before/after, payload spans, label_invalidated): `evidence/default.json`, `evidence/patched.json`.

## Lineage
No files under `data/seeds/`, `data/interim/` or `data/curated/` were read or written. Baseline artifacts keep their existing hashes.

## Next smallest action
Wire `PayloadPreservingTextCleaner` into the EN curation path (`src/curate/build_en_v0.py`) for injection-labelled rows, then rerun the fixture as a gate before the CuratorKIT export.

## Wiring
`build_en_v0.py --run-curatorkit` calls this fixture in `patched` mode as a pre-export gate (abort on non-zero) and runs CuratorKIT with PayloadPreservingTextCleaner. See data/CURATION_PLAN.md.
