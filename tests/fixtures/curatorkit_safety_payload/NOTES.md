# CuratorKIT safety-payload preservation fixture

Credit: CuratorKIT issue 42, "strip_html (on by default) silently removes tag-based prompt injections from safety datasets" (open), https://github.com/Lexsi-Labs/CuratorKIT/issues/42. This fixture reuses its repro row (`issue42_repro`). No upstream issue or PR was opened.

## Classification
- Bug reproduced on the pinned vendor CuratorKIT (v0.1.21, `curatorkit/normalizers/clean.py`, `_HTML_TAG_RE = <[^>]+>`, default `strip_html: True`).
- Default cleaner: 7 of 7 injection rows lose their payload (label invalidated). Extra finding: benign `x < 5 aur y > 3` is corrupted to `x 3`.
- What we ship here is a local behavior change (opt-in subclass), not an upstream fix and not a config workaround. The only config workaround (`strip_html: false`) is rejected because it disables HTML cleaning globally.

## Local patch
`src/curate/safe_clean.py`: `PayloadPreservingTextCleaner` overrides `_strip_html` to remove only an allowlist of benign formatting tags (b, i, u, em, strong, p, br, span, div, small, sup, sub). All other tag-shaped spans are kept verbatim. `payload_spans()` is the detector used for label invalidation. Vendor code is untouched; the default `TextCleaner` behavior is unchanged for anyone not opting in.

Nov 7 extension: attributed formatting tags can also carry instructions. The
previous local patch stripped a synthetic `span role=system` wrapper and did not
detect its loss. The candidate preserves attributed tags plus opening/closing
tags of the same name within that prompt. Bare formatting still cleans normally.
This is a conservative text-cleaning policy, not an HTML sanitizer or a model
defense. Other configured Unicode and whitespace normalization still applies.
Executable before/after fixture evidence is under
`runs/curatorkit-attributes-fixture-2026-11-07-r3/fixture_results.json`.

Reproduce in an isolated environment with CuratorKIT pinned to the package named
in the fixture result:

```bash
python scripts/verify_payload_attributes.py --previous-commit 66a0fd3ec1a6028d4a02432edeb246b65fb18826 --out /tmp/hs-attribute-fixture
```

The previous project commit must exist locally. No upstream files are modified
and no upstream issue or PR is opened by this command.

## Cases (`cases.jsonl`)

The closing-tag extension was executed against CuratorKIT 0.1.21 in
`runs/curatorkit-closing-attributes-fixture-2026-11-07/fixture_results.json`.
It compares the candidate against project commit `99c4b35`, including attributes
on closing tags. Prior fixture artifacts describe their historical source hashes.

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
