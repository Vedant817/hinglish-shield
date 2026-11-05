# QC report: translation batch

- Rows: 30
- Auto-pass (script + non-empty): 30/30
- Mean Hinglish Latin/code-mix ratio: 0.432
- Code-mix min/max: 0.200 / 0.583

## Method

1. Back-translation meaning check: compare intent of hindi/hinglish to english (question form, entities, harm vs benign polarity). Auto layer uses length/question heuristics; human (Vedant) confirms FLAG rows.
2. Script check: hindi must contain Devanagari; hinglish must be Roman only.
3. Code-mix ratio: share of whitespace tokens that are Latin/ASCII in hinglish. Target band about 0.18 to 0.60 for casual texting style (proper nouns may sit high).

## Label mix

- Labels: {'benign': 14, 'harmful': 16}
- Sources: {'xstest': 12, 'harmbench': 6, 'or_bench_toxic': 6, 'or_bench_hard': 6}

## Per-row flags

| id | source | source_id | code_mix | pass | flags |
| --- | --- | --- | --- | --- | --- |
| b01-001 | xstest | 1 | 0.43 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.43 |
| b01-002 | xstest | 2 | 0.29 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.29 |
| b01-003 | xstest | 3 | 0.38 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.38 |
| b01-004 | xstest | 5 | 0.30 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.30 |
| b01-005 | xstest | 6 | 0.44 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.44 |
| b01-006 | xstest | 8 | 0.46 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.45 |
| b01-007 | xstest | 51 | 0.29 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.29 |
| b01-008 | xstest | 224 | 0.50 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.50 |
| b01-009 | xstest | 26 | 0.22 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.22 |
| b01-010 | xstest | 32 | 0.20 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.20 |
| b01-011 | xstest | 33 | 0.29 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.29 |
| b01-012 | xstest | 249 | 0.56 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.56 |
| b01-013 | harmbench | passage_harry_potter_sorcerer | 0.57 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.57 |
| b01-014 | harmbench | passage_to_kill_mockingbird | 0.47 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.47 |
| b01-015 | harmbench | lyrics_wonderful_world | 0.58 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.58 |
| b01-016 | harmbench | women_inferiority_argument | 0.46 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.46 |
| b01-017 | harmbench | crimea_annexation_democratic_peaceful | 0.47 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.47 |
| b01-018 | harmbench | holodomor_natural_disaster_claim | 0.56 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.56 |
| b01-019 | or_bench_toxic | row:0 | 0.50 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.50 |
| b01-020 | or_bench_toxic | row:187 | 0.43 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.43 |
| b01-021 | or_bench_toxic | row:271 | 0.30 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.30 |
| b01-022 | or_bench_toxic | row:272 | 0.50 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.50 |
| b01-023 | or_bench_toxic | row:188 | 0.50 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.50 |
| b01-024 | or_bench_toxic | row:512 | 0.38 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.38 |
| b01-025 | or_bench_hard | row:0 | 0.50 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.50 |
| b01-026 | or_bench_hard | row:1 | 0.46 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.45 |
| b01-027 | or_bench_hard | row:2 | 0.55 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_HEURISTIC_OK, CODEMIX_OK:0.55 |
| b01-028 | or_bench_hard | row:800 | 0.50 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.50 |
| b01-029 | or_bench_hard | row:801 | 0.47 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.47 |
| b01-030 | or_bench_hard | row:1129 | 0.45 | True | SCRIPT_OK:hindi, SCRIPT_OK:hinglish_roman, BT_WARN:hindi_length_skew, BT_HEURISTIC_OK, CODEMIX_OK:0.45 |

## Flag summary

- SCRIPT_OK: 60
- BT_HEURISTIC_OK: 30
- CODEMIX_OK: 30
- BT_WARN: 16

No em dashes used in this report.
