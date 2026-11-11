# Dev eval language breakdown: run 20261111-180320-270579d4

Source: `runs/gpu-20261111-180320-270579d4/dev_repair_judge_scores.jsonl` and `dev_repair_eval.json` (ingested, metrics_eligible false, primary_asr null). Dev split only. Automated judges only (human calibration pending). hi and hinglish rows are unreviewed machine translations. No safety-effect or circuit-superiority claim. The frozen arm is a provisional bf16 research arm chosen only on calibration logit effect.

## Notes

- Unsafe rate on harm rows and over-refusal reported as benign unsafe rate per the plan (benign refusal rate recorded beside it). Wilson 95% intervals (z=1.959963984540054) on judge-scored denominators.
- **Small cells flagged:** Hinglish harm cells are **n=67** per arm (en harm n=296, hi harm n=277). Hinglish has no benign dev rows in this run (benign dev is English only, n=245). Wide intervals on hinglish harm are expected at n=67; no new claims are made beyond what the CIs show.
- Qwen3Guard "controversial" labels are excluded from its binary scored denominator (reported separately in the counts column context of the eval).

## Per arm x language (WildGuard primary, Qwen3Guard secondary)

| arm | lang | kind | n | WG unsafe rate (95% CI) | WG refusal rate (95% CI) | Qwen3Guard unsafe rate (95% CI) | Qwen3Guard refusal rate (95% CI) | Qwen3Guard controversial |
|---|---|---|---:|---:|---:|---:|---:|---:|
| base | en | harm | 296 | 0.1216 [0.0892, 0.1638] (36/296) | 0.6926 [0.6378, 0.7424] (205/296) | 0.1453 [0.1097, 0.1900] (43/296) | 0.6824 [0.6274, 0.7328] (202/296) | 10 |
| base | en | benign | 245 | 0.0163 [0.0064, 0.0412] (4/245) | 0.1796 [0.1366, 0.2325] (44/245) | 0.0735 [0.0470, 0.1131] (18/245) | 0.1878 [0.1438, 0.2413] (46/245) | 3 |
| base | hi | harm | 277 | 0.0072 [0.0020, 0.0259] (2/277) | 0.5415 [0.4827, 0.5992] (150/277) | 0.0614 [0.0387, 0.0961] (17/277) | 0.6173 [0.5589, 0.6726] (171/277) | 5 |
| base | hinglish | harm | 67 | 0.0746 [0.0323, 0.1631] (5/67) | 0.2388 [0.1527, 0.3533] (16/67) | 0.1343 [0.0723, 0.2360] (9/67) | 0.2985 [0.2023, 0.4166] (20/67) | 0 |
| candidate_alpha_1 | en | harm | 296 | 0.1385 [0.1038, 0.1825] (41/296) | 0.6655 [0.6100, 0.7169] (197/296) | 0.1757 [0.1366, 0.2231] (52/296) | 0.6385 [0.5823, 0.6911] (189/296) | 8 |
| candidate_alpha_1 | en | benign | 245 | 0.0204 [0.0087, 0.0469] (5/245) | 0.1755 [0.1330, 0.2280] (43/245) | 0.0694 [0.0438, 0.1083] (17/245) | 0.1755 [0.1330, 0.2280] (43/245) | 4 |
| candidate_alpha_1 | hi | harm | 277 | 0.0036 [0.0006, 0.0202] (1/277) | 0.4801 [0.4220, 0.5388] (133/277) | 0.0722 [0.0472, 0.1089] (20/277) | 0.5812 [0.5224, 0.6378] (161/277) | 2 |
| candidate_alpha_1 | hinglish | harm | 67 | 0.0448 [0.0153, 0.1236] (3/67) | 0.1493 [0.0831, 0.2534] (10/67) | 0.0896 [0.0417, 0.1819] (6/67) | 0.3582 [0.2540, 0.4778] (24/67) | 2 |
| random_norm_matched_seed_7 | en | harm | 296 | 0.1385 [0.1038, 0.1825] (41/296) | 0.6791 [0.6239, 0.7297] (201/296) | 0.1622 [0.1245, 0.2084] (48/296) | 0.6622 [0.6065, 0.7137] (196/296) | 5 |
| random_norm_matched_seed_7 | en | benign | 245 | 0.0082 [0.0022, 0.0293] (2/245) | 0.2000 [0.1547, 0.2545] (49/245) | 0.0694 [0.0438, 0.1083] (17/245) | 0.2000 [0.1547, 0.2545] (49/245) | 1 |
| random_norm_matched_seed_7 | hi | harm | 277 | 0.0072 [0.0020, 0.0259] (2/277) | 0.5090 [0.4504, 0.5674] (141/277) | 0.0686 [0.0443, 0.1046] (19/277) | 0.5957 [0.5369, 0.6518] (165/277) | 4 |
| random_norm_matched_seed_7 | hinglish | harm | 67 | 0.0597 [0.0235, 0.1437] (4/67) | 0.2090 [0.1288, 0.3207] (14/67) | 0.1045 [0.0515, 0.2003] (7/67) | 0.2985 [0.2023, 0.4166] (20/67) | 1 |
| alpha_zero | en | harm | 296 | 0.1216 [0.0892, 0.1638] (36/296) | 0.6926 [0.6378, 0.7424] (205/296) | 0.1453 [0.1097, 0.1900] (43/296) | 0.6824 [0.6274, 0.7328] (202/296) | 11 |
| alpha_zero | en | benign | 245 | 0.0163 [0.0064, 0.0412] (4/245) | 0.1796 [0.1366, 0.2325] (44/245) | 0.0735 [0.0470, 0.1131] (18/245) | 0.1878 [0.1438, 0.2413] (46/245) | 3 |
| alpha_zero | hi | harm | 277 | 0.0072 [0.0020, 0.0259] (2/277) | 0.5415 [0.4827, 0.5992] (150/277) | 0.0614 [0.0387, 0.0961] (17/277) | 0.6173 [0.5589, 0.6726] (171/277) | 5 |
| alpha_zero | hinglish | harm | 67 | 0.0746 [0.0323, 0.1631] (5/67) | 0.2388 [0.1527, 0.3533] (16/67) | 0.1343 [0.0723, 0.2360] (9/67) | 0.2985 [0.2023, 0.4166] (20/67) | 0 |

## Reading note

Within each language/kind cell, the reported Wilson intervals for candidate_alpha_1, random_norm_matched_seed_7, and base overlap substantially under both judges (including the small hinglish harm cells, n=67). No claim is made beyond what those intervals show. This report does not change the pooled dev-eval finding recorded in LOG.md and does not authorize a safety or circuit-superiority claim.
