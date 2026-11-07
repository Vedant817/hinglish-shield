# Independent runtime-evidence review

Completed by Grok 4.7 high. Scope is the immutable PR21/PR22 heads below, not model safety, multilingual accuracy, original fp32 parity credit, or acceptance of a later merge commit. The earlier interrupted review remains historical and did not supply this verdict. Existing raw artifacts, provenance labels and receipts remain unchanged.

+PR21 and PR22 are acceptable for this runtime-evidence gate. No in-scope blocker. The interrupted review [Review measured parity and real WildGuard fixture evidence](<t3-thread://v2/a5fdbfcb-a18f-41e7-a307-f7e7557f4761/thread:delegated-task:command%3Amcp%3A3c416cbc-dea0-446b-b377-308322c03aa1%3Adelegate-task%3Ahs-real-runtime-evidence-review-v8-wg>) never reached a verdict; the checks below were recomputed from the committed blobs.

## PR21: accept

Head `23c53c87e3da39d43f8492d77ea4a1ac3d55ac5e` on parent `ea04bf61628c51b7a7c802b9276dc3be6b474b10`. The notebook bytes match that parent. All 12 ingestion file hashes match the committed blobs. The embedded prereg SHA-256 `2945f03cbf33f6eb9af075bf61a8913f05191ddc159e9215950aac98bf2766f6` matches `docs/research/PREREG_baseline.md` at both `ea04bf6` and this head. `runtime_prereg_file_available` is false. The protocol SHA-256 `6bf1b4fb6651d14133380a2985bdf0d81b5d7b3d73b2e19b1b14f8c4b9281e20` matches the notebook `D4_PROTOCOL` dict.

Recomputed bf16, from `parity_variant_bf16.json` and `decide_bf16_variant` / `decide_bf16_long` / `combine_bf16_gate` (`notebooks/gpu_parity_and_baseline.ipynb` lines 586, 622, 669, 1685):

- Each of the four short KLs is finite, nonnegative, and `<= 2 *` its own floor. Top-1 is `1.0` against `0.99`. The stored short decision is `provisional: false` with rule `kl <= 2 * floor (per prompt)`.
- The short mean is exactly `0.0004105035332031548`. That is above the original `1e-4` tolerance. `used_fp32_tol` is false. This is variant evidence only.
- Long `sliding_window_4196` KL is `4.146346327615902e-06`, under `1e-2`. `long_floor` is null, so the long leg stays provisional. The combined variant is a provisional pass.
- Original fp32 short and the 4196 forward are `incomplete` (`ran_short: false`, `long_ran: false`) after the D5 32 GiB preflight. `locate_credit_granted` is false. `require_original_credit` still requires original status `pass`.

D6 source semantics: `_load_tl_bf16` refuses a live donor, and both orchestrations pass `keep_hf_donor=False` (lines 1560-1566, 1663, 1680). `fold_ln` is omitted at the call. In TransformerLens 3.8.0, `from_pretrained_no_processing` defaults `fold_ln=False`, which matches `versions.json` and the recorded flag. Pins are `CohereLabs/tiny-aya-base` `d62ac17e241c871fc52e9a95d7b0e2615189d7d5` for parity and `CohereLabs/tiny-aya-global` `af89d219b53ed9b13b8a4645f9c8028973510324` for the fp16 guard.

Fp16 guard: 20 `cal_*` rows, every KL finite and nonnegative, recomputed mean exactly `8.59616297930188e-06`, top-1 `1.0`, greedy32 `20/20`, hooks attached, thresholds mean `<= 1e-2`, top-1 `>= 0.98`, greedy32 `>= 17`. The guard passes. Those rows are hand-written fixtures. Metrics in `metrics_stub.json` stay null. Seven prediction rows are `fixture` / `example` at `max_new_tokens` 64.

Low note, not a blocker. `runtime_provenance.json` line 12 names the check `bf16_provisional_short_and_long`, and `LOG.md` says the short and long measurements satisfy the provisional variant rule. The stored short object uses the floor rule. The overall variant is provisional because the long floor is absent. Smallest repair is a provenance label fix. Leave the pinned prereg bytes alone. `docs/research/PREREG_baseline.md` line 110 still says nothing in D4 has been measured; that is the pre-run text inside the pinned hash. Results live under `runs/kaggle-parity-v8-2026-11-07/`.

The 12 hashes show the committed files are internally consistent. They do not authenticate the Kaggle machine. `archive_sha256` was not recomputed; the zip is not in git. The scoped dependency check passed and the global environment check is recorded as failed.

## PR22: accept

Head `31d82aaf26cbd647b5d81c58451b0735e14f4c2b` on `6ebc9bf`. All 4 ingestion hashes match. Manifest, environment, and metrics pin source `6ebc9bf0f067d4164a82af1c3bc9fada55353a13`, judge `allenai/wildguard` revision `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`, profile `hs_wildguard_v1`, `max_tokens` 128, AuditKIT `1.1.3`, CUDA float16, status `parsed`. The config hash matches `sha256(json.dumps(config, sort_keys=True))`.

Both rows match the hand-written `FIXTURES` in `src/eval/smoke_wildguard.py` and `parse_wildguard_output` (`src/eval/judges.py` lines 71-85, 129-135, 171-174). Both `harmful_response` values are `no`, both labels are `safe`, and both values are `0.0`. One row has `harmful_request: yes` and `response_refusal: yes`, which is a refused request. There is no unsafe-response positive control. A late duplicate field and a missing field both fail that parser. ASR, over-refusal, and capability are null. `benchmark_metrics_eligible` is false.

`configs/eval/judges.yaml` lines 12-18 record `RUNTIME-VERIFIED` with `effective_profile: hs_wildguard_v1` and `validation_scope: English fixture load and parsing only`, and `multilingual_human_calibration: unvalidated`. `configs/eval/baseline.yaml` stays `proposed`. `run_baseline.py` still stops in the stub and does not write scores. Ledger eligibility in `aggregate_dev.py` still requires `artifact_kind: run` plus calibration. This registry flag does not make the fixture score-eligible.

Same attestation limit: the receipt binds committed JSON to itself and to the source adapter at `6ebc9bf`. It does not prove which machine produced the file.

Absent multilingual validation, human calibration, judge accuracy, ASR, and the original fp32 gate are project limits already stated in the artifacts. They are not code defects in these two heads.

Parent audit `a9e51b524f9e1b46958aa67c147346b3f751a78f` on open PR23 was read with `git show` and is not merged. Its git-context logs record main `259 passed, 7 subtests`, isolated integration `260 passed, 7 subtests`, and smoke and family-split exit 0. This review did not rerun that suite.

https://github.com/Vedant817/hinglish-shield/pull/21 head 23c53c87e3da39d43f8492d77ea4a1ac3d55ac5e merged, integrity/GitGuardian/Kilo SUCCESS, reviews none; https://github.com/Vedant817/hinglish-shield/pull/22 head 31d82aaf26cbd647b5d81c58451b0735e14f4c2b open, integrity/GitGuardian/Kilo SUCCESS, reviews none
