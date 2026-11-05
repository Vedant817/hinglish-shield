# Seed sources for Hinglish Shield (Lexsi Track 2)

Pulled: 2026-11-06 IST
Location: `/workspace/hinglish-shield/data/seeds/`

This folder holds raw seed prompts and behaviors only. No model weights or attack artifacts.

---

## 1. Aya red-teaming (Cohere Labs)

- URL: https://huggingface.co/datasets/CohereLabs/aya_redteaming
- How pulled: `huggingface_hub.hf_hub_download` (anonymous) for each language JSONL plus `README.md`
- License: Apache 2.0 (from HF dataset card YAML `license: apache-2.0` and README "License: Apache 2.0")
- Approximate counts (JSONL lines = items):
  - `aya_hin.jsonl` (Hindi): 915
  - `aya_eng.jsonl` (English): 987
  - `aya_arb.jsonl` (Arabic): 900
  - `aya_fra.jsonl` (French): 813
  - `aya_spa.jsonl` (Spanish): 782
  - `aya_rus.jsonl` (Russian): 1007
  - `aya_srp.jsonl` (Serbian): 1006
  - `aya_tgl.jsonl` (Filipino): 1009
  - Total across languages: 7419
- Hindi / HI subset: saved as `aya_redteaming/aya_hin.jsonl`. Each row has `"language": "Hindi"`. No extra filter step needed; the upstream split is already per language.
- Notes: Dataset tagged `not-for-all-audiences` on the HF card. Full multilingual seed set kept for reference; Hindi is the primary Track 2 seed.

## 2. HarmBench (behaviors only)

- URL: https://github.com/centerforaisafety/HarmBench
- How pulled: shallow git clone with sparse checkout of `data/`, then copied behavior CSVs plus `LICENSE` and `README.md`. Did not keep multimodal images, copyright classifier hashes, optimizer targets, or Python tooling.
- License: MIT (from repo `LICENSE`: "MIT License", Copyright (c) 2024 centerforaisafety)
- Approximate counts (CSV data rows, header excluded; counted with Python `csv` because some fields are multiline):
  - `behavior_datasets/harmbench_behaviors_text_all.csv`: 400
  - `behavior_datasets/harmbench_behaviors_text_test.csv`: 320
  - `behavior_datasets/harmbench_behaviors_text_val.csv`: 80
  - `behavior_datasets/harmbench_behaviors_multimodal_all.csv`: 110 (behavior metadata only; image files not downloaded)
  - `behavior_datasets/2_behaviors.csv`: 2
  - extras: `advbench_behaviors.csv` 520, `adv_training_behaviors.csv` 100, `tdc2023_test_phase_behaviors.csv` 50
- Primary seed for this project: text behaviors (`harmbench_behaviors_text_all.csv`, 400 items).

## 3. XSTest

- URL: https://github.com/paul-rottger/xstest
- How pulled: shallow git clone, then kept `xstest_prompts.csv`, `LICENSE`, and `readme.md` only (dropped model completions and evaluation artifacts).
- License: Creative Commons Attribution 4.0 International (CC BY 4.0). Confirmed from repo `LICENSE` ("Attribution 4.0 International" / Creative Commons Attribution 4.0 International Public License) and `readme.md` ("XSTest prompts are subject to Creative Commons Attribution 4.0 International (CC-BY-4.0) license.").
- Approximate counts: `xstest_prompts.csv` has 450 data rows (header: id, prompt, type, label, focus, note).

## 4. OR-Bench

- URL: https://huggingface.co/datasets/bench-llm/or-bench
- How pulled: `huggingface_hub.hf_hub_download` (anonymous) for the three CSV configs plus `README.md`. Image assets under `images/` were not downloaded.
- License: CC BY 4.0 (from HF dataset card YAML `license: "cc-by-4.0"`)
- Approximate counts (CSV data rows, header excluded):
  - `or-bench-80k.csv`: 80359
  - `or-bench-hard-1k.csv`: 1319
  - `or-bench-toxic.csv`: 655
- Notes: Filenames say "80k" / "hard-1k"; measured row counts above are what is on disk after pull.

---

## Auth / blockers

- Hugging Face pulls succeeded anonymously. No gated auth was required for Aya or OR-Bench.
- Hub warned about unauthenticated rate limits; downloads completed without a token.
- No blockers for Hindi Aya: `aya_hin.jsonl` is present and countable.

## Layout

```
data/seeds/
  SOURCES.md
  MANIFEST.json
  aya_redteaming/     # JSONL per language + README.md
  harmbench/          # LICENSE, README.md, behavior_datasets/*.csv
  xstest/             # xstest_prompts.csv, LICENSE, readme.md
  or_bench/           # or-bench-*.csv + README.md
```
