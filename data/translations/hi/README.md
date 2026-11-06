# HI translations drop zone (Data Engineer A)

Put HI batch CSVs or JSONL here. Owned by DE A (`src/curate/translate_seeds.py`).

Expected CSV columns:

`id, source, source_id, english, hindi, hinglish, label, category, qc_notes`

After batches land, Data Engineer B joins on EN `item_id` / DE A `id`, copies EN `split`, and writes `data/interim/hi/` then curated `hs_*_hi/` folders.

Do not re-sample splits here.
