# Hinglish translations drop zone (Data Engineer A)

Put Hinglish batch CSVs or JSONL here. Owned by DE A (`src/curate/translate_seeds.py`).

Expected CSV columns:

`id, source, source_id, english, hindi, hinglish, label, category, qc_notes`

After batches land, Data Engineer B joins on EN ids, copies EN `split`, and writes `data/interim/hinglish/` then curated `hs_*_hinglish/` folders.

Do not re-sample splits here.
