---
title: Hinglish Shield
emoji: 🛡️
colorFrom: teal
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Cross-lingual safety repair demo for EN / HI / Hinglish
---

# Hinglish Shield

Gradio demo for Lexsi Hackathon Track 2. Compare base Tiny Aya vs a repaired
Hinglish Shield path, inspect MOCK ASR / over-refusal results, peek inside
refusal directions, and browse dataset placeholders.

## Run locally

```bash
cd space
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Or from the repo root: `make space`

## Demo mode

This Space ships with **MOCK / PLACEHOLDER** metrics and canned compare-tab
responses. It does **not** load Tiny Aya until endpoints are wired. Real numbers
should replace `mock_data.py` from `runs/<id>/metrics.json`.
