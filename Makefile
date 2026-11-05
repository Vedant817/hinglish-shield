.PHONY: install smoke all space

install:
	python3 -m pip install -r requirements.txt

smoke:
	python3 scripts/smoke.py

# Full pipeline stages are owned by other roles; keep the entrypoint honest.
all: smoke
	@echo "smoke OK. Next stages (data, baseline, locate, repair) are role-owned Task Board items."

# Gradio demo (MOCK metrics until runs/<id>/metrics.json and src/demo are wired).
space:
	cd space && python3 -m pip install -r requirements.txt && python3 app.py
