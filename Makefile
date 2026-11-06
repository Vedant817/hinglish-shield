.PHONY: install smoke all

install:
	python3 -m pip install -r requirements.txt

smoke:
	python3 scripts/smoke.py

# Full pipeline stages are owned by other roles; keep the entrypoint honest.
all: smoke
	@echo "smoke OK. Next stages (data, baseline, locate, repair) are role-owned Task Board items."

# DE A: family-split checks on staging translation CSVs (DE B v2 checker when present)
.PHONY: check-family-splits-translations
check-family-splits-translations:
	python3 scripts/check_translation_family_splits.py
