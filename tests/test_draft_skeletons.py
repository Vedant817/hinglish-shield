"""Article/pitch skeleton contracts. Placeholders only, never numbers."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "docs" / "drafts" / "ARTICLE.md"
PITCH = ROOT / "docs" / "drafts" / "PITCH.md"


def test_skeletons_reference_metrics_json_and_stay_unrun():
    for path in (ARTICLE, PITCH):
        text = path.read_text(encoding="utf-8")
        assert "Not run" in text
        assert "runs/<run-id>/metrics.json" in text or "runs/<id>/metrics.json" in text
        assert "\u2014" not in text
        assert re.search(r"\[\[METRICS", text) is not None
        assert "do not" in text.lower() or "Do not" in text
    assert "Do not publish or submit" in ARTICLE.read_text(encoding="utf-8")
    assert "not approved" in PITCH.read_text(encoding="utf-8").lower()


def test_skeletons_contain_no_numeric_result_claims():
    for path in (ARTICLE, PITCH):
        text = path.read_text(encoding="utf-8")
        # Percentages, plain decimals, and ASR verdicts must stay out of drafts.
        assert not re.search(r"\d+(\.\d+)?\s*%", text)
        assert not re.search(r"ASR\s*(of|is|was|=)\s*\d", text, re.IGNORECASE)
        assert "metrics_eligible" in text or "metrics.json" in text
