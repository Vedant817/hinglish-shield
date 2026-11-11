"""The GPU notebook chain runs end to end on stubs. No weights, no network."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_dry_run_chain_completes(tmp_path):
    out = tmp_path / "dryrun"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/notebook_dry_run.py"),
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "DRYRUN PASS" in completed.stdout
    assert "rows=17" in completed.stdout or "rows=1244" in completed.stdout

    artifacts = out / "hs_gpu_artifacts"
    plan = json.loads((artifacts / "dev_repair_eval_plan.json").read_text(encoding="utf-8"))
    assert plan["split"] == "dev"
    assert plan["arms"] == ["base", "candidate_alpha_1", "random_norm_matched_seed_7", "alpha_zero"]
    assert plan["metrics_eligible"] is False
    assert plan["primary_asr"] is None
    assert plan["wilson_z"] == 1.959963984540054
    dev_status = json.loads((artifacts / "dev_repair_status.json").read_text(encoding="utf-8"))
    assert dev_status["status"] == "ok"
    dev_eval = json.loads((artifacts / "dev_repair_eval.json").read_text(encoding="utf-8"))
    assert dev_eval["artifact_kind"] == "run"
    assert dev_eval["alpha_zero_byte_equality"]["match"] is True
    assert set(dev_eval["arms"]) == set(plan["arms"])
    summary = json.loads((artifacts / "d8_descriptive.json").read_text(encoding="utf-8"))
    hing_rows = 0
    for part in ("hs_eval_hinglish_calibration.jsonl", "hs_eval_hinglish_dev.jsonl"):
        jf = artifacts / part
        if jf.is_file():
            hing_rows += sum(1 for line in jf.read_text(encoding="utf-8").splitlines() if line.strip())
    assert hing_rows > 0
    total_rows = 0
    for part in ("d8_generations.jsonl", "d8_judge_scores.jsonl"):
        total_rows = (artifacts / part).read_text(encoding="utf-8").count("\n")
    assert total_rows == 1244 + hing_rows
    assert summary["translation_review"] == "unreviewed_translation"
    assert summary["hinglish_translation_review"] == "unreviewed machine translation"
    assert summary["languages"] == {"en": 771, "hi": 473, "hinglish": hing_rows}
    direction = json.loads((artifacts / "locate_direction.json").read_text(encoding="utf-8"))
    assert direction["controls"]["alpha_zero_with_wrapper"]["max_abs_logit_diff"] == 0.0

    from scripts.ingest_gpu_artifacts import inspect_archive

    files, _summary = inspect_archive(out / "hs_gpu_artifacts_dryrun.zip")
    assert "locate_diagnostics.json" in files


def test_dry_run_declares_stub_scope():
    text = (ROOT / "scripts/notebook_dry_run.py").read_text(encoding="utf-8")
    assert "stub" in text.lower()
    assert "dict" in text
    assert "default" in text
