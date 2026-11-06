import copy
import json
from pathlib import Path

import pytest
import jsonschema

from src.eval.metrics_io import is_fixture, validate_metrics


def fixture():
    return json.loads(Path("runs/_example/metrics.json").read_text())


def run():
    data = fixture()
    data.update(artifact_kind="run", split="dev", full_commit="a" * 40,
                config_hash="b" * 64, model_revision="c" * 40, dataset_hash="d" * 64,
                method="base", n_intended=4, n_generated=3, n_errors=1,
                n_judged=2, n_judge_failures=1, harmful_count=4,
                judge_version="pinned-fixture-contract")
    for judge in data["judges"]:
        judge["version"] = "fixture-contract-only"
    return data


def test_missing_kind_is_not_admitted():
    data = fixture()
    del data["artifact_kind"]
    assert is_fixture(data)
    with pytest.raises(jsonschema.ValidationError):
        validate_metrics(data)


def test_setup_smoke_is_excluded_even_with_dev_split():
    data = fixture()
    data.update(artifact_kind="setup-smoke", split="dev")
    validate_metrics(data)
    assert is_fixture(data)


@pytest.mark.parametrize("key", ["full_commit", "model_revision", "dataset_hash", "n_intended", "judge_version"])
def test_runs_need_pins_and_counts(key):
    data = run()
    data[key] = None
    with pytest.raises(jsonschema.ValidationError):
        validate_metrics(data)


def test_count_conservation_not_clipping():
    data = run()
    validate_metrics(data)
    for updates in ({"n_generated": 5}, {"n_judged": 3}, {"harmful_count": 5}):
        invalid = copy.deepcopy(data)
        invalid.update(updates)
        with pytest.raises(ValueError):
            validate_metrics(invalid)


def test_sensitivity_fields_match_config_contract():
    data = run()
    data["metrics"].update(asr_bound_all_safe=0.0, asr_bound_all_harmful=1.0, language_gap_hi=None)
    validate_metrics(data)


def test_run_metrics_need_null_reasons_and_finite_values():
    data = run()
    data["null_reasons"] = {}
    with pytest.raises(ValueError, match="null reason"):
        validate_metrics(data)
    data = run()
    data["metrics"]["language_gap"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_metrics(data)
