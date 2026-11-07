"""The dependency probe never executes the notebook or fetches model weights."""
import ast
import json
from pathlib import Path

import pytest

from scripts import kaggle_dependency_preflight as probe


def test_current_install_list_still_agrees_with_preflight_without_executing_cells(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "notebooks/gpu_parity_and_baseline.ipynb"
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    # Current numeric gates may change while the dependency-only probe remains
    # bound to the older immutable notebook. Here compare install arguments;
    # the runtime probe retains its real source hash and rejects drift.
    monkeypatch.setattr(probe, "NOTEBOOK_SHA256", probe.sha256(raw))
    pins = probe.notebook_pins(raw)
    notebook = json.loads(raw)
    source = next("".join(c["source"]) for c in notebook["cells"] if "def pip_install" in "".join(c.get("source", [])))
    tree = ast.parse(source)
    install_call = next(n.value for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Name) and n.value.func.id == "pip_install")
    # Evaluate the literal list/f-strings only, never the cell or an import.
    expected = eval(compile(ast.Expression(install_call.args[0]), "notebook_install_args", "eval"), {"__builtins__": {}}, pins)
    assert probe.install_arguments(pins) == expected


def test_notebook_drift_is_refused_before_any_install():
    with pytest.raises(ValueError, match="hash mismatch"):
        probe.notebook_pins(b'{"cells":[]}')


def test_failure_still_archives_source_and_diagnostics(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("synthetic fetch failure")
    monkeypatch.setattr(probe.urllib.request, "urlopen", unavailable)
    assert probe.main(tmp_path) == 2
    directory = next(p for p in tmp_path.iterdir() if p.is_dir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["phase"] == "fetch_pinned_notebook"
    assert manifest["model_weights_loaded"] is False
    assert manifest["model_inference"] is False
    assert manifest["files_sha256"]["preflight_source.py"] == probe.sha256(Path(probe.__file__).read_bytes())
    assert list(tmp_path.glob("*.zip"))
    assert not (directory / "metrics.json").exists()
