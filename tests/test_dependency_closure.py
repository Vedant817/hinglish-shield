from types import SimpleNamespace

from scripts.dependency_closure import check_dependency_closure


def dist(version, *requires):
    return SimpleNamespace(version=version, requires=list(requires))


def test_transitive_constraints_and_missing_dependencies_are_not_ignored():
    packages = {"root": dist("1", "mid>=1"), "mid": dist("1", "leaf<2"), "leaf": dist("3")}
    report = check_dependency_closure(["root"], lookup=packages.__getitem__)
    assert report["passed"] is False
    assert report["errors"][0]["package"] == "mid"
    del packages["leaf"]
    assert check_dependency_closure(["root"], lookup=packages.__getitem__)["passed"] is False


def test_requested_extras_are_checked_but_unrelated_broken_packages_are_not_certified():
    packages = {"root": dist("1", "child[feature]>=1"),
                "child": dist("1", 'leaf>=2; extra == "feature"'), "leaf": dist("1"),
                "unrelated": dist("1", "absent")}
    report = check_dependency_closure(["root"], lookup=packages.__getitem__)
    assert report["passed"] is False
    packages["leaf"] = dist("2")
    report = check_dependency_closure(["root"], lookup=packages.__getitem__)
    assert report["passed"] is True
    assert "unrelated" not in report["checked_versions"]


def test_inactive_marker_and_cycles_are_handled():
    packages = {"root": dist("1", "child>=1", 'missing; python_version < "1"'),
                "child": dist("1", "root>=1")}
    assert check_dependency_closure(["root"], lookup=packages.__getitem__)["passed"] is True


def test_notebook_uses_the_same_checker_source():
    import ast
    import inspect
    import json
    from pathlib import Path
    script = ast.parse(inspect.getsource(check_dependency_closure)).body[0]
    notebook = json.loads(Path("notebooks/gpu_parity_and_baseline.ipynb").read_text(encoding="utf-8"))
    found = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            found.extend(n for n in ast.parse("".join(cell["source"])).body
                         if isinstance(n, ast.FunctionDef) and n.name == "check_dependency_closure")
    assert len(found) == 1
    assert ast.dump(script) == ast.dump(found[0])
