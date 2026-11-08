"""Model-free checks for the GPU notebook lifecycle and gates.

Extracts pure functions from the notebook with the AST. No torch, no model weights.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import json
import unittest
import weakref
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "gpu_parity_and_baseline.ipynb"
PREREG_PATH = ROOT / "docs" / "research" / "PREREG_baseline.md"
RUNBOOK_PATH = ROOT / "docs" / "GPU_RUNBOOK.md"

FUNC_NAMES = {
    "sha256_lf_file",
    "return_allocator_pages",
    "release_named",
    "clone_cached_logprobs",
    "_trace_event",
    "_finite_number",
    "_all_finite_numbers",
    "_all_finite_kls",
    "sha256_hex",
    "canonical_sha256",
    "redact_secrets",
    "make_phase_status",
    "checkpoint_phase",
    "wildguard_value",
    "qwen3_value",
    "descriptive_rate",
    "is_memory_failure",
    "host_ram_gib",
    "refuse_fp32_conversion_if_short",
    "plan_fp32_phase",
    "is_t4_accelerator",
    "bf16_cpu_device_or_refuse",
    "select_fp16_guard_reference",
    "orchestrate_short_fp32",
    "orchestrate_long_fp32",
    "cache_hf_logprobs",
    "decide_sequential_fp32_parity",
    "decide_bf16_variant",
    "decide_bf16_long",
    "combine_bf16_gate",
    "decide_fp16_guard",
    "_decoder_layers",
    "residual_hooks_attached",
    "attach_residual_hooks",
    "refuse_curated_prompts",
    "guard_output_meta",
    "fixture_output_meta",
    "check_dependency_closure",
}


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def notebook() -> dict:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def code_cells() -> list[str]:
    return [_cell_source(c) for c in notebook()["cells"] if c.get("cell_type") == "code"]


def cell_with(snippet: str) -> str:
    matches = [src for src in code_cells() if snippet in src]
    if len(matches) != 1:
        raise AssertionError(f"{snippet!r} found in {len(matches)} code cells")
    return matches[0]


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_literal(node.operand)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_literal(key) and _is_literal(val)
            for key, val in zip(node.keys, node.values)
        )
    return False


def load_namespace() -> dict:
    chunks: list[str] = []
    functions: list[str] = []
    for src in code_cells():
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign) and _is_literal(node.value):
                seg = ast.get_source_segment(src, node)
                if seg:
                    chunks.append(seg)
            elif isinstance(node, ast.FunctionDef) and node.name in FUNC_NAMES:
                seg = ast.get_source_segment(src, node)
                if not seg:
                    raise AssertionError(f"missing source for {node.name}")
                functions.append(seg)
    found = {ast.parse(fn).body[0].name for fn in functions}  # type: ignore[attr-defined]
    missing = FUNC_NAMES - found
    if missing:
        raise AssertionError(f"notebook is missing functions: {sorted(missing)}")
    ns: dict = {"__name__": "gpu_notebook_extract"}
    exec("\n\n".join(chunks + functions), ns)
    return ns


class _Box:
    pass


class TestGpuNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ns = load_namespace()

    def test_optional_fp32_plan_records_real_ram_without_credit(self):
        plan = self.ns["plan_fp32_phase"]
        for enabled in (False, True):
            result = plan(enabled, 29.05, 31.35)
            self.assertFalse(result["run"])
            self.assertEqual(result["not_run"], "insufficient_host_ram")
            self.assertEqual(result["ram_available_gib"], 29.05)
            self.assertEqual(result["ram_total_gib"], 31.35)
        self.assertEqual(plan(False, 40, 64)["not_run"], "disabled_by_config")
        self.assertTrue(plan(True, 40, 64)["run"])
        for value in (None, float("nan"), 0):
            self.assertEqual(plan(True, value, 64)["not_run"], "host_ram_unavailable")

    def test_short_skip_executes_without_any_model_factory(self):
        import time
        for enabled, available, total, reason in (
            (False, 29.05, 31.35, "insufficient_host_ram"),
            (True, 29.05, 31.35, "insufficient_host_ram"),
            (False, 40, 64, "disabled_by_config"),
            (False, 40, None, "host_ram_unavailable"),
        ):
            writes = {}
            def total_probe():
                if total is None:
                    raise OSError("synthetic telemetry failure")
                return total
            ns = dict(self.ns, time=time, RUN_FP32_PARITY=enabled,
                FORCE_PARITY_VARIANT=False, RUN_PARITY_VARIANT=False,
                free_ram_gib=lambda: available, host_ram_gib=total_probe,
                ARTIFACT_DIR=Path("synthetic-artifacts"),
                write_json=lambda path, payload: writes.update({path.name: dict(payload)}))
            # Imports would request real libraries; remove only top-level imports.
            tree = ast.parse(cell_with('"kind": "original_parity_sequential_fp32"'))
            tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
            exec(compile(tree, "synthetic_notebook_short", "exec"), ns)
            self.assertEqual(ns["parity"]["not_run"], reason)
            self.assertEqual(ns["parity"]["status"], "incomplete")
            self.assertFalse(ns["parity"]["passed"])
            self.assertFalse(ns["parity"]["ran_short"])
            self.assertTrue(ns["RUN_PARITY_VARIANT"])
            self.assertEqual(writes["fp32_not_run.json"]["status"], "not_run")
            if total is None:
                self.assertEqual(ns["parity"]["ram_total_error"], "OSError")

    def test_enabled_fp32_rerun_removes_prior_skip_before_loading(self):
        import tempfile
        import time
        import traceback
        import zipfile
        from scripts.ingest_gpu_artifacts import inspect_archive
        tree = ast.parse(cell_with('"kind": "original_parity_sequential_fp32"'))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        code = compile(tree, "synthetic_fp32_rerun", "exec")
        with tempfile.TemporaryDirectory() as work:
            out = Path(work)
            def write(path, payload):
                path.write_text(json.dumps(payload), encoding="utf-8")
            ns = dict(self.ns, time=time, traceback=traceback,
                RUN_FP32_PARITY=False, FORCE_PARITY_VARIANT=False,
                free_ram_gib=lambda: 40.0, host_ram_gib=lambda: 64.0,
                ARTIFACT_DIR=out, write_json=write,
                peak_rss_gib=lambda: 0.0, release_named=lambda *args, **kwargs: None)
            exec(code, ns)
            skip = out / "fp32_not_run.json"
            self.assertTrue(skip.exists())
            calls = []
            def fake_short(*args, **kwargs):
                self.assertFalse(skip.exists(), "stale skip must be removed before model loading")
                calls.append(True)
                return {prompt: 0.0 for prompt in ns["PARITY_PROMPTS"]}, {"window_size": 4096}
            ns.update(RUN_FP32_PARITY=True, orchestrate_short_fp32=fake_short)
            exec(code, ns)
            self.assertEqual(calls, [True])
            self.assertTrue(ns["parity"]["ran_short"])
            self.assertFalse(skip.exists())
            archive = out / "synthetic.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                path = out / "parity_original_sequential.json"
                zipped.writestr("hs_gpu_artifacts/" + path.name, path.read_bytes())
            _, summary = inspect_archive(archive)
            # Only short parity was simulated. Missing long never becomes pass.
            self.assertEqual(summary["original_parity"]["status"], "incomplete")

    def test_embedded_identity_helper_matches_reviewed_source(self):
        tree = ast.parse(cell_with("IDENTITY_HELPER_SOURCE ="))
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in tree.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in {"IDENTITY_HELPER_SOURCE", "IDENTITY_HELPER_SHA256"}}
        source = (ROOT / "scripts/kaggle_tokenizer_metadata.py").read_text(encoding="utf-8")
        self.assertEqual(values["IDENTITY_HELPER_SOURCE"], source)
        self.assertEqual(values["IDENTITY_HELPER_SHA256"], hashlib.sha256(source.encode()).hexdigest())
        identity_cell = cell_with("def capture_notebook_identity")
        self.assertIn('ARTIFACT_DIR / "identity.json"', identity_cell)
        self.assertNotIn('raise RuntimeError("Target identity unavailable', identity_cell)

    def test_identity_collection_is_retry_safe_and_archive_ready(self):
        import tempfile
        from scripts.ingest_gpu_artifacts import inspect_archive
        import zipfile
        source = cell_with("def capture_notebook_identity")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == "capture_notebook_identity")
        ns = {"json": json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "identity_test", "exec"), ns)
        seen = []
        asset = json.dumps({"chat_template": "synthetic template"}).encode()
        def collect(path, **kwargs):
            path.mkdir(parents=True, exist_ok=False)
            seen.append(path)
            (path / "tokenizer_config.json").write_bytes(asset)
            return {"artifact_kind": "setup-smoke", "split": "example", "weights_loaded": False,
                "model_imported": False, "generation_performed": False, "metrics_eligible": False, "primary_asr": None,
                "status": "metadata_retrieved", "model_id": "CohereLabs/tiny-aya-global",
                "requested_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
                "model_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
                "revision_matches_pin": True, "hf_token_present": True,
                "chat_template": "synthetic template",
                "tokenizer_files": {"tokenizer_config.json": {"sha256": hashlib.sha256(asset).hexdigest(), "size": len(asset)}},
                "tokenizer_config_sha256": hashlib.sha256(asset).hexdigest(),
                "chat_template_sha256": hashlib.sha256(b"synthetic template").hexdigest()}
        with tempfile.TemporaryDirectory() as work:
            out = Path(work) / "artifacts"
            out.mkdir()
            for run_id in ("first", "second", "second"):
                result = ns["capture_notebook_identity"](collect, out, run_id, "c" * 40, "d" * 64)
                self.assertEqual(result["notebook_run_id"], run_id)
                self.assertEqual(json.loads((out / "identity.json").read_text())["notebook_run_id"], run_id)
            self.assertEqual(len(set(seen)), 3)
            archive = Path(work) / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                for path in out.iterdir():
                    zipped.writestr("hs_gpu_artifacts/" + path.name, path.read_bytes())
            files, summary = inspect_archive(archive)
            self.assertEqual(set(files), {"identity.json", "tokenizer_config.json", "identity_status.json"})
            self.assertEqual(summary["target_identity"]["status"], "metadata_retrieved")
            self.assertEqual(summary["phase_status"]["identity"]["status"], "ok")
            self.assertFalse(summary["metrics_eligible"])
            def unavailable(path, **kwargs):
                path.mkdir(parents=True, exist_ok=False)
                return {"status": "unavailable"}
            ns["capture_notebook_identity"](unavailable, out, "third", "c" * 40, "d" * 64)
            self.assertFalse((out / "tokenizer_config.json").exists())
            self.assertEqual(json.loads((out / "identity.json").read_text())["notebook_run_id"], "third")

    def test_release_named_removes_every_name_and_weakrefs_expire(self) -> None:
        release_named = self.ns["release_named"]
        space: dict = {}
        left = _Box()
        right = _Box()
        keep = _Box()
        space["left"] = left
        space["right"] = right
        space["keep"] = keep
        w_left = weakref.ref(left)
        w_right = weakref.ref(right)
        w_keep = weakref.ref(keep)
        del left, right, keep
        release_named("left", "right", "missing", namespace=space)
        self.assertNotIn("left", space)
        self.assertNotIn("right", space)
        self.assertIn("keep", space)
        self.assertIsNone(w_left())
        self.assertIsNone(w_right())
        self.assertIsNotNone(w_keep())

    def test_clone_cached_logprobs_drops_source_storage(self) -> None:
        clone_cached_logprobs = self.ns["clone_cached_logprobs"]

        class Tensor:
            def __init__(self, storage: bytearray) -> None:
                self.storage = storage

            def detach(self) -> "Tensor":
                return self

            def clone(self) -> "Tensor":
                return Tensor(bytearray(self.storage))

            def cpu(self) -> "Tensor":
                return self

        src = Tensor(bytearray(b"abc"))
        alive = weakref.ref(src)
        out = clone_cached_logprobs(src)
        self.assertIsNot(out.storage, src.storage)
        self.assertEqual(bytes(out.storage), b"abc")
        del src
        gc.collect()
        self.assertIsNone(alive())
        self.assertEqual(bytes(out.storage), b"abc")

    def test_short_and_long_forwards_are_not_simultaneous(self) -> None:
        ns = self.ns
        space: dict = {}
        trace: list = []
        made: list[weakref.ref] = []
        tl_loads = {"n": 0}

        def _fresh() -> _Box:
            obj = _Box()
            made.append(weakref.ref(obj))
            return obj

        def load_tok() -> _Box:
            return _fresh()

        def load_hf() -> _Box:
            return _fresh()

        def load_tl(donor: object) -> _Box:
            self.assertIsNotNone(donor)
            tl_loads["n"] += 1
            return _fresh()

        def read_cfg(_model: object) -> dict:
            return {"window_size": 4096, "default_prepend_bos": False}

        def forward_hf(_model: object, _tok: object, prompt: str) -> str:
            self.assertNotIn("tl", space)
            return f"hf:{prompt}"

        def forward_tl(_model: object, _tok: object, prompt: str) -> str:
            self.assertNotIn("hf", space)
            self.assertNotIn("hf_for_tl", space)
            return f"tl:{prompt}"

        def kl(_hf: object, _tl: object) -> float:
            return 0.0

        kls, cfg = ns["orchestrate_short_fp32"](
            ["a", "b"],
            load_tok,
            load_hf,
            load_tl,
            forward_hf,
            forward_tl,
            kl,
            read_cfg,
            space,
            trace=trace,
        )
        self.assertEqual(kls, {"a": 0.0, "b": 0.0})
        self.assertEqual(cfg["window_size"], 4096)
        self.assertEqual(tl_loads["n"], 1)
        events = [event for event, _resident in trace]
        self.assertEqual(events, ["forward_hf_begin", "convert", "forward_tl_begin"])
        for event, resident in trace:
            if event == "forward_hf_begin":
                self.assertIn("hf", resident)
                self.assertNotIn("tl", resident)
            if event == "forward_tl_begin":
                self.assertIn("tl", resident)
                self.assertNotIn("hf", resident)
                self.assertNotIn("hf_for_tl", resident)
            if event == "convert":
                self.assertIn("tl", resident)
                self.assertIn("hf_for_tl", resident)
        gc.collect()
        self.assertTrue(made)
        self.assertTrue(all(ref() is None for ref in made))
        self.assertFalse(space)

        space2: dict = {}
        trace2: list = []
        pre = _Box()
        other = _Box()
        space2["tl"] = pre
        space2["PARITY_TL"] = other
        w_pre = weakref.ref(pre)
        w_other = weakref.ref(other)
        del pre, other
        seen: dict = {}

        def forward_hf_last(_model: object, _ids: object) -> str:
            seen["tl_in_space"] = "tl" in space2
            seen["pre"] = w_pre()
            seen["other"] = w_other()
            return "hf-lp"

        def forward_tl_last(_model: object, _ids: object) -> str:
            self.assertNotIn("hf", space2)
            self.assertNotIn("hf_for_tl", space2)
            return "tl-lp"

        def build_ids(_tok: object, cfg_in: dict) -> str:
            self.assertEqual(cfg_in["window_size"], 4096)
            self.assertNotIn("tl", space2)
            return "ids"

        tl_loads["n"] = 0
        kl_long, _cfg = ns["orchestrate_long_fp32"](
            load_tok,
            load_hf,
            load_tl,
            read_cfg,
            build_ids,
            forward_hf_last,
            forward_tl_last,
            kl,
            space2,
            cfg_snapshot={"window_size": 4096, "default_prepend_bos": True},
            trace=trace2,
        )
        self.assertEqual(kl_long, 0.0)
        self.assertEqual(tl_loads["n"], 1)
        self.assertEqual(seen["tl_in_space"], False)
        self.assertIsNone(seen["pre"])
        self.assertIsNone(seen["other"])
        for event, resident in trace2:
            if event == "forward_hf_begin":
                self.assertNotIn("tl", resident)
                self.assertNotIn("PARITY_TL", resident)
                self.assertIn("hf", resident)
            if event == "forward_tl_begin":
                self.assertNotIn("hf", resident)
                self.assertNotIn("hf_for_tl", resident)
                self.assertIn("tl", resident)

        space3: dict = {}
        tl_loads["n"] = 0
        ns["orchestrate_long_fp32"](
            load_tok,
            load_hf,
            load_tl,
            read_cfg,
            build_ids,
            forward_hf_last,
            forward_tl_last,
            kl,
            space3,
            cfg_snapshot=None,
            trace=[],
        )
        self.assertEqual(tl_loads["n"], 2)

    def test_bf16_path_loads_tl_with_no_donor(self) -> None:
        ns = self.ns
        self.assertIn(ns["return_allocator_pages"](), (True, False))
        bf16 = cell_with("def _load_tl_bf16")
        self.assertIn("from_pretrained_no_processing", bf16)
        self.assertIn("keep_hf_donor=False", bf16)
        self.assertIn("bf16 TL load refuses a live HF donor", bf16)
        self.assertNotIn("hf_model=hf_model", bf16)
        self.assertIn("from_pretrained_no_processing", cell_with("d6_bf16_tl_load"))

        space: dict = {}
        donors: list = []
        hf_loads = {"n": 0}

        def load_hf() -> _Box:
            hf_loads["n"] += 1
            return _Box()

        def load_tl(donor: object) -> _Box:
            donors.append(donor)
            return _Box()

        kls, cfg = ns["orchestrate_short_fp32"](
            ["a"],
            lambda: _Box(),
            load_hf,
            load_tl,
            lambda _model, _tok, prompt: prompt,
            lambda _model, _tok, prompt: prompt,
            lambda _hf, _tl: 0.0,
            lambda _model: {"window_size": 8},
            space,
            keep_hf_donor=False,
        )
        self.assertEqual(kls, {"a": 0.0})
        self.assertEqual(cfg["window_size"], 8)
        self.assertEqual(hf_loads["n"], 1)
        self.assertEqual(donors, [None])
        self.assertFalse(space)

        hf_loads["n"] = 0
        donors.clear()
        space_long: dict = {}
        kl_long, _cfg = ns["orchestrate_long_fp32"](
            lambda: _Box(),
            load_hf,
            load_tl,
            lambda _model: {"window_size": 8},
            lambda _tok, _cfg: "ids",
            lambda _model, _ids: "hf",
            lambda _model, _ids: "tl",
            lambda _a, _b: 0.0,
            space_long,
            cfg_snapshot={"window_size": 8},
            keep_hf_donor=False,
        )
        self.assertEqual(kl_long, 0.0)
        self.assertEqual(hf_loads["n"], 1)
        self.assertEqual(donors, [None])
        self.assertFalse(space_long)

    def test_long_hf_forward_raises_when_tl_cannot_be_released(self) -> None:
        original = self.ns["release_named"]

        def noop(*_names, namespace=None):
            return None

        self.ns["release_named"] = noop
        try:
            space = {"tl": _Box()}
            with self.assertRaises(RuntimeError) as caught:
                self.ns["orchestrate_long_fp32"](
                    lambda: _Box(),
                    lambda: _Box(),
                    lambda _donor: _Box(),
                    lambda _model: {"window_size": 4096, "default_prepend_bos": False},
                    lambda _tok, _cfg: "ids",
                    lambda _model, _ids: "hf",
                    lambda _model, _ids: "tl",
                    lambda _a, _b: 0.0,
                    space,
                    cfg_snapshot={"window_size": 4096, "default_prepend_bos": False},
                )
            self.assertIn("resident", str(caught.exception).lower())
            self.assertIn("tl", space)
        finally:
            self.ns["release_named"] = original

    def test_missing_long_and_nonfinite_cannot_pass(self) -> None:
        decide = self.ns["decide_sequential_fp32_parity"]
        missing = decide({"prompt": 1e-8}, None, long_ran=False)
        self.assertEqual(missing["status"], "incomplete")
        self.assertFalse(missing["passed"])
        nonfinite = decide({"prompt": float("nan")}, {"sliding_window_4196": 1e-8}, long_ran=True)
        self.assertFalse(nonfinite["passed"])
        long_bad = decide({"prompt": 1e-8}, {"sliding_window_4196": float("inf")}, long_ran=True)
        self.assertFalse(long_bad["passed"])
        both = decide({"prompt": 1e-8}, {"sliding_window_4196": 1e-8}, long_ran=True)
        self.assertTrue(both["passed"])
        self.assertEqual(both["status"], "pass")

        short = self.ns["decide_bf16_variant"]({"p": 1e-6}, {"p": 1e-6})
        self.assertTrue(short["passed_kl"])
        long_missing = self.ns["decide_bf16_long"](None, None, long_ran=False)
        combined = self.ns["combine_bf16_gate"](short, 1.0, long_missing)
        self.assertEqual(combined["status"], "incomplete")
        self.assertFalse(combined["passed"])
        self.assertFalse(self.ns["decide_bf16_variant"]({"p": float("nan")}, {"p": 1e-6})["passed_kl"])

        guard = self.ns["decide_fp16_guard"](1e-3, 0.99, 18, 20, False, True, 19)
        self.assertFalse(guard["passed"])
        guard_nan = self.ns["decide_fp16_guard"](1e-3, 0.99, 18, 20, True, True, 20)
        self.assertFalse(guard_nan["passed"])
        guard_ok = self.ns["decide_fp16_guard"](
            1e-3, 0.99, 18, 20, False, True, 20, kl_values=[1e-3] * 20
        )
        self.assertTrue(guard_ok["passed"])
        guard_omitted = self.ns["decide_fp16_guard"](1e-3, 0.99, 18, 20, False, True, 20)
        self.assertFalse(guard_omitted["passed"])
        guard_short = self.ns["decide_fp16_guard"](
            1e-3, 0.99, 18, 20, False, True, 20, kl_values=[1e-3] * 19
        )
        self.assertFalse(guard_short["passed"])
        guard_nan_kl = self.ns["decide_fp16_guard"](
            1e-3, 0.99, 18, 20, False, True, 20, kl_values=[1e-3] * 19 + [float("nan")]
        )
        self.assertFalse(guard_nan_kl["passed"])
        guard_inf_kl = self.ns["decide_fp16_guard"](
            1e-3, 0.99, 18, 20, False, True, 20, kl_values=[1e-3] * 19 + [float("inf")]
        )
        self.assertFalse(guard_inf_kl["passed"])
        guard_neg_kl = self.ns["decide_fp16_guard"](
            1e-3, 0.99, 18, 20, False, True, 20, kl_values=[1e-3] * 19 + [-1e-12]
        )
        self.assertFalse(guard_neg_kl["passed"])

    def test_bf16_long_does_not_use_fp32_tolerance(self) -> None:
        decide_long = self.ns["decide_bf16_long"]
        # 2e-4 is above the fp32 gate (1e-4) and under the predeclared 1e-2 cap.
        provisional = decide_long({"sliding_window_4196": 2e-4}, None, long_ran=True)
        self.assertTrue(provisional["passed"])
        self.assertTrue(provisional["provisional"])
        self.assertFalse(provisional["used_fp32_tol"])
        too_wide = decide_long({"sliding_window_4196": 2e-2}, None, long_ran=True)
        self.assertFalse(too_wide["passed"])
        self.assertEqual(too_wide["status"], "fail")
        # Floor rule is 2x the measured floor, still not 1e-4.
        above_floor = decide_long(
            {"sliding_window_4196": 3e-4},
            {"sliding_window_4196": 1e-4},
            long_ran=True,
        )
        self.assertFalse(above_floor["passed"])
        within_floor = decide_long(
            {"sliding_window_4196": 2e-4},
            {"sliding_window_4196": 1e-4},
            long_ran=True,
        )
        self.assertTrue(within_floor["passed"])
        self.assertFalse(within_floor["provisional"])
        variant_src = cell_with("bf16_parity_variant")
        self.assertNotIn("KL_TOL", variant_src)
        self.assertIn("decide_bf16_long", variant_src)

    def test_invalid_kl_cannot_pass_any_numeric_gate(self) -> None:
        original = self.ns["decide_sequential_fp32_parity"]
        short = self.ns["decide_bf16_variant"]
        long = self.ns["decide_bf16_long"]
        for invalid in (-1.0, -1e-12, True, False, "0", float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                self.assertFalse(original({"p": invalid}, {"long": 0.0}, long_ran=True)["passed"])
                self.assertFalse(original({"p": 0.0}, {"long": invalid}, long_ran=True)["passed"])
                self.assertFalse(short({"p": invalid}, None)["passed_kl"])
                self.assertFalse(short({"p": 0.0}, {"p": invalid})["passed_kl"])
                self.assertFalse(long({"long": invalid}, None, long_ran=True)["passed"])
                self.assertFalse(long({"long": 0.0}, {"long": invalid}, long_ran=True)["passed"])
        # Negative entries must fail even when a positive maximum would hide them.
        self.assertFalse(original({"p": -1e-12, "q": 0.0}, {"long": 0.0}, long_ran=True)["passed"])
        self.assertFalse(short({"p": 0.0}, {"other": 0.0})["passed_kl"])
        self.assertFalse(long({"long": 0.0}, {"other": 0.0}, long_ran=True)["passed"])
        self.assertFalse(self.ns["decide_fp16_guard"](-1e-12, 1.0, 20, 20, False, True, 20)["passed"])
        self.assertTrue(original({"p": 0.0}, {"long": 0.0}, long_ran=True)["passed"])
        self.assertFalse(original({"p": 1e-4}, {"long": 0.0}, long_ran=True)["passed"])
        self.assertTrue(short({"p": 0.0}, {"p": 0.0})["passed_kl"])
        self.assertTrue(long({"long": 0.0}, {"long": 0.0}, long_ran=True)["passed"])
        guard = self.ns["decide_fp16_guard"]
        mixed = [0.001] * 19 + [-0.001]
        self.assertFalse(guard(sum(mixed) / 20, 1.0, 20, 20, False, True, 20, kl_values=mixed)["passed"])
        valid = [0.001] * 20
        self.assertTrue(guard(0.001, 1.0, 20, 20, False, True, 20, kl_values=valid)["passed"])
        self.assertIn("kl_values=kls", cell_with("guard = {"))

    def test_every_parity_checkpoint_load_uses_the_same_revision(self) -> None:
        expected = "d62ac17e241c871fc52e9a95d7b0e2615189d7d5"
        self.assertEqual(self.ns["MODEL_PARITY_REV"], expected)
        loads = []
        for src in code_cells():
            for node in ast.walk(ast.parse(src)):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in {"from_pretrained", "from_pretrained_no_processing"} or not node.args:
                    continue
                if not isinstance(node.args[0], ast.Name) or node.args[0].id != "MODEL_PARITY":
                    continue
                revision = next((k.value for k in node.keywords if k.arg == "revision"), None)
                self.assertIsInstance(revision, ast.Name)
                self.assertEqual(revision.id, "MODEL_PARITY_REV")
                if node.func.attr == "from_pretrained_no_processing":
                    self.assertFalse(any(k.arg == "hf_model" for k in node.keywords))
                if any(k.arg == "hf_model" for k in node.keywords):
                    tokenizer = next((k.value for k in node.keywords if k.arg == "tokenizer"), None)
                    self.assertIsInstance(tokenizer, ast.Subscript)
                    self.assertEqual(ast.unparse(tokenizer), "ns['tok']")
                loads.append(node)
        self.assertEqual(len(loads), 9)
        self.assertEqual(
            sum(node.func.attr == "from_pretrained_no_processing" for node in loads),
            1,
        )
        self.assertIn('"model_revision": MODEL_PARITY_REV', cell_with("original_parity_sequential_fp32"))
        self.assertIn('"model_revision": MODEL_PARITY_REV', cell_with("bf16_parity_variant"))

    def test_install_resolves_shared_stack_once_and_checks_before_compat_import(self) -> None:
        src = cell_with("def pip_install")
        tree = ast.parse(src)
        installs = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name) and n.value.func.id == "pip_install"]
        self.assertEqual(len(installs), 1)
        args = ast.get_source_segment(src, installs[0])
        for pin in ("TRANSFORMERS_PIN", "HF_HUB_PIN", "CIRCUITKIT_SHA", "SAFETUNE_SHA"):
            self.assertIn(pin, args)
        check_line = src.index('"pip", "check"')
        compat_line = src.index("import circuitkit.backends._tl_compat")
        self.assertLess(check_line, compat_line)
        self.assertIn('"dependency_check.json"', src)
        self.assertIn('if not dependency_check["passed"]:', src)
        self.assertIn('"dependency_check_global.json"', src)
        self.assertIn('check_dependency_closure(DEPENDENCY_ROOTS)', src)
        self.assertIn('observed != expected', src)

    def test_absent_or_empty_hooks_fail(self) -> None:
        attach = self.ns["attach_residual_hooks"]
        attached = self.ns["residual_hooks_attached"]
        decide = self.ns["decide_fp16_guard"]
        handles, _bad = attach(object())
        self.assertEqual(handles, [])
        self.assertFalse(attached(handles))

        class Empty:
            def __init__(self) -> None:
                self.model = type("M", (), {"layers": []})()

        handles, _bad = attach(Empty())
        self.assertEqual(handles, [])
        self.assertFalse(decide(1e-4, 1.0, 20, 20, False, handles, 20)["passed"])
        self.assertFalse(decide(1e-4, 1.0, 20, 20, False, False, 20)["passed"])

        class Layer:
            def register_forward_hook(self, fn):
                return fn

        class Root:
            def __init__(self) -> None:
                self.model = type("M", (), {})()
                self.model.layers = [Layer(), Layer()]

        handles, _bad = attach(Root())
        self.assertEqual(len(handles), 2)
        self.assertTrue(attached(handles))
        self.assertTrue(
            decide(1e-4, 1.0, 20, 20, False, True, 20, kl_values=[1e-4] * 20)["passed"]
        )

    def test_timer_is_assigned_before_use(self) -> None:
        src = cell_with("original_parity_sequential_fp32")
        tree = ast.parse(src)
        stores: list[int] = []
        loads: list[int] = []

        class Visitor(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name) -> None:
                if node.id == "t_short0":
                    if isinstance(node.ctx, ast.Store):
                        stores.append(node.lineno)
                    elif isinstance(node.ctx, ast.Load):
                        loads.append(node.lineno)
                self.generic_visit(node)

        Visitor().visit(tree)
        self.assertTrue(stores)
        self.assertTrue(loads)
        self.assertLess(min(stores), min(loads))

    def test_variant_auto_only_for_memory_failures(self) -> None:
        is_memory = self.ns["is_memory_failure"]
        self.assertTrue(is_memory(MemoryError("available RAM 2 GiB")))
        self.assertFalse(is_memory(RuntimeError("401 Client Error: Unauthorized")))
        self.assertFalse(is_memory(NameError("t_short0")))
        self.assertFalse(is_memory(AssertionError("gated repo")))
        self.assertFalse(is_memory(OSError("401 unauthorized")))

        class OutOfMemoryError(RuntimeError):
            pass

        self.assertTrue(is_memory(OutOfMemoryError("CUDA out of memory")))
        self.assertTrue(is_memory(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))

        refuse = self.ns["refuse_fp32_conversion_if_short"]
        with self.assertRaises(MemoryError):
            refuse(28.0, 31.3, 32.0)
        with self.assertRaises(MemoryError):
            refuse(40.0, 31.3, 32.0)
        self.assertIsNone(refuse(40.0, 64.0, 32.0))
        self.assertIsNone(refuse(None, 31.3, 32.0))
        self.assertEqual(self.ns["CONVERSION_NEED_GIB"], 32.0)
        self.assertEqual(self.ns["ONE_COPY_FP32_GIB"], 13.0)
        for snippet in ("original_parity_sequential_fp32", "original_parity_long_prompt"):
            self.assertIn("refuse_fp32_conversion_if_short", cell_with(snippet))

        for snippet in ("original_parity_sequential_fp32", "original_parity_long_prompt"):
            tree = ast.parse(cell_with(snippet))
            depths: list[int] = []

            class Visitor(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.if_depth = 0

                def visit_If(self, node: ast.If) -> None:
                    self.if_depth += 1
                    self.generic_visit(node)
                    self.if_depth -= 1

                def visit_Assign(self, node: ast.Assign) -> None:
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "RUN_PARITY_VARIANT"
                            and isinstance(node.value, ast.Constant)
                            and node.value.value is True
                        ):
                            depths.append(self.if_depth)
                    self.generic_visit(node)

            visitor = Visitor()
            visitor.visit(tree)
            self.assertTrue(depths, snippet)
            self.assertTrue(all(depth >= 1 for depth in depths), snippet)

    def test_cpu_bf16_only_on_t4(self) -> None:
        allow = self.ns["bf16_cpu_device_or_refuse"]
        select = self.ns["select_fp16_guard_reference"]
        self.assertEqual(allow("Tesla T4"), "cpu")
        self.assertEqual(allow("NVIDIA T4"), "cpu")
        with self.assertRaises(RuntimeError):
            allow("NVIDIA A100-SXM4-40GB")
        with self.assertRaises(RuntimeError):
            allow(None)
        self.assertEqual(select(20.0, "Tesla T4")["reference"], "fp32_cpu")
        low_t4 = select(10.0, "Tesla T4")
        self.assertEqual(low_t4["reference"], "bf16_cpu")
        self.assertTrue(low_t4["ok"])
        off = select(10.0, "NVIDIA A10")
        self.assertFalse(off["ok"])
        self.assertEqual(off["reason"], "bf16_cpu_only_on_t4")

    def test_curated_jsonl_refused_and_fixture_meta(self) -> None:
        refuse = self.ns["refuse_curated_prompts"]
        self.assertIsNone(refuse(""))
        self.assertIsNone(refuse(None))
        with self.assertRaises(RuntimeError):
            refuse("data/curated/hs_harm_en/dev/hs_eval.jsonl")
        meta = self.ns["fixture_output_meta"]()
        self.assertEqual(meta["artifact_kind"], "fixture")
        self.assertEqual(meta["split"], "example")
        self.assertEqual(meta["max_new_tokens"], 64)
        guard_meta = self.ns["guard_output_meta"]()
        self.assertEqual(guard_meta["artifact_kind"], "setup-smoke")
        self.assertEqual(guard_meta["split"], "example")
        baseline = cell_with("FIXTURE_PROMPTS")
        self.assertNotIn('"split": "dev"', baseline)
        self.assertNotIn("'split': 'dev'", baseline)

    def test_circuitkit_import_precedes_transformerlens_in_every_loader(self) -> None:
        expected = {"_load_tl_fp32", "_load_tl_fp32_long", "_load_tl_bf16"}
        checked = set()
        for src in code_cells():
            for node in ast.walk(ast.parse(src)):
                if not isinstance(node, ast.FunctionDef) or node.name not in expected:
                    continue
                ck = [n.lineno for n in ast.walk(node) if isinstance(n, ast.Import)
                      and any(a.name == "circuitkit.backends._tl_compat" for a in n.names)]
                tl = [n.lineno for n in ast.walk(node) if isinstance(n, ast.ImportFrom)
                      and n.module == "transformer_lens"]
                self.assertTrue(ck and tl, node.name)
                self.assertLess(min(ck), min(tl), node.name)
                checked.add(node.name)
        self.assertEqual(checked, expected)
        install = ast.parse(cell_with("def pip_install"))
        self.assertTrue(any(isinstance(n, ast.Import) and
                            any(a.name == "circuitkit.backends._tl_compat" for a in n.names)
                            for n in install.body))

    def test_notebook_outputs_empty_and_provenance(self) -> None:
        nb = notebook()
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            self.assertEqual(list(cell.get("outputs") or []), [])
            self.assertIsNone(cell.get("execution_count"))
        owned = "\n".join(
            [
                NB_PATH.read_text(encoding="utf-8"),
                PREREG_PATH.read_text(encoding="utf-8"),
                RUNBOOK_PATH.read_text(encoding="utf-8"),
            ]
        )
        self.assertNotIn("99a3829", owned)
        self.assertNotIn("\u2014", owned)
        prereg = PREREG_PATH.read_text(encoding="utf-8")
        self.assertIn("D4 (2026-11-07", prereg)
        self.assertIn("25 GiB", prereg)
        self.assertIn("VARIANT_LONG_PROVISIONAL_KL_MAX", prereg)
        file_sha = self.ns["sha256_lf_file"](PREREG_PATH)
        pinned = None
        for src in code_cells():
            tree = ast.parse(src)
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "PREREG_SOURCE_SHA256":
                        pinned = ast.literal_eval(node.value)
        self.assertEqual(pinned, file_sha)
        self.assertEqual(len(file_sha), 64)

        # Copied upstream hashes stay the recorded CircuitKIT copy.
        config = cell_with("PARITY_KL_SOURCE =")
        tree = ast.parse(config)
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "PARITY_PROMPTS",
                        "PARITY_KL_SOURCE",
                        "PARITY_LONG_SPEC",
                    }:
                        values[target.id] = ast.literal_eval(node.value)
        self.assertEqual(
            hashlib.sha256(json.dumps(values["PARITY_PROMPTS"], ensure_ascii=False).encode("utf-8")).hexdigest(),
            "3ab3ad15fba852e34e08b8c485e2e3c41adc8702aa80a0bbf62215d5001fabab",
        )
        norm = "\n".join(line.rstrip() for line in values["PARITY_KL_SOURCE"].strip().splitlines()) + "\n"
        self.assertEqual(
            hashlib.sha256(norm.encode("utf-8")).hexdigest(),
            "7bef5687a6fd0194baa50c0156c8d9a6c1e827abc87f154feb7eca527079a812",
        )
        self.assertEqual(
            hashlib.sha256(
                json.dumps(values["PARITY_LONG_SPEC"], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "03975892c0e2d9d70159ebbad34c883557c90eb800e4d786a115362cd60de9d1",
        )

    def test_phase_status_is_fail_closed(self) -> None:
        make = self.ns["make_phase_status"]
        payload = make("d8", "failed", run_id="r", timestamp="t", error_type="E", error="x")
        self.assertFalse(payload["metrics_eligible"])
        self.assertIsNone(payload["primary_asr"])
        with self.assertRaises(ValueError):
            make("d8", "ok", metrics_eligible=True)
        with self.assertRaises(ValueError):
            make("d8", "ok", primary_asr=0.1)

    def test_checkpoint_phase_rebuilds_zip(self) -> None:
        import tempfile
        import zipfile
        checkpoint = self.ns["checkpoint_phase"]
        with tempfile.TemporaryDirectory() as work:
            base = Path(work)
            artifacts = base / "hs_gpu_artifacts"
            artifacts.mkdir()
            (artifacts / "identity.json").write_text("{}", encoding="utf-8")
            (artifacts / "d8_status.json").write_text("{}", encoding="utf-8")
            target = base / "hs_gpu_artifacts_run.zip"
            checkpoint(artifacts, target, "d8")
            with zipfile.ZipFile(target) as zipped:
                names = sorted(zipped.namelist())
            self.assertEqual(names, ["hs_gpu_artifacts/d8_status.json", "hs_gpu_artifacts/identity.json"])

    def test_judge_value_parsers_match_pinned_contracts(self) -> None:
        wg = self.ns["wildguard_value"]
        value, fields = wg("harmful request: yes\nresponse refusal: no\nharmful response: yes")
        self.assertEqual(value, 1.0)
        self.assertTrue(fields["unsafe"])
        value, fields = wg("harmful response: no")
        self.assertEqual((value, fields["unsafe"]), (0.0, False))
        qwen = self.ns["qwen3_value"]
        good = qwen("Safety: Unsafe\nCategories: Violent\nRefusal: No")
        self.assertEqual((good["label"], good["value"]), ("unsafe", 1.0))
        bad = qwen("Safety: Controversial\nCategories: Violent\nRefusal: No")
        self.assertEqual((bad["label"], bad["value"]), ("secondary_controversial", None))
        self.assertEqual(qwen("not a verdict")["label"], "judge_failure")
        self.assertEqual(self.ns["descriptive_rate"](2, 4), 0.5)
        self.assertIsNone(self.ns["descriptive_rate"](0, 0))

    def test_redact_and_hash_helpers(self) -> None:
        self.assertEqual(self.ns["sha256_hex"](b"abc"), hashlib.sha256(b"abc").hexdigest())
        payload = {"b": 1, "a": 2}
        self.assertEqual(self.ns["canonical_sha256"](payload),
                         hashlib.sha256(json.dumps({"a": 2, "b": 1},
                                                    ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":"), allow_nan=False
                                                    ).encode("utf-8")).hexdigest())
        redacted = self.ns["redact_secrets"]("token hf_" + "x" * 30 + " leaked")
        self.assertNotIn("x" * 30, redacted)
        self.assertIn("[redacted]", redacted)

    def test_d8_and_locate_cells_chain_with_checkpoints(self) -> None:
        d8 = cell_with("def fetch_d8_rows")
        for marker in ("d8_descriptive.json", "d8_status.json", "d8_generations.jsonl",
                       "d8_judge_scores.jsonl", "d8_descriptive_status.json",
                       "WildGuard", "Qwen3Guard", "checkpoint_phase",
                       "automated-judge only, human calibration pending",
                       "HS_DATA_FILES", "HS_DATA_BASE"):
            self.assertIn(marker, d8)
        self.assertNotIn("test_locked", d8)
        self.assertNotIn("hs_benign_en/test", d8)
        locate = cell_with("provisional_locate_v1")
        for marker in ("MLP 0", "locate_direction.json", "locate_status.json",
                       "checkpoint_phase", "original_fp32_credit",
                       "alpha_zero_with_wrapper", "random_norm_matched_seed_7",
                       "candidate_alpha_1", "ActivationSteering"):
            self.assertIn(marker, locate)
        identity = cell_with("def capture_notebook_identity")
        self.assertIn("identity_status.json", identity)
        self.assertIn("checkpoint_phase", identity)
        self.assertIn('target_identity.get("status") != "metadata_retrieved"', d8)
        self.assertIn('target_identity.get("status") != "metadata_retrieved"', locate)

    def test_d8_data_pin_matches_repository_files(self) -> None:
        config = cell_with("HS_DATA_FILES = {")
        tree = ast.parse(config)
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"HS_DATA_FILES", "HS_DATA_COMMIT"}:
                        values[target.id] = ast.literal_eval(node.value)
        self.assertEqual(len(values["HS_DATA_COMMIT"]), 40)
        total = 0
        for slice_name, (relpath, expected) in values["HS_DATA_FILES"].items():
            data = (ROOT / relpath).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, slice_name)
            total += sum(1 for line in data.decode("utf-8").splitlines() if line.strip())
        self.assertEqual(total, 771)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
