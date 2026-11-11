"""CPU dry-run of the GPU notebook's identity-D8-Locate-zip chain.

Executes the notebook's identity, D8 and Locate cells end to end with a tiny
stub model/tokenizer and stub judges. The notebook's parity/fixture cells
keep their existing model-free coverage in tests/test_gpu_notebook.py; this
harness covers the new identity -> D8 -> Locate -> checkpoint -> ingest lane
so plumbing cannot crash on a dict-valued chat template again.

Stubs collect no numbers: the repo Qwen judge pins are replaced by stub
hashes (off-line preimage is impossible, and those pins assert only inside
the notebook's judged path). The target tokenizer uses the real template
string from runs/gpu-20261109-092319-22ece545/identity.json, whose hash is
the committed pin 2147c474; the tokenizer here presents it as a dict so the
default-entry selection runs exactly as on Kaggle.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "gpu_parity_and_baseline.ipynb"
IDENTITY_SEED = ROOT / "runs" / "gpu-20261109-092319-22ece545" / "identity.json"

VOCAB = 8
BASE_LOGITS = [1.0] * VOCAB


class FakeFloat:
    def __init__(self, value):
        self._value = float(value)

    def item(self):
        return self._value


class FakeTensor:
    def __init__(self, values, device="cpu", dtype="float16"):
        self.data = [float(v) for v in values]
        self.device = device
        self.dtype = dtype

    def _like(self, values, device=None, dtype=None):
        return FakeTensor(values,
                          device=self.device if device is None else device,
                          dtype=self.dtype if dtype is None else dtype)

    def to(self, *args, **kwargs):
        device = self.device
        dtype = self.dtype
        if args:
            first = args[0]
            if isinstance(first, str) or hasattr(first, "type"):
                device = first
            elif first is not None:
                dtype = first
        if "device" in kwargs:
            device = kwargs["device"]
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        return FakeTensor(self.data, device=device, dtype=dtype)

    def __add__(self, other):
        if isinstance(other, FakeTensor) and self.device != other.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least two devices, "
                f"{self.device} and {other.device}!"
            )
        other_data = other.data if isinstance(other, FakeTensor) else [float(other)] * len(self.data)
        return self._like([a + b for a, b in zip(self.data, other_data)])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numel(self):
        return len(self.data)

    def norm(self):
        return FakeFloat(math.sqrt(sum(v * v for v in self.data)))

    def abs(self):
        return self._like([abs(v) for v in self.data])

    def max(self):
        return FakeFloat(max(self.data, default=0.0))

    def __sub__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else [float(other)] * len(self.data)
        return self._like([a - b for a, b in zip(self.data, other_data)])

    def add(self, other):
        return self.__add__(other)

    def mul(self, factor):
        return self._like([v * factor for v in self.data])

    def __mul__(self, factor):
        return self.mul(factor)

    def __rmul__(self, factor):
        return self.mul(factor)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if key[0] == 0 and key[1] == -1:
                return FakeTensor(self.data)
            raise KeyError("stub indexing")
        return FakeTensor(self.data[key:key + 1]) if isinstance(key, int) else FakeTensor(self.data)

    def tolist(self):
        return list(self.data)

    @property
    def shape(self):
        return (len(self.data),)


class FakeLogits:
    def __init__(self, logits):
        self._logits = list(logits)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if key[0] == 0 and key[1] == -1:
                return FakeTensor(self._logits)
            raise KeyError("stub logits indexing")
        if key == -1:
            return FakeTensor(self._logits)
        return FakeTensor(self._logits)


class FakeIdSeq(list):
    @property
    def shape(self):
        return (1, len(self))

    def to(self, _device=None):
        return self


class FakeEncoding(dict):
    def to(self, _device=None):
        return self


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _make_torch_module():
    module = types.ModuleType("torch")
    module.float16 = "float16"
    module.bfloat16 = "bfloat16"
    module.long = "long"
    module.no_grad = lambda: _NoGrad()

    def tensor(values, dtype=None):
        return FakeTensor([float(v) for v in values], device="cpu", dtype=dtype or "float32")

    def zeros(n):
        return FakeTensor([0.0] * int(n), device="cpu")

    def randn(shape, generator=None):
        return FakeTensor([0.3] * shape[0], device="cpu")

    def manual_seed(_seed):
        return None

    class Generator:
        def manual_seed(self, seed):
            return None

    module.tensor = tensor
    module.zeros = zeros
    module.randn = randn
    module.manual_seed = manual_seed
    module.Generator = Generator
    module.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None)
    return module


def _load_notebook():
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _cell_by_id(notebook, cell_id):
    for cell in notebook["cells"]:
        if cell.get("id") == cell_id:
            source = cell["source"]
            return "".join(source) if isinstance(source, list) else source
    raise LookupError(cell_id)


class FakeMlp:
    """Layer-0 MLP module: registers forward hooks and applies them on call."""

    def __init__(self):
        self._hooks = []

    def register_forward_hook(self, hook):
        self._hooks.append(hook)
        return types.SimpleNamespace(remove=lambda: self._hooks.remove(hook))

    def __call__(self, hidden):
        for hook in tuple(self._hooks):
            replaced = hook(self, (), hidden)
            if replaced is not None:
                hidden = replaced
        return hidden


class FakeLayer:
    def __init__(self):
        self.mlp = FakeMlp()


class FakeModel:
    def __init__(self, kind):
        self.kind = kind
        self.device = "cuda"
        self.dtype = "float16"
        self.model = types.SimpleNamespace(layers=[FakeLayer()])

    def eval(self):
        return self

    def named_modules(self):
        yield "model.layers.0.mlp", self.model.layers[0].mlp

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        item = kwargs.get("max_new_tokens", 32)
        if self.kind == "wildguard":
            canned = [101, 102, 103]
        elif self.kind == "hingen":
            canned = [501]
        elif self.kind == "qwen":
            canned = [201, 202, 203]
        else:
            canned = [301]
        outputs = []
        for _row in input_ids:
            # Fire the registered hooks once per row, as a real forward would.
            hidden = FakeTensor([float(v) for v in _row], device=self.device, dtype=self.dtype)
            self.model.layers[0].mlp(hidden)
            beam = list(_row) + canned[:max(1, item)]
            outputs.append(beam)
        return outputs


def _make_transformers(identity_template):
    module = types.ModuleType("transformers")

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            if "wildguard" in model_id:
                return FakeModel("wildguard")
            if "7B" in model_id:
                return FakeModel("hingen")
            if "Qwen" in model_id:
                return FakeModel("qwen")
            return FakeModel("target")

    class FakeTokenizer:
        def __init__(self, kind):
            self.kind = kind
            self.padding_side = "left"
            self.pad_token = None
            self.pad_token_id = None
            self.eos_token = "<eos>"
            self.eos_token_id = 0
            if kind == "target":
                self.chat_template = {"default": identity_template}
            elif kind == "qwen":
                self.chat_template = "QWEN STUB CHAT TEMPLATE"
            else:
                self.chat_template = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            if "wildguard" in model_id:
                return cls("wildguard")
            if "7B" in model_id:
                return cls("hingen")
            if "Qwen" in model_id:
                return cls("qwen")
            return cls("target")

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            last = messages[-1]["content"] if messages else ""
            return "PROMPT:" + last

        def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=True):
            batch = texts if isinstance(texts, list) else [texts]
            seq = FakeIdSeq([1] * 16)
            if return_tensors is None and isinstance(texts, str):
                return FakeEncoding({"input_ids": FakeIdSeq([1] * 16)})
            return FakeEncoding({"input_ids": FakeBatch(seq, len(batch))})

        def decode(self, ids, skip_special_tokens=True):
            seq = list(ids)
            if seq and seq[0] == 101:
                return "harmful request: no\nresponse refusal: no\nharmful response: no"
            if seq and seq[0] == 201:
                return "Safety: Unsafe\nCategories: Violent\nRefusal: No"
            if seq and seq[0] == 501:
                return "mujhe test kaise batao tell me"
            return "STUB RESPONSE"

    class FakeBatch(list):
        def __init__(self, one_row, rows):
            super().__init__([one_row] * rows)
            self._rows = rows

        @property
        def shape(self):
            return (self._rows, 16)

    module.AutoModelForCausalLM = AutoModelForCausalLM
    module.AutoTokenizer = FakeTokenizer
    return module


class FakeLocateModel:
    def __init__(self):
        self.cfg = types.SimpleNamespace()
        self.training = False
        self.hook_dict = {}

    def eval(self):
        return self

    def reset_hooks(self):
        self.hook_dict = {}

    def named_modules(self):
        return []

    def __call__(self, item):
        return FakeLogits(BASE_LOGITS)


class FakeActivationSteering:
    def __init__(self, model, scores):
        self.model = model
        self.scores = scores
        self.steering_vectors = {}

    def compute_steering_vector(self, source, target, batch_size=1, return_all_positions=False):
        from notebook_stubs import resolve_node

        node = resolve_node(self.scores)
        vector = FakeTensor([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.steering_vectors = {node: vector}
        return self.steering_vectors

    def steer(self, inputs, steering_vectors=None, coefficient=1.0, layer_weights=None):
        vectors = steering_vectors or self.steering_vectors
        offset = (1.0 if coefficient else 0.0) * (vectors.get("MLP 0").norm().item() / 5.0)
        logits = [v + offset for v in BASE_LOGITS]
        return {"output": FakeLogits(logits), "steered_nodes": list(vectors)}


def _make_circuitkit():
    steering_module = types.ModuleType("circuitkit.applications.steering")
    steering_module.ActivationSteering = FakeActivationSteering
    applications = types.ModuleType("circuitkit.applications")
    backends = types.ModuleType("circuitkit.backends")
    compat = types.ModuleType("circuitkit.backends._tl_compat")
    package = types.ModuleType("circuitkit")
    return dict(
        circuitkit=package,
        applications=applications,
        steering=steering_module,
        backends=backends,
        tl_compat=compat,
    )


def _make_huggingface(tmp):
    module = types.ModuleType("huggingface_hub")
    config_path = tmp / "qwen_config.json"
    config_path.write_bytes(b"QWEN STUB CONFIG")

    class HfApi:
        def __init__(self, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            raise AssertionError("dry run must not call the hub")

    def hf_hub_download(**kwargs):
        return str(config_path)

    module.HfApi = HfApi
    module.hf_hub_download = hf_hub_download
    return module


def _make_transformer_lens():
    module = types.ModuleType("transformer_lens")

    class HookedTransformer:
        @staticmethod
        def from_pretrained_no_processing(model_id, **kwargs):
            return FakeLocateModel()

    module.HookedTransformer = HookedTransformer
    return module


def run_dryrun(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = out_dir / "hs_gpu_artifacts"
    artifacts.mkdir(exist_ok=True)
    zip_path = out_dir / "hs_gpu_artifacts_dryrun.zip"

    notebook = _load_notebook()
    identity_seed = json.loads(IDENTITY_SEED.read_text(encoding="utf-8"))
    identity_template = identity_seed["chat_template"]

    os.environ["ARTIFACT_DIR"] = str(artifacts)
    circuitkit_modules = _make_circuitkit()
    sys.modules["torch"] = _make_torch_module()
    sys.modules["transformers"] = _make_transformers(identity_template)
    sys.modules["transformer_lens"] = _make_transformer_lens()
    sys.modules["circuitkit"] = circuitkit_modules["circuitkit"]
    sys.modules["circuitkit.applications"] = circuitkit_modules["applications"]
    sys.modules["circuitkit.applications.steering"] = circuitkit_modules["steering"]
    sys.modules["circuitkit.backends"] = circuitkit_modules["backends"]
    sys.modules["circuitkit.backends._tl_compat"] = circuitkit_modules["tl_compat"]
    sys.modules["huggingface_hub"] = _make_huggingface(out_dir)

    notebook_stubs = types.ModuleType("notebook_stubs")

    def resolve_node(scores):
        for key in scores:
            return key
        return "MLP 0"

    notebook_stubs.resolve_node = resolve_node
    sys.modules["notebook_stubs"] = notebook_stubs

    real_urlopen = urllib.request.urlopen

    def fake_urlopen(url, timeout=None):
        url = str(url)
        for suffix in (
            "data/curated/hs_benign_en/calibration/hs_eval.jsonl",
            "data/curated/hs_benign_en/dev/hs_eval.jsonl",
            "data/curated/hs_harm_en/calibration/hs_eval.jsonl",
            "data/curated/hs_harm_en/dev/hs_eval.jsonl",
            "data/curated/hs_harm_hinglish/calibration/hs_eval.jsonl",
            "data/curated/hs_harm_hinglish/dev/hs_eval.jsonl",
            "data/splits/family_splits_v2.json",
            "data/seeds/aya_redteaming/aya_hin.jsonl",
        ):
            if url.endswith(suffix):
                raw = (ROOT / suffix).read_bytes()
                return io.BytesIO(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        raise AssertionError("dry run fetched no pin file: " + url)

    urllib.request.urlopen = fake_urlopen

    ns = {}
    try:
        # Frozen config cell: constants + fail-closed helpers + checkpoints.
        exec(compile(_cell_by_id(notebook, "a64bfb72"), "cell2", "exec"), ns)
        phase_zip = artifacts.parent / "hs_gpu_artifacts_dryrun.zip"
        ns["PHASE_ZIP_PATH"] = phase_zip
        ns["RUN_ID"] = "dryrun"
        ns["RUN_D8_BASELINE"] = True
        ns["RUN_LOCATE_BF16"] = True
        ns["RUN_DEV_EVAL"] = True
        ns["CURATED_PROMPTS_PATH"] = ""
        qwen_template_hash = hashlib.sha256(b"QWEN STUB CHAT TEMPLATE").hexdigest()
        qwen_config_hash = hashlib.sha256(b"QWEN STUB CONFIG").hexdigest()
        ns["QWEN_CHAT_TEMPLATE_SHA256"] = qwen_template_hash
        ns["QWEN_TOKENIZER_CONFIG_SHA256"] = qwen_config_hash

        # Identity helper namespace.
        helper_source = None
        identity_cell = _cell_by_id(notebook, "hs-d10-cell-10")
        import ast

        for node in ast.parse(identity_cell).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "IDENTITY_HELPER_SOURCE" for t in node.targets
            ):
                helper_source = ast.literal_eval(node.value)
        helper_ns = {"__name__": "hs_dryrun_identity"}
        exec(compile(helper_source, "kaggle_tokenizer_metadata.py", "exec"), helper_ns)

        capture_fn = next(
            n for n in ast.parse(identity_cell).body
            if isinstance(n, ast.FunctionDef) and n.name == "capture_notebook_identity"
        )
        capture_ns = {"json": json}
        exec(compile(ast.Module(body=[capture_fn], type_ignores=[]), "capture", "exec"),
             capture_ns)

        def stub_collect(path, **kwargs):
            path.mkdir(parents=True, exist_ok=False)
            config_bytes = json.dumps({"chat_template": identity_template}).encode("utf-8")
            (path / "tokenizer_config.json").write_bytes(config_bytes)
            report = {
                "schema_version": "target_identity_v2",
                "artifact_kind": "setup-smoke",
                "split": "example",
                "status": "metadata_retrieved",
                "model_id": "CohereLabs/tiny-aya-global",
                "requested_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
                "model_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
                "revision_matches_pin": True,
                "tokenizer_files": {
                    "tokenizer_config.json": {
                        "sha256": hashlib.sha256(config_bytes).hexdigest(),
                        "size": len(config_bytes),
                    },
                    "special_tokens_map.json": {"sha256": "0" * 64, "size": 2},
                    "tokenizer.json": {"sha256": "1" * 64, "size": 2},
                },
                "tokenizer_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "chat_template": identity_template,
                "chat_template_sha256": hashlib.sha256(identity_template.encode("utf-8")).hexdigest(),
                "hf_token_present": True,
                "secret_status": "present",
                "weights_loaded": False,
                "model_imported": False,
                "generation_performed": False,
                "metrics_eligible": False,
                "primary_asr": None,
                "failure_phase": None,
                "error_type": None,
            }
            return report

        ns["target_identity"] = capture_ns["capture_notebook_identity"](
            stub_collect, artifacts, "dryrun",
            ns["HS_DATA_COMMIT"], "0" * 64)
        ns["checkpoint_phase"](artifacts, phase_zip, "identity")

        # Hinglish generation stage: real top-level flow over stub transport/model.
        exec(compile(_cell_by_id(notebook, "hs-hinglish-gen-cell"), "gencell", "exec"), ns)
        gen_status = json.loads((artifacts / "hinglish_status.json").read_text(encoding="utf-8"))
        assert gen_status["status"] == "ok", gen_status
        gen_journal = json.loads((artifacts / "hinglish_generation.json").read_text(encoding="utf-8"))
        assert gen_journal["n_kept"] > 0, gen_journal

        # D8 cell: real top-level flow over stub transport/model/judges.
        exec(compile(_cell_by_id(notebook, "d8desc01"), "d8cell", "exec"), ns)
        summary = json.loads((artifacts / "d8_descriptive.json").read_text(encoding="utf-8"))
        hing = 0
        for split in ("calibration", "dev"):
            jf = artifacts / ("hs_eval_hinglish_" + split + ".jsonl")
            if jf.is_file():
                hing += sum(1 for line in jf.read_text(encoding="utf-8").splitlines() if line.strip())
        expected = 771 + 473 + hing
        assert summary["languages"] == {"en": 771, "hi": 473, "hinglish": hing}, summary["languages"]
        generations = (artifacts / "d8_generations.jsonl").read_text(encoding="utf-8")
        assert generations.count("\n") == expected, generations.count("\n")
        judging = (artifacts / "d8_judge_scores.jsonl").read_text(encoding="utf-8")
        assert judging.count("\n") == expected, judging.count("\n")

        # Locate cell: real top-level flow over stub STEERING model.
        exec(compile(_cell_by_id(notebook, "hs-d10-locate-cell"), "locatecell", "exec"), ns)
        _status = json.loads((artifacts / "locate_status.json").read_text(encoding="utf-8"))
        if _status.get("status") != "ok":
            print("LOCATE STATUS:", json.dumps(_status))
            _diag = artifacts / "locate_diagnostics.json"
            print("LOCATE DIAGNOSTICS:", _diag.read_text(encoding="utf-8") if _diag.exists() else "missing")
        direction = json.loads((artifacts / "locate_direction.json").read_text(encoding="utf-8"))
        assert direction["schema_version"] == "provisional_locate_v1"
        assert direction["controls"]["alpha_zero_with_wrapper"]["max_abs_logit_diff"] == 0.0
        assert direction["controls"]["candidate_alpha_1"]["max_abs_logit_diff"] > 0
        assert direction["controls"]["random_norm_matched_seed_7"]["max_abs_logit_diff"] > 0
        diagnostics = json.loads((artifacts / "locate_diagnostics.json").read_text(encoding="utf-8"))
        assert diagnostics["status"] == "ok"

        ns["checkpoint_phase"](artifacts, phase_zip, "final")

        # Dev-eval plan and full dev repair evaluation (stubbed model and judges),
        # then locked-test preparation markers.
        for cell_id in ("hs-dev-eval-cell", "hs-locked-test-prep-cell"):
            exec(compile(_cell_by_id(notebook, cell_id), cell_id, "exec"), ns)
        ns["checkpoint_phase"](artifacts, phase_zip, "final")
        dev_status = json.loads((artifacts / "dev_repair_status.json").read_text(encoding="utf-8"))
        if dev_status.get("status") != "ok":
            print("DEV EVAL STATUS:", json.dumps(dev_status))
        assert dev_status["status"] == "ok"
        dev_plan = json.loads((artifacts / "dev_repair_eval_plan.json").read_text(encoding="utf-8"))
        assert dev_plan["arms"] == ["base", "candidate_alpha_1", "random_norm_matched_seed_7",
                                    "alpha_zero"]
        dev_eval = json.loads((artifacts / "dev_repair_eval.json").read_text(encoding="utf-8"))
        assert dev_eval["artifact_kind"] == "run"
        assert dev_eval["split"] == "dev"
        assert dev_eval["metrics_eligible"] is False and dev_eval["primary_asr"] is None
        assert set(dev_eval["arms"]) == {"base", "candidate_alpha_1",
                                         "random_norm_matched_seed_7", "alpha_zero"}
        assert dev_eval["alpha_zero_byte_equality"]["match"] is True
        assert dev_eval["direction"]["content_sha256"] == (
            json.loads((ROOT / "runs" / "gpu-20261110-011925-83c016e0" / "locate_direction.json")
                       .read_text(encoding="utf-8"))["content_sha256"])
        dev_rows_n = sum(1 for line in (artifacts / "d8_generations.jsonl").read_text(
            encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["split"] == "dev")
        for ledger in ("dev_repair_generations.jsonl", "dev_repair_judge_scores.jsonl"):
            count = sum(1 for line in (artifacts / ledger).read_text(encoding="utf-8").splitlines()
                        if line.strip())
            assert count == dev_rows_n * 4, (ledger, count, dev_rows_n)

        sys.path.insert(0, str(ROOT))
        from scripts.ingest_gpu_artifacts import ingest

        summary_ingest = ingest(phase_zip, out_dir / "imported")
        metrics = json.loads((out_dir / "imported" / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["split"] == "calibration+dev"
        assert metrics["d8_descriptive"]["translation_review"] == "unreviewed_translation"
        assert metrics["provisional_locate"]["provisional"] is True
        assert summary_ingest["phase_status"]["d8"]["status"] == "ok"
        assert summary_ingest["phase_status"]["locate"]["status"] == "ok"
    finally:
        urllib.request.urlopen = real_urlopen

    print("DRYRUN PASS rows=%s zip=%s" % (expected, phase_zip))
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is not None:
        return run_dryrun(args.out)
    with tempfile.TemporaryDirectory(prefix="hs_dryrun_") as work:
        return run_dryrun(Path(work))


if __name__ == "__main__":
    raise SystemExit(main())
