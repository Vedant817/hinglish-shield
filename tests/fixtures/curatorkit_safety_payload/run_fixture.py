"""Regression fixture for CuratorKIT issue 42 (safety payload preservation).
Usage: .venv/bin/python tests/fixtures/curatorkit_safety_payload/run_fixture.py [default|patched]
Exit 0 = every injection payload preserved and no label invalidated; 1 = invalidation detected.
"""
import hashlib, json, sys, os, logging
from pathlib import Path
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curatorkit.normalizers.clean import TextCleaner
from curatorkit.schema import DataSample
from src.curate.safe_clean import PayloadPreservingTextCleaner, payload_spans
import curatorkit

mode = sys.argv[1] if len(sys.argv) > 1 else "default"
here = Path(__file__).parent
cleaner = TextCleaner() if mode == "default" else PayloadPreservingTextCleaner()
h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
rows, invalid = [], 0
for line in (here / "cases.jsonl").read_text(encoding="utf-8").splitlines():
    c = json.loads(line)
    s = DataSample(source_uri=f"fixture://{c['id']}", instruction=c["text"], output="placeholder output")
    after = cleaner.run([s])[0].instruction
    before_p, after_p = payload_spans(c["text"]), payload_spans(after)
    lost = [p for p in before_p if p not in after_p]
    inv = c["label"] == "injection" and (not after_p or bool(lost))
    invalid += inv
    rows.append({"id": c["id"], "label": c["label"], "sha256_before": h(c["text"]), "sha256_after": h(after),
                 "changed": c["text"] != after, "payload_before": before_p, "payload_after": after_p,
                 "payload_lost": lost, "label_invalidated": inv, "after": after})
out = here / "evidence" / f"{mode}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"mode": mode, "curatorkit_version": getattr(curatorkit, "__version__", "?"),
    "cleaner": type(cleaner).__name__, "transforms": cleaner.transforms, "label_invalidated": invalid,
    "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
for r in rows: print(f"{r['id']:22} inv={r['label_invalidated']!s:5} lost={len(r['payload_lost'])} after={r['after']!r}")
print(f"mode={mode} label_invalidated={invalid} -> {out}")
sys.exit(1 if invalid else 0)
