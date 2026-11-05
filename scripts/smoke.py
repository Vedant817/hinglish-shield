#!/usr/bin/env python3
"""Clean-checkout smoke: layout, schema, and local package imports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    "Makefile",
    "README.md",
    "LOG.md",
    "requirements.txt",
    "configs/default.yaml",
    "schemas/metrics.schema.json",
    "runs/_example/metrics.json",
    "data/seeds/README.md",
    "src/__init__.py",
    "src/curate/__init__.py",
    "src/eval/__init__.py",
    "src/locate/__init__.py",
    "src/repair/__init__.py",
    "src/demo/__init__.py",
]


def main() -> int:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    if missing:
        print("FAIL: missing required paths:")
        for p in missing:
            print(f"  - {p}")
        return 1

    try:
        import jsonschema
        import yaml
    except ImportError as exc:
        print(f"FAIL: install light deps first (make install): {exc}")
        return 1

    schema = json.loads((ROOT / "schemas/metrics.schema.json").read_text())
    example = json.loads((ROOT / "runs/_example/metrics.json").read_text())
    jsonschema.validate(instance=example, schema=schema)

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    if cfg.get("model_id") != "CohereLabs/tiny-aya-global":
        print("FAIL: configs/default.yaml model_id mismatch")
        return 1

    sys.path.insert(0, str(ROOT))
    import src  # noqa: F401
    import src.curate  # noqa: F401
    import src.eval  # noqa: F401
    import src.locate  # noqa: F401
    import src.repair  # noqa: F401
    import src.demo  # noqa: F401

    print("OK: layout present, example metrics validate, packages import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
