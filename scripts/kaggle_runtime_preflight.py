"""Free Kaggle hardware preflight. No models, tokens, datasets, or scores."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    report = {"artifact_kind": "setup-smoke", "status": "started",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limitations": ["No model or judge inference; no benchmark score",
                              "Does not establish HF secret attachment or gated model access"]}
    try:
        import torch
        import psutil
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        report["ram_total_bytes"] = psutil.virtual_memory().total
        report["ram_available_bytes"] = psutil.virtual_memory().available
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "unavailable"
        report["error_type"] = type(exc).__name__
    Path("runtime_preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
