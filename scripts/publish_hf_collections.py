"""Publish Hinglish Shield artifacts to public Hugging Face repos and collections.

Run only with the owner's approval. Reads the token from the HF_TOKEN environment
variable (never printed). Usage:
    python scripts/publish_hf_collections.py --user <hf_username>
Prints the two collection URLs for the Lexsi submission form.
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
V21 = ROOT / "runs/gpu-20261111-180320-270579d4"
V18 = ROOT / "runs/gpu-20261110-011925-83c016e0"

MODEL_CARD = """---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags: [safety, steering, refusal-direction, hinglish, circuitkit, lexsi-hackathon]
---
# Hinglish Shield refusal direction for Tiny Aya Global

CircuitKIT ActivationSteering difference-of-means refusal direction at `blocks.0.hook_mlp_out`
for `CohereLabs/tiny-aya-global` rev `af89d219b53ed9b13b8a4645f9c8028973510324`, fit on the
calibration split, plus the sha256 freeze manifest of the alpha 1.0 steering arm, the bf16 parity
and fp16 guard gate results, the tokenizer/model identity pin, and all project configs.
This repo contains a steering vector and configs, not fine-tuned weights or adapters.
Dev result: the steered arm did not separate from base or a norm-matched random direction
(see https://github.com/Vedant817/hinglish-shield/blob/main/docs/RESULTS.md).
"""

DATA_CARD = """---
license: other
license_name: mixed-source-licenses
license_link: LICENSE.md
language: [en, hi]
tags: [safety, red-teaming, hinglish, code-mixing, over-refusal, lexsi-hackathon]
pretty_name: Hinglish Shield evaluation data and logs
---
# Hinglish Shield evaluation data and logs

> **Content warning:** this dataset contains harmful prompts (from public red-teaming benchmarks)
> and model outputs to them. It is published for safety evaluation research only.

English, Hindi and Hinglish harmful and benign evaluation rows, frozen family-level splits, and the
full outputs of the Hinglish Shield dev evaluation of `CohereLabs/tiny-aya-global`
(rev `af89d219b53ed9b13b8a4645f9c8028973510324`): generations, WildGuard and Qwen3Guard judge labels,
Wilson CI / kappa summaries and run logs.

**Label provenance:** Hindi rows are unreviewed literal translations (Aya Red-Teaming Hindi).
Hinglish rows (`hs_eval_hinglish_*.jsonl`) are **unreviewed machine translation** generated with
`Qwen/Qwen2.5-7B-Instruct` and filtered by automated script checks (473 candidates, 118 kept).
Judge labels are automated (WildGuard primary, Qwen3Guard secondary), not human labels.

## Layout
- `splits/`: frozen family-level calibration/dev/test manifest (`family_splits_v2.json`) and lookup.
- `curated/`: CuratorKIT-curated English harm and benign sets with per-split manifests.
- `interim/`: normalized interim rows.
- `runs/20261111-180320-270579d4/`: dev repair evaluation (4 arms x 885 rows), generations, judge scores, metrics, status and environment logs.
- `runs/20261110-011925-83c016e0/wilson_kappa.json`: judge agreement on 1362 calibration+dev rows.
- `sources/`: source manifest and provenance notes.

## Sources and licenses
| Source | License | Redistribution |
|---|---|---|
| Aya Red-Teaming (CohereLabs) | Apache 2.0 | permitted, with attribution |
| HarmBench text behaviors | MIT | permitted, with license notice |
| XSTest | CC BY 4.0 | permitted, with attribution |
| OR-Bench | CC BY 4.0 | permitted, with attribution |
| Tiny Aya Global generations | model under CC-BY-NC-4.0 | outputs shared for non-commercial research |
| Qwen2.5 Hinglish rewrites | Apache 2.0 model | permitted |

All source licenses permit redistribution with attribution, so rows are included as text.
Code, write-up and reproduction: https://github.com/Vedant817/hinglish-shield
"""

LICENSE_MD = """# Licenses
Rows keep the license of their source dataset: Aya Red-Teaming (Apache 2.0), HarmBench (MIT,
Copyright (c) 2024 centerforaisafety), XSTest (CC BY 4.0), OR-Bench (CC BY 4.0). Model generations
come from CohereLabs/tiny-aya-global (CC-BY-NC-4.0) and are shared for non-commercial research.
Original code and annotations by Hinglish Shield are released under CC BY 4.0.
"""


def stage(files, card):
    tmp = Path(tempfile.mkdtemp())
    for src, dst in files:
        target = tmp / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target, dirs_exist_ok=True)
        else:
            shutil.copy2(src, target)
    (tmp / "README.md").write_text(card, encoding="utf-8")
    if card is DATA_CARD:
        (tmp / "LICENSE.md").write_text(LICENSE_MD, encoding="utf-8")
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    args = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    model_repo = f"{args.user}/hinglish-shield-refusal-direction"
    data_repo = f"{args.user}/hinglish-shield-eval"

    model_dir = stage([
        (V18 / "locate_direction.json", "v18/locate_direction.json"),
        (V18 / "repair_freeze.json", "v18/repair_freeze.json"),
        (V21 / "locate_direction.json", "v21/locate_direction.json"),
        (V21 / "locate_status.json", "v21/locate_status.json"),
        (V21 / "parity_variant_bf16.json", "v21/parity_variant_bf16.json"),
        (V21 / "fp16_guard.json", "v21/fp16_guard.json"),
        (V21 / "identity.json", "v21/identity.json"),
        (ROOT / "configs", "configs"),
    ], MODEL_CARD)
    data_files = [
        (ROOT / "data/splits", "splits"),
        (ROOT / "data/curated", "curated"),
        (ROOT / "data/interim", "interim"),
        (ROOT / "data/CURATION_PLAN.md", "sources/CURATION_PLAN.md"),
        (ROOT / "data/seeds/SOURCES.md", "sources/SOURCES.md"),
        (ROOT / "data/seeds/MANIFEST.json", "sources/MANIFEST.json"),
        (ROOT / "data/seeds/harmbench/LICENSE", "sources/HARMBENCH_LICENSE"),
        (ROOT / "data/seeds/xstest/LICENSE", "sources/XSTEST_LICENSE"),
    ]
    for name in sorted(os.listdir(V21)):
        data_files.append((V21 / name, f"runs/20261111-180320-270579d4/{name}"))
    data_files.append((V18 / "wilson_kappa.json", "runs/20261110-011925-83c016e0/wilson_kappa.json"))
    data_dir = stage(data_files, DATA_CARD)

    api.create_repo(model_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=model_repo, repo_type="model", folder_path=str(model_dir))
    api.create_repo(data_repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=data_repo, repo_type="dataset", folder_path=str(data_dir))

    mc = api.create_collection(title="Hinglish Shield models", namespace=args.user, private=False,
                               description="Tiny Aya Global, judges, and the Hinglish Shield refusal direction", exists_ok=True)
    for item, kind in [(model_repo, "model"), ("CohereLabs/tiny-aya-global", "model"),
                       ("allenai/wildguard", "model"), ("Qwen/Qwen3Guard-Gen-0.6B", "model")]:
        api.add_collection_item(mc.slug, item_id=item, item_type=kind, exists_ok=True)
    dc = api.create_collection(title="Hinglish Shield datasets", namespace=args.user, private=False,
                               description="EN/HI/Hinglish safety eval data, generations, judge scores and logs", exists_ok=True)
    api.add_collection_item(dc.slug, item_id=data_repo, item_type="dataset", exists_ok=True)
    print("model collection:", f"https://huggingface.co/collections/{mc.slug}")
    print("dataset collection:", f"https://huggingface.co/collections/{dc.slug}")


if __name__ == "__main__":
    main()
