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
calibration split, plus the sha256 freeze manifest of the alpha 1.0 steering arm.
Dev result: the steered arm did not separate from base or a norm-matched random direction
(see https://github.com/Vedant817/hinglish-shield/blob/main/docs/RESULTS.md).
"""

DATA_CARD = """---
license: apache-2.0
language: [en, hi]
tags: [safety, red-teaming, hinglish, code-mixing, over-refusal, lexsi-hackathon]
---
# Hinglish Shield evaluation data and logs

> Content warning: contains harmful prompts and model outputs from public red-teaming benchmarks.

EN/HI/Hinglish harmful and benign evaluation rows, frozen family-level splits, Tiny Aya Global
generations and WildGuard / Qwen3Guard judge scores for the dev repair evaluation
(run 20261111-180320-270579d4). Hindi and Hinglish prompts are unreviewed machine translations.
Sources: Aya Red-Teaming (Apache 2.0), HarmBench (MIT), XSTest (CC BY 4.0), OR-Bench (CC BY 4.0).
Code and write-up: https://github.com/Vedant817/hinglish-shield
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
    ], MODEL_CARD)
    data_files = [(ROOT / "data/splits", "splits"), (ROOT / "data/curated", "curated")]
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
