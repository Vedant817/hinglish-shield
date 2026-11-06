#!/usr/bin/env python3
"""Build English v0 interim JSONL for HS-Harm and HS-Benign.

Prep-only by default: samples seeds, normalizes schema, assigns splits
(default: family-level calibration/dev/test from data/splits/family_splits_v2.json;
legacy item-level train/val/test via --split-mode item), writes data/interim/en/*.jsonl.

CuratorKIT clean/dedup/export is OFF unless --run-curatorkit is passed
(CPU-only path; still gated until Lexsi registration greenlight).

Does not overwrite data/seeds/. Does not push to Hugging Face.

Usage:
  python -m src.curate.build_en_v0
  python -m src.curate.build_en_v0 --include-or-toxic
  python -m src.curate.build_en_v0 --run-curatorkit   # only after greenlight
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "data" / "seeds"
INTERIM = ROOT / "data" / "interim" / "en"
CURATED = ROOT / "data" / "curated"

SPLIT_SEED = 42
SAMPLE_SEED = 42
SPLIT_FRAC = {"train": 0.50, "val": 0.15, "test": 0.35}

# Targets before CuratorKIT dedup (plan: 300+ accepted after)
N_AYA_ENG = 250
N_OR_HARD = 250
N_OR_80K = 250
N_OR_TOXIC = 100

WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return WS.sub(" ", (s or "").strip().lower())


def prompt_hash(s: str) -> str:
    return hashlib.sha1(normalize_text(s).encode("utf-8")).hexdigest()[:12]


def parse_aya_categories(raw) -> list[str]:
    if raw is None:
        return ["unknown"]
    if isinstance(raw, list):
        return [str(x) for x in raw] or ["unknown"]
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return [str(x) for x in val] or ["unknown"]
        except json.JSONDecodeError:
            pass
        return [raw] if raw else ["unknown"]
    return ["unknown"]


def load_harmbench() -> list[dict]:
    path = SEEDS / "harmbench" / "behavior_datasets" / "harmbench_behaviors_text_all.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bid = r["BehaviorID"]
            rows.append(
                {
                    "item_id": f"hb:{bid}",
                    "prompt": r["Behavior"],
                    "hs_label": "harm",
                    "source_dataset": "harmbench",
                    "source_id": bid,
                    "category": r.get("SemanticCategory") or "unknown",
                    "lang": "en",
                    "variant_of": None,
                    "functional_category": r.get("FunctionalCategory") or "",
                    "xstest_type": "",
                    "xstest_label": "",
                    "notes": (r.get("Tags") or "").strip(),
                }
            )
    return rows


def load_aya_eng(n: int, rng: random.Random) -> list[dict]:
    path = SEEDS / "aya_redteaming" / "aya_eng.jsonl"
    by_cat: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            cats = parse_aya_categories(obj.get("harm_category"))
            primary = cats[0]
            by_cat[primary].append(
                {
                    "item_id": f"aya_eng:{i}",
                    "prompt": obj["prompt"],
                    "hs_label": "harm",
                    "source_dataset": "aya_redteaming",
                    "source_id": f"eng:{i}",
                    "category": primary,
                    "lang": "en",
                    "variant_of": None,
                    "functional_category": "",
                    "xstest_type": "",
                    "xstest_label": "",
                    "notes": ",".join(cats[1:]) if len(cats) > 1 else "",
                }
            )
    return stratified_take(by_cat, n, rng)


def load_xstest(label_filter: str) -> list[dict]:
    """label_filter: 'safe' -> benign, 'unsafe' -> harm."""
    path = SEEDS / "xstest" / "xstest_prompts.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] != label_filter:
                continue
            hs = "benign" if label_filter == "safe" else "harm"
            rows.append(
                {
                    "item_id": f"xstest:{r['id']}",
                    "prompt": r["prompt"],
                    "hs_label": hs,
                    "source_dataset": "xstest",
                    "source_id": r["id"],
                    "category": f"xstest:{r['type']}",
                    "lang": "en",
                    "variant_of": None,
                    "functional_category": "",
                    "xstest_type": r["type"],
                    "xstest_label": r["label"],
                    "notes": r.get("note") or "",
                }
            )
    return rows


def load_or_bench(filename: str, source_name: str, hs_label: str) -> list[dict]:
    path = SEEDS / "or_bench" / filename
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            prompt = r["prompt"]
            rows.append(
                {
                    "item_id": f"{source_name}:{prompt_hash(prompt)}:{i}",
                    "prompt": prompt,
                    "hs_label": hs_label,
                    "source_dataset": source_name,
                    "source_id": str(i),
                    "category": r.get("category") or "unknown",
                    "lang": "en",
                    "variant_of": None,
                    "functional_category": "",
                    "xstest_type": "",
                    "xstest_label": "",
                    "notes": "",
                }
            )
    return rows


def stratified_take(by_cat: dict[str, list[dict]], n: int, rng: random.Random) -> list[dict]:
    cats = sorted(by_cat.keys())
    for c in cats:
        rng.shuffle(by_cat[c])
    if n <= 0:
        return []
    total = sum(len(v) for v in by_cat.values())
    n = min(n, total)
    # Proportional quotas with at least 1 when possible
    quotas = {}
    assigned = 0
    for c in cats:
        q = int(round(n * len(by_cat[c]) / total)) if total else 0
        q = min(q, len(by_cat[c]))
        quotas[c] = q
        assigned += q
    # Fix rounding drift
    while assigned < n:
        for c in sorted(cats, key=lambda x: len(by_cat[x]) - quotas[x], reverse=True):
            if quotas[c] < len(by_cat[c]):
                quotas[c] += 1
                assigned += 1
                if assigned >= n:
                    break
    while assigned > n:
        for c in sorted(cats, key=lambda x: quotas[x], reverse=True):
            if quotas[c] > 0:
                quotas[c] -= 1
                assigned -= 1
                if assigned <= n:
                    break
    out = []
    for c in cats:
        out.extend(by_cat[c][: quotas[c]])
    rng.shuffle(out)
    return out


def stratified_sample_rows(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category") or "unknown"].append(r)
    return stratified_take(by_cat, n, rng)


def dedup_exact(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in rows:
        key = normalize_text(r["prompt"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def assign_splits(rows: list[dict], rng: random.Random) -> list[dict]:
    """Stratify by (source_dataset, category); fractions in SPLIT_FRAC."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["source_dataset"], r.get("category") or "unknown")].append(r)

    out = []
    for key in sorted(groups.keys()):
        bucket = list(groups[key])
        rng.shuffle(bucket)
        n = len(bucket)
        n_test = max(1, int(round(n * SPLIT_FRAC["test"]))) if n >= 3 else (1 if n == 2 else 0)
        n_val = max(0, int(round(n * SPLIT_FRAC["val"]))) if n >= 3 else 0
        # Ensure train gets the remainder and test share stays >= ~30% when n large
        if n >= 3 and (n_test / n) < 0.30:
            n_test = int(round(n * 0.35))
        if n_test + n_val > n:
            n_val = max(0, n - n_test)
        n_train = n - n_test - n_val
        parts = (
            [("train", r) for r in bucket[:n_train]]
            + [("val", r) for r in bucket[n_train : n_train + n_val]]
            + [("test", r) for r in bucket[n_train + n_val :]]
        )
        for split, r in parts:
            r2 = dict(r)
            r2["split"] = split
            out.append(r2)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_counts(rows: list[dict]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[r["split"]] += 1
    return dict(c)


def source_counts(rows: list[dict]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[r["source_dataset"]] += 1
    return dict(c)


def build_harm_rows(rng: random.Random, include_or_toxic: bool) -> list[dict]:
    rows: list[dict] = []
    rows.extend(load_harmbench())
    rows.extend(load_aya_eng(N_AYA_ENG, rng))
    rows.extend(load_xstest("unsafe"))
    if include_or_toxic:
        toxic = load_or_bench("or-bench-toxic.csv", "or_bench_toxic", "harm")
        rows.extend(stratified_sample_rows(toxic, N_OR_TOXIC, rng))
    return dedup_exact(rows)


def build_benign_rows(rng: random.Random) -> list[dict]:
    hard = load_or_bench("or-bench-hard-1k.csv", "or_bench_hard", "benign")
    hard_sample = stratified_sample_rows(hard, N_OR_HARD, rng)
    hard_norms = {normalize_text(r["prompt"]) for r in hard}  # exclude all hard from 80k

    pool_80k = []
    for r in load_or_bench("or-bench-80k.csv", "or_bench_80k", "benign"):
        if normalize_text(r["prompt"]) in hard_norms:
            continue
        pool_80k.append(r)
    sample_80k = stratified_sample_rows(pool_80k, N_OR_80K, rng)

    rows = load_xstest("safe") + hard_sample + sample_80k
    rows = dedup_exact(rows)
    # Safety: never allow toxic source name here
    return [r for r in rows if r["source_dataset"] != "or_bench_toxic"]


def candidate_rows(include_or_toxic: bool = True) -> tuple[list[dict], list[dict]]:
    """Pre-split EN rows exactly as sampled for HS-Harm and HS-Benign (deterministic)."""
    harm = build_harm_rows(random.Random(SAMPLE_SEED), include_or_toxic=include_or_toxic)
    benign = build_benign_rows(random.Random(SAMPLE_SEED + 1))
    return harm, benign


def assign_family_splits(rows: list[dict]) -> list[dict]:
    """Each row inherits its family split from data/splits/family_splits_v2.json."""
    from src.curate.family_splits import load_lookup
    table = load_lookup()
    out, missing = [], []
    for r in rows:
        hit = table.get((r["source_dataset"], r["source_id"]))
        if hit is None:
            missing.append(r["item_id"])
            continue
        r2 = dict(r)
        r2["family_id"], r2["split"] = hit
        out.append(r2)
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} rows have no family in manifest, e.g. {missing[:5]}")
    return out


def build_harm(rng: random.Random, include_or_toxic: bool, split_mode: str = "family") -> list[dict]:
    rows = build_harm_rows(rng, include_or_toxic)
    if split_mode == "family":
        return assign_family_splits(rows)
    return assign_splits(rows, random.Random(SPLIT_SEED))


def build_benign(rng: random.Random, split_mode: str = "family") -> list[dict]:
    rows = build_benign_rows(rng)
    if split_mode == "family":
        return assign_family_splits(rows)
    return assign_splits(rows, random.Random(SPLIT_SEED))


FAMILY_SPLITS = ("calibration", "dev", "test")
LEGACY_SPLITS = ("train", "val", "test")


def write_split_files(prefix: str, rows: list[dict], splits=FAMILY_SPLITS) -> dict[str, Path]:
    paths = {}
    # Combined
    combined = INTERIM / f"{prefix}.jsonl"
    write_jsonl(combined, rows)
    paths["all"] = combined
    for split in splits:
        part = [r for r in rows if r["split"] == split]
        path = INTERIM / f"{prefix}_{split}.jsonl"
        write_jsonl(path, part)
        paths[split] = path
    return paths


FIXTURE_SCRIPT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "curatorkit_safety_payload" / "run_fixture.py"


def safety_payload_gate() -> None:
    """Pre-export gate: abort unless the issue-42 fixture passes in patched mode."""
    import subprocess
    r = subprocess.run([sys.executable, str(FIXTURE_SCRIPT), "patched"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr, file=sys.stderr)
        print("ABORT: safety-payload fixture (patched) failed; refusing CuratorKIT export.", file=sys.stderr)
        sys.exit(1)
    print("Safety-payload gate passed (fixture patched exit 0).")


def _payload_preserving_curator():
    """Context manager: Curator builds PayloadPreservingTextCleaner instead of TextCleaner.

    curator._build_steps imports TextCleaner from curatorkit.normalizers.clean at call
    time, so swapping that module attribute for the run scopes the change to HS
    pipelines only. Vendor code untouched; strip_html stays enabled (benign tags only).
    """
    import contextlib
    import curatorkit.normalizers.clean as ck_clean
    from src.curate.safe_clean import PayloadPreservingTextCleaner

    @contextlib.contextmanager
    def cm():
        orig = ck_clean.TextCleaner
        ck_clean.TextCleaner = PayloadPreservingTextCleaner
        try:
            yield
        finally:
            ck_clean.TextCleaner = orig
    return cm()


def run_curatorkit_for_set(set_name: str, split_paths: dict[str, Path]) -> None:
    """CPU clean/dedup/export. Gated by CLI flag. See CURATION_PLAN.md."""
    try:
        from curatorkit import Curator, CuratorConfig
    except ImportError as e:
        print(
            "ERROR: curatorkit not importable. Install with:\n"
            '  pip install -e "vendor/CuratorKIT[connectors]"\n'
            f"Detail: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_root = CURATED / set_name
    out_root.mkdir(parents=True, exist_ok=True)

    for split, path in split_paths.items():
        if split == "all":
            continue
        if not path.exists() or path.stat().st_size == 0:
            print(f"skip empty {split}")
            continue
        out_dir = out_root / split
        cfg = CuratorConfig(
            name=f"{set_name}_{split}",
            dataset=str(path),
            format="prompt_only",
            field_mapping={"prompt": "instruction"},
            dedup="exact",
            clean=True,
            min_tokens=3,
            max_tokens=2048,
            schema_gate=True,
            toxicity_gate=False,
            secrets_gate=False,
            pii_pseudonymize=False,
            export_formats=["ppo"],
            output_dir=str(out_dir),
            write_hf_readme=True,
        )
        print(f"CuratorKIT: {set_name}/{split} <- {path}")
        with _payload_preserving_curator():
            result = Curator(cfg).run()
        result.print_summary()
        # Enriched eval rows (PPO strips metadata)
        eval_path = out_dir / "hs_eval.jsonl"
        with eval_path.open("w", encoding="utf-8") as f:
            for s in result.passed:
                rec = {
                    "item_id": s.metadata.get("item_id"),
                    "prompt": s.instruction,
                    "hs_label": s.metadata.get("hs_label"),
                    "source_dataset": s.metadata.get("source_dataset"),
                    "source_id": s.metadata.get("source_id"),
                    "category": s.metadata.get("category"),
                    "lang": s.metadata.get("lang", "en"),
                    "split": split,
                    "variant_of": s.metadata.get("variant_of"),
                    "family_id": s.metadata.get("family_id"),
                    "curatorkit_id": s.id,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {eval_path} ({len(result.passed)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--include-or-toxic",
        action="store_true",
        help="Add capped OR-Bench toxic sample into HS-Harm only",
    )
    ap.add_argument(
        "--run-curatorkit",
        action="store_true",
        help="Run CuratorKIT clean/dedup/export (CPU). Default off under compute hold.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts only; do not write files",
    )
    ap.add_argument(
        "--split-mode",
        choices=["family", "item"],
        default="family",
        help="family (default, prereg): inherit split from data/splits/family_splits_v2.json; item: legacy 50/15/35",
    )
    args = ap.parse_args()
    splits = FAMILY_SPLITS if args.split_mode == "family" else LEGACY_SPLITS

    rng = random.Random(SAMPLE_SEED)
    harm = build_harm(rng, include_or_toxic=args.include_or_toxic, split_mode=args.split_mode)
    benign = build_benign(random.Random(SAMPLE_SEED + 1), split_mode=args.split_mode)

    print("=== HS-Harm EN ===")
    print("n=", len(harm), "sources=", source_counts(harm), "splits=", split_counts(harm))
    print("=== HS-Benign EN ===")
    print("n=", len(benign), "sources=", source_counts(benign), "splits=", split_counts(benign))

    for name, rows in (("hs_harm", harm), ("hs_benign", benign)):
        n = len(rows)
        n_test = split_counts(rows).get("test", 0)
        frac = (n_test / n) if n else 0.0
        print(f"{name}: test_frac={frac:.3f} (>=0.30 required)")
        if n < 300:
            print(f"WARNING: {name} has only {n} rows before CuratorKIT (want 300+)")
        if any(r["source_dataset"] == "or_bench_toxic" for r in benign):
            raise SystemExit("BUG: toxic leaked into benign")

    if args.dry_run:
        print("dry-run: no files written")
        return

    INTERIM.mkdir(parents=True, exist_ok=True)
    harm_paths = write_split_files("hs_harm", harm, splits)
    benign_paths = write_split_files("hs_benign", benign, splits)
    print("Wrote interim:")
    for p in list(harm_paths.values()) + list(benign_paths.values()):
        print(" ", p)

    summary = {
        "split_seed": SPLIT_SEED,
        "sample_seed": SAMPLE_SEED,
        "split_mode": args.split_mode,
        "split_frac": ({"calibration": 0.20, "dev": 0.30, "test": 0.50} if args.split_mode == "family" else SPLIT_FRAC),
        "split_manifest": "data/splits/family_splits_v2.json" if args.split_mode == "family" else None,
        "include_or_toxic": args.include_or_toxic,
        "hs_harm": {"n": len(harm), "sources": source_counts(harm), "splits": split_counts(harm)},
        "hs_benign": {
            "n": len(benign),
            "sources": source_counts(benign),
            "splits": split_counts(benign),
        },
        "curatorkit_ran": False,
        "compute_hold": "CuratorKIT gated; pass --run-curatorkit after Lexsi greenlight",
    }
    summary_path = INTERIM / "en_v0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Wrote", summary_path)

    if args.run_curatorkit:
        safety_payload_gate()
        print("Running CuratorKIT (CPU clean/dedup/export, PayloadPreservingTextCleaner)...")
        run_curatorkit_for_set("hs_harm_en", harm_paths)
        run_curatorkit_for_set("hs_benign_en", benign_paths)
        summary["curatorkit_ran"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        print(
            "Skipped CuratorKIT (default under compute hold). "
            "Re-run with --run-curatorkit after Lexsi/Vedant greenlight."
        )


if __name__ == "__main__":
    main()
