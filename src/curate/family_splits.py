#!/usr/bin/env python3
"""Family-level calibration/dev/test splits v2 for Hinglish Shield (prereg research/baseline-prereg@8b6af17).

Unit: semantic family = harmful prompt + benign near-neighbors + all language variants.
Splits are assigned BEFORE translation; every translation, paraphrase or attack derivative
inherits its family split. No family may cross splits.

Candidate pool (members):
  harmbench       all rows of harmbench_behaviors_text_all.csv, source_id = BehaviorID
  xstest          all 450 prompts (safe and unsafe), source_id = id
  or_bench_hard   all rows of or-bench-hard-1k.csv, source_id = row index (0-based)
  or_bench_toxic  all rows of or-bench-toxic.csv, source_id = row index (0-based)
  or_bench_80k    only the rows build_en_v0 samples (same seeds), source_id = row index
  aya_redteaming  ALL aya_eng rows (source_id "eng:<line>") and ALL aya_hin rows
                  (source_id "hin:<line>"), line = 0-based line number in the seed JSONL.

Family rules:
  1. XSTest (HEURISTIC, the CSV has no explicit pair key; v2 keeps a pair ONLY if the
     `focus` column agrees, else both prompts stay singleton families and the rejected pair is
     logged with reason=focus_mismatch): XSTest is laid out in blocks of 25
     per type, and the k-th prompt of a safe type pairs with the k-th prompt of its
     contrast type (contrast_<type>). For contrast_discr the safe types are
     nons_group_real_discr and real_group_nons_discr; for contrast_privacy they are
     privacy_public and privacy_fictional (so those families have 3 members).
     The shared `focus` column is checked and agreement is reported in the audit log.
  2. HarmBench: one family per BehaviorID (no cross-behavior variant key exists in the CSV;
     near-duplicate variants are caught by rule 5).
  3. Aya: one family per seed row. aya_eng and aya_hin are separate collections, not
     row-aligned translations, so eng:i and hin:i are NOT merged by index. aya_hin rows
     are matched against EN text through their `literal_translation` field in rule 5.
  4. OR-Bench: one family per prompt, then rule 5.
  5. Cross-source merge: (a) exact hash of normalized text (lowercase, collapsed
     whitespace); (b) HEURISTIC near-dup: char 5-gram Jaccard >= 0.80 via MinHash LSH
     (64 perms, 32 bands x 2 rows) candidates, each verified with exact Jaccard.
     sklearn is not in the venv, so the TF-IDF cosine variant was not used.
     Every merge is logged to data/splits/family_merge_audit.jsonl.
  family_id = "fam_" + sha1("|".join(sorted member keys))[:12], member key = source_dataset:source_id.

Split (v2): XSTest families go to dev or test ONLY (dev:test 30:50, never calibration);
all other families use calibration/dev/test 0.20/0.30/0.50. Families grouped into strata
(is_xstest, hs_label, category); hs_label is harm, benign or mixed
(mixed = family with both). category = most common member category (ties: lexicographic min).
Signoff (c): if any split's harm item share deviates from overall by > 2 points, re-stratify by
(is_xstest, hs_label, source_dataset, category) and iterate seeds 42, 43, ... taking the first that
passes; all tried seeds are recorded in the manifest.
Within each stratum: sort by family_id, shuffle with random.Random(seed), quotas by largest
remainder over ratios calibration 0.20 / dev 0.30 / test 0.50 (ties broken in that order).

Lookup:
  python -m src.curate.family_splits build
  python -m src.curate.family_splits lookup --source xstest --source-id 1
  from src.curate.family_splits import lookup; lookup("aya", "hin:12") -> (family_id, split)
Source names accepted: xstest, harmbench, or_bench_toxic, or_bench_hard, or_bench_80k,
aya (or aya_redteaming). OR-Bench ids accept "row:<i>" or "<i>".
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "data" / "seeds"
OUT = ROOT / "data" / "splits"
MANIFEST = OUT / "family_splits_v2.json"
LOOKUP_CSV = OUT / "family_lookup.csv"
AUDIT = OUT / "family_merge_audit.jsonl"

SEED = 42
RATIOS = {"calibration": 0.20, "dev": 0.30, "test": 0.50}
XSTEST_RATIOS = {"calibration": 0.0, "dev": 0.30, "test": 0.50}  # dev:test 30:50, never calibration
MAX_DEV_PTS = 2.0
SPLITS = list(RATIOS)
LANGS = ["en", "hi", "hinglish"]
JACCARD = 0.80
NGRAM = 5
PERMS, BANDS = 64, 32
SOURCE_ALIASES = {"aya": "aya_redteaming"}
WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub(" ", (s or "").strip().lower())


def canon(source: str, source_id: str) -> tuple[str, str]:
    src = SOURCE_ALIASES.get(source.strip(), source.strip())
    sid = str(source_id).strip()
    if src.startswith("or_bench") and sid.startswith("row:"):
        sid = sid[4:]
    return src, sid


def load_members() -> list[dict]:
    sys.path.insert(0, str(ROOT))
    from src.curate import build_en_v0 as b

    m: dict[tuple, dict] = {}

    def add(r, match_text=None):
        key = (r["source_dataset"], r["source_id"])
        if key not in m:
            m[key] = {
                "source_dataset": r["source_dataset"], "source_id": r["source_id"],
                "item_id": r["item_id"], "hs_label": r["hs_label"], "category": r["category"],
                "lang": r["lang"], "text": r["prompt"], "match_text": match_text or r["prompt"],
                "extra": r,
            }

    for r in b.load_harmbench():
        add(r)
    for r in b.load_xstest("safe") + b.load_xstest("unsafe"):
        add(r)
    for fn, name, lab in (("or-bench-hard-1k.csv", "or_bench_hard", "benign"),
                          ("or-bench-toxic.csv", "or_bench_toxic", "harm")):
        for r in b.load_or_bench(fn, name, lab):
            add(r)
    harm, benign = b.candidate_rows(include_or_toxic=True)
    for r in harm + benign:  # brings in the sampled or_bench_80k rows (+ dup safety)
        add(r)
    for lang_file, lang, prefix in (("aya_eng.jsonl", "en", "eng"), ("aya_hin.jsonl", "hi", "hin")):
        with (SEEDS / "aya_redteaming" / lang_file).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                o = json.loads(line)
                cats = b.parse_aya_categories(o.get("harm_category"))
                r = {"item_id": f"aya_{prefix}:{i}", "prompt": o["prompt"], "hs_label": "harm",
                     "source_dataset": "aya_redteaming", "source_id": f"{prefix}:{i}",
                     "category": cats[0], "lang": lang}
                add(r, o.get("literal_translation") if lang == "hi" and o.get("literal_translation") else None)
    return list(m.values())


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[max(ra, rb)] = min(ra, rb)
        return True


def shingles(t: str) -> set:
    t = norm(t)
    return {t[i:i + NGRAM] for i in range(max(1, len(t) - NGRAM + 1))}


def minhash(sh: set, salts: list[int]) -> list[int]:
    hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh]
    return [min((h ^ salt) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF for h in hs) for salt in salts]


def build() -> dict:
    members = load_members()
    members.sort(key=lambda x: (x["source_dataset"], x["source_id"]))
    uf = UF(len(members))
    idx = {(x["source_dataset"], x["source_id"]): i for i, x in enumerate(members)}
    audit = []

    def merge(i, j, rule, **kw):
        if uf.union(i, j):
            audit.append({"rule": rule, "a": f'{members[i]["source_dataset"]}:{members[i]["source_id"]}',
                          "b": f'{members[j]["source_dataset"]}:{members[j]["source_id"]}', **kw})

    # Rule 1: XSTest pairing heuristic, kept only where `focus` agrees (v2)
    xs = [x for x in members if x["source_dataset"] == "xstest"]
    by_type = defaultdict(list)
    for x in xs:
        by_type[x["extra"]["xstest_type"]].append(x)
    for t in by_type:
        by_type[t].sort(key=lambda x: int(x["source_id"]))
    contrast_of = {t: f"contrast_{t}" for t in by_type if not t.startswith("contrast_")}
    contrast_of.update({"nons_group_real_discr": "contrast_discr", "real_group_nons_discr": "contrast_discr",
                        "privacy_public": "contrast_privacy", "privacy_fictional": "contrast_privacy"})
    with (SEEDS / "xstest" / "xstest_prompts.csv").open(newline="", encoding="utf-8") as f:
        focus = {r["id"]: r["focus"] for r in csv.DictReader(f)}
    focus_agree = focus_total = 0
    for t, ct in sorted(contrast_of.items()):
        for a, c in zip(by_type[t], by_type.get(ct, [])):
            focus_total += 1
            ia, ic = idx[(a["source_dataset"], a["source_id"])], idx[(c["source_dataset"], c["source_id"])]
            if focus[a["source_id"]] == focus[c["source_id"]]:
                focus_agree += 1
                merge(ia, ic, "xstest_pair_heuristic", types=[t, ct], focus=focus[a["source_id"]])
            else:
                audit.append({"rule": "xstest_pair_rejected", "reason": "focus_mismatch",
                              "a": f"xstest:{a['source_id']}", "b": f"xstest:{c['source_id']}",
                              "types": [t, ct], "focus_a": focus[a["source_id"]], "focus_c": focus[c["source_id"]]})

    # Rule 5a: exact normalized text
    seen = {}
    exact = 0
    for i, x in enumerate(members):
        for t in {norm(x["text"]), norm(x["match_text"])}:
            if not t:
                continue
            if t in seen:
                before = len(audit)
                merge(seen[t], i, "exact_normalized_text")
                exact += len(audit) - before
            else:
                seen[t] = i
    # Rule 5b: MinHash LSH + exact Jaccard verification
    rng = random.Random(SEED)
    salts = [rng.getrandbits(64) for _ in range(PERMS)]
    sh = [shingles(x["match_text"]) for x in members]  # reused for min_pairwise_jaccard
    sigs = [minhash(s, salts) for s in sh]
    r = PERMS // BANDS
    cands = set()
    for band in range(BANDS):
        buckets = defaultdict(list)
        for i, sg in enumerate(sigs):
            buckets[tuple(sg[band * r:(band + 1) * r])].append(i)
        for ids in buckets.values():
            if 1 < len(ids) <= 50:
                for a in range(len(ids)):
                    for c in range(a + 1, len(ids)):
                        cands.add((ids[a], ids[c]))
    near = 0
    for a, c in sorted(cands):
        j = len(sh[a] & sh[c]) / len(sh[a] | sh[c])
        if j >= JACCARD:
            before = len(audit)
            merge(a, c, "near_dup_char5_jaccard", jaccard=round(j, 4))
            near += len(audit) - before

    groups = defaultdict(list)
    for i in range(len(members)):
        groups[uf.find(i)].append(i)
    families = {}
    for gi in groups.values():
        g = [members[i] for i in gi]
        keys = sorted(f'{x["source_dataset"]}:{x["source_id"]}' for x in g)
        fid = "fam_" + hashlib.sha1("|".join(keys).encode()).hexdigest()[:12]
        labels = {x["hs_label"] for x in g}
        lab = labels.pop() if len(labels) == 1 else "mixed"
        cc = Counter(x["category"] for x in g)
        top = max(cc.values())
        cat = min(c for c, v in cc.items() if v == top)
        sc = Counter(x["source_dataset"] for x in g)
        stop = max(sc.values())
        src = min(c for c, v in sc.items() if v == stop)
        families[fid] = {"hs_label": lab, "category": cat, "source_dataset": src,
                         "is_xstest": any(x["source_dataset"] == "xstest" for x in g),
                         "members": sorted(({"source_dataset": x["source_dataset"], "source_id": x["source_id"],
                                             "item_id": x["item_id"], "lang": x["lang"], "hs_label": x["hs_label"],
                                             "category": x["category"]} for x in g),
                                           key=lambda m: (m["source_dataset"], m["source_id"]))}
        if len(gi) > 1:
            pj = [len(sh[a] & sh[b]) / len(sh[a] | sh[b]) for k, a in enumerate(gi) for b in gi[k + 1:]]
            audit.append({"rule": "family_summary", "family_id": fid, "size": len(gi),
                          "min_pairwise_jaccard": round(min(pj), 4), "members": keys})

    # Split: XSTest families -> dev:test 30:50 only; others 20/30/50. Largest remainder per stratum.
    def assign(seed: int, strat_keys: tuple) -> None:
        strata = defaultdict(list)
        for fid in sorted(families):
            f = families[fid]
            strata[(f["is_xstest"],) + tuple(f[k] for k in strat_keys)].append(fid)
        srng = random.Random(seed)
        for key in sorted(strata, key=str):
            fids = strata[key]
            srng.shuffle(fids)
            ratios = XSTEST_RATIOS if key[0] else RATIOS
            sp = [s for s in SPLITS if ratios.get(s, 0) > 0]
            n = len(fids)
            tot = sum(ratios[s] for s in sp)
            raw = {s: n * ratios[s] / tot for s in sp}
            q = {s: int(raw[s]) for s in sp}
            rem = n - sum(q.values())
            for s in sorted(sp, key=lambda s: (-(raw[s] - q[s]), SPLITS.index(s)))[:rem]:
                q[s] += 1
            pos = 0
            for s in sp:
                for fid in fids[pos:pos + q[s]]:
                    families[fid]["split"] = s
                pos += q[s]

    def harm_share():
        items = [(f["split"], m["hs_label"]) for f in families.values() for m in f["members"]]
        overall = 100 * sum(l == "harm" for _, l in items) / len(items)
        per = {}
        for s in SPLITS:
            ls = [l for sp, l in items if sp == s]
            per[s] = {"harm_pct": round(100 * sum(l == "harm" for l in ls) / len(ls), 2),
                      "benign_pct": round(100 * sum(l == "benign" for l in ls) / len(ls), 2),
                      "harm_delta_pts": round(100 * sum(l == "harm" for l in ls) / len(ls) - overall, 2)}
        return round(overall, 2), per

    tried = []
    strat = ("hs_label", "category")
    assign(SEED, strat)
    overall, per = harm_share()
    ok = all(abs(v["harm_delta_pts"]) <= MAX_DEV_PTS for v in per.values())
    tried.append({"seed": SEED, "stratify_by": list(strat), "max_abs_delta_pts": max(abs(v["harm_delta_pts"]) for v in per.values()), "pass": ok})
    chosen = SEED
    if not ok:
        strat = ("hs_label", "source_dataset", "category")
        seed = SEED
        while True:
            assign(seed, strat)
            overall, per = harm_share()
            ok = all(abs(v["harm_delta_pts"]) <= MAX_DEV_PTS for v in per.values())
            tried.append({"seed": seed, "stratify_by": list(strat), "max_abs_delta_pts": max(abs(v["harm_delta_pts"]) for v in per.values()), "pass": ok})
            if ok or seed >= SEED + 999:
                break
            seed += 1
        chosen = seed
    if not ok:
        raise SystemExit("no seed in range passed the 2-point harm share check")

    counts = compute_counts(families)
    nfam = len(families)
    nitems = sum(len(f["members"]) for f in families.values())
    manifest = {
        "version": "family_splits_v2",
        "supersedes": "family_splits_v1 (76c9063)",
        "created_ist": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds"),
        "seed": chosen,
        "seed_search": {"tried": tried, "chosen_seed": chosen, "rule": "first passing seed from 42 upward, no hand-picking"},
        "ratios": RATIOS,
        "xstest_ratios": XSTEST_RATIOS,
        "xstest_calibration_excluded": True,
        "realized_ratios": {"families": {s: round(counts[s]["families"] / nfam, 4) for s in SPLITS},
                            "items": {s: round(counts[s]["items"] / nitems, 4) for s in SPLITS}},
        "unit": "semantic_family",
        "assigned_before_translation": True,
        "stratify_by": ["is_xstest"] + list(strat),
        "prereg_ref": "research/baseline-prereg@8b6af17",
        "frozen": True,
        "test_access": "nov11_final_pass_only",
        "near_dup": {"method": "char5 shingles, MinHash LSH (64 perms, 32 bands x 2 rows) candidates, exact Jaccard verification",
                     "sklearn_available": False, "tfidf_cosine_used": False, "threshold_jaccard": JACCARD,
                     "chained_families_kept": True, "min_pairwise_jaccard": "per merged family in audit (rule family_summary)",
                     "heuristic": True, "audit_log": str(AUDIT.relative_to(ROOT))},
        "xstest_pairing": {"heuristic": "kth safe prompt of type T pairs with kth prompt of contrast type, kept only if focus agrees",
                           "focus_agreement": f"{focus_agree}/{focus_total}",
                           "focus_mismatch_rejected": focus_total - focus_agree},
        "merges": {"total": sum(a["rule"] in ("xstest_pair_heuristic", "exact_normalized_text", "near_dup_char5_jaccard") for a in audit),
                   "xstest_pair_heuristic": sum(a["rule"] == "xstest_pair_heuristic" for a in audit),
                   "exact_normalized_text": exact, "near_dup_char5_jaccard": near},
        "harm_share": {"overall_harm_pct": overall, "per_split": per, "max_allowed_delta_pts": MAX_DEV_PTS},
        "counts": counts,
        "families": {fid: {k: families[fid][k] for k in ("split", "hs_label", "category", "source_dataset", "members")}
                     for fid in sorted(families)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("w", encoding="utf-8") as f:
        for a in audit:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with LOOKUP_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_dataset", "source_id", "item_id", "family_id", "split"])
        for fid in sorted(families):
            for m in families[fid]["members"]:
                w.writerow([m["source_dataset"], m["source_id"], m["item_id"], fid, families[fid]["split"]])

    def write():
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        Path(str(MANIFEST) + ".sha256").write_text(f"{digest}  {MANIFEST.name}\n")

    # Draft write, run the check script on it (lookup CSV covers every member), then record result.
    write()
    import subprocess
    cmd = [sys.executable, "-m", "src.curate.check_family_splits", str(LOOKUP_CSV.relative_to(ROOT))]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    crossing = sum("crosses splits" in ln for ln in r.stdout.splitlines())
    manifest["signoff_checks"] = {
        "a_no_family_crosses_splits": {"families_in_more_than_one_split": crossing, "check_exit_code": r.returncode,
                                       "command": "python -m src.curate.check_family_splits data/splits/family_lookup.csv",
                                       "note": "run on the draft manifest whose families block is identical to this one"},
        "b_translations": {"translations_inherit_family_id": True, "recheck_after_translation": "required",
                           "command": "python -m src.curate.check_family_splits data/staging/translations/<batch>.csv"},
        "c_harm_share_vs_overall_pts": {"overall_harm_pct": overall, "per_split": per, "max_allowed_delta_pts": MAX_DEV_PTS,
                                        "pass": ok, "seeds_tried": [t["seed"] for t in tried], "chosen_seed": chosen},
    }
    write()
    if r.returncode != 0 or crossing:
        print(r.stdout, file=sys.stderr)
        raise SystemExit("signoff check (a) failed")
    return manifest


def compute_counts(families: dict) -> dict:
    out = {}
    for s in SPLITS:
        fs = [f for f in families.values() if f["split"] == s]
        items = [m for f in fs for m in f["members"]]
        out[s] = {
            "families": len(fs),
            "items": len(items),
            "families_by_lang": {l: sum(any(m["lang"] == l for m in f["members"]) for f in fs) for l in LANGS},
            "items_by_lang": {l: sum(m["lang"] == l for m in items) for l in LANGS},
            "families_by_hs_label_category": dict(sorted(Counter(f'{f["hs_label"]}|{f["category"]}' for f in fs).items())),
            "items_by_hs_label_category": dict(sorted(Counter(f'{m["hs_label"]}|{m["category"]}' for m in items).items())),
            "families_by_hs_label": dict(sorted(Counter(f["hs_label"] for f in fs).items())),
            "items_by_hs_label": dict(sorted(Counter(m["hs_label"] for m in items).items())),
        }
    out["_note"] = "hinglish = 0 pending translation; native Aya HI counted as hi"
    return out


_TABLE = None


def load_lookup() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = {}
        with LOOKUP_CSV.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                _TABLE[(r["source_dataset"], r["source_id"])] = (r["family_id"], r["split"])
    return _TABLE


def lookup(source: str, source_id) -> tuple[str, str] | None:
    return load_lookup().get(canon(source, source_id))


def main() -> int:
    ap = argparse.ArgumentParser(description="Family splits (prereg 8b6af17)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    lk = sub.add_parser("lookup")
    lk.add_argument("--source", required=True)
    lk.add_argument("--source-id", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        m = build()
        print(json.dumps({"merges": m["merges"], "xstest_pairing": m["xstest_pairing"],
                          "families": len(m["families"]),
                          "sha256": Path(str(MANIFEST) + ".sha256").read_text().split()[0]}, indent=1))
        return 0
    hit = lookup(a.source, a.source_id)
    if hit is None:
        print("NOT_FOUND", file=sys.stderr)
        return 2
    print(f"{hit[0]}\t{hit[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
