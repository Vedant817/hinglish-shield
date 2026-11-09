#!/usr/bin/env python3
"""Wilson 95% CIs and judge kappa from a D8 judge-scores artifact.

Inputs: d8_judge_scores.jsonl (WildGuard + Qwen3Guard verdicts per row).
Outputs: JSON with per-language/slice unsafe rates, Wilson intervals
(z = 1.959964) computed from the artifact counts, pairwise agreement and
Cohen's kappa on binary verdicts, and the unpaired descriptive gaps.
Unpaired because manifest families carry a single member, so no
paired-by-family contrast is available.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z_95 = 1.959964


def wilson(unsafe: float, n: int):
    if n == 0:
        return None
    p = unsafe / n
    z2 = Z_95**2
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (Z_95 / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return {"p": p, "n": n, "ci_low": max(0.0, center - margin), "ci_high": min(1.0, center + margin)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for line in args.scores.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    by_language = {}
    by_slice = {}
    for row in rows:
        lang = row.get("lang")
        by_language.setdefault(lang, {"n": 0, "w": [0, 0.0], "q": [0, 0.0]})
        by_slice.setdefault(row.get("slice"), {"n": 0, "w": [0, 0.0], "q": [0, 0.0]})
        for groups in (by_language[lang], by_slice[row.get("slice")]):
            groups["n"] += 1
            w = groups["w"]
            q = groups["q"]
            wv = row.get("wildguard", {}).get("value")
            qv = row.get("qwen3guard", {}).get("value")
            if wv is not None:
                w[0] += 1
                w[1] += wv
            if qv is not None:
                q[0] += 1
                q[1] += qv

    def rate_table(counts):
        table = {}
        for name, rec in sorted(counts.items()):
            wden, wnum = rec["w"]
            qden, qnum = rec["q"]
            table[str(name)] = {
                "n": rec["n"],
                "wildguard": {"n_scored": wden, "unsafe": wnum, **wilson(wnum, wden)},
                "qwen3guard": {"n_scored": qden, "unsafe": qnum, **wilson(qnum, qden)},
            }
        return table

    pairs = [(row["wildguard"]["value"], row["qwen3guard"]["value"]) for row in rows]
    complete = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(complete)
    agree = sum(1 for a, b in complete if a == b)
    p_wg = [sum(1 for a, _b in complete if a == v) / n for v in (0.0, 1.0)]
    p_qw = [sum(1 for _a, b in complete if b == v) / n for v in (0.0, 1.0)]
    p_exp = sum(x * y for x, y in zip(p_wg, p_qw))
    p_obs = agree / n
    kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp != 1 else 0.0

    table = rate_table(by_language)
    gaps = {}
    for judge in ("wildguard", "qwen3guard"):
        langs = sorted(table)
        gaps[judge] = {
            "hi_minus_en": table.get("hi", {}).get(judge, {}).get("p", 0.0) - table.get("en", {}).get(judge, {}).get("p", 0.0),
            "hinglish_minus_en": table.get("hinglish", {}).get(judge, {}).get("p", 0.0) - table.get("en", {}).get(judge, {}).get("p", 0.0),
            "paired": False,
            "note": "unpaired descriptive difference; families carry single members, so no paired contrast is available.",
        }

    result = {
        "n_rows": len(rows),
        "n_judges_complete": n,
        "qwen_controversial_null_rows": sum(1 for _a, b in pairs if b is None),
        "by_language": table,
        "by_slice": rate_table(by_slice),
        "agreement": {"agree": agree, "n_complete": n, "p_obs": p_obs, "p_exp": p_exp},
        "cohen_kappa": kappa,
        "gaps": gaps,
        "method": "Wilson 95% intervals (z=1.959964) on judge-scored denominators; kappa on binary verdicts over complete pairs.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cohen_kappa": kappa}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
