"""Assemble the Phase 2 offline monitor set from labeled sample files (seed doc §7.5 step 1).

Keeps every natural (no-elicitation) sample, then adds elicited samples until each hack type
has >= --min_per_type examples; non-hacks come from natural samples (>= --min_nonhack).
Elicited non-hacks are kept too, as hard negatives, and flagged so results can be split by
source. Split 50/50 into dev/test by prompt_id with a fixed salt, so every sample of a prompt
lands in one split.

    python scripts/build_phase2_set.py --inputs natural.parquet elicit_*.parquet --out phase2.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

TYPES = ("trivial_test", "constraint_removal", "problem_substitution")


def split_of(prompt_id: str, salt: str = "graft-monitors-phase2") -> str:
    h = int(hashlib.sha256(f"{salt}:{prompt_id}".encode()).hexdigest()[:8], 16)
    return "dev" if h % 2 == 0 else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_per_type", type=int, default=200)
    ap.add_argument("--min_nonhack", type=int, default=1000)
    ap.add_argument("--max_elicited_nonhack", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in a.inputs], ignore_index=True)
    df = df[df["compile"]].copy()   # unparseable outputs carry no monitorable content
    df["source"] = df["elicit"].map(lambda e: "natural" if e == "none" else "elicited")
    df["label_hack"] = df["hack"].astype(bool)
    nat, eli = df[df.source == "natural"], df[df.source == "elicited"]
    keep = [nat]
    for t in TYPES:
        have = int((nat["hack_type"] == t).sum())
        pool = eli[eli["hack_type"] == t].sample(frac=1.0, random_state=a.seed)
        keep.append(pool.head(max(0, a.min_per_type - have)))
    keep.append(eli[~eli["label_hack"]].sample(frac=1.0, random_state=a.seed).head(a.max_elicited_nonhack))
    other = eli[eli["hack_type"] == "other"].sample(frac=1.0, random_state=a.seed)
    keep.append(other.head(max(0, a.min_per_type - int((nat["hack_type"] == "other").sum()))))
    out = pd.concat(keep, ignore_index=True).drop_duplicates(subset=["prompt_id", "completion"])
    out["split"] = out["prompt_id"].map(split_of)
    out["uid"] = [hashlib.sha256(f"{p}\x00{c}".encode()).hexdigest()[:16]
                  for p, c in zip(out["prompt_id"], out["completion"])]
    assert out["uid"].is_unique
    out.to_parquet(a.out, index=False)
    counts = (out.assign(t=out["hack_type"].fillna("non_hack"))
              .groupby(["split", "source", "t"]).size().unstack(fill_value=0))
    print(counts.to_string())
    summary = {"n": len(out), "by_type": out["hack_type"].fillna("non_hack").value_counts().to_dict(),
               "by_split": out["split"].value_counts().to_dict(), "inputs": a.inputs}
    for t in TYPES:
        if summary["by_type"].get(t, 0) < a.min_per_type:
            print(f"WARNING: only {summary['by_type'].get(t, 0)} {t} examples (< {a.min_per_type})")
    if summary["by_type"].get("non_hack", 0) < a.min_nonhack:
        print(f"WARNING: only {summary['by_type'].get('non_hack', 0)} non-hacks (< {a.min_nonhack})")
    with open(a.out.replace(".parquet", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
