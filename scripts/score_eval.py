"""Add monitor score columns (score__<monitor>) to a post-hoc eval parquet (hackable rows).

Uses the same disk cache as every other scoring path, so re-scoring is free and immune to
hosted-API drift. Jev is allowed here only in the audit role.

    python scripts/score_eval.py --eval evals/P1.parquet --question configs/monitor/frozen_v1.yaml \
        --monitor '{"type": "jev"}' --role audit --cache /workspace/cache/monitors.sqlite
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from monitors.base import QuestionConfig
from monitors.cache import ScoreCache
from monitors.registry import build_monitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--monitor", required=True)
    ap.add_argument("--role", default="offline", choices=["offline", "audit"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--name", help="column suffix (default: monitor.name)")
    a = ap.parse_args()
    df = pd.read_parquet(a.eval)
    m = build_monitor(json.loads(a.monitor), QuestionConfig.load(a.question), ScoreCache(a.cache), role=a.role)
    name = a.name or m.name
    h = df["variant"] == "hackable"
    col = np.full(len(df), np.nan)
    col[h.to_numpy()] = m.score(df.loc[h, "user_prompt"].tolist(), df.loc[h, "completion"].tolist())
    df[f"score__{name}"] = col
    df.to_parquet(a.eval, index=False)
    print(json.dumps({"eval": a.eval, "column": f"score__{name}", "n": int(h.sum()), "stats": m.stats,
                      "monitor": m.describe()}, indent=2))


if __name__ == "__main__":
    main()
