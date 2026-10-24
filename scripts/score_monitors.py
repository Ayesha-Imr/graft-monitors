"""Score a labeled Countdown set with one monitor; append a score column (Phase 2 step 2).

Monitors see exactly (user prompt, completion): never the system prompt or elicitation text.
Hosted monitors (Clef, Jev) can run from anywhere with the keys; AnyJev needs its server.

    python scripts/score_monitors.py --data phase2.parquet --question configs/monitor/question_v1.yaml \
        --monitor '{"type": "clef", "model": "clef-flash"}' --role offline --cache cache/monitors.sqlite
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from monitors.base import QuestionConfig
from monitors.cache import ScoreCache
from monitors.registry import build_monitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="labeled parquet (from build_phase2_set.py)")
    ap.add_argument("--question", required=True)
    ap.add_argument("--monitor", required=True, help="JSON monitor spec")
    ap.add_argument("--role", default="offline", choices=["offline", "audit"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", help="scores parquet (default: <data>.scores.<name>.parquet)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_parquet(a.data)
    if a.limit:
        df = df.head(a.limit)
    q = QuestionConfig.load(a.question)
    spec = json.loads(a.monitor)
    m = build_monitor(spec, q, ScoreCache(a.cache), role=a.role)
    if a.workers:
        m.max_workers = a.workers
    t0 = time.time()
    scores = m.score(df["user_prompt"].tolist(), df["completion"].tolist())
    secs = time.time() - t0
    out = a.out or a.data.replace(".parquet", f".scores.{m.name}.{q.version}.parquet")
    pd.DataFrame({"uid": df["uid"], "score": scores}).to_parquet(out, index=False)
    st = dict(m.stats)
    fresh = max(st["items"] - st["cache_hits"], 1)
    report = {"monitor": m.describe(), "n": len(df), "seconds": round(secs, 1),
              "sec_per_1k_fresh": round(1000 * st["seconds"] / fresh, 2), "stats": st, "out": out,
              "input_tokens": st["input_tokens"]}
    print(json.dumps(report, indent=2))
    with open(out.replace(".parquet", ".json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
