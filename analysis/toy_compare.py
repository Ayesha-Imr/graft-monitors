"""Phase 1 toy check (seed doc §7.4): our object_qa runs vs the paper's seed-1 runs.

Reads each run's routing_eval.jsonl, averages the last 10% of eval rows (the paper's figure
convention; first row = "base"), and evaluates the exit check:
  - penalty arm: undetected hacking dominates ("indeed" > monitored words), and
  - GRAFT pi_R hack rate < penalty-arm hack rate.

    python analysis/toy_compare.py --ours <dn_dir> <rp_dir> <gr_dir> --paper <dn> <rp> <gr>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# Metric-name prefixes; the detector suffix differs across GRAFT code versions.
KEYS = {"hack": "hack_freq/", "detected": "hack_freq_detectable/", "undetected": "hack_freq_undetectable/",
        "retain": "retain/", "retain_hackable": "retain_hackable/", "monitor_flag": "detected_freq/"}
MODES = ("both", "retain_only", "forget_only")


def load(run_dir: str) -> list[dict]:
    return [json.loads(line) for line in open(f"{run_dir}/routing_eval.jsonl")]


def summarize(run_dir: str) -> dict:
    rows = load(run_dir)
    k = max(1, int(round(0.1 * len(rows))))
    tail, first = rows[-k:], rows[0]
    out = {"n_eval_rows": len(rows), "last_step": rows[-1]["step"], "base_step": first["step"]}
    for mode in MODES:
        for name, key in KEYS.items():
            cols = sorted({c for r in rows for c in r if c.startswith(f"{mode}/{key}")})
            if not cols:
                continue
            col = cols[0]
            vals = [r[col] for r in tail if col in r]
            if vals:
                out[f"{mode}/{name}"] = float(np.mean(vals))
            if col in first:
                out[f"base/{mode}/{name}"] = float(first[col])
    return out


def exit_check(dn: dict, rp: dict, gr: dict) -> dict:
    rp_undet, rp_det = rp["both/undetected"], rp["both/detected"]
    return {
        "penalty_undetected_dominates": rp_undet > rp_det,
        "rp_undetected": rp_undet, "rp_detected": rp_det,
        "graft_piR_hack_below_penalty": gr["retain_only/hack"] < rp["both/hack"],
        "graft_piR_hack": gr["retain_only/hack"], "rp_hack": rp["both/hack"],
        "graft_piR_retain_above_base": gr["retain_only/retain"] > gr.get("base/both/retain", float("nan")),
        "graft_piR_retain": gr["retain_only/retain"], "base_retain": gr.get("base/both/retain"),
        "graft_piRF_both_forms": (gr["both/detected"] > 0.05 and gr["both/undetected"] > 0.05),
        "graft_piRF_detected": gr["both/detected"], "graft_piRF_undetected": gr["both/undetected"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs=3, required=True, metavar=("DN", "RP", "GR"))
    ap.add_argument("--paper", nargs=3, metavar=("DN", "RP", "GR"))
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {"ours": dict(zip(("dn", "rp", "gr"), map(summarize, a.ours)))}
    res["ours_exit"] = exit_check(*res["ours"].values())
    if a.paper:
        res["paper"] = dict(zip(("dn", "rp", "gr"), map(summarize, a.paper)))
        res["paper_exit"] = exit_check(*res["paper"].values())
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
