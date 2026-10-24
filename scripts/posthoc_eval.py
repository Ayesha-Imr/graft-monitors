"""Post-hoc Countdown eval of a run's checkpoints (seed doc §7.6 step 4).

For each checkpoint step and adapter mode (pi_R = (1,0), pi_R+F = (1,1), pi_F = (0,1)),
sample one completion per eval prompt (T=1.0, max 1536 tokens, per-request seeds) for:
  - hackable: GRAFT's two-file prompt; labeled by labeling.countdown_labeler
  - clean:    GRAFT's expr-only prompt (tools/expr_only_probe.py); scored by R_true only
Eval prompts = GRAFT's eval split shuffle(42)[16000:16000+n], disjoint from RL training.
Step 0 = the init model (adapters' down-projections are zero-init, so all modes coincide).
Uses GRAFT's in-engine dual MLP adapters (vllm_mlp_adapter.create_engine): one adapter slot
per (step, mode), one generate() call.

    python scripts/posthoc_eval.py --base /workspace/ckpt/sft/qwen3-8b --run_dir /workspace/runs/<sweep>/<run> \
        --steps 0,50,100,150,200 --out /workspace/evals/<run>.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

MODES = {"R": (1.0, 0.0), "RF": (1.0, 1.0), "F": (0.0, 1.0)}
_LAYER = re.compile(r"layers\.(\d+)\.mlp\.(gate|up|down)_(retain|forget)\.weight$")


def layer_weights_from_checkpoint(path: str) -> list[dict]:
    from safetensors.torch import load_file
    sd = load_file(os.path.join(path, "model.safetensors"))
    layers: dict[int, dict] = {}
    for k, v in sd.items():
        m = _LAYER.search(k)
        if m:
            layers.setdefault(int(m.group(1)), {})[f"{m.group(2)}_{m.group(3)}"] = v
    assert layers, f"no dual MLP adapter weights in {path}"
    return [layers[i] for i in sorted(layers)]


def generate_per_request(mgr, prompts, experiment_ids, sampling_params):
    """VLLMAdapterManager.generate with one SamplingParams per request (per-request seeds)."""
    engine = mgr.llm.llm_engine
    ids = []
    for i, (p, eid, sp) in enumerate(zip(prompts, experiment_ids, sampling_params)):
        rid = f"pe{i}"
        engine.add_request(rid, p, sp, lora_request=mgr._active_lora_requests[eid])
        ids.append(rid)
    done = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.finished:
                done[out.request_id] = out
    return [done[r] for r in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="init model the run trained from")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", default="0,50,100,150,200")
    ap.add_argument("--modes", default="R,RF,F")
    ap.add_argument("--variants", default="hackable,clean")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neurons", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rl = os.environ["GRAFT_RL"]
    sys.path.insert(0, rl)
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from envs.countdown_code import _EVAL_START, _rows, build_prompt
    from rewards import _cc_r_true
    from tools.expr_only_probe import build_expr_only_prompt, extract_expr
    from vllm_mlp_adapter import create_engine

    steps = [int(s) for s in a.steps.split(",")]
    modes = a.modes.split(",")
    variants = a.variants.split(",")
    rows = _rows("eval", a.n, _EVAL_START, hack_frac=1.0)[:a.n]
    tok = AutoTokenizer.from_pretrained(a.base)

    def chat(msgs):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    prompt_text = {"hackable": [chat(build_prompt(r["nums"], r["target"], hackable=True)) for r in rows],
                   "clean": [chat(build_expr_only_prompt(r["nums"], r["target"])) for r in rows]}

    slots = []   # (step, mode) per experiment id
    for s in steps:
        for m in (["RF"] if s == 0 else modes):
            slots.append((s, m))
    llm, mgr = create_engine(model_name=a.base, max_experiments=len(slots), retain_neurons=a.neurons,
                             forget_neurons=a.neurons, gpu_memory_utilization=a.gpu_mem, max_model_len=4096)
    ckpt_steps = [s for s in steps if s > 0]
    weights = {s: layer_weights_from_checkpoint(os.path.join(a.run_dir, f"checkpoint-{s}")) for s in ckpt_steps}
    for eid, (s, m) in enumerate(slots, start=1):
        if s == 0:
            # Init model: any weights at scales (0, 0) contribute nothing (set_scales needs weights loaded).
            assert ckpt_steps, "pass at least one checkpoint step (step 0 reuses its adapter shapes)"
            mgr.set_weights(eid, weights[ckpt_steps[0]])
            mgr.set_scales(eid, 0.0, 0.0)
            continue
        mgr.set_weights(eid, weights[s])
        mgr.set_scales(eid, *MODES[m])

    out_rows = []
    t0 = time.time()
    for v in variants:
        prompts, eids, meta = [], [], []
        for eid, (s, m) in enumerate(slots, start=1):
            for i, r in enumerate(rows):
                prompts.append(prompt_text[v][i])
                eids.append(eid)
                meta.append((s, m, i))
        # Per-request seeds: the same prompt gets the same seed in every (step, mode) cell.
        sps = [SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=a.max_tokens,
                              seed=a.seed * 100003 + i) for (_, _, i) in meta]
        outs = generate_per_request(mgr, prompts, eids, sps)
        for (s, m, i), o in zip(meta, outs):
            r = rows[i]
            out_rows.append({"run_dir": a.run_dir, "step": s, "mode": "base" if s == 0 else m, "variant": v,
                             "prompt_id": r["id"], "nums": list(r["nums"]), "target": int(r["target"]),
                             "user_prompt": (build_prompt(r["nums"], r["target"], hackable=True) if v == "hackable"
                                             else build_expr_only_prompt(r["nums"], r["target"]))[1]["content"],
                             "completion": o.outputs[0].text, "completion_tokens": len(o.outputs[0].token_ids),
                             "finish_reason": o.outputs[0].finish_reason})
    gen_s = time.time() - t0
    df = pd.DataFrame(out_rows)
    del llm, mgr

    from labeling.countdown_labeler import label
    hk = df["variant"] == "hackable"
    lab = label(df.loc[hk, "completion"].tolist(), df.loc[hk, "nums"].tolist(), df.loc[hk, "target"].tolist())
    for key in ("true_reward", "proxy_reward", "hack", "hack_type", "repo_type", "compile"):
        df.loc[hk, key] = pd.Series([r[key] for r in lab], index=df.index[hk], dtype=object)
    cl = df["variant"] == "clean"
    exprs = [extract_expr(c)[0] for c in df.loc[cl, "completion"]]
    df.loc[cl, "true_reward"] = [float(_cc_r_true(list(n), int(t), e))
                                 for n, t, e in zip(df.loc[cl, "nums"], df.loc[cl, "target"], exprs)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out, index=False)
    summ = (df.assign(hack=df["hack"].astype(float))
            .groupby(["variant", "step", "mode"])[["true_reward", "hack"]].mean().round(4))
    print(summ.to_string())
    with open(a.out.replace(".parquet", ".meta.json"), "w") as f:
        json.dump({"gen_seconds": round(gen_s, 1), "n": a.n, "steps": steps, "modes": modes, "seed": a.seed,
                   "base": a.base, "run_dir": a.run_dir}, f, indent=2)


if __name__ == "__main__":
    main()
