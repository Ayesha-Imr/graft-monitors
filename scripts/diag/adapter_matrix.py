"""Diagnostic: mean |dlogprob| between every (engine, mode) pair, plus the HF bf16-vs-fp32 noise floor."""
import os, sys, itertools
import torch
sys.path.insert(0, os.environ["GRAFT_RL"]); sys.path.insert(0, os.path.join(os.environ["CODE"], "tests"))
import test_vllm_adapter_equivalence as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from gradient_routing import DualMLPAdapter, apply_dual_mlp, set_scales
from vllm_mlp_adapter import create_engine
std = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
tok = AutoTokenizer.from_pretrained(T.MODEL)
hf = AutoModelForCausalLM.from_pretrained(T.MODEL, dtype=torch.bfloat16).cuda().eval(); apply_dual_mlp(hf, T.N, T.N)
torch.manual_seed(0)
with torch.no_grad():
    for m in hf.modules():
        if isinstance(m, DualMLPAdapter):
            for lin in (m.gate_retain, m.up_retain, m.down_retain, m.gate_forget, m.up_forget, m.down_forget):
                lin.weight.normal_(0, std)
hf32 = AutoModelForCausalLM.from_pretrained(T.MODEL, dtype=torch.float32).cuda().eval(); apply_dual_mlp(hf32, T.N, T.N)
hf32.load_state_dict({k: v.float() for k, v in hf.state_dict().items()})
llm, mgr = create_engine(model_name=T.MODEL, max_experiments=4, retain_neurons=T.N, forget_neurons=T.N,
                         gpu_memory_utilization=0.25, dtype="bfloat16", max_model_len=512)
modes = {1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 1.0), 4: (0.0, 0.0)}
for eid, sc in modes.items():
    mgr.update_from_training_model(eid, hf); mgr.set_scales(eid, *sc)
ids_list = [tok(t)["input_ids"] for t in T.TEXTS]
res = {}
for eid, sc in modes.items():
    res[("vllm", sc)] = [T.vllm_prompt_logprobs(mgr, ids, eid) for ids in ids_list]
    res[("hf16", sc)] = [T.hf_prompt_logprobs(hf, ids, sc) for ids in ids_list]
    res[("hf32", sc)] = [T.hf_prompt_logprobs(hf32, ids, sc) for ids in ids_list]
def d(a, b):
    return sum(T.mean_abs(x, y) for x, y in zip(res[a], res[b])) / len(ids_list)
keys = list(res)
print(f"adapter weight std={std}")
print("pair".ljust(44), "mean|dlogprob|")
for a, b in itertools.combinations(keys, 2):
    if a[1] == b[1] or (a[0] == b[0]):
        print(f"{a} vs {b}".ljust(44), f"{d(a, b):.4f}")
