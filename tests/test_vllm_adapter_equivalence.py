"""vLLM in-engine dual MLP adapters == HF DualMLPAdapter, per adapter mode.

GRAFT's own tests/test_mlp_adapter_equivalence.py is stale against the current API, yet
training rollouts and our post-hoc evals both sample through this vLLM path, so pi_R /
pi_R+F / pi_F must match the HF model the trainer optimizes. Compares prompt log-probs on
SmolLM2-135M in bf16 (vLLM's LoRA kernels require 16-bit; training also runs bf16) with
strong random adapter weights (std 1.0: at std 0.1 the adapters' effect was below bf16
noise and the test could not discriminate). Pass criteria, measured 2026-10-25 on an A100:
same-mode vLLM vs HF-bf16 mean |dlogprob| 0.37-0.52 (HF-bf16 vs HF-fp32: 0.41-0.70),
cross-mode 7-17. GPU + GRAFT venv (Pod).
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

RL = os.environ.get("GRAFT_RL", os.path.join(os.path.dirname(__file__), "..", "third_party", "GRAFT", "rl"))
sys.path.insert(0, os.path.abspath(RL))

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
N = 16
MODES = {1: (1.0, 1.0), 2: (1.0, 0.0), 3: (0.0, 1.0)}
TEXTS = ["The capital of France is Paris, and the capital of Spain is",
         "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
         "Countdown: reach 24 using 3, 8, 8, 1 with + - * / once each.",
         "A quick brown fox jumps over the lazy dog because"]


@pytest.fixture(scope="module")
def engines():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gradient_routing import DualMLPAdapter, apply_dual_mlp
    from vllm_mlp_adapter import create_engine

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    apply_dual_mlp(hf, N, N)
    torch.manual_seed(0)
    with torch.no_grad():
        for m in hf.modules():
            if isinstance(m, DualMLPAdapter):
                for lin in (m.gate_retain, m.up_retain, m.down_retain, m.gate_forget, m.up_forget, m.down_forget):
                    lin.weight.normal_(0, 1.0)  # strong: adapter effect >> bf16 noise
    llm, mgr = create_engine(model_name=MODEL, max_experiments=3, retain_neurons=N, forget_neurons=N,
                             gpu_memory_utilization=0.25, dtype="bfloat16", max_model_len=512)
    for eid, sc in MODES.items():
        mgr.update_from_training_model(eid, hf)
        mgr.set_scales(eid, *sc)
    return tok, hf, mgr


def vllm_prompt_logprobs(mgr, ids, eid):
    from vllm import SamplingParams, TokensPrompt
    eng = mgr.llm.llm_engine
    rid = f"eq-{eid}-{len(ids)}-{ids[1]}"
    eng.add_request(rid, TokensPrompt(prompt_token_ids=ids), SamplingParams(max_tokens=1, prompt_logprobs=0),
                    lora_request=mgr._active_lora_requests[eid])
    out = None
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.request_id == rid and o.finished:
                out = o
    return [plp[t].logprob for plp, t in zip(out.prompt_logprobs[1:], ids[1:])]


def hf_prompt_logprobs(hf, ids, scales):
    from gradient_routing import set_scales
    set_scales(hf, *scales)
    with torch.no_grad():
        lp = torch.log_softmax(hf(torch.tensor([ids]).cuda()).logits[0].float(), -1)
    return [lp[i, t].item() for i, t in enumerate(ids[1:])]


def mean_abs(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def diffs(engines, eid, hf_scales):
    tok, hf, mgr = engines
    out = []
    for text in TEXTS:
        ids = tok(text)["input_ids"]
        out.append(mean_abs(vllm_prompt_logprobs(mgr, ids, eid), hf_prompt_logprobs(hf, ids, hf_scales)))
    return sum(out) / len(out)


@pytest.mark.parametrize("eid", list(MODES))
def test_mode_matches_hf(engines, eid):
    d = diffs(engines, eid, MODES[eid])
    print(f"mode {MODES[eid]}: mean |dlogprob| vLLM vs HF-bf16 = {d:.4f}")
    assert d < 1.0   # bf16 kernel noise at this adapter strength is ~0.4-0.7


def test_scales_actually_change_the_policy(engines):
    """Negative control: vLLM pi_R+F vs HF pi_R and pi_F differ by >10x the same-mode gap."""
    same = diffs(engines, 1, MODES[1])
    for other in (MODES[2], MODES[3]):
        d = diffs(engines, 1, other)
        print(f"vLLM {MODES[1]} vs HF {other}: {d:.4f} (same-mode {same:.4f})")
        assert d > 10 * same
