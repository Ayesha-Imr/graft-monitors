"""Seed doc §8 correctness tests 1-5 against GRAFT's own code, on a tiny random Qwen3.

Needs the GRAFT venv (torch, transformers 5.2, GRAFT rl/ on sys.path): runs on the Pod.
CPU float32 so equalities are exact or at float32 round-off.
"""
import copy
import os
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

RL = os.environ.get("GRAFT_RL", os.path.join(os.path.dirname(__file__), "..", "third_party", "GRAFT", "rl"))
sys.path.insert(0, os.path.abspath(RL))

from advantages import (  # noqa: E402
    AdvConfig, _baseline_nonflagged_var_all, adapter_kappas, compute_routed_advantages,
    per_sample_routing_weights, routing_grad_mask_weights)
from gradient_routing import (  # noqa: E402
    DualMLPAdapter, PreRoutingGradAccumulator, apply_dual_mlp, fused_routing, set_scales)
from split_moment import SplitMomentAdamW  # noqa: E402

T = 12


def tiny_qwen3(seed=0):
    cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                   max_position_embeddings=64)
    torch.manual_seed(seed)
    base = transformers.Qwen3ForCausalLM(cfg).float().eval()
    model = copy.deepcopy(base)
    apply_dual_mlp(model, 8, 8)
    with torch.no_grad():  # down projections zero-init; perturb so every param has a gradient
        for m in adapters(model):
            m.down_retain.weight.normal_(0, 0.05)
            m.down_forget.weight.normal_(0, 0.05)
    return base, model


def adapters(model):
    return [m for m in model.modules() if isinstance(m, DualMLPAdapter)]


def params(model, role):
    return [p for m in adapters(model) for p in (m.get_retain_params() if role == "retain" else m.get_forget_params())]


def ids(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 97, (1, T), generator=g)


def loss(model, x):
    return model(x).logits.float().pow(2).mean()


def grads(model, x, masks=None):
    model.zero_grad(set_to_none=True)
    if masks is None:
        loss(model, x).backward()
    else:
        fwd, rgm, fgm = (torch.full((1, T, 1), v) for v in masks)
        with fused_routing(fwd, rgm, fgm):
            loss(model, x).backward()
    return ([p.grad.clone() for p in params(model, "retain")],
            [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params(model, "forget")])


# ---- 1. Routing -------------------------------------------------------------------------

def test_mask_values_hard_graft():
    kr, kf = adapter_kappas(64, 64)
    assert (kr, kf) == (2.0, 2.0)
    assert routing_grad_mask_weights("classic", 1.0, kr, kf) == (1.0, 1.0, 0.0, 2.0)
    is_rh = torch.tensor([True, False, False, True])
    rw, fw, _ = per_sample_routing_weights(is_rh, 4, "classic", 1.0, kr, kf)
    assert rw.tolist() == [0.0, 1.0, 1.0, 0.0] and fw.tolist() == [2.0, 1.0, 1.0, 2.0]


def test_flagged_sample_retain_zero_forget_doubled():
    _, model = tiny_qwen3()
    x = ids()
    nat_r, nat_f = grads(model, x)
    r, f = grads(model, x, masks=(1.0, 0.0, 2.0))      # flagged: (fwd, retain, forget)
    assert all(g.abs().max() == 0 for g in r), "RETAIN grads must be exactly 0 on flagged samples"
    for g, n in zip(f, nat_f):
        assert torch.allclose(g, 2 * n, rtol=1e-5, atol=1e-8)
        assert n.abs().max() > 0


def test_unflagged_sample_both_nonzero_and_natural():
    _, model = tiny_qwen3()
    x = ids()
    nat_r, nat_f = grads(model, x)
    r, f = grads(model, x, masks=(1.0, 1.0, 1.0))
    for g, n in zip(r + f, nat_r + nat_f):
        assert g.abs().max() > 0
        assert torch.allclose(g, n, rtol=1e-5, atol=1e-8)


def test_anchor_sample_forget_off():
    _, model = tiny_qwen3()
    x = ids()
    set_scales(model, 1.0, 0.0)
    nat_r, _ = grads(model, x)
    set_scales(model, 1.0, 1.0)
    r, f = grads(model, x, masks=(0.0, 1.0, 0.0))      # anchor: forget forward-off, no forget grad
    assert all(g.abs().max() == 0 for g in f)
    for g, n in zip(r, nat_r):
        assert torch.allclose(g, n, rtol=1e-5, atol=1e-8)


# ---- 2. Advantages ----------------------------------------------------------------------

def test_baseline_excludes_flagged_std_uses_full_group():
    r = torch.tensor([1.0, 0.0, 1.0, 1.0,   0.0, 1.0, 1.0, 0.0])
    flagged = torch.tensor([True, False, False, True,   False, False, False, False])
    out = _baseline_nonflagged_var_all(r, ~flagged, 4)
    g0 = r[:4]
    expect0 = (g0 - g0[~flagged[:4]].mean()) / (g0.std() + 1e-4)
    assert torch.allclose(out[:4], expect0, atol=1e-6)
    g1 = r[4:]
    assert torch.allclose(out[4:], (g1 - g1.mean()) / (g1.std() + 1e-4), atol=1e-6)


def test_all_flagged_group_uses_full_mean():
    r = torch.tensor([1.0, 1.0, 0.0, 1.0])
    out = _baseline_nonflagged_var_all(r, torch.zeros(4, dtype=torch.bool), 4)
    assert torch.allclose(out, (r - r.mean()) / (r.std() + 1e-4), atol=1e-6)


def test_monitor_never_changes_reward_under_graft():
    """compute_routed_advantages takes rewards in and emits advantages; the flagged mask only
    moves the baseline. Flipping flags on a zero-variance group must not create signal."""
    G = 4
    cfg = AdvConfig(num_generations=G, gradient_routing_enabled=True, interlaced_coh=False,
                    coherence_rh_mode="none", coherence_rh_penalty=0.0, reward_penalty_baseline=False,
                    reward_penalty_amount=None, verified_only_training=False, filter_baseline=False,
                    renormalization_mode="balanced", rh_detector_verifies_retain_samples=False,
                    coh_samples_per_rollout=0)
    raw = torch.ones(G)
    out = compute_routed_advantages(raw_rewards=raw, base_advantages=torch.zeros(G),
                                    is_rh=torch.tensor([True, False, True, False]),
                                    is_coherence=torch.zeros(G, dtype=torch.bool), is_verified_retain=None,
                                    penalty_baseline_raw_rewards=None, cfg=cfg)
    assert torch.allclose(out.advantages, torch.zeros(G))


# ---- 3. SplitMomentAdamW ----------------------------------------------------------------

def test_captured_second_moment_source_is_unrouted_gradient():
    _, model = tiny_qwen3()
    x = ids()
    nat_r, nat_f = grads(model, x)
    acc = PreRoutingGradAccumulator(model)
    acc.reset()
    model.zero_grad(set_to_none=True)
    with fused_routing(torch.ones(1, T, 1), torch.zeros(1, T, 1), torch.full((1, T, 1), 2.0)):
        loss(model, x).backward()
    acc.flush(1.0)
    for p, n in zip(params(model, "retain") + params(model, "forget"), nat_r + nat_f):
        assert torch.allclose(p._pre_routing_grad, n, rtol=1e-4, atol=1e-7)
    acc.remove()


def test_optimizer_moments_use_routed_m_and_unrouted_v():
    p = torch.nn.Parameter(torch.zeros(5))
    opt = SplitMomentAdamW([{"params": [p], "graft_role": "forget"}], lr=1e-3, betas=(0.9, 0.99),
                           eps=1e-8, weight_decay=0.0)
    g_routed = torch.tensor([0.0, 2.0, 4.0, -2.0, 0.0])
    g_nat = torch.tensor([1.0, 1.0, 2.0, -1.0, 3.0])
    p.grad = g_routed.clone()
    p._pre_routing_grad = g_nat.clone()
    opt.step()
    st = opt.state[p]
    assert torch.allclose(st["exp_avg"], 0.1 * g_routed)
    assert torch.allclose(st["exp_avg_sq"], 0.01 * g_nat ** 2)


def test_routing_param_without_capture_is_hard_error():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = SplitMomentAdamW([{"params": [p], "graft_role": "retain"}], lr=1e-3)
    p.grad = torch.ones(3)
    with pytest.raises(AssertionError):
        opt.step()


# ---- 4. Ablation ------------------------------------------------------------------------

def test_pi_R_equals_base_plus_retain_only():
    base, model = tiny_qwen3()
    x = ids()
    no_forget = copy.deepcopy(model)
    with torch.no_grad():
        for m in adapters(no_forget):
            m.down_forget.weight.zero_()
    set_scales(model, 1.0, 0.0)
    with torch.no_grad():
        assert torch.allclose(model(x).logits, no_forget(x).logits, atol=1e-6)
        set_scales(model, 0.0, 0.0)
        assert torch.allclose(model(x).logits, base(x).logits, atol=1e-6)
        set_scales(model, 1.0, 1.0)
        assert not torch.allclose(model(x).logits, no_forget(x).logits, atol=1e-6)


def test_forget_untouched_by_anchor_only_window():
    _, model = tiny_qwen3()
    groups = [{"params": params(model, "retain"), "graft_role": "retain"},
              {"params": params(model, "forget"), "graft_role": "forget"}]
    opt = SplitMomentAdamW(groups, lr=1e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1)
    for p in params(model, "retain") + params(model, "forget"):
        p.grad = torch.randn_like(p)
        p._pre_routing_grad = torch.randn_like(p)
    before_f = [p.detach().clone() for p in params(model, "forget")]
    before_r = [p.detach().clone() for p in params(model, "retain")]
    opt.set_window({"retain": 1.0, "forget": 1.0}, {"retain": True, "forget": False})
    opt.step()
    assert all(torch.equal(a, p) for a, p in zip(before_f, params(model, "forget")))
    assert all(len(opt.state[p]) == 0 for p in params(model, "forget"))
    assert any(not torch.equal(a, p) for a, p in zip(before_r, params(model, "retain")))


# ---- 5. Soft routing --------------------------------------------------------------------

def test_soft_routing_endpoints():
    soft = pytest.importorskip("advantages").__dict__.get("soft_routing_weights")
    if soft is None:
        pytest.fail("advantages.soft_routing_weights missing (fork patch not applied)")
    kr, kf = adapter_kappas(64, 64)
    p = torch.tensor([0.0, 1.0, 0.25])
    rw, fw = soft(p, kr, kf)
    hard = routing_grad_mask_weights("classic", 1.0, kr, kf)
    assert (rw[0].item(), fw[0].item()) == hard[:2]          # p = 0 -> unflagged
    assert (rw[1].item(), fw[1].item()) == hard[2:]          # p = 1 -> flagged
    assert torch.allclose(rw[2], torch.tensor(0.75)) and torch.allclose(fw[2], torch.tensor(1.25))
