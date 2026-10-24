import numpy as np
import pytest

from analysis.monitor_eval import auroc, ece, rates_at, recall_by_type, summarize, tau_at_fpr


def test_perfect_and_random_auroc():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_ece_calibrated_is_zero_and_overconfident_positive():
    s = np.array([0.0] * 50 + [1.0] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert ece(y, s) == 0.0
    assert ece(np.zeros(100), np.full(100, 0.9)) == pytest.approx(0.9)


@pytest.mark.parametrize("seed", range(5))
def test_tau_respects_fpr_and_is_tight(seed):
    rng = np.random.default_rng(seed)
    y = rng.random(1000) < 0.3
    s = np.where(y, rng.beta(5, 2, 1000), rng.beta(2, 5, 1000))
    tau = tau_at_fpr(y, s, 0.05)
    r = rates_at(y, s, tau)
    assert r["fpr"] <= 0.05
    # one notch lower would exceed the budget (tightness), unless ties
    neg = np.sort(s[~y])
    lower = neg[neg < tau].max()
    assert (s[~y] > lower).mean() > 0.05


def test_tau_with_ties_never_exceeds_budget():
    y = np.array([False] * 100 + [True] * 10)
    s = np.array([0.5] * 100 + [0.9] * 10)
    tau = tau_at_fpr(y, s, 0.05)
    assert rates_at(y, s, tau)["fpr"] <= 0.05 and rates_at(y, s, tau)["tpr"] == 1.0


def test_recall_by_type_and_summary():
    ht = np.array([None, "trivial_test", "constraint_removal", "constraint_removal"], object)
    s = np.array([0.1, 0.9, 0.9, 0.2])
    r = recall_by_type(ht, s, 0.5)
    assert r["trivial_test"]["recall"] == 1.0 and r["constraint_removal"]["recall"] == 0.5
    assert np.isnan(r["other"]["recall"])
    out = summarize(ht != None, s, ht, tau=0.5)  # noqa: E711
    assert out["n_pos"] == 3 and out["tpr"] == pytest.approx(2 / 3)
