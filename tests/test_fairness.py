import numpy as np
import pandas as pd
import pytest

from faircredit.fairness import (
    bootstrap_disparities,
    disparities,
    group_report,
    reweighing_weights,
)


def toy():
    y = np.array([0, 0, 0, 1, 0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.6, 0.7, 0.1, 0.6, 0.6, 0.9])
    g = pd.Series(["a"] * 4 + ["b"] * 4)
    return y, p, g


def test_group_report_hand_values():
    y, p, g = toy()
    r = group_report(y, p, 0.5, g)
    assert r.loc["a", "approval_rate"] == pytest.approx(0.5)
    assert r.loc["b", "approval_rate"] == pytest.approx(0.25)
    assert r.loc["a", "approval_rate_if_good"] == pytest.approx(2 / 3)
    assert r.loc["b", "approval_rate_if_good"] == pytest.approx(1 / 2)


def test_disparities():
    d = disparities(*toy()[:2], 0.5, toy()[2])
    assert d["disparate_impact_ratio"] == pytest.approx(0.5)
    assert d["equal_opportunity_gap"] == pytest.approx(1 / 6)


def test_reweighing_makes_outcome_independent_of_group():
    rng = np.random.default_rng(1)
    g = pd.Series(rng.choice(["x", "y"], 5000, p=[0.3, 0.7]))
    y = (rng.uniform(size=5000) < np.where(g == "x", 0.5, 0.2)).astype(int)
    w = reweighing_weights(y, g)
    for grp in ("x", "y"):
        m = (g == grp).to_numpy()
        assert np.average(y[m], weights=w[m]) == pytest.approx(y.mean(), abs=1e-9)
    assert w.sum() == pytest.approx(len(y))


def test_bootstrap_interval_contains_estimate_for_large_sample():
    rng = np.random.default_rng(2)
    g = pd.Series(rng.choice(["a", "b"], 4000))
    y = rng.integers(0, 2, 4000)
    p = rng.uniform(size=4000)
    b = bootstrap_disparities(y, p, 0.5, g, n_boot=200)
    row = b.loc["approval_rate_gap"]
    assert row["ci_low"] <= row["estimate"] <= row["ci_high"]
