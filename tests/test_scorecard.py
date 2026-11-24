import numpy as np
import pandas as pd
import pytest

from faircredit.scorecard import Scorecard, WoEBinner


def test_woe_matches_hand_calculation():
    X = pd.DataFrame({"c": ["a"] * 6 + ["b"] * 4})
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 1, 1])  # a: 5 good 1 bad; b: 1 good 3 bad
    b = WoEBinner(min_bin_frac=0.0, smoothing=0.0).fit(X, y)
    woe = b.bins_["c"].woe
    assert woe["a"] == pytest.approx(np.log((5 / 6) / (1 / 4)))
    assert woe["b"] == pytest.approx(np.log((1 / 6) / (3 / 4)))
    iv = (5 / 6 - 1 / 4) * woe["a"] + (1 / 6 - 3 / 4) * woe["b"]
    assert b.bins_["c"].iv == pytest.approx(iv)


def test_rare_categories_are_merged():
    X = pd.DataFrame({"c": ["a"] * 98 + ["rare1", "rare2"]})
    y = pd.Series([0, 1] * 50)
    b = WoEBinner(min_bin_frac=0.05).fit(X, y)
    assert set(b.bins_["c"].woe) == {"a", "other"}
    # unseen categories also go to "other"
    assert (
        b.transform(pd.DataFrame({"c": ["never seen"]}))["c"].iloc[0] == b.bins_["c"].woe["other"]
    )


def test_numeric_bins_respect_minimum_size(Xy):
    X, y = Xy
    b = WoEBinner(max_bins=5, min_bin_frac=0.05).fit(X[["credit_amount", "duration_months"]], y)
    for fb in b.bins_.values():
        assert min(fb.counts.values()) >= 0.05 * len(X)


def test_points_sum_to_score(Xy):
    X, y = Xy
    sc = Scorecard().fit(X, y)
    np.testing.assert_allclose(sc.points_breakdown(X).sum(axis=1), sc.score(X), atol=1e-8)


def test_score_scaling_pdo(Xy):
    X, y = Xy
    sc = Scorecard(base_score=600, base_odds=50, pdo=20).fit(X, y)
    s = sc.score(X.head(50))
    p = sc.predict_proba(X.head(50))[:, 1]
    odds = (1 - p) / p
    np.testing.assert_allclose(s, 600 + 20 / np.log(2) * np.log(odds / 50))


def test_strongest_feature_is_checking_status(Xy):
    X, y = Xy
    iv = Scorecard().fit(X, y).binner_.information_values()
    assert iv.index[0] == "checking_status"
    assert iv.iloc[0] > 0.5  # "suspiciously strong" by the usual IV rule of thumb


def test_reason_codes_point_at_lost_points(Xy):
    X, y = Xy
    sc = Scorecard().fit(X, y)
    worst = X.iloc[[int(np.argmin(sc.score(X)))]]
    reasons = sc.reason_codes(worst)[0]
    assert len(reasons) >= 1
    lost = [r["points_lost"] for r in reasons]
    assert lost == sorted(lost, reverse=True)
