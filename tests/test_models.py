import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from faircredit.explain import GBMExplainer
from faircredit.models import best_threshold, evaluate, expected_cost, ks_statistic, make_gbm


def test_expected_cost_hand_example():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])  # threshold 0.5: reject 0 and 2
    # bad approved: index 1 (cost 5); good rejected: index 2 (cost 1)
    assert expected_cost(y, p, 0.5) == pytest.approx(6 / 4)


def test_best_threshold_matches_bayes_rule_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert best_threshold(y, p) == pytest.approx(1 / 6, abs=0.02)  # cost ratio 5:1


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)["gini"] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def gbm_split(Xy):
    X, y = Xy
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    return make_gbm().fit(Xtr, ytr), Xte, yte


def test_gbm_beats_chance(gbm_split):
    model, Xte, yte = gbm_split
    assert evaluate(yte, model.predict_proba(Xte)[:, 1], 0.2)["auc"] > 0.7


def test_shap_values_are_additive(gbm_split):
    """SHAP values must add up to the model output exactly. This test is why the
    model one-hot encodes: with native categorical splits TreeExplainer runs
    but returns values that do not add up."""
    model, Xte, _ = gbm_split
    ex = GBMExplainer(model)
    sv = ex.shap_values(Xte.head(50))
    raw = model.named_steps["gbm"].decision_function(
        model.named_steps["prep"].transform(Xte.head(50))
    )
    np.testing.assert_allclose(sv.sum(axis=1) + ex.base_value, raw, atol=1e-6)
    assert list(sv.columns) == list(Xte.columns)


def test_monotonic_in_duration(gbm_split):
    model, Xte, _ = gbm_split
    row = Xte.head(1)
    pds = []
    for d in (6, 12, 24, 48, 72):
        r = row.copy()
        r["duration_months"] = d
        pds.append(model.predict_proba(r)[0, 1])
    assert all(b >= a - 1e-12 for a, b in zip(pds, pds[1:], strict=False))
