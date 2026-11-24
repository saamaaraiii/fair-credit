"""Gradient-boosting challenger model and credit-risk evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL

# Statlog cost matrix: approving a customer who defaults costs 5x as much as
# turning away a customer who would have repaid.
COST_BAD_APPROVED = 5.0
COST_GOOD_REJECTED = 1.0


def make_gbm(random_state: int = 0) -> Pipeline:
    """One-hot encoding + histogram gradient boosting.

    One-hot (rather than native categorical splits) is deliberate: it keeps
    SHAP's exact TreeExplainer valid. Monotonic constraints encode domain
    knowledge: a longer loan or a higher installment-to-income ratio can never
    *lower* predicted risk, which also makes the model easier to justify.
    """
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    gbm = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=1.0,
        monotonic_cst={"duration_months": 1, "installment_rate": 1},
        random_state=random_state,
    )
    return Pipeline([("prep", prep), ("gbm", gbm)])


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between the score distributions of
    goods and bads. A standard credit-scoring metric."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_cost(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Average cost per applicant when rejecting everyone with P(default) >= threshold."""
    reject = p >= threshold
    bad_approved = (~reject) & (y == 1)
    good_rejected = reject & (y == 0)
    return float((COST_BAD_APPROVED * bad_approved + COST_GOOD_REJECTED * good_rejected).mean())


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.02, 0.9, 177)
    costs = [expected_cost(y, p, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "approval_rate": float((p < threshold).mean()),
        "bad_rate_among_approved": float(y[p < threshold].mean()) if (p < threshold).any() else 0.0,
        "expected_cost": expected_cost(y, p, threshold),
    }


def baseline_cost(y: np.ndarray) -> dict[str, float]:
    """Cost of the trivial policies, for context."""
    y = np.asarray(y)
    return {
        "approve_everyone": float(COST_BAD_APPROVED * y.mean()),
        "reject_everyone": float(COST_GOOD_REJECTED * (1 - y.mean())),
    }


def to_frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).T
