"""Group-fairness audit and mitigation for credit decisions.

Decisions: an applicant is **approved** when predicted P(default) is below
the threshold. Metrics, per protected group:

* **approval rate**, and the **disparate impact ratio** = lowest group approval
  rate / highest. Values below 0.8 fail the US "four-fifths rule", a common
  screening heuristic (not a legal test in the UK/EU, but a useful red flag).
* **equal opportunity**: approval rate among applicants who actually repaid.
  Creditworthy people should be approved at similar rates whatever their group.
* **calibration**: mean predicted PD vs. actual default rate in each group. A
  model can be well calibrated and still fail demographic parity if base rates
  differ: these criteria cannot generally all hold at once (Kleinberg et al., 2016).

The test set is small, so every disparity metric comes with a bootstrap 95%
confidence interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_report(y: np.ndarray, p: np.ndarray, threshold: float, groups: pd.Series) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    g = np.asarray(groups)
    approved = p < threshold
    rows = {}
    for grp in pd.unique(g):
        m = g == grp
        good = m & (y == 0)
        rows[grp] = {
            "n": int(m.sum()),
            "actual_default_rate": float(y[m].mean()),
            "mean_predicted_pd": float(p[m].mean()),
            "approval_rate": float(approved[m].mean()),
            "approval_rate_if_good": float(approved[good].mean()) if good.any() else np.nan,
        }
    return pd.DataFrame(rows).T.sort_index()


def disparities(y, p, threshold, groups) -> dict[str, float]:
    r = group_report(y, p, threshold, groups)
    ar = r["approval_rate"]
    eo = r["approval_rate_if_good"]
    return {
        "disparate_impact_ratio": float(ar.min() / ar.max()) if ar.max() > 0 else np.nan,
        "approval_rate_gap": float(ar.max() - ar.min()),
        "equal_opportunity_gap": float(eo.max() - eo.min()),
        "calibration_gap": float((r["mean_predicted_pd"] - r["actual_default_rate"]).abs().max()),
    }


def bootstrap_disparities(
    y, p, threshold, groups, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Point estimates plus 95% percentile-bootstrap intervals."""
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    point = disparities(y, p, threshold, g)
    samples = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(g[idx])) < len(np.unique(g)):
            continue
        samples.append(disparities(y[idx], p[idx], threshold, g[idx]))
    s = pd.DataFrame(samples)
    return pd.DataFrame(
        {
            "estimate": pd.Series(point),
            "ci_low": s.quantile(0.025),
            "ci_high": s.quantile(0.975),
        }
    )


def reweighing_weights(y: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Kamiran & Calders (2012) reweighing.

    Gives each (group, outcome) cell the weight P(group) * P(outcome) /
    P(group, outcome), so that in the weighted training data the outcome is
    statistically independent of the protected attribute. The protected
    attribute is only used at *training* time, never to make a decision.
    """
    y = np.asarray(y)
    g = np.asarray(groups)
    w = np.empty(len(y), dtype=float)
    for grp in np.unique(g):
        for out in (0, 1):
            m = (g == grp) & (y == out)
            if m.any():
                w[m] = (g == grp).mean() * (y == out).mean() / m.mean()
    return w
