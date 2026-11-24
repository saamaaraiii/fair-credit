"""A traditional points-based credit scorecard.

1. **Weight of Evidence (WoE) binning**: each numeric feature is cut into
   quantile bins and each categorical feature keeps its categories (rare ones
   merged). Every bin is replaced by

       WoE = ln( share of goods in bin / share of bads in bin )

   so a higher WoE means safer. Each feature's **Information Value**
   IV = sum((%good - %bad) * WoE) measures its predictive power.
2. **Logistic regression** on the WoE-encoded features.
3. **Scaling to points**: score = offset + factor * ln(odds of good), with
   ``factor = PDO / ln 2``. The default anchors a score of 600 at odds of
   50:1 and adds 20 points every time the odds double. The total is split
   exactly into points per feature and bin, so a credit officer can read the
   whole model off one table.

This is the kind of model banks have used for decades, because every decision
can be traced and explained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


@dataclass
class FeatureBins:
    name: str
    kind: str  # "numeric" or "categorical"
    edges: np.ndarray | None  # numeric bin edges (inner cut points)
    mapping: dict[str, str] | None  # categorical value -> bin label
    woe: dict[str, float]
    iv: float
    counts: dict[str, int]
    bad_rates: dict[str, float]


class WoEBinner:
    def __init__(self, max_bins: int = 5, min_bin_frac: float = 0.05, smoothing: float = 0.5):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.smoothing = smoothing
        self.bins_: dict[str, FeatureBins] = {}

    @staticmethod
    def _numeric_label(edges: np.ndarray, i: int) -> str:
        lo = "-inf" if i == 0 else f"{edges[i - 1]:g}"
        hi = "inf" if i == len(edges) else f"{edges[i]:g}"
        return f"[{lo}, {hi})"

    def _assign(self, fb: FeatureBins, values: pd.Series) -> pd.Series:
        if fb.kind == "numeric":
            idx = np.searchsorted(fb.edges, values.to_numpy(dtype=float), side="right")
            return pd.Series([self._numeric_label(fb.edges, i) for i in idx], index=values.index)
        return values.astype(str).map(lambda v: fb.mapping.get(v, "other"))

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> WoEBinner:
        y = pd.Series(np.asarray(y), index=X.index)
        w = pd.Series(
            np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float),
            index=X.index,
        )
        n_good, n_bad = w[y == 0].sum(), w[y == 1].sum()
        for col in X.columns:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s):
                qs = np.unique(np.quantile(s, np.linspace(0, 1, self.max_bins + 1)[1:-1]))
                edges = qs[qs > s.min()]
                # merge bins that are too small
                while len(edges):
                    counts = np.bincount(
                        np.searchsorted(edges, s, side="right"), minlength=len(edges) + 1
                    )
                    small = np.where(counts < self.min_bin_frac * len(s))[0]
                    if not len(small):
                        break
                    edges = np.delete(edges, min(small[0], len(edges) - 1))
                fb = FeatureBins(col, "numeric", edges, None, {}, 0.0, {}, {})
            else:
                freq = s.astype(str).value_counts(normalize=True)
                mapping = {v: (v if f >= self.min_bin_frac else "other") for v, f in freq.items()}
                fb = FeatureBins(col, "categorical", None, mapping, {}, 0.0, {}, {})
            labels = self._assign(fb, s)
            iv = 0.0
            for lab, grp in y.groupby(labels):
                gw = w.loc[grp.index]
                g = gw[grp == 0].sum() + self.smoothing
                b = gw[grp == 1].sum() + self.smoothing
                pg, pb = g / n_good, b / n_bad
                woe = float(np.log(pg / pb))
                fb.woe[lab] = woe
                fb.counts[lab] = int(len(grp))
                fb.bad_rates[lab] = float(np.average(grp, weights=gw))
                iv += (pg - pb) * woe
            fb.iv = float(iv)
            self.bins_[col] = fb
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, fb in self.bins_.items():
            labels = self._assign(fb, X[col])
            out[col] = labels.map(fb.woe).fillna(0.0).astype(float)  # unseen -> neutral
        return pd.DataFrame(out, index=X.index)

    def bin_labels(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self._assign(fb, X[c]) for c, fb in self.bins_.items()})

    def information_values(self) -> pd.Series:
        return pd.Series({c: fb.iv for c, fb in self.bins_.items()}).sort_values(ascending=False)


class Scorecard(BaseEstimator, ClassifierMixin):
    """WoE logistic-regression scorecard. ``predict_proba[:, 1]`` is P(default)."""

    def __init__(
        self,
        max_bins: int = 5,
        min_iv: float = 0.02,
        C: float = 1.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        pdo: float = 20.0,
    ):
        self.max_bins = max_bins
        self.min_iv = min_iv
        self.C = C
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> Scorecard:
        self.binner_ = WoEBinner(max_bins=self.max_bins).fit(X, y, sample_weight)
        iv = self.binner_.information_values()
        self.features_ = [c for c in iv.index if iv[c] >= self.min_iv]  # drop weak features
        W = self.binner_.transform(X)[self.features_]
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000)
        self.lr_.fit(W, y, sample_weight=sample_weight)
        self.classes_ = self.lr_.classes_
        self.factor_ = self.pdo / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        W = self.binner_.transform(X)[self.features_]
        return self.lr_.predict_proba(W)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------ points
    def points_table(self) -> pd.DataFrame:
        """Points awarded for every (feature, bin). Scores are the row sums."""
        n = len(self.features_)
        b0 = self.lr_.intercept_[0]
        rows = []
        for coef, col in zip(self.lr_.coef_[0], self.features_, strict=True):
            fb = self.binner_.bins_[col]
            for lab, woe in fb.woe.items():
                pts = -self.factor_ * coef * woe + (self.offset_ - self.factor_ * b0) / n
                rows.append(
                    dict(
                        feature=col,
                        bin=lab,
                        woe=woe,
                        bad_rate=fb.bad_rates[lab],
                        count=fb.counts[lab],
                        points=pts,
                    )
                )
        return pd.DataFrame(rows)

    def score(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        """Credit score: offset + factor * ln(odds of being good)."""
        p_bad = self.predict_proba(X)[:, 1]
        return self.offset_ + self.factor_ * np.log((1 - p_bad) / p_bad)

    def points_breakdown(self, X: pd.DataFrame) -> pd.DataFrame:
        labels = self.binner_.bin_labels(X)[self.features_]
        table = self.points_table().set_index(["feature", "bin"])["points"]
        return pd.DataFrame(
            {c: [table.get((c, lab), np.nan) for lab in labels[c]] for c in self.features_},
            index=X.index,
        )

    def reason_codes(self, X: pd.DataFrame, top: int = 4) -> list[list[dict]]:
        """Adverse-action reasons: the features where the applicant lost the most
        points compared with the best bin available for that feature."""
        table = self.points_table()
        best = table.groupby("feature")["points"].max()
        labels = self.binner_.bin_labels(X)[self.features_]
        breakdown = self.points_breakdown(X)
        out = []
        for i in range(len(X)):
            lost = (best[self.features_] - breakdown.iloc[i]).sort_values(ascending=False)
            out.append(
                [
                    dict(
                        feature=f,
                        value=str(X.iloc[i][f]),
                        bin=labels.iloc[i][f],
                        points_lost=round(float(v), 1),
                    )
                    for f, v in lost.head(top).items()
                    if v > 0.05
                ]
            )
        return out
