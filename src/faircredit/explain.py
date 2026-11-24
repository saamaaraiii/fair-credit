"""SHAP explanations for the gradient-boosting model, mapped back to the
original (pre-one-hot) features so they read like the scorecard's reasons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .data import CATEGORICAL, FEATURES


class GBMExplainer:
    def __init__(self, pipeline: Pipeline):
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["gbm"]
        self.tree = shap.TreeExplainer(self.model)
        encoder = self.prep.named_transformers_["cat"]
        self.columns_ = list(self.prep.get_feature_names_out())
        owner = {}
        for feat, cats in zip(CATEGORICAL, encoder.categories_, strict=True):
            for c in cats:
                owner[f"{feat}_{c}"] = feat
        self.owner_ = [owner.get(col, col) for col in self.columns_]

    @property
    def base_value(self) -> float:
        """Average model output in log-odds of default."""
        return float(np.ravel(self.tree.expected_value)[0])

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP contributions in log-odds of default, one column per original
        feature (one-hot columns summed back to their feature). Positive values
        push towards default. Rows sum to ``decision_function - base_value``."""
        Z = self.prep.transform(X[FEATURES])
        sv = np.asarray(self.tree.shap_values(Z))
        df = pd.DataFrame(sv, columns=self.columns_, index=X.index)
        return df.T.groupby(self.owner_).sum().T[FEATURES]

    def reason_codes(self, X: pd.DataFrame, top: int = 4) -> list[list[dict]]:
        sv = self.shap_values(X)
        out = []
        for i in range(len(X)):
            row = sv.iloc[i].sort_values(ascending=False)
            out.append(
                [
                    dict(feature=f, value=str(X.iloc[i][f]), impact_log_odds=round(float(v), 3))
                    for f, v in row.head(top).items()
                    if v > 0.01
                ]
            )
        return out
