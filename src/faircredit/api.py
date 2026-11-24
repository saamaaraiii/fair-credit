"""REST API serving credit decisions with reasons.

    uvicorn faircredit.api:app --reload
    # then open http://127.0.0.1:8000/docs

Every decision comes with its reasons. Under UK GDPR Article 22 and the EU AI
Act (where credit scoring is classed as high-risk), an automated credit
decision has to be explainable to the applicant.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .data import CODES, FEATURES

BUNDLE_PATH = Path(__file__).resolve().parents[2] / "models" / "bundle.joblib"


def _choices(col: str):
    return Literal[tuple(CODES[col].values())]  # type: ignore[valid-type]


class Applicant(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "checking_status": "< 0",
                "duration_months": 24,
                "credit_history": "existing repaid duly",
                "purpose": "used car",
                "credit_amount": 4000,
                "savings": "< 100",
                "employment_since": "1-4 years",
                "installment_rate": 3,
                "other_debtors": "none",
                "residence_since": 2,
                "property": "car or other",
                "other_installment_plans": "none",
                "housing": "own",
                "existing_credits": 1,
                "job": "skilled employee",
                "num_dependents": 1,
                "telephone": "no",
            }
        }
    )

    checking_status: _choices("checking_status") = Field(description="Current account balance (DM)")
    duration_months: int = Field(ge=1, le=120)
    credit_history: _choices("credit_history")
    purpose: _choices("purpose")
    credit_amount: float = Field(gt=0, le=1_000_000)
    savings: _choices("savings") = Field(description="Savings balance (DM)")
    employment_since: _choices("employment_since")
    installment_rate: int = Field(ge=1, le=4, description="Instalment as % of income band (1-4)")
    other_debtors: _choices("other_debtors")
    residence_since: int = Field(ge=1, le=4)
    property: _choices("property")
    other_installment_plans: _choices("other_installment_plans")
    housing: _choices("housing")
    existing_credits: int = Field(ge=1, le=10)
    job: _choices("job")
    num_dependents: int = Field(ge=0, le=10)
    telephone: _choices("telephone")


class Reason(BaseModel):
    feature: str
    value: str
    impact: float = Field(description="Points lost (scorecard) or log-odds added (gbm)")


class Decision(BaseModel):
    model: str
    model_version: str
    probability_of_default: float
    credit_score: float | None = Field(description="Scorecard points (scorecard models only)")
    decision: Literal["approve", "decline"]
    threshold: float
    reasons: list[Reason] = Field(description="Main factors working against the applicant")


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        from .train import train_and_evaluate

        train_and_evaluate(make_plots=False, n_boot=50)
    return joblib.load(BUNDLE_PATH)


@lru_cache(maxsize=1)
def gbm_explainer():
    from .explain import GBMExplainer

    return GBMExplainer(load_bundle()["models"]["gbm"])


app = FastAPI(
    title="fair-credit",
    version=__version__,
    description="Explainable, fairness-audited credit decisions on the Statlog German Credit data.",
)

ModelName = Literal["scorecard", "gbm", "scorecard_reweighed"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/models")
def models() -> dict:
    b = load_bundle()
    return {
        name: {"threshold": b["thresholds"][name], **b["metrics"][name]} for name in b["models"]
    }


@app.post("/v1/score", response_model=Decision)
def score(applicant: Applicant, model: ModelName = Query("scorecard")) -> Decision:
    b = load_bundle()
    if model not in b["models"]:
        raise HTTPException(404, f"unknown model {model}")
    X = pd.DataFrame([applicant.model_dump()])[FEATURES]
    est = b["models"][model]
    pd_ = float(est.predict_proba(X)[0, 1])
    thr = float(b["thresholds"][model])

    if model == "gbm":
        reasons = [
            Reason(feature=r["feature"], value=r["value"], impact=r["impact_log_odds"])
            for r in gbm_explainer().reason_codes(X)[0]
        ]
        points = None
    else:
        reasons = [
            Reason(feature=r["feature"], value=r["value"], impact=r["points_lost"])
            for r in est.reason_codes(X)[0]
        ]
        points = round(float(est.score(X)[0]), 1)

    return Decision(
        model=model,
        model_version=b["version"],
        probability_of_default=round(pd_, 4),
        credit_score=points,
        decision="approve" if pd_ < thr else "decline",
        threshold=round(thr, 4),
        reasons=reasons,
    )
