"""Statlog German Credit data (Hofmann, 1994; UCI ML Repository, CC BY 4.0).

1,000 loan applicants, 20 attributes, labelled good (700) or bad (300) credit
risk. The raw file uses opaque codes such as ``A11``; they are decoded here to
readable labels from the dataset documentation.

Protected characteristics are separated from model features:

* ``sex`` is derived from ``personal_status`` (which mixes sex and marital status)
* ``age_group`` is under 25 vs 25 and over
* ``foreign_worker``

These are **not** given to the models ("fairness through unawareness") but are
kept to audit the models' decisions for disparate impact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "german.data"

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "class",
]  # fmt: skip

CODES = {
    "checking_status": {"A11": "< 0", "A12": "0-200", "A13": ">= 200", "A14": "no account"},
    "credit_history": {
        "A30": "no credits / all repaid", "A31": "all repaid at this bank",
        "A32": "existing repaid duly", "A33": "past delays", "A34": "critical account",
    },
    "purpose": {
        "A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio/tv",
        "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
        "A48": "retraining", "A49": "business", "A410": "other",
    },
    "savings": {"A61": "< 100", "A62": "100-500", "A63": "500-1000", "A64": ">= 1000",
                "A65": "unknown / none"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
                         "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status": {
        "A91": "male: divorced/separated", "A92": "female: divorced/separated/married",
        "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings agreement / life insurance",
                 "A123": "car or other", "A124": "unknown / none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee", "A174": "management / self-employed"},
    "telephone": {"A191": "no", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}  # fmt: skip

CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings", "employment_since",
    "other_debtors", "property", "other_installment_plans", "housing", "job", "telephone",
]  # fmt: skip
NUMERIC = [
    "duration_months", "credit_amount", "installment_rate", "residence_since",
    "existing_credits", "num_dependents",
]  # fmt: skip
FEATURES = CATEGORICAL + NUMERIC
PROTECTED = ["sex", "age_group", "foreign_worker"]
# foreign_worker = "no" covers only 37 of 1,000 applicants, too few to audit reliably
AUDITED = ["sex", "age_group"]
TARGET = "default"  # 1 = bad credit risk


def load_german_credit(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load and decode the dataset. Returns features, protected attributes and target."""
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMNS)
    for col, mapping in CODES.items():
        df[col] = df[col].map(mapping)
    df[TARGET] = (df["class"] == 2).astype(int)
    df["sex"] = df["personal_status"].str.split(":").str[0]
    df["age_group"] = (df["age"] < 25).map({True: "under 25", False: "25 and over"})
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df[FEATURES + PROTECTED + ["age", TARGET]]
