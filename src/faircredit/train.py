"""Train, evaluate and audit both models, then save the artefacts.

    python -m faircredit.train            # writes models/, reports/ and docs/ charts

Protocol: a stratified 70/30 train/test split. Decision thresholds are chosen
on 5-fold out-of-fold predictions **inside the training set** (minimising the
5:1 misclassification cost), so the test set is touched only once, for the
final numbers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import __version__
from .data import AUDITED, FEATURES, TARGET, load_german_credit
from .fairness import bootstrap_disparities, group_report, reweighing_weights
from .models import baseline_cost, best_threshold, evaluate, make_gbm
from .scorecard import Scorecard

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def _oof(model, X, y, sample_weight=None, weight_param="sample_weight"):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    params = {weight_param: sample_weight} if sample_weight is not None else None
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", params=params)[:, 1]


def train_and_evaluate(out_dir: Path = ROOT, make_plots: bool = True, n_boot: int = 1000) -> dict:
    df = load_german_credit()
    train, test = train_test_split(df, test_size=0.3, stratify=df[TARGET], random_state=SEED)
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    # --- models: champion scorecard, challenger GBM, and a fairness-reweighed scorecard
    w_age = reweighing_weights(ytr, train["age_group"])
    specs = {
        "scorecard": (Scorecard(), None, "sample_weight"),
        "gbm": (make_gbm(SEED), None, "gbm__sample_weight"),
        "scorecard_reweighed": (Scorecard(), w_age, "sample_weight"),
    }
    fitted, thresholds, metrics, test_pred = {}, {}, {}, {}
    for name, (model, w, wparam) in specs.items():
        oof = _oof(model, Xtr, ytr, w, wparam)
        thresholds[name] = best_threshold(ytr.to_numpy(), oof)
        fit_kw = {wparam: w} if w is not None else {}
        fitted[name] = model.fit(Xtr, ytr, **fit_kw)
        p = fitted[name].predict_proba(Xte)[:, 1]
        test_pred[name] = p
        metrics[name] = {
            "cv_auc": evaluate(ytr, oof, thresholds[name])["auc"],
            **evaluate(yte, p, thresholds[name]),
        }

    # --- fairness audit on the test set
    fairness = {}
    for name in specs:
        fairness[name] = {}
        for attr in AUDITED:
            rep = group_report(yte, test_pred[name], thresholds[name], test[attr])
            dis = bootstrap_disparities(
                yte, test_pred[name], thresholds[name], test[attr], n_boot=n_boot, seed=SEED
            )
            fairness[name][attr] = {
                "groups": rep.to_dict(orient="index"),
                "disparities": dis.to_dict(orient="index"),
            }

    sc: Scorecard = fitted["scorecard"]
    report = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "n_train": len(train),
            "n_test": len(test),
            "default_rate": float(df[TARGET].mean()),
        },
        "baselines_test": baseline_cost(yte),
        "metrics_test": metrics,
        "thresholds": thresholds,
        "fairness_test": fairness,
        "information_values": sc.binner_.information_values().round(4).to_dict(),
        "scorecard_features": sc.features_,
    }

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": fitted, "thresholds": thresholds, "version": __version__, "metrics": metrics},
        out_dir / "models" / "bundle.joblib",
    )
    (out_dir / "reports" / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    sc.points_table().round(2).to_csv(out_dir / "reports" / "scorecard_points.csv", index=False)

    if make_plots:
        _plots(out_dir / "docs", fitted, test, test_pred, thresholds)
    return report


def _plots(out: Path, fitted, test, test_pred, thresholds) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve

    from .explain import GBMExplainer

    out.mkdir(parents=True, exist_ok=True)
    yte = test[TARGET]
    labels = {
        "scorecard": "WoE scorecard",
        "gbm": "Gradient boosting",
        "scorecard_reweighed": "Scorecard (reweighed)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, p in test_pred.items():
        fpr, tpr, _ = roc_curve(yte, p)
        auc = evaluate(yte, p, thresholds[name])["auc"]
        axes[0].plot(fpr, tpr, label=f"{labels[name]} (AUC {auc:.3f})")
        frac, mean = calibration_curve(yte, p, n_bins=8, strategy="quantile")
        axes[1].plot(mean, frac, "o-", label=labels[name])
    axes[0].plot([0, 1], [0, 1], "k:")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test set)")
    axes[1].plot([0, 0.8], [0, 0.8], "k:")
    axes[1].set(
        xlabel="Predicted P(default)",
        ylabel="Observed default rate",
        title="Calibration (test set)",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_calibration.png", dpi=130)

    sv = GBMExplainer(fitted["gbm"]).shap_values(test[FEATURES])
    imp = sv.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#1f77b4")
    ax.set(xlabel="mean |SHAP| (log-odds of default)", title="Gradient boosting: global importance")
    fig.tight_layout()
    fig.savefig(out / "shap_importance.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, attr in zip(axes, AUDITED, strict=True):
        width = 0.27
        groups = sorted(test[attr].unique())
        x = np.arange(len(groups))
        for i, name in enumerate(test_pred):
            rep = group_report(yte, test_pred[name], thresholds[name], test[attr]).loc[groups]
            ax.bar(x + (i - 1) * width, rep["approval_rate"], width, label=labels[name])
        ax.set_xticks(x, groups)
        ax.set_ylim(0, 1)
        ax.set_title(f"Approval rate by {attr.replace('_', ' ')}")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fairness_approval_rates.png", dpi=130)

    sc = fitted["scorecard"]
    scores = sc.score(test[FEATURES])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(scores.min(), scores.max(), 30)
    ax.hist(scores[yte == 0], bins, alpha=0.6, label="repaid")
    ax.hist(scores[yte == 1], bins, alpha=0.6, label="defaulted")
    cut = sc.offset_ + sc.factor_ * np.log((1 - thresholds["scorecard"]) / thresholds["scorecard"])
    ax.axvline(cut, color="k", ls="--", label=f"cut-off ≈ {cut:.0f}")
    ax.set(
        xlabel="Credit score (600 = 50:1 odds, +20 pts doubles odds)",
        ylabel="Applicants",
        title="Scorecard score distribution (test set)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "score_distribution.png", dpi=130)


def _print_summary(r: dict) -> None:
    m = pd.DataFrame(r["metrics_test"]).T
    cols = [
        "cv_auc",
        "auc",
        "gini",
        "ks",
        "brier",
        "threshold",
        "approval_rate",
        "bad_rate_among_approved",
        "expected_cost",
    ]
    print("Test-set performance\n")
    print(m[cols].round(3).to_markdown())
    print(f"\nBaseline cost per applicant: {r['baselines_test']}\n")
    for attr in AUDITED:
        print(f"Fairness by {attr}")
        rows = {}
        for name, f in r["fairness_test"].items():
            d = f[attr]["disparities"]
            rows[name] = {
                k: f"{v['estimate']:.2f} [{v['ci_low']:.2f}, {v['ci_high']:.2f}]"
                for k, v in d.items()
                if k in ("disparate_impact_ratio", "equal_opportunity_gap")
            }
        print(pd.DataFrame(rows).T.to_markdown())
        groups = r["fairness_test"]["scorecard"][attr]["groups"]
        print(pd.DataFrame(groups).T.round(3).to_markdown())
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT)
    args = ap.parse_args()
    _print_summary(train_and_evaluate(args.out, make_plots=not args.no_plots))
