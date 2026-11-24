# Model card: fair-credit v0.1.0

Format follows Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model details

| | |
|---|---|
| Models | (1) WoE logistic-regression scorecard (champion); (2) histogram gradient boosting with monotonic constraints (challenger); (3) scorecard trained with Kamiran-Calders reweighing on age group |
| Output | Probability of default (PD); scorecard also returns points (600 = 50:1 good:bad odds, PDO 20) |
| Decision rule | Decline if PD ≥ threshold. The threshold minimises expected cost under a 5:1 cost matrix and is chosen on out-of-fold training predictions |
| Explanations | Scorecard: points lost per feature against the best bin. Gradient boosting: exact TreeSHAP values in log-odds |
| Version | 0.1.0 |

## Intended use

An **educational and portfolio** demonstration of explainable, fairness-aware credit risk modelling. **Not for real lending decisions.** The data is from 1994 Germany, contains only 1,000 records and does not represent any current population or product.

## Training data

Statlog German Credit (UCI, CC BY 4.0): 1,000 consumer loans, 30% classed as bad risk. 17 features (account status, credit history, loan purpose, amount, duration, savings, employment, property and so on). Stratified 70/30 train/test split, seed 42.

**Excluded from model inputs:** sex and marital status (`personal_status`), age, foreign-worker status. These are kept only for auditing.

## Performance (test set, n = 300)

| Model | AUC | Gini | KS | Brier | Expected cost |
|---|---:|---:|---:|---:|---:|
| Scorecard | 0.785 | 0.569 | 0.462 | 0.164 | 0.553 |
| Gradient boosting | 0.777 | 0.555 | 0.443 | 0.169 | 0.520 |
| Scorecard + reweighing | 0.785 | 0.571 | 0.475 | 0.164 | 0.570 |

## Fairness evaluation (test set)

| Model | Group | Disparate impact ratio (95% CI) |
|---|---|---|
| Scorecard | sex | 0.90 (0.68–0.99) |
| Scorecard | age < 25 | 0.80 (0.53–0.99) |
| Gradient boosting | age < 25 | 0.71 (0.42–0.98) |
| Scorecard + reweighing | age < 25 | 0.89 (0.57–0.99) |

The foreign-worker minority group (n = 37 overall, 9 in test) is too small to evaluate.

## Limitations and risks

- **Small sample**: every fairness interval is wide. None of the observed disparities is statistically conclusive in either direction.
- **Proxy discrimination**: leaving out protected attributes does not stop the model from learning age through correlated features.
- **Group miscalibration**: without age as an input, under-25s' risk is under-predicted (31.7% predicted vs 47.5% observed). Making approval rates more equal and keeping calibration within each group are conflicting goals.
- **Historical bias**: labels reflect past lending decisions and outcomes, which may themselves have been biased. Applicants who were declined never got a loan, so their outcomes are missing (selection bias / reject inference is not handled).
- **Dated, non-representative data**: currency (DM), products and demographics are from 1994 Germany.

## Recommendations for any real use

Retrain on representative, recent data with reject inference. Agree fairness metrics and tolerances with compliance and legal teams before deployment. Monitor approval rates, calibration and population stability by group after deployment. Keep a human review route for declined applicants.
