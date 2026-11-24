Test-set performance

|                     |   cv_auc |   auc |   gini |    ks |   brier |   threshold |   approval_rate |   bad_rate_among_approved |   expected_cost |
|:--------------------|---------:|------:|-------:|------:|--------:|------------:|----------------:|--------------------------:|----------------:|
| scorecard           |    0.762 | 0.785 |  0.569 | 0.462 |   0.164 |       0.2   |           0.447 |                     0.112 |           0.553 |
| gbm                 |    0.769 | 0.777 |  0.555 | 0.443 |   0.169 |       0.145 |           0.36  |                     0.083 |           0.52  |
| scorecard_reweighed |    0.76  | 0.785 |  0.571 | 0.475 |   0.164 |       0.155 |           0.39  |                     0.111 |           0.57  |

Baseline cost per applicant: {'approve_everyone': 1.5, 'reject_everyone': 0.7}

Fairness by sex
|                     | disparate_impact_ratio   | equal_opportunity_gap   |
|:--------------------|:-------------------------|:------------------------|
| scorecard           | 0.90 [0.68, 0.99]        | 0.01 [0.00, 0.17]       |
| gbm                 | 0.94 [0.66, 0.99]        | 0.00 [0.00, 0.17]       |
| scorecard_reweighed | 0.91 [0.66, 1.00]        | 0.01 [0.00, 0.17]       |
|        |   n |   actual_default_rate |   mean_predicted_pd |   approval_rate |   approval_rate_if_good |
|:-------|----:|----------------------:|--------------------:|----------------:|------------------------:|
| female | 101 |                 0.347 |               0.308 |           0.416 |                   0.561 |
| male   | 199 |                 0.276 |               0.289 |           0.462 |                   0.569 |

Fairness by age_group
|                     | disparate_impact_ratio   | equal_opportunity_gap   |
|:--------------------|:-------------------------|:------------------------|
| scorecard           | 0.80 [0.53, 0.99]        | 0.06 [0.00, 0.25]       |
| gbm                 | 0.71 [0.42, 0.98]        | 0.06 [0.00, 0.25]       |
| scorecard_reweighed | 0.89 [0.57, 0.99]        | 0.02 [0.00, 0.23]       |
|             |   n |   actual_default_rate |   mean_predicted_pd |   approval_rate |   approval_rate_if_good |
|:------------|----:|----------------------:|--------------------:|----------------:|------------------------:|
| 25 and over | 241 |                 0.257 |               0.29  |           0.465 |                   0.575 |
| under 25    |  59 |                 0.475 |               0.317 |           0.373 |                   0.516 |

