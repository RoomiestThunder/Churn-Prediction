# Telco Customer Churn Prediction

A leakage-free scikit-learn `Pipeline` that predicts telecom customer churn and
turns the prediction into a business decision: which customers the marketing team
should call before they cancel.

The cost of a **false negative** (missing a real churner) is ~**5× higher** than
a false positive (calling a happy customer), so the model is optimised for
**recall on the churn class** and the decision threshold is tuned to a business
recall target — not the default 0.5.

## Headline results (held-out test set, scored once)

| Metric | Value | Target | Pass |
|--------|------:|:------:|:----:|
| Recall (churn)    | **0.799** | ≥ 0.75 | ✅ |
| Precision (churn) | **0.506** | ≥ 0.45 | ✅ |
| ROC-AUC           | 0.845 | — | — |
| Average precision | 0.662 | — | — |

Model: **Gradient Boosting** at a frozen decision threshold of **0.253**.
Full breakdown and plots in [`results/results.md`](results/results.md).

## Why not accuracy?

The data is imbalanced — only **~27% churn**. A model that blindly predicts
"no churn" for everyone already scores **73.5% accuracy** while catching *zero*
churners. Accuracy is therefore discarded. Both `DummyClassifier` baselines
(`most_frequent`, `stratified`) are beaten by every real model on recall and
ROC-AUC (see the CV table below).

## Repository structure

```
.
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw IBM Telco dataset (7,043 rows)
│   └── new_customers.csv                       # 5 sample rows for predict.py (no Churn)
├── notebook/
│   └── EDA.ipynb                               # exploratory analysis (not evaluated)
├── scripts/
│   ├── preprocessing.py                        # custom transformer + Pipeline builder
│   ├── train.py                                # train, tune, threshold, evaluate, save
│   └── predict.py                              # score new customers with the saved Pipeline
├── results/
│   ├── churn_pipeline.pkl                      # fitted Pipeline (created by train.py)
│   ├── threshold.json                          # frozen decision threshold
│   ├── predictions.csv                         # output of predict.py
│   ├── results.md                              # metrics + business translation
│   └── plots/                                  # PR curve, ROC curve, confusion matrix
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. Environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Train, tune, threshold and evaluate (writes results/)
python scripts/train.py

# 3. Score new customers (writes results/predictions.csv)
python scripts/predict.py
```

Both scripts are run **from the repo root**. The custom feature-engineering
transformer lives in `scripts/preprocessing.py` so the pickled Pipeline can be
unpickled by `predict.py`.

## How leakage is prevented

Every preprocessing step — imputation, scaling, one-hot encoding **and all
feature engineering** — lives **inside a single `Pipeline`**:

```
TelcoFeatureEngineer → ColumnTransformer(num: impute+scale, cat: one-hot) → classifier
```

- The `Pipeline` is fitted on the **training fold only** inside cross-validation,
  so no test/validation statistic ever leaks into preprocessing.
- The trap column `TotalCharges` (empty strings on tenure-0 rows) is coerced to
  numeric `NaN` and imputed by a `SimpleImputer(median)` **inside** the Pipeline.
- The decision threshold is chosen on **out-of-fold training** probabilities and
  the test set is touched exactly once, at the very end.

## Engineered features (justification)

All three are computed row-locally inside `TelcoFeatureEngineer`, so they are
reproduced identically at inference:

| Feature | Definition | Why |
|---------|-----------|-----|
| `charges_per_tenure` | `TotalCharges / max(tenure, 1)` | average monthly spend, decoupled from how long the customer has stayed |
| `n_services` | count of active services | proxy for how embedded the customer is; counts a service as active when its value is **neither `No` nor `No internet service`/`No phone service`** (a naive `== "Yes"` would wrongly score DSL/Fiber as 0) |
| `tenure_bucket` | bins `0-12 / 13-24 / 25-48 / 49+` | lets linear models capture the strongly non-linear "new customers churn most" effect |

## Model selection (5-fold stratified CV, train only)

Mean ± std on the churn (positive) class. Only the final Pipeline step is swapped.

| Model | recall | precision | f1 | roc_auc |
| --- | --- | --- | --- | --- |
| Dummy (most_frequent) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Dummy (stratified) | 0.278 ± 0.025 | 0.275 ± 0.025 | 0.276 ± 0.025 | 0.507 ± 0.017 |
| Logistic Regression | 0.797 ± 0.035 | 0.518 ± 0.016 | 0.628 ± 0.021 | 0.846 ± 0.011 |
| Random Forest | 0.464 ± 0.031 | 0.640 ± 0.033 | 0.538 ± 0.029 | 0.827 ± 0.011 |
| Gradient Boosting | 0.522 ± 0.025 | 0.662 ± 0.035 | 0.583 ± 0.026 | **0.847 ± 0.011** |

**Gradient Boosting** wins on CV ROC-AUC and is tuned with `GridSearchCV`
(`scoring='average_precision'`, two+ hyperparameters over the full Pipeline):
best params `learning_rate=0.05, max_depth=2, n_estimators=300`.

> Logistic Regression already reaches high recall at its default cutoff because
> of `class_weight='balanced'`. Gradient Boosting is preferred for its stronger
> ranking (ROC-AUC / average precision); the recall target is then met by tuning
> the threshold rather than relying on the default 0.5.

## Threshold tuning

The default 0.5 cutoff is not a business decision. Using out-of-fold training
probabilities (`cross_val_predict(..., method='predict_proba')`), the
precision-recall curve is scanned for the **highest-precision threshold with
recall ≥ 0.80**. The frozen threshold (**0.253**) is then applied to the test set
exactly once.

**Business translation:** at this threshold the model flags ~419 customers per
1,000, of whom ~212 are real churners → about 419 retention calls per 1,000
customers to catch the large majority of at-risk customers before they cancel.

## Dataset

IBM Telco Customer Churn dataset (~7,000 rows, 21 columns). Direct download used
in this project (no account required), as listed in the assignment resources:
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
(saved as `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`). Also on
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
