"""
Train, tune, threshold and evaluate the Telco churn model.

Run from the repo root:

    python scripts/train.py

Pipeline of work:
  1. Stratified 80/20 train/test split (test touched exactly once, at the end).
  2. DummyClassifier baselines (most_frequent, stratified).
  3. Stratified 5-fold CV comparison of 3 classifiers (recall/precision/F1/AUC).
  4. Light GridSearchCV hyperparameter tuning on the best classifier.
  5. Threshold tuning on out-of-fold training probabilities for recall >= 0.80.
  6. Single final evaluation on the held-out test set at the frozen threshold.
  7. Persist the fitted Pipeline + threshold and write results/results.md + plots.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
    f1_score, make_scorer, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
    train_test_split,
)

from preprocessing import build_pipeline, load_raw, split_features_target

# ----------------------------------------------------------------------------
# Paths (robust to the current working directory)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
PIPE_PATH = RESULTS / "churn_pipeline.pkl"
THR_PATH = RESULTS / "threshold.json"

RANDOM_STATE = 42
RECALL_TARGET = 0.80          # business target used to pick the threshold
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}

# Candidate classifiers (swapped into the final Pipeline step).
CANDIDATES = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE,
        n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# Small hyperparameter grids (full-Pipeline 'model__' naming) per classifier.
GRIDS = {
    "Logistic Regression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__solver": ["lbfgs", "liblinear"],
    },
    "Random Forest": {
        "model__n_estimators": [300, 500],
        "model__max_depth": [6, 10, None],
        "model__min_samples_leaf": [1, 5],
    },
    "Gradient Boosting": {
        "model__n_estimators": [150, 300],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
}


def cv_table(X_train, y_train) -> pd.DataFrame:
    """5-fold stratified CV of dummies + candidates; mean +/- std per metric."""
    rows = {}
    dummies = {
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
        "Dummy (stratified)": DummyClassifier(
            strategy="stratified", random_state=RANDOM_STATE),
    }
    for name, model in {**dummies, **CANDIDATES}.items():
        res = cross_validate(build_pipeline(model), X_train, y_train,
                             cv=CV, scoring=SCORING, n_jobs=-1)
        rows[name] = {
            m: f"{res[f'test_{m}'].mean():.3f} +/- {res[f'test_{m}'].std():.3f}"
            for m in SCORING
        }
    return pd.DataFrame(rows).T[list(SCORING)]


def best_candidate(X_train, y_train) -> str:
    """Pick the real classifier with the highest mean CV ROC-AUC."""
    scores = {}
    for name, model in CANDIDATES.items():
        res = cross_validate(build_pipeline(model), X_train, y_train,
                             cv=CV, scoring="roc_auc", n_jobs=-1)
        scores[name] = res["test_score"].mean()
    return max(scores, key=scores.get)


def pick_threshold(y_true, proba) -> float:
    """Highest-precision threshold whose recall meets RECALL_TARGET."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    # rec/prec have len = len(thr)+1; align with thresholds via index.
    candidates = [(thr[i], prec[i]) for i in range(len(thr))
                  if rec[i] >= RECALL_TARGET]
    if not candidates:                      # fallback: best F1
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return float(thr[int(np.argmax(f1[:-1]))])
    return float(max(candidates, key=lambda t: t[1])[0])


def save_pr_curve(y_true, proba, threshold):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="steelblue", label="PR curve")
    idx = int(np.argmin(np.abs(thr - threshold)))
    ax.scatter(rec[idx], prec[idx], color="red", zorder=5,
               label=f"threshold = {threshold:.3f}")
    ax.axvline(RECALL_TARGET, ls="--", color="grey", alpha=0.6,
               label=f"recall target = {RECALL_TARGET}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall (out-of-fold, train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "precision_recall_curve.png", dpi=120)
    plt.close(fig)


def save_roc_curve(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange",
            label=f"ROC (AUC = {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "roc_curve.png", dpi=120)
    plt.close(fig)


def save_confusion(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No churn", "Churn"],
        colorbar=False, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix (test set, frozen threshold)")
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    # 1. Load + split -------------------------------------------------------
    df = load_raw(DATA)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    majority_acc = max(y.mean(), 1 - y.mean())
    print(f"Rows: {len(df)} | churn rate: {y.mean():.1%} "
          f"| majority-class accuracy baseline: {majority_acc:.1%}")

    # 2-3. Baselines + candidate comparison --------------------------------
    print("\n5-fold stratified CV (mean +/- std):")
    table = cv_table(X_train, y_train)
    print(table.to_string())

    # 4. Tune the best candidate -------------------------------------------
    winner = best_candidate(X_train, y_train)
    print(f"\nBest candidate by CV ROC-AUC: {winner}")
    grid = GridSearchCV(
        build_pipeline(clone(CANDIDATES[winner])), GRIDS[winner],
        scoring="average_precision", cv=CV, n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV average-precision: {grid.best_score_:.3f}")
    best_pipe = grid.best_estimator_

    # 5. Threshold tuning on out-of-fold TRAIN probabilities ---------------
    oof_proba = cross_val_predict(
        clone(best_pipe), X_train, y_train, cv=CV,
        method="predict_proba", n_jobs=-1)[:, 1]
    threshold = pick_threshold(y_train, oof_proba)
    save_pr_curve(y_train, oof_proba, threshold)
    print(f"\nFrozen threshold (recall >= {RECALL_TARGET} on OOF train): "
          f"{threshold:.3f}")

    # 6. Final evaluation on the held-out TEST set (once) ------------------
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    t_recall = recall_score(y_test, test_pred)
    t_prec = precision_score(y_test, test_pred)
    t_auc = roc_auc_score(y_test, test_proba)
    t_ap = average_precision_score(y_test, test_proba)
    cm = confusion_matrix(y_test, test_pred)
    cm_df = pd.DataFrame(
        cm, index=["True: No churn", "True: Churn"],
        columns=["Pred: No churn", "Pred: Churn"])
    save_roc_curve(y_test, test_proba)
    save_confusion(y_test, test_pred)

    print("\n=== TEST SET (frozen threshold, scored once) ===")
    print(f"Recall (churn):    {t_recall:.3f}  (target >= 0.75)")
    print(f"Precision (churn): {t_prec:.3f}  (target >= 0.45)")
    print(f"ROC-AUC:           {t_auc:.3f}")
    print(f"Avg precision:     {t_ap:.3f}")
    print("Confusion matrix:")
    print(cm_df.to_string())

    # 7. Persist pipeline + threshold + write results.md -------------------
    joblib.dump(best_pipe, PIPE_PATH)
    THR_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))
    write_results_md(table, winner, grid, threshold, majority_acc,
                     t_recall, t_prec, t_auc, t_ap, cm_df, len(y_test),
                     int(test_pred.sum()), int(cm[1, 1]))
    print(f"\nSaved pipeline   -> {PIPE_PATH.relative_to(ROOT)}")
    print(f"Saved threshold  -> {THR_PATH.relative_to(ROOT)}")
    print(f"Saved report     -> {(RESULTS / 'results.md').relative_to(ROOT)}")


def df_to_md(df, index_label=""):
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join([index_label] + [str(c) for c in cols]) + " |"
    sep = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
    rows = [
        "| " + " | ".join([str(idx)] + [str(df.loc[idx, c]) for c in cols]) + " |"
        for idx in df.index
    ]
    return "\n".join([header, sep, *rows])


def write_results_md(table, winner, grid, threshold, majority_acc,
                     recall, prec, auc, ap, cm_df, n_test, n_flagged, n_tp):
    per_1000 = n_flagged / n_test * 1000
    tp_1000 = n_tp / n_test * 1000
    md = f"""# Results — Telco Churn Model

## Why not accuracy
The classes are imbalanced (~{(1-majority_acc)*100:.0f}% churn). A model that
always predicts "no churn" already scores **{majority_acc:.1%} accuracy** while
catching zero churners. Accuracy is therefore meaningless here; we optimise for
**recall on the churn class** (a missed churner costs ~5x a false alarm) and
report ROC-AUC / average precision for ranking quality.

## Cross-validation comparison (5-fold stratified, train only)
Mean +/- std on the churn (positive) class.

{df_to_md(table, "Model")}

Both DummyClassifier baselines are beaten by every real model on recall and
ROC-AUC, which justifies discarding accuracy as the headline metric.

## Best model & tuning
* Selected by CV ROC-AUC: **{winner}**
* GridSearchCV (scoring = average precision) best params: `{grid.best_params_}`
* Best CV average precision: {grid.best_score_:.3f}

## Threshold
Chosen on out-of-fold **training** probabilities (never the test set) as the
highest-precision threshold with recall >= {RECALL_TARGET:.2f}.
**Frozen threshold = {threshold:.3f}.**

## Final test-set performance (scored once)
| Metric | Value | Target |
|--------|------:|:------:|
| Recall (churn)    | {recall:.3f} | >= 0.75 |
| Precision (churn) | {prec:.3f} | >= 0.45 |
| ROC-AUC           | {auc:.3f} | — |
| Average precision | {ap:.3f} | — |

Confusion matrix:

{df_to_md(cm_df)}

## Business translation
At the frozen threshold the model flags about **{per_1000:.0f} customers per
1000**, of whom roughly **{tp_1000:.0f} are real churners**. For a 1000-customer
weekly batch that is ~{per_1000:.0f} retention calls to catch ~{tp_1000:.0f} of
the at-risk customers — the marketing team trades a manageable call volume for
catching the large majority of churners before they cancel.

![PR curve](plots/precision_recall_curve.png)
![ROC curve](plots/roc_curve.png)
![Confusion matrix](plots/confusion_matrix.png)
"""
    (RESULTS / "results.md").write_text(md)


if __name__ == "__main__":
    main()
