"""
Preprocessing & feature engineering for the Telco churn project.

All feature engineering lives INSIDE the scikit-learn Pipeline as a custom
transformer, so it is reproduced identically at training and inference time and
cannot leak. These class definitions live in `scripts/` so the pickled Pipeline
can be unpickled by `predict.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Identifier and target are removed before the pipeline ever sees the data.
ID_COL = "customerID"
TARGET = "Churn"

# Service columns used to build `n_services`. A service is "active" when its
# value is neither "No" nor one of the "No * service" sentinels. A naive
# `== "Yes"` count would silently score InternetService (DSL/Fiber optic) as 0.
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]
_INACTIVE = {"No", "No internet service", "No phone service"}

# Column groups AFTER feature engineering (consumed by the ColumnTransformer).
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges",
            "charges_per_tenure", "n_services"]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_bucket",
]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-local feature engineering on the raw Telco frame.

    Stateless (``fit`` learns nothing), so it is leakage-proof by construction.

    Engineered features
    -------------------
    * ``charges_per_tenure`` — TotalCharges / max(tenure, 1); average spend per
      month a customer has been active. Captures spend intensity independent of
      how long they have stayed.
    * ``n_services`` — count of active services; a proxy for how embedded the
      customer is in the product. More services generally means stickier.
    * ``tenure_bucket`` — coarse tenure bins; lets linear models capture the
      strongly non-linear "new customers churn most" effect.

    Also coerces the trap column ``TotalCharges`` (empty strings on tenure-0
    rows) to numeric NaN, which the downstream ``SimpleImputer`` fills.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Trap: TotalCharges loads as object because of empty strings.
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        # Feature 1: spend per active month (guard against tenure == 0).
        X["charges_per_tenure"] = X["TotalCharges"] / X["tenure"].clip(lower=1)

        # Feature 2: number of active services (handles the DSL/Fiber trap).
        active = pd.DataFrame(
            {c: ~X[c].isin(_INACTIVE) for c in SERVICE_COLS}
        )
        X["n_services"] = active.sum(axis=1).astype(int)

        # Feature 3: tenure bucket as a categorical string.
        X["tenure_bucket"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12", "13-24", "25-48", "49+"],
        ).astype(str)

        return X


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: median-impute + scale numerics, one-hot categoricals.

    Every fit happens on the training fold only because the whole thing is run
    inside the Pipeline / cross-validation.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def build_pipeline(model) -> Pipeline:
    """Full leakage-free pipeline: feature engineering → preprocessing → model."""
    return Pipeline([
        ("features", TelcoFeatureEngineer()),
        ("prep", build_preprocessor()),
        ("model", model),
    ])


def load_raw(path) -> pd.DataFrame:
    """Load the raw CSV and encode the target Churn (Yes/No → 1/0)."""
    df = pd.read_csv(path)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split_features_target(df: pd.DataFrame):
    """Return X (features, no id/target) and y (target)."""
    X = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])
    y = df[TARGET] if TARGET in df.columns else None
    return X, y
