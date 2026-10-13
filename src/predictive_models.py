"""
predictive_models.py
====================
Exploratory predictive modeling for blood pressure reduction (delta_sbp).

Statistical & Methodological Disclaimer:
----------------------------------------
These machine learning models evaluate purely prognostic and predictive capability.
They DO NOT estimate causal treatment effects, control Type I error, or validate
regulatory clinical efficacy. Evaluating predictive performance serves solely as an
exploratory benchmark to assess non-linear feature interactions and variance explanation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_ml_pipeline(estimator) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with strict leakage-free preprocessing.

    Parameters
    ----------
    estimator : BaseEstimator
        Scikit-learn regressor.

    Returns
    -------
    Pipeline
        Full processing and estimation pipeline.
    """
    num_cols = ["sbp_0", "age", "bmi"]
    cat_cols = ["site", "treatment", "device_flag", "sex"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def evaluate_predictive_models(
    df: pd.DataFrame,
    test_size: float = 0.3,
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Pipeline]]:
    """
    Train and evaluate Ridge, Random Forest, and Gradient Boosting against a Dummy baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed or clean trial DataFrame with `delta_sbp`.
    test_size : float, default 0.3
        Holdout test set fraction.
    cv_splits : int, default 5
        Number of cross-validation folds.
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Pipeline]]
        - Table of holdout test metrics (R², MAE, RMSE).
        - Table of cross-validated metrics (Mean ± SD).
        - Dictionary of fitted pipelines.
    """
    features = ["sbp_0", "age", "bmi", "site", "treatment", "device_flag", "sex"]
    target = "delta_sbp"

    work_df = df.dropna(subset=features + [target]).copy()
    X = work_df[features]
    y = work_df[target]

    models = {
        "Dummy (Mean Baseline)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=3, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }

    # 1. Holdout Evaluation (reproducing original 70/30 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    holdout_records = []
    fitted_pipes = {}

    dummy_mae = None
    for name, model in models.items():
        pipe = build_ml_pipeline(model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

        if name == "Dummy (Mean Baseline)":
            dummy_mae = mae
            beats_dummy = "BASELINE"
        else:
            beats_dummy = "YES" if (dummy_mae is not None and mae < dummy_mae) else "NO"

        holdout_records.append({
            "Model": name,
            "Test R²": round(r2, 3),
            "Test MAE (mmHg)": round(mae, 2),
            "Test RMSE (mmHg)": round(rmse, 2),
            "Beats Dummy MAE": beats_dummy,
        })
        fitted_pipes[name] = pipe

    holdout_df = pd.DataFrame(holdout_records)

    # 2. 5-Fold Cross Validation Evaluation
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_records = []

    for name, model in models.items():
        pipe = build_ml_pipeline(model)
        scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
        cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

        r2_mean = float(cv_res["test_r2"].mean())
        r2_sd = float(cv_res["test_r2"].std())
        mae_mean = float(-cv_res["test_neg_mean_absolute_error"].mean())
        mae_sd = float(cv_res["test_neg_mean_absolute_error"].std())
        rmse_mean = float(-cv_res["test_neg_root_mean_squared_error"].mean())
        rmse_sd = float(cv_res["test_neg_root_mean_squared_error"].std())

        cv_records.append({
            "Model": name,
            "CV R² (Mean ± SD)": f"{r2_mean:.3f} ± {r2_sd:.3f}",
            "CV MAE (Mean ± SD)": f"{mae_mean:.2f} ± {mae_sd:.2f}",
            "CV RMSE (Mean ± SD)": f"{rmse_mean:.2f} ± {rmse_sd:.2f}",
        })

    cv_df = pd.DataFrame(cv_records)

    return holdout_df, cv_df, fitted_pipes


def compute_feature_importance(
    pipeline: Pipeline,
    df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance on the full dataset.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn pipeline.
    df : pd.DataFrame
        Trial DataFrame.
    random_state : int, default 42
        Seed for permutation.

    Returns
    -------
    pd.DataFrame
        Table of features sorted by mean importance score.
    """
    features = ["sbp_0", "age", "bmi", "site", "treatment", "device_flag", "sex"]
    work_df = df.dropna(subset=features + ["delta_sbp"]).copy()
    X = work_df[features]
    y = work_df["delta_sbp"]

    result = permutation_importance(
        pipeline, X, y, n_repeats=10, random_state=random_state, scoring="neg_mean_absolute_error"
    )

    records = []
    for col, mean_imp, std_imp in zip(features, result.importances_mean, result.importances_std):
        records.append({
            "Feature": col,
            "Importance (Δ MAE)": round(float(mean_imp), 3),
            "Std Dev": round(float(std_imp), 3),
        })

    return pd.DataFrame(records).sort_values("Importance (Δ MAE)", ascending=False).reset_index(drop=True)
