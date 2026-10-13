"""
missing_data.py
===============
Missing data diagnostics, dropout logistic modeling, single vs multiple
imputation, and assumption-based tipping-point MNAR sensitivity analysis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def fit_dropout_logistic_regression(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, object]:
    """
    Fit a multivariable logistic regression model to assess predictors of trial dropout.

    Model:
        logit(P(Dropout = 1)) ~ treatment + site + age + bmi + sbp_0 + sex

    Statistical Note:
        Observed baseline predictors of dropout assess Missing at Random (MAR)
        vs Missing Completely at Random (MCAR). Observed data alone CANNOT rule out
        or confirm Missing Not at Random (MNAR), which depends on unobserved post-baseline
        outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame with `dropout_week` column.

    Returns
    -------
    Tuple[pd.DataFrame, BinaryResultsWrapper]
        - Summary table with coefficients, Odds Ratios (OR), 95% CIs, and p-values.
        - Fitted statsmodels Logit results object.
    """
    df_model = df.copy()
    # Define binary dropout indicator: 1 if dropped out (has dropout_week), 0 if completed
    df_model["dropout"] = df_model["dropout_week"].notna().astype(int)

    # Impute missing age/bmi with median for the logistic model to keep sample size maximal
    for c in ["age", "bmi"]:
        df_model[c] = df_model[c].fillna(df_model[c].median())

    formula = "dropout ~ C(treatment, Treatment(reference='Placebo')) + C(site) + age + bmi + sbp_0 + C(sex)"
    logit_model = smf.logit(formula, data=df_model).fit(disp=False)

    params = logit_model.params
    conf = logit_model.conf_int()
    pvals = logit_model.pvalues
    bse = logit_model.bse

    results_list = []
    for var in params.index:
        coef = params[var]
        or_val = np.exp(coef)
        ci_lower = np.exp(conf.loc[var, 0])
        ci_upper = np.exp(conf.loc[var, 1])
        pval = pvals[var]
        se = bse[var]

        results_list.append({
            "Predictor": var,
            "Coefficient": round(coef, 4),
            "Std Error": round(se, 4),
            "Odds Ratio": round(or_val, 4),
            "95% CI Lower": round(ci_lower, 4),
            "95% CI Upper": round(ci_upper, 4),
            "p-value": round(pval, 6),
        })

    summary_df = pd.DataFrame(results_list)
    return summary_df, logit_model


def impute_median_unconditional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the original notebook's site-level median imputation.

    Methodological Warning:
        This single imputation groups solely by `site` and does NOT condition
        on treatment arm. At Site 3 (76.7% missing at Week 12), this imputes identical
        values to both TrX and Placebo arms, artificially crushing the treatment effect
        toward zero. Maintained solely as a baseline sensitivity benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trial DataFrame.

    Returns
    -------
    pd.DataFrame
        Imputed DataFrame with missing age dropped (N=177).
    """
    df_imp = df.dropna(subset=["age"]).copy()
    for col in ["bmi", "sbp_4", "sbp_8", "sbp_12"]:
        df_imp[col] = df_imp[col].fillna(df_imp.groupby("site")[col].transform("median"))
    df_imp["delta_sbp"] = df_imp["sbp_0"] - df_imp["sbp_12"]
    return df_imp


def impute_multiple_conditioned(
    df: pd.DataFrame,
    n_imputations: int = 10,
    random_state: int = 42,
) -> List[pd.DataFrame]:
    """
    Perform multivariate imputation conditioned on treatment arm, baseline SBP,
    site, age, BMI, sex, and device to preserve the treatment difference.

    Uses IterativeImputer (Chained Equations with Bayesian Ridge estimator).

    Parameters
    ----------
    df : pd.DataFrame
        Raw trial DataFrame.
    n_imputations : int, default 10
        Number of imputed datasets to generate for Rubin's rules pooling.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    List[pd.DataFrame]
        List of `n_imputations` completed trial DataFrames.
    """
    df_base = df.copy()

    # Numeric coding for categorical covariates for imputer
    feature_cols = [
        "sbp_0", "sbp_4", "sbp_8", "sbp_12", "age", "bmi",
        "site", "device_flag", "deviation_flag"
    ]
    work_df = df_base[feature_cols].copy()
    work_df["is_trx"] = (df_base["treatment"] == "TrX").astype(int)
    work_df["is_male"] = (df_base["sex"] == "M").astype(int)

    imputed_datasets = []
    for i in range(n_imputations):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=random_state + i * 17,
            sample_posterior=True,
        )
        imputed_array = imputer.fit_transform(work_df)
        df_completed = df_base.copy()

        # Update imputed columns
        for idx, col in enumerate(work_df.columns):
            if col in df_completed.columns:
                df_completed[col] = imputed_array[:, idx]

        df_completed["delta_sbp"] = df_completed["sbp_0"] - df_completed["sbp_12"]
        imputed_datasets.append(df_completed)

    return imputed_datasets


def run_tipping_point_sensitivity(
    df: pd.DataFrame,
    delta_penalties: Optional[np.ndarray] = None,
    sites: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Perform a delta-adjustment (tipping point) sensitivity analysis under MNAR assumptions.

    Clinical Rationale:
        Under potential Missing Not at Random (MNAR) dropout, patients who drop out
        in the active treatment arm may experience less benefit than observed completers
        (e.g., due to rebound hypertension, non-compliance, or adverse events).
        We penalize the unobserved Week 12 SBP reduction in the TrX arm by delta mmHg
        and determine at what delta the treatment effect loses statistical significance.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame (complete cases or imputed baseline).
    delta_penalties : np.ndarray, optional
        Grid of penalty values (mmHg) to subtract from TrX dropouts.
        Defaults to [0, 2, 4, 6, 8, 10, 12, 14, 16] mmHg.
    sites : List[int], optional
        Sites to include in the sensitivity model. Defaults to [1, 2].

    Returns
    -------
    pd.DataFrame
        Table with delta penalty, adjusted TrX effect, SE, 95% CI, p-value,
        and significance status at alpha = 0.05.
    """
    if delta_penalties is None:
        delta_penalties = np.arange(0.0, 18.0, 2.0)
    if sites is None:
        sites = [1, 2]

    # Baseline imputed dataset as starting point
    base_data = impute_median_unconditional(df)
    base_data = base_data[base_data["site"].isin(sites)].copy()

    # Identify TrX patients whose SBP_12 was originally missing
    trx_dropout_mask = (base_data["treatment"] == "TrX") & (
        df.loc[base_data.index, "sbp_12"].isna()
    )

    records = []
    formula = "delta_sbp ~ C(treatment, Treatment(reference='Placebo')) + sbp_0 + C(site) + C(device_flag) + age + bmi + C(sex)"

    for delta in delta_penalties:
        temp_df = base_data.copy()
        # Penalize TrX dropouts: reduce delta_sbp by delta (or increase SBP_12 by delta)
        temp_df.loc[trx_dropout_mask, "delta_sbp"] -= delta
        temp_df.loc[trx_dropout_mask, "sbp_12"] += delta

        model = smf.ols(formula, data=temp_df).fit()
        coef = float(model.params["C(treatment, Treatment(reference='Placebo'))[T.TrX]"])
        se = float(model.bse["C(treatment, Treatment(reference='Placebo'))[T.TrX]"])
        pval = float(model.pvalues["C(treatment, Treatment(reference='Placebo'))[T.TrX]"])
        ci = model.conf_int().loc["C(treatment, Treatment(reference='Placebo'))[T.TrX]"]

        records.append({
            "Delta Penalty (mmHg)": delta,
            "TrX Effect Estimate (mmHg)": round(coef, 2),
            "Std Error": round(se, 2),
            "95% CI Lower": round(float(ci[0]), 2),
            "95% CI Upper": round(float(ci[1]), 2),
            "p-value": round(pval, 6),
            "Significant (p < 0.05)": "YES" if pval < 0.05 else "NO",
        })

    return pd.DataFrame(records)
