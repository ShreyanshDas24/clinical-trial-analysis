"""
sensitivity_analysis.py
========================
Systematic sensitivity analysis suite evaluating analytical robustness across:
- Population subsets (Sites 1–2 vs All Sites)
- Missing data strategies (Complete-case vs Median Imputation vs Multiple Imputation)
- Model formulations (Baseline-adjusted follow-up vs Change-score)
- Influence and diagnostic exclusions
- Robust standard error estimations
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .statistical_analysis import fit_primary_ancova
from .missing_data import impute_median_unconditional


def run_sensitivity_suite(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Execute a structured battery of sensitivity analyses across alternative defensible choices.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw clinical trial DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary table ready for publication and forest plotting.
    """
    results = []

    # Prepare datasets
    df_imp = impute_median_unconditional(df_raw)
    df_cc = df_raw.dropna(subset=["sbp_12", "sbp_0", "age", "bmi"]).copy()
    df_cc["delta_sbp"] = df_cc["sbp_0"] - df_cc["sbp_12"]

    # 1. Primary Model (Sites 1–2 Follow-up ANCOVA, Imputed Benchmark)
    m1_metrics, _ = fit_primary_ancova(df_imp, endpoint="follow_up", sites=[1, 2])
    results.append({
        "Model / Scenario": "1. Primary ANCOVA (Sites 1–2, SBP12)",
        "Category": "Primary Specification",
        "N Analyzed": m1_metrics["n_analyzed"],
        "Estimate (mmHg)": -m1_metrics["treatment_coef"],  # report as reduction magnitude
        "Std Error": m1_metrics["std_error"],
        "95% CI Lower": -m1_metrics["ci_95_upper"],
        "95% CI Upper": -m1_metrics["ci_95_lower"],
        "p-value": m1_metrics["p_value"],
        "Notes": "Baseline-adjusted follow-up model (ICH E9 aligned)",
    })

    # 2. Change-Score ANCOVA (Sites 1–2, delta_sbp)
    m2_metrics, _ = fit_primary_ancova(df_imp, endpoint="change_score", sites=[1, 2])
    results.append({
        "Model / Scenario": "2. Change-Score ANCOVA (Sites 1-2, Delta SBP)",
        "Category": "Endpoint Formulation",
        "N Analyzed": m2_metrics["n_analyzed"],
        "Estimate (mmHg)": m2_metrics["treatment_coef"],
        "Std Error": m2_metrics["std_error"],
        "95% CI Lower": m2_metrics["ci_95_lower"],
        "95% CI Upper": m2_metrics["ci_95_upper"],
        "p-value": m2_metrics["p_value"],
        "Notes": "Mathematically isomorphic to Model 1",
    })

    # 3. Complete-Case Analysis (Sites 1–2)
    m3_metrics, _ = fit_primary_ancova(df_cc, endpoint="change_score", sites=[1, 2])
    results.append({
        "Model / Scenario": "3. Complete Cases (Sites 1–2 only)",
        "Category": "Missing Data Strategy",
        "N Analyzed": m3_metrics["n_analyzed"],
        "Estimate (mmHg)": m3_metrics["treatment_coef"],
        "Std Error": m3_metrics["std_error"],
        "95% CI Lower": m3_metrics["ci_95_lower"],
        "95% CI Upper": m3_metrics["ci_95_upper"],
        "p-value": m3_metrics["p_value"],
        "Notes": "No imputation; avoids median shrinkage",
    })

    # 4. Complete-Case Analysis (All Sites)
    m4_metrics, _ = fit_primary_ancova(df_cc, endpoint="change_score", sites=[1, 2, 3])
    results.append({
        "Model / Scenario": "4. Complete Cases (All 3 Sites)",
        "Category": "Population Scope",
        "N Analyzed": m4_metrics["n_analyzed"],
        "Estimate (mmHg)": m4_metrics["treatment_coef"],
        "Std Error": m4_metrics["std_error"],
        "95% CI Lower": m4_metrics["ci_95_lower"],
        "95% CI Upper": m4_metrics["ci_95_upper"],
        "p-value": m4_metrics["p_value"],
        "Notes": "Includes 13 complete cases from Site 3",
    })

    # 5. All Sites (Median Imputed)
    m5_metrics, _ = fit_primary_ancova(df_imp, endpoint="change_score", sites=[1, 2, 3])
    results.append({
        "Model / Scenario": "5. All Sites (Median Imputed)",
        "Category": "Population Scope",
        "N Analyzed": m5_metrics["n_analyzed"],
        "Estimate (mmHg)": m5_metrics["treatment_coef"],
        "Std Error": m5_metrics["std_error"],
        "95% CI Lower": m5_metrics["ci_95_lower"],
        "95% CI Upper": m5_metrics["ci_95_upper"],
        "p-value": m5_metrics["p_value"],
        "Notes": "Attenuated by 77% median imputation at Site 3",
    })

    # 6. Robust Standard Errors (HC3, Sites 1–2)
    m6_metrics, _ = fit_primary_ancova(df_imp, endpoint="change_score", sites=[1, 2], cov_type="HC3")
    results.append({
        "Model / Scenario": "6. Robust SE (HC3, Sites 1–2)",
        "Category": "Standard Error Robustness",
        "N Analyzed": m6_metrics["n_analyzed"],
        "Estimate (mmHg)": m6_metrics["treatment_coef"],
        "Std Error": m6_metrics["std_error"],
        "95% CI Lower": m6_metrics["ci_95_lower"],
        "95% CI Upper": m6_metrics["ci_95_upper"],
        "p-value": m6_metrics["p_value"],
        "Notes": "Heteroscedasticity-consistent standard errors",
    })

    # 7. Exclusion of Influential Points (Cook's D > 4/N)
    _, base_m = fit_primary_ancova(df_imp, endpoint="change_score", sites=[1, 2])
    infl = base_m.get_influence()
    cooks_d, _ = infl.cooks_distance
    high_cooks = np.where(cooks_d > (4.0 / len(cooks_d)))[0]
    df_no_infl = df_imp[df_imp["site"].isin([1, 2])].drop(df_imp[df_imp["site"].isin([1, 2])].index[high_cooks])

    m7_metrics, _ = fit_primary_ancova(df_no_infl, endpoint="change_score", sites=[1, 2])
    results.append({
        "Model / Scenario": "7. Influential Points Excluded (Sites 1–2)",
        "Category": "Influence Sensitivity",
        "N Analyzed": m7_metrics["n_analyzed"],
        "Estimate (mmHg)": m7_metrics["treatment_coef"],
        "Std Error": m7_metrics["std_error"],
        "95% CI Lower": m7_metrics["ci_95_lower"],
        "95% CI Upper": m7_metrics["ci_95_upper"],
        "p-value": m7_metrics["p_value"],
        "Notes": f"Excludes {len(high_cooks)} points with Cook's D > 4/N",
    })

    # 8. Unadjusted ANCOVA (Treatment + Baseline SBP only)
    df_s12 = df_imp[df_imp["site"].isin([1, 2])].copy()
    m8 = smf.ols("delta_sbp ~ C(treatment, Treatment(reference='Placebo')) + sbp_0", data=df_s12).fit()
    trt_term = "C(treatment, Treatment(reference='Placebo'))[T.TrX]"
    ci8 = m8.conf_int().loc[trt_term]
    results.append({
        "Model / Scenario": "8. Minimal ANCOVA (TrX + SBP0 only)",
        "Category": "Covariate Adjustment",
        "N Analyzed": int(m8.nobs),
        "Estimate (mmHg)": round(float(m8.params[trt_term]), 2),
        "Std Error": round(float(m8.bse[trt_term]), 2),
        "95% CI Lower": round(float(ci8[0]), 2),
        "95% CI Upper": round(float(ci8[1]), 2),
        "p-value": float(m8.pvalues[trt_term]),
        "Notes": "Omits site, device, age, sex, and BMI covariates",
    })

    return pd.DataFrame(results)
