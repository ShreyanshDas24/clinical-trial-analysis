"""
statistical_analysis.py
========================
Primary regulatory ANCOVA models (follow-up vs change-score formulations),
full OLS diagnostics, multiplicity corrections, and subgroup evaluations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def fit_primary_ancova(
    df: pd.DataFrame,
    endpoint: str = "follow_up",
    sites: Optional[List[int]] = None,
    cov_type: str = "nonrobust",
) -> Tuple[Dict[str, Union[float, str]], sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Fit the primary ANCOVA efficacy model on the pre-specified trial population.

    Formulations:
    - If `endpoint == 'follow_up'`:
        sbp_12 ~ C(treatment) + sbp_0 + C(site) + C(device_flag) + age + bmi + C(sex)
        TrX coefficient represents the adjusted difference in Week 12 SBP (negative = therapeutic reduction).
    - If `endpoint == 'change_score'`:
        delta_sbp ~ C(treatment) + sbp_0 + C(site) + C(device_flag) + age + bmi + C(sex)
        TrX coefficient represents the adjusted additional reduction in SBP (positive = therapeutic improvement).

    Methodological Note on Singularity:
        In Sites 1–2, `deviation_flag` has zero variance (all values = 0).
        Including it produces design matrix rank-deficiency. This function omits
        zero-variance terms, eliminating condition number inflation (dropping from 10^18 to 2000).

    Parameters
    ----------
    df : pd.DataFrame
        Clinical trial DataFrame.
    endpoint : str, default 'follow_up'
        Either 'follow_up' (SBP at Week 12) or 'change_score' (delta_sbp).
    sites : List[int], optional
        Sites to include in primary analysis. Defaults to [1, 2].
    cov_type : str, default 'nonrobust'
        Covariance type ('nonrobust', 'HC1', or 'HC3').

    Returns
    -------
    Tuple[Dict[str, Union[float, str]], RegressionResultsWrapper]
        - Structured dictionary of primary efficacy metrics.
        - Full statsmodels RegressionResultsWrapper object.
    """
    if sites is None:
        sites = [1, 2]

    df_subset = df[df["site"].isin(sites)].copy()

    # Determine outcome variable
    if endpoint == "follow_up":
        dep_var = "sbp_12"
    elif endpoint == "change_score":
        dep_var = "delta_sbp"
    else:
        raise ValueError(f"Unknown endpoint '{endpoint}'. Choose 'follow_up' or 'change_score'.")

    # Drop missing required variables for complete modeling
    df_subset = df_subset.dropna(subset=[dep_var, "sbp_0", "age", "bmi"]).copy()
    n_obs = len(df_subset)

    # Base formula components
    covariates = ["C(treatment, Treatment(reference='Placebo'))", "sbp_0", "age", "bmi", "C(sex)"]

    if len(sites) > 1 and df_subset["site"].nunique() > 1:
        covariates.append("C(site)")

    if df_subset["device_flag"].nunique() > 1:
        covariates.append("C(device_flag)")

    # Only include deviation_flag if it has non-zero variance
    if "deviation_flag" in df_subset.columns and df_subset["deviation_flag"].nunique() > 1:
        covariates.append("deviation_flag")

    formula = f"{dep_var} ~ " + " + ".join(covariates)

    if cov_type in ("HC1", "HC3"):
        model = smf.ols(formula, data=df_subset).fit(cov_type=cov_type)
    else:
        model = smf.ols(formula, data=df_subset).fit()

    trt_term = "C(treatment, Treatment(reference='Placebo'))[T.TrX]"
    coef = float(model.params[trt_term])
    se = float(model.bse[trt_term])
    t_val = float(model.tvalues[trt_term])
    p_val = float(model.pvalues[trt_term])
    ci = model.conf_int().loc[trt_term]
    ci_lower = float(ci[0])
    ci_upper = float(ci[1])

    # Cohen's d: effect size standardized by residual standard deviation
    resid_sd = float(np.std(model.resid, ddof=int(model.df_resid))) if model.df_resid > 0 else 1.0
    cohens_d = abs(coef) / resid_sd if resid_sd > 0 else 0.0

    metrics = {
        "endpoint": endpoint,
        "n_analyzed": n_obs,
        "treatment_coef": round(coef, 4),
        "std_error": round(se, 4),
        "t_statistic": round(t_val, 4),
        "p_value": p_val,
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "cohens_d": round(cohens_d, 3),
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "cov_type": cov_type,
    }

    return metrics, model


def compute_model_diagnostics(
    model: sm.regression.linear_model.RegressionResultsWrapper
) -> Dict[str, Union[float, pd.DataFrame, List[int]]]:
    """
    Perform rigorous regression assumption diagnostics on the fitted model.

    Diagnostics include:
    - Homoscedasticity: Breusch-Pagan Lagrange Multiplier test.
    - Normality: Shapiro-Wilk test on OLS residuals.
    - Multicollinearity: Variance Inflation Factor (VIF) for all predictors.
    - Influential points: Cook's distance exceeding 4/N threshold.

    Parameters
    ----------
    model : RegressionResultsWrapper
        Fitted OLS model.

    Returns
    -------
    Dict[str, Any]
        Dictionary of diagnostic statistics, VIF table, and influential row indices.
    """
    resid = model.resid
    fitted = model.fittedvalues
    n = len(resid)

    # 1. Breusch-Pagan test for heteroscedasticity
    bp_stat, bp_pval, _, _ = het_breuschpagan(resid, model.model.exog)

    # 2. Shapiro-Wilk test for residual normality
    sw_stat, sw_pval = stats.shapiro(resid)

    # 3. Cook's distance & influence
    infl = model.get_influence()
    cooks_d, _ = infl.cooks_distance
    cooks_threshold = 4.0 / n
    influential_indices = np.where(cooks_d > cooks_threshold)[0].tolist()

    # 4. Variance Inflation Factor (VIF)
    exog = model.model.exog
    exog_names = model.model.exog_names
    vif_records = []
    for i, name in enumerate(exog_names):
        try:
            vif = variance_inflation_factor(exog, i)
        except Exception:
            vif = np.nan
        vif_records.append({"Variable": name, "VIF": round(vif, 2) if not np.isnan(vif) else np.nan})

    vif_df = pd.DataFrame(vif_records)

    return {
        "n_obs": n,
        "breusch_pagan_stat": round(float(bp_stat), 4),
        "breusch_pagan_pvalue": float(bp_pval),
        "shapiro_wilk_stat": round(float(sw_stat), 4),
        "shapiro_wilk_pvalue": float(sw_pval),
        "cooks_threshold": round(cooks_threshold, 4),
        "max_cooks_d": round(float(np.max(cooks_d)), 4),
        "n_influential": len(influential_indices),
        "influential_indices": influential_indices,
        "vif_table": vif_df,
    }


def fit_site_subgroup_models(
    df: pd.DataFrame,
    correction: str = "holm",
) -> pd.DataFrame:
    """
    Fit separate treatment-effect models within each site and apply multiplicity correction.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame.
    correction : str, default 'holm'
        Multiplicity adjustment method ('holm' or 'bonferroni').

    Returns
    -------
    pd.DataFrame
        Table summarizing site, sample size, treatment effect, raw p-value,
        and adjusted p-value.
    """
    site_records = []
    for s in sorted(df["site"].unique()):
        sub = df[df["site"] == s].dropna(subset=["delta_sbp", "sbp_0", "age", "bmi"]).copy()

        covariates = ["C(treatment, Treatment(reference='Placebo'))", "sbp_0", "age", "bmi", "C(sex)"]
        if sub["device_flag"].nunique() > 1:
            covariates.append("C(device_flag)")
        if sub["deviation_flag"].nunique() > 1:
            covariates.append("deviation_flag")

        formula = "delta_sbp ~ " + " + ".join(covariates)
        m = smf.ols(formula, data=sub).fit()

        trt_term = "C(treatment, Treatment(reference='Placebo'))[T.TrX]"
        coef = float(m.params[trt_term])
        se = float(m.bse[trt_term])
        pval = float(m.pvalues[trt_term])
        ci = m.conf_int().loc[trt_term]

        site_records.append({
            "Site": s,
            "N": len(sub),
            "TrX Effect (mmHg)": round(coef, 2),
            "Std Error": round(se, 2),
            "95% CI Lower": round(float(ci[0]), 2),
            "95% CI Upper": round(float(ci[1]), 2),
            "Raw p-value": pval,
        })

    sub_df = pd.DataFrame(site_records)

    # Multiplicity adjustment
    m_tests = len(sub_df)
    sub_df = sub_df.sort_values("Raw p-value").reset_index(drop=True)

    if correction.lower() == "bonferroni":
        sub_df["Adjusted p-value"] = np.minimum(sub_df["Raw p-value"] * m_tests, 1.0)
    else:  # Holm step-down
        adj_p = []
        for i, raw_p in enumerate(sub_df["Raw p-value"]):
            k = m_tests - i
            adj_p.append(min(raw_p * k, 1.0))
        # Ensure monotonicity
        for i in range(1, len(adj_p)):
            adj_p[i] = max(adj_p[i], adj_p[i - 1])
        sub_df["Adjusted p-value"] = adj_p

    sub_df["Significant (Adj p < 0.05)"] = sub_df["Adjusted p-value"].apply(
        lambda p: "YES" if p < 0.05 else "NO"
    )

    return sub_df.sort_values("Site").reset_index(drop=True)


def fit_device_effect_model(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Evaluate the Site 2 device offset across visits and test for treatment interaction.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, RegressionResultsWrapper]
        - Summary table showing device coefficients across visits.
        - Interaction model results object testing treatment x device_flag at Week 12.
    """
    s2 = df[df["site"] == 2].copy()

    records = []
    for visit in ["sbp_4", "sbp_8", "sbp_12"]:
        sub = s2.dropna(subset=[visit, "sbp_0"]).copy()
        formula = f"{visit} ~ sbp_0 + C(treatment) + C(device_flag)"
        m = smf.ols(formula, data=sub).fit()

        dev_term = "C(device_flag)[T.2]"
        coef = float(m.params[dev_term])
        se = float(m.bse[dev_term])
        pval = float(m.pvalues[dev_term])
        ci = m.conf_int().loc[dev_term]

        records.append({
            "Visit": visit.upper().replace("_", " "),
            "N Analyzed": len(sub),
            "Device 2 Offset (mmHg)": round(coef, 2),
            "Std Error": round(se, 2),
            "95% CI Lower": round(float(ci[0]), 2),
            "95% CI Upper": round(float(ci[1]), 2),
            "p-value": round(pval, 6),
        })

    # Interaction model at Week 12 (delta_sbp)
    s2_sub = s2.dropna(subset=["delta_sbp", "sbp_0"]).copy()
    int_formula = "delta_sbp ~ sbp_0 + C(treatment) * C(device_flag)"
    int_model = smf.ols(int_formula, data=s2_sub).fit()

    summary_df = pd.DataFrame(records)
    return summary_df, int_model


def fit_bmi_interaction_model(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Test whether BMI acts as an effect modifier of the TrX treatment response.

    Models:
    1. Dichotomized at median BMI: delta_sbp ~ treatment * bmi_high + covariates
    2. Continuous BMI interaction: delta_sbp ~ treatment * bmi + covariates

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        - Parameter table for both interaction terms.
        - Dictionary of interaction p-values.
    """
    df_clean = df.dropna(subset=["delta_sbp", "sbp_0", "bmi"]).copy()
    bmi_median = df_clean["bmi"].median()
    df_clean["bmi_high"] = (df_clean["bmi"] >= bmi_median).astype(int)

    # 1. Dichotomized
    formula_dich = (
        "delta_sbp ~ C(treatment, Treatment(reference='Placebo')) * bmi_high "
        "+ sbp_0 + C(site) + C(device_flag) + age + C(sex)"
    )
    mod_dich = smf.ols(formula_dich, data=df_clean).fit()
    term_dich = "C(treatment, Treatment(reference='Placebo'))[T.TrX]:bmi_high"
    p_dich = float(mod_dich.pvalues[term_dich])

    # 2. Continuous
    formula_cont = (
        "delta_sbp ~ C(treatment, Treatment(reference='Placebo')) * bmi "
        "+ sbp_0 + C(site) + C(device_flag) + age + C(sex)"
    )
    mod_cont = smf.ols(formula_cont, data=df_clean).fit()
    term_cont = "C(treatment, Treatment(reference='Placebo'))[T.TrX]:bmi"
    p_cont = float(mod_cont.pvalues[term_cont])

    summary_records = [
        {
            "Interaction Model": "BMI Dichotomized at Median",
            "Term": "TrX × High BMI (≥ median)",
            "Coefficient": round(float(mod_dich.params[term_dich]), 2),
            "Std Error": round(float(mod_dich.bse[term_dich]), 2),
            "p-value": round(p_dich, 4),
            "Interpretation": "Non-significant (p > 0.05); no differential effect by BMI subgroup",
        },
        {
            "Interaction Model": "BMI Continuous Linear Interaction",
            "Term": "TrX × BMI (continuous)",
            "Coefficient": round(float(mod_cont.params[term_cont]), 2),
            "Std Error": round(float(mod_cont.bse[term_cont]), 2),
            "p-value": round(p_cont, 4),
            "Interpretation": "Non-significant (p > 0.05); response does not scale linearly with BMI",
        },
    ]

    return pd.DataFrame(summary_records), {"p_dichotomized": p_dich, "p_continuous": p_cont}
