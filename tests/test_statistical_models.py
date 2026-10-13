"""
test_statistical_models.py
===========================
Unit tests for primary ANCOVA, model isomorphism, regression diagnostics,
multiplicity adjustments, device effects, and interaction tests.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.data_quality import load_and_validate_trial_data
from src.missing_data import impute_median_unconditional
from src.statistical_analysis import (
    fit_primary_ancova,
    compute_model_diagnostics,
    fit_site_subgroup_models,
    fit_device_effect_model,
    fit_bmi_interaction_model,
)


@pytest.fixture
def trial_df():
    path = Path(__file__).resolve().parent.parent / "trialx_data.csv"
    return load_and_validate_trial_data(path)


@pytest.fixture
def imputed_df(trial_df):
    return impute_median_unconditional(trial_df)


def test_primary_ancova_followup_and_changescore_isomorphism(imputed_df):
    """
    Verify mathematical equivalence between baseline-adjusted follow-up ANCOVA
    and change-score ANCOVA on Sites 1-2.
    """
    m_fu, _ = fit_primary_ancova(imputed_df, endpoint="follow_up", sites=[1, 2])
    m_cs, _ = fit_primary_ancova(imputed_df, endpoint="change_score", sites=[1, 2])

    assert m_fu["n_analyzed"] == 119
    assert m_cs["n_analyzed"] == 119

    # Magnitudes must be exactly equal
    assert pytest.approx(abs(m_fu["treatment_coef"]), 0.001) == 7.812
    assert pytest.approx(abs(m_cs["treatment_coef"]), 0.001) == 7.812

    # Standard errors must be identical
    assert pytest.approx(m_fu["std_error"], 0.0001) == m_cs["std_error"]
    assert pytest.approx(m_fu["std_error"], 0.001) == 1.391

    # T-statistics must be opposite in sign but identical in magnitude
    assert pytest.approx(m_fu["t_statistic"], 0.001) == -5.617
    assert pytest.approx(m_cs["t_statistic"], 0.001) == 5.617

    # P-values must match exactly
    assert pytest.approx(m_fu["p_value"], rel=1e-5) == m_cs["p_value"]
    assert m_fu["p_value"] < 1e-6


def test_singularity_eliminated_in_primary_model(imputed_df):
    """Verify that removing deviation_flag eliminates design matrix near-singularity."""
    _, model = fit_primary_ancova(imputed_df, endpoint="change_score", sites=[1, 2])
    cond_no = model.condition_number

    # Condition number must be well below singular threshold (< 5000 vs 10^18 with deviation_flag)
    assert cond_no < 5000.0
    # No NaN or near-zero eigenvalues
    assert not np.isnan(cond_no)


def test_regression_diagnostics(imputed_df):
    """Verify OLS diagnostic test values (Breusch-Pagan, Shapiro-Wilk, VIF)."""
    _, model = fit_primary_ancova(imputed_df, endpoint="change_score", sites=[1, 2])
    diag = compute_model_diagnostics(model)

    # Breusch-Pagan test confirms homoscedasticity (p > 0.05)
    assert diag["breusch_pagan_pvalue"] > 0.05

    # Shapiro-Wilk detects modest skew in residuals (W ~ 0.97, p ~ 0.01)
    assert diag["shapiro_wilk_stat"] > 0.95

    # Multicollinearity check: all VIFs (excluding Intercept) must be < 2.5
    vif_df = diag["vif_table"]
    non_intercept_vif = vif_df[vif_df["Variable"] != "Intercept"]["VIF"].dropna()
    assert (non_intercept_vif < 2.5).all()

    # Influence check: exactly 4 observations exceed 4/N
    assert diag["n_influential"] == 4


def test_complete_case_treatment_effect(trial_df):
    """Verify complete case estimate on Sites 1-2 (N=107)."""
    df_cc = trial_df.dropna(subset=["sbp_12", "sbp_0", "age", "bmi"]).copy()
    m_cc, _ = fit_primary_ancova(df_cc, endpoint="change_score", sites=[1, 2])

    assert m_cc["n_analyzed"] == 107
    assert pytest.approx(m_cc["treatment_coef"], 0.01) == 9.65
    assert pytest.approx(m_cc["std_error"], 0.01) == 1.28
    assert m_cc["p_value"] < 1e-10


def test_site_subgroups_and_multiplicity(imputed_df):
    """Verify site-level subgroup estimates and Holm-Bonferroni correction."""
    site_table = fit_site_subgroup_models(imputed_df, correction="holm")

    assert len(site_table) == 3

    s1 = site_table[site_table["Site"] == 1].iloc[0]
    s2 = site_table[site_table["Site"] == 2].iloc[0]
    s3 = site_table[site_table["Site"] == 3].iloc[0]

    assert s1["Significant (Adj p < 0.05)"] == "YES"
    assert s2["Significant (Adj p < 0.05)"] == "YES"
    assert s3["Significant (Adj p < 0.05)"] == "NO"

    assert pytest.approx(s1["TrX Effect (mmHg)"], 0.1) == 9.29
    assert pytest.approx(s2["TrX Effect (mmHg)"], 0.1) == 6.81


def test_site2_device_offset_and_interaction(trial_df):
    """Verify Site 2 device offset across visits and lack of treatment interaction."""
    dev_table, int_model = fit_device_effect_model(trial_df)

    assert len(dev_table) == 3
    # Significant device offset at Week 4 and Week 8
    w4_row = dev_table[dev_table["Visit"] == "SBP 4"].iloc[0]
    w8_row = dev_table[dev_table["Visit"] == "SBP 8"].iloc[0]
    assert w4_row["p-value"] < 0.001
    assert w8_row["p-value"] < 0.001
    assert pytest.approx(w4_row["Device 2 Offset (mmHg)"], 0.2) == 5.52

    # Treatment x Device interaction at Week 12 must be non-significant
    int_p = int_model.pvalues.get("C(treatment)[T.TrX]:C(device_flag)[T.2]", 1.0)
    assert int_p > 0.50


def test_bmi_interactions_are_nonsignificant(imputed_df):
    """Verify that BMI does not act as an effect modifier of TrX."""
    _, p_dict = fit_bmi_interaction_model(imputed_df)

    assert p_dict["p_dichotomized"] > 0.40
    assert p_dict["p_continuous"] > 0.20
