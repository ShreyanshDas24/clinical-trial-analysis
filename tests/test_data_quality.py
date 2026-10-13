"""
test_data_quality.py
====================
Unit tests for data validation, schema integrity, baseline comparability,
missingness auditing, and outlier detection.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_quality import (
    load_and_validate_trial_data,
    compute_baseline_balance,
    summarize_missingness,
    detect_iqr_outliers,
)


@pytest.fixture
def raw_data_path():
    path = Path(__file__).resolve().parent.parent / "trialx_data.csv"
    assert path.is_file(), f"trialx_data.csv not found at {path}"
    return path


@pytest.fixture
def trial_df(raw_data_path):
    return load_and_validate_trial_data(raw_data_path)


def test_load_and_validate_dimensions(trial_df):
    """Assert dataset has exactly 180 patients and all expected columns."""
    assert len(trial_df) == 180
    assert trial_df["patient_id"].nunique() == 180
    assert "delta_sbp" in trial_df.columns
    assert set(trial_df["site"].unique()) == {1, 2, 3}
    assert set(trial_df["treatment"].unique()) == {"Placebo", "TrX"}


def test_delta_sbp_direction(trial_df):
    """Verify delta_sbp = sbp_0 - sbp_12 (positive means SBP reduction)."""
    valid = trial_df.dropna(subset=["sbp_0", "sbp_12"])
    expected = valid["sbp_0"] - valid["sbp_12"]
    pd.testing.assert_series_equal(valid["delta_sbp"], expected, check_names=False)


def test_baseline_balance_statistics(trial_df):
    """Verify that baseline continuous variables show no significant difference between arms."""
    table, p_values = compute_baseline_balance(trial_df)

    assert len(table) == 4
    # All baseline comparability tests must be non-significant (p > 0.05) confirming randomization
    assert p_values["age_ttest_p"] > 0.05
    assert p_values["bmi_ttest_p"] > 0.05
    assert p_values["sbp_0_ttest_p"] > 0.05
    assert p_values["sex_chi2_p"] > 0.05
    assert p_values["site_baseline_sbp_kw_p"] > 0.05


def test_missingness_structure(trial_df):
    """Verify exact counts of missing values by site and monotone dropout pattern."""
    tables = summarize_missingness(trial_df)
    by_site = tables["by_site"]

    # Site 1 and 2 low missingness
    assert by_site.loc[1, "sbp_12"] <= 10.0
    assert by_site.loc[2, "sbp_12"] <= 10.0

    # Site 3 catastrophic missingness (76.67%)
    assert pytest.approx(by_site.loc[3, "sbp_12"], 0.1) == 76.67

    # Monotone dropout: every patient with missing sbp_4 is also missing sbp_12
    missing_w4 = trial_df[trial_df["sbp_4"].isna()]
    assert missing_w4["sbp_12"].isna().all()


def test_outlier_detection_iqr(trial_df):
    """Verify IQR outlier detection flags exactly the 2 verified outliers at Week 8."""
    summary, flagged = detect_iqr_outliers(trial_df)

    assert len(flagged) == 2
    flagged_patients = set(flagged["patient_id"])
    assert flagged_patients == {"P054", "P105"}

    # No outliers at baseline or Week 12
    assert summary.loc[summary["Variable"] == "sbp_0", "Outlier Count"].iloc[0] == 0
    assert summary.loc[summary["Variable"] == "sbp_12", "Outlier Count"].iloc[0] == 0


def test_validation_rejects_corrupted_data(tmp_path):
    """Verify validator raises ValueError on out-of-bounds blood pressures or duplicate IDs."""
    bad_df = pd.DataFrame({
        "patient_id": ["P001", "P001"],  # Duplicate
        "site": [1, 1],
        "treatment": ["Placebo", "TrX"],
        "age": [50.0, 50.0],
        "sex": ["M", "F"],
        "bmi": [25.0, 25.0],
        "sbp_0": [140.0, 350.0],  # Out of clinical bounds
        "sbp_4": [140.0, 140.0],
        "sbp_8": [140.0, 140.0],
        "sbp_12": [140.0, 140.0],
        "dropout_week": [np.nan, np.nan],
        "device_flag": [1, 1],
        "deviation_flag": [0, 0],
    })
    bad_csv = tmp_path / "bad.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError):
        load_and_validate_trial_data(bad_csv)
