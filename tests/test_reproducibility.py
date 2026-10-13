"""
test_reproducibility.py
=======================
Tests for deterministic reproducibility, tipping-point sensitivity properties,
and multiple imputation stability.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.data_quality import load_and_validate_trial_data
from src.missing_data import (
    impute_median_unconditional,
    impute_multiple_conditioned,
    run_tipping_point_sensitivity,
)
from src.predictive_models import evaluate_predictive_models


@pytest.fixture
def trial_df():
    path = Path(__file__).resolve().parent.parent / "trialx_data.csv"
    return load_and_validate_trial_data(path)


def test_predictive_modeling_seed_reproducibility(trial_df):
    """Assert that running predictive models with random_state=42 yields deterministic results."""
    df_imp = impute_median_unconditional(trial_df)

    res1, _, _ = evaluate_predictive_models(df_imp, random_state=42)
    res2, _, _ = evaluate_predictive_models(df_imp, random_state=42)

    pd.testing.assert_frame_equal(res1, res2)


def test_tipping_point_monotonicity_and_robustness(trial_df):
    """
    Verify that increasing MNAR delta penalty on TrX dropouts monotonically reduces
    the estimated effect, and confirm that the effect is robust past 6 mmHg.
    """
    penalties = np.arange(0.0, 16.0, 2.0)
    tip_df = run_tipping_point_sensitivity(trial_df, delta_penalties=penalties, sites=[1, 2])

    assert len(tip_df) == len(penalties)

    # Monotonic decrease in estimated effect
    estimates = tip_df["TrX Effect Estimate (mmHg)"].tolist()
    assert all(estimates[i] >= estimates[i + 1] for i in range(len(estimates) - 1))

    # Treatment effect remains statistically significant at least up to delta = 8 mmHg
    row_delta_8 = tip_df[tip_df["Delta Penalty (mmHg)"] == 8.0].iloc[0]
    assert row_delta_8["Significant (p < 0.05)"] == "YES"
    assert row_delta_8["TrX Effect Estimate (mmHg)"] > 6.0


def test_multiple_imputation_preserves_sample_size_and_arms(trial_df):
    """Verify that multiple imputation produces datasets with no missing values and preserved balance."""
    imputed_list = impute_multiple_conditioned(trial_df, n_imputations=3, random_state=42)

    assert len(imputed_list) == 3
    for imp_df in imputed_list:
        assert len(imp_df) == 180
        assert imp_df["sbp_12"].isna().sum() == 0
        assert imp_df["delta_sbp"].isna().sum() == 0
        # Balance preserved
        assert (imp_df["treatment"] == "Placebo").sum() == 90
        assert (imp_df["treatment"] == "TrX").sum() == 90
