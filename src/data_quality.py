"""
data_quality.py
===============
Data ingestion, validation, baseline balance assessment, missingness auditing,
and outlier detection for the multi-site clinical trial dataset.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_COLUMNS = [
    "patient_id",
    "site",
    "treatment",
    "age",
    "sex",
    "bmi",
    "sbp_0",
    "sbp_4",
    "sbp_8",
    "sbp_12",
    "dropout_week",
    "device_flag",
    "deviation_flag",
]

CLINICAL_BOUNDS = {
    "age": (18.0, 100.0),
    "bmi": (10.0, 60.0),
    "sbp_0": (70.0, 250.0),
    "sbp_4": (70.0, 250.0),
    "sbp_8": (70.0, 250.0),
    "sbp_12": (70.0, 250.0),
}


def load_and_validate_trial_data(
    filepath: Union[str, Path]
) -> pd.DataFrame:
    """
    Load and perform strict clinical integrity validation on trialx_data.csv.

    Parameters
    ----------
    filepath : str or Path
        Path to the trialx_data.csv dataset.

    Returns
    -------
    pd.DataFrame
        Validated clinical trial DataFrame with derived `delta_sbp` column.

    Raises
    ------
    FileNotFoundError
        If dataset file does not exist.
    ValueError
        If schema is invalid, IDs are duplicated, or values violate clinical bounds.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Clinical trial dataset not found at: {path}")

    df = pd.read_csv(path)

    # 1. Schema check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")

    # 2. Key uniqueness check
    if df["patient_id"].duplicated().any():
        dups = df["patient_id"][df["patient_id"].duplicated()].tolist()
        raise ValueError(f"Duplicate patient_id found in dataset: {dups}")

    # 3. Categorical value validity
    if not set(df["treatment"].dropna().unique()).issubset({"Placebo", "TrX"}):
        raise ValueError(f"Unexpected treatment labels: {df['treatment'].unique()}")

    if not set(df["site"].dropna().unique()).issubset({1, 2, 3}):
        raise ValueError(f"Unexpected site identifiers: {df['site'].unique()}")

    if not set(df["sex"].dropna().unique()).issubset({"M", "F"}):
        raise ValueError(f"Unexpected sex labels: {df['sex'].unique()}")

    if not set(df["device_flag"].dropna().unique()).issubset({1, 2}):
        raise ValueError(f"Unexpected device_flag values: {df['device_flag'].unique()}")

    if not set(df["deviation_flag"].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected deviation_flag values: {df['deviation_flag'].unique()}")

    # 4. Plausible clinical bounds check
    for col, (lower, upper) in CLINICAL_BOUNDS.items():
        vals = df[col].dropna()
        out_of_bounds = vals[(vals < lower) | (vals > upper)]
        if not out_of_bounds.empty:
            raise ValueError(
                f"Variable '{col}' has values outside plausible range [{lower}, {upper}]: "
                f"{out_of_bounds.to_dict()}"
            )

    # 5. Endpoint construction
    # delta_sbp = sbp_0 - sbp_12 represents SBP reduction (positive = improvement)
    df = df.copy()
    df["delta_sbp"] = df["sbp_0"] - df["sbp_12"]

    return df


def compute_baseline_balance(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Assess comparability between treatment arms (Placebo vs TrX) at baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame containing baseline covariates.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        - Table summarizing means, standard deviations, and p-values by arm.
        - Dictionary of test p-values (t-test, Mann-Whitney, Chi-square, Kruskal-Wallis).
    """
    placebo = df[df["treatment"] == "Placebo"]
    trx = df[df["treatment"] == "TrX"]

    summary_rows = []
    p_values = {}

    continuous_vars = [
        ("age", "Age (years)"),
        ("bmi", "BMI (kg/m^2)"),
        ("sbp_0", "Baseline SBP (mmHg)"),
    ]

    for col, label in continuous_vars:
        p_vals = placebo[col].dropna()
        t_vals = trx[col].dropna()

        t_res = stats.ttest_ind(p_vals, t_vals, equal_var=False)
        mw_res = stats.mannwhitneyu(p_vals, t_vals, alternative="two-sided")

        p_values[f"{col}_ttest_p"] = float(t_res.pvalue)
        p_values[f"{col}_mannwhitney_p"] = float(mw_res.pvalue)

        summary_rows.append({
            "Variable": label,
            "Placebo N": len(p_vals),
            "Placebo Mean (SD)": f"{p_vals.mean():.2f} ({p_vals.std():.2f})",
            "TrX N": len(t_vals),
            "TrX Mean (SD)": f"{t_vals.mean():.2f} ({t_vals.std():.2f})",
            "Welch t-test p": f"{t_res.pvalue:.4f}",
            "Mann-Whitney p": f"{mw_res.pvalue:.4f}",
        })

    # Categorical: Sex balance
    contingency_sex = pd.crosstab(df["treatment"], df["sex"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency_sex)
    p_values["sex_chi2_p"] = float(chi2_p)

    summary_rows.append({
        "Variable": "Sex (% Male)",
        "Placebo N": len(placebo["sex"].dropna()),
        "Placebo Mean (SD)": f"{(placebo['sex'] == 'M').mean() * 100:.1f}%",
        "TrX N": len(trx["sex"].dropna()),
        "TrX Mean (SD)": f"{(trx['sex'] == 'M').mean() * 100:.1f}%",
        "Welch t-test p": "—",
        "Mann-Whitney p": f"Chi² p = {chi2_p:.4f}",
    })

    # Site balance for baseline SBP (Kruskal-Wallis across sites)
    kw_res = stats.kruskal(*[group["sbp_0"].dropna().values for _, group in df.groupby("site")])
    p_values["site_baseline_sbp_kw_p"] = float(kw_res.pvalue)

    table = pd.DataFrame(summary_rows)
    return table, p_values


def summarize_missingness(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Produce comprehensive missingness audit tables overall, by site, and by arm.

    Parameters
    ----------
    df : pd.DataFrame
        Trial dataset.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys 'overall', 'by_site', 'by_treatment', and 'by_site_and_treatment'.
    """
    total_n = len(df)
    focus_vars = ["age", "bmi", "sbp_0", "sbp_4", "sbp_8", "sbp_12", "dropout_week"]

    # Overall
    overall = pd.DataFrame({
        "Missing N": df[focus_vars].isna().sum(),
        "Missing %": (df[focus_vars].isna().mean() * 100).round(2),
    })

    # By Site
    by_site = (
        df.groupby("site")[focus_vars]
        .apply(lambda g: (g.isna().mean() * 100).round(2), include_groups=False)
    )

    # By Treatment Arm
    by_treatment = (
        df.groupby("treatment")[focus_vars]
        .apply(lambda g: (g.isna().mean() * 100).round(2), include_groups=False)
    )

    # SBP_12 missingness by site and treatment
    by_site_trt = (
        df.groupby(["site", "treatment"])["sbp_12"]
        .agg(
            total_n="count",
            missing_n=lambda x: x.isna().sum(),
            missing_pct=lambda x: round(x.isna().mean() * 100, 2),
        )
    )

    return {
        "overall": overall,
        "by_site": by_site,
        "by_treatment": by_treatment,
        "by_site_and_treatment": by_site_trt,
    }


def detect_iqr_outliers(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    k: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect statistical outliers using the standard Tukey IQR rule.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame.
    cols : List[str], optional
        Columns to test. Defaults to continuous numeric variables.
    k : float, default 1.5
        IQR multiplier threshold.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Summary table with quartiles, IQR, bounds, and outlier count per variable.
        - Detail table of flagged outlier records with patient ID and values.
    """
    if cols is None:
        cols = ["age", "bmi", "sbp_0", "sbp_4", "sbp_8", "sbp_12"]

    summary_records = []
    flagged_records = []

    for col in cols:
        vals = df[col].dropna()
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
        iqr = q75 - q25
        lower = q25 - k * iqr
        upper = q75 + k * iqr

        outliers = df.loc[vals[(vals < lower) | (vals > upper)].index]
        n_outliers = len(outliers)

        summary_records.append({
            "Variable": col,
            "N Observed": len(vals),
            "Q25": round(q25, 2),
            "Q75": round(q75, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
            "Outlier Count": n_outliers,
        })

        for _, row in outliers.iterrows():
            flagged_records.append({
                "patient_id": row["patient_id"],
                "site": row["site"],
                "treatment": row["treatment"],
                "variable": col,
                "value": row[col],
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
            })

    summary_df = pd.DataFrame(summary_records)
    flagged_df = pd.DataFrame(flagged_records)
    return summary_df, flagged_df
