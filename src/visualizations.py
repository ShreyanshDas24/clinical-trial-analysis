"""
visualizations.py
=================
Publication-grade statistical visualizations for the clinical trial analysis.
Includes missingness heatmaps, distribution plots, 4-panel regression diagnostics,
forest plots for sensitivity analyses, and predictive modeling benchmarks.
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# Set publication style
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})
sns.set_theme(style="whitegrid", palette="muted")


def plot_missingness_overview(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate a 2-panel figure showing missingness patterns by site and visit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Panel 1: Missingness % heatmap across visits by site
    visit_cols = ["sbp_0", "sbp_4", "sbp_8", "sbp_12"]
    site_miss = (
        df.groupby("site")[visit_cols]
        .apply(lambda g: g.isna().mean() * 100, include_groups=False)
    )
    site_miss.index = [f"Site {s}" for s in site_miss.index]
    site_miss.columns = ["Week 0", "Week 4", "Week 8", "Week 12"]

    sns.heatmap(
        site_miss,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Missing Rate (%)"},
        vmin=0,
        vmax=100,
        ax=axes[0],
    )
    axes[0].set_title("Missing SBP Measurements by Site and Visit", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Clinical Center")

    # Panel 2: Week 12 Dropout Rate by Treatment Arm and Site
    df_temp = df.copy()
    df_temp["w12_missing"] = df_temp["sbp_12"].isna().astype(int) * 100

    sns.barplot(
        data=df_temp,
        x="site",
        y="w12_missing",
        hue="treatment",
        palette=["#4575b4", "#d73027"],
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_title("Week 12 Attrition Rate by Site & Arm", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Clinical Site")
    axes[1].set_ylabel("Attrition Rate at Week 12 (%)")
    axes[1].set_xticks(range(len(sorted(df["site"].unique()))))
    axes[1].set_xticklabels([f"Site {s}" for s in sorted(df["site"].unique())])
    axes[1].legend(title="Treatment Arm")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_baseline_and_followup_distributions(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot baseline vs Week 12 SBP distributions split by treatment arm and site.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Baseline SBP
    sns.boxplot(
        data=df,
        x="site",
        y="sbp_0",
        hue="treatment",
        palette=["#4575b4", "#d73027"],
        ax=axes[0],
        boxprops=dict(alpha=0.8),
    )
    axes[0].set_title("Baseline SBP Distribution (Week 0)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Clinical Site")
    axes[0].set_ylabel("Systolic Blood Pressure (mmHg)")
    axes[0].set_xticks(range(len(sorted(df["site"].unique()))))
    axes[0].set_xticklabels([f"Site {s}" for s in sorted(df["site"].unique())])
    axes[0].legend(title="Arm")

    # Panel 2: Blood Pressure Reduction (delta_sbp)
    df_clean = df.dropna(subset=["delta_sbp"]).copy()
    sns.boxplot(
        data=df_clean,
        x="site",
        y="delta_sbp",
        hue="treatment",
        palette=["#4575b4", "#d73027"],
        ax=axes[1],
        boxprops=dict(alpha=0.8),
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Blood Pressure Reduction (Delta SBP = SBP0 - SBP12)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Clinical Site")
    axes[1].set_ylabel("SBP Reduction from Baseline (mmHg)")
    axes[1].set_xticks(range(len(sorted(df["site"].unique()))))
    axes[1].set_xticklabels([f"Site {s}" for s in sorted(df["site"].unique())])
    axes[1].legend(title="Arm")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_forest_sensitivity(
    sensitivity_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate a publication-grade Forest Plot comparing treatment effect estimates
    and 95% confidence intervals across all sensitivity models.
    """
    df_plot = sensitivity_df.copy().iloc[::-1].reset_index(drop=True)
    y_positions = np.arange(len(df_plot))

    fig, ax = plt.subplots(figsize=(11, 6))

    # Reference line at 0 (null effect)
    ax.axvline(0, color="#d95f02", linestyle="--", linewidth=1.2, label="Null Effect (0 mmHg)")

    # Plot confidence interval bars and points
    for i, row in df_plot.iterrows():
        est = row["Estimate (mmHg)"]
        lower = row["95% CI Lower"]
        upper = row["95% CI Upper"]
        color = "#1b7837" if row["p-value"] < 0.05 else "#7570b3"

        ax.plot([lower, upper], [i, i], color=color, linewidth=2.2, solid_capstyle="round")
        ax.plot(est, i, marker="s", markersize=8, color=color)

        # Value annotation
        p_str = f"p={row['p-value']:.2e}" if row['p-value'] < 0.001 else f"p={row['p-value']:.4f}"
        label = f"{est:.2f} [{lower:.2f}, {upper:.2f}] (N={row['N Analyzed']}, {p_str})"
        ax.text(max(upper + 0.4, 13.0), i, label, va="center", fontsize=9, color="#222222")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_plot["Model / Scenario"], fontsize=10, fontweight="medium")
    ax.set_xlabel("Adjusted Treatment Effect: TrX vs Placebo Reduction (mmHg)", fontsize=11, fontweight="bold")
    ax.set_title("Sensitivity Analysis Forest Plot — TrX Efficacy Across Analytical Scenarios", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(-2, 19)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_model_diagnostics(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate a 4-panel regression diagnostic panel:
    1. Residuals vs Fitted
    2. Normal Q-Q Plot
    3. Scale-Location (Spread vs Level)
    4. Cook's Distance
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    resid = model.resid
    fitted = model.fittedvalues
    student_resid = model.get_influence().resid_studentized_internal
    cooks_d, _ = model.get_influence().cooks_distance
    n = len(resid)

    # 1. Residuals vs Fitted
    sns.scatterplot(x=fitted, y=resid, alpha=0.7, color="#2b83ba", ax=axes[0, 0])
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    sns.regplot(x=fitted, y=resid, scatter=False, lowess=True, ax=axes[0, 0], line_kws={"color": "darkred", "lw": 1.5})
    axes[0, 0].set_title("Residuals vs Fitted (Linearity & Homoscedasticity)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")

    # 2. Q-Q Plot
    sm.qqplot(student_resid, line="45", ax=axes[0, 1], color="#2b83ba")
    axes[0, 1].set_title("Normal Q–Q (Residual Normality)", fontsize=11, fontweight="bold")

    # 3. Scale-Location
    sqrt_abs_resid = np.sqrt(np.abs(student_resid))
    sns.scatterplot(x=fitted, y=sqrt_abs_resid, alpha=0.7, color="#2b83ba", ax=axes[1, 0])
    sns.regplot(x=fitted, y=sqrt_abs_resid, scatter=False, lowess=True, ax=axes[1, 0], line_kws={"color": "darkred", "lw": 1.5})
    axes[1, 0].set_title("Scale-Location (Constant Variance Check)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("√|Studentized Residuals|")

    # 4. Cook's Distance
    threshold = 4.0 / n
    axes[1, 1].stem(np.arange(n), cooks_d, markerfmt=" ", linefmt="C0-", basefmt="k-")
    axes[1, 1].axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold 4/N ({threshold:.3f})")
    axes[1, 1].set_title("Cook's Distance (Influential Observations)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Observation Index")
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].legend()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_site2_device_effect(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the Site 2 calibration offset between Device 1 and Device 2 across visits.
    """
    s2 = df[df["site"] == 2].copy()
    melted = s2.melt(
        id_vars=["patient_id", "device_flag", "treatment"],
        value_vars=["sbp_0", "sbp_4", "sbp_8", "sbp_12"],
        var_name="visit",
        value_name="sbp",
    ).dropna()
    melted["visit"] = melted["visit"].map({
        "sbp_0": "Week 0 (Baseline)",
        "sbp_4": "Week 4",
        "sbp_8": "Week 8",
        "sbp_12": "Week 12",
    })
    melted["device_label"] = melted["device_flag"].map({1: "Device 1 (Standard)", 2: "Device 2 (Shifted)"})

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.pointplot(
        data=melted,
        x="visit",
        y="sbp",
        hue="device_label",
        palette=["#2b83ba", "#d7191c"],
        dodge=0.25,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        ax=ax,
    )
    ax.set_title("Site 2 Blood Pressure Measurement Shift by Device Cohort", fontsize=12, fontweight="bold")
    ax.set_xlabel("Trial Visit")
    ax.set_ylabel("Mean Systolic Blood Pressure (mmHg)")
    ax.legend(title="Measurement Device")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_predictive_model_comparison(
    holdout_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Bar plot comparing predictive model metrics against the Dummy Baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.barplot(
        data=holdout_df,
        x="Model",
        y="Test R²",
        hue="Model",
        palette="Blues_r",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Holdout Test R² Comparison", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("R² (Out-of-sample)")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(
        data=holdout_df,
        x="Model",
        y="Test MAE (mmHg)",
        hue="Model",
        palette="Greens_r",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Holdout Test MAE Comparison (mmHg)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Mean Absolute Error (mmHg)")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
