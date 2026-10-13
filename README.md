# Clinical Trial Analysis: Contaminated Multi-Site Phase II RCT
### Biostatistical Efficacy Analysis, Data Quality Auditing, and Reproducible Sensitivity Modeling

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-16%20passed-success.svg)](tests/)
[![Regulatory Alignment](https://img.shields.io/badge/Guideline-ICH%20E9%20Aligned-darkgreen.svg)](docs/statistical_methodology.md)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> [!IMPORTANT]
> **Simulated Dataset Disclosure**: This repository analyzes a synthetic clinical trial dataset (`trialx_data.csv`) generated for statistical education, methodological auditing, and data science portfolio evaluation. All findings, model scores, confidence intervals, and clinical discussions represent methodological demonstrations on simulated data and must **never** be cited or interpreted as evidence of real-world pharmacotherapeutic efficacy.

---

## 1. Executive Summary & Clinical Research Problem

### The Clinical Research Question
In essential hypertension management, evaluating whether a novel therapeutic agent (**TrX**) achieves a clinically meaningful and statistically significant reduction in Systolic Blood Pressure (SBP in mmHg) compared to Placebo is the standard objective of a Phase II Randomized Controlled Trial (RCT).

$$\text{Primary Hypothesis } H_0: \mu_{\text{TrX}} - \mu_{\text{Placebo}} = 0 \quad \text{vs.} \quad H_1: \mu_{\text{TrX}} - \mu_{\text{Placebo}} < 0 \quad (\text{SBP reduction at Week 12})$$

### The Methodological Challenge: Trial Contamination
Standard trial analyses often assume uniform data quality and missing-at-random mechanisms. This trial exhibits realistic real-world trial contamination across three clinical sites ($N = 180$; 60 patients/site; 1:1 randomization):
1. **Severe Multi-Center Attrition**: Site 3 experienced catastrophic participant discontinuation (76.7% missing primary endpoint at Week 12) combined with 15 GCP protocol deviations.
2. **Nested Device Measurement Bias**: Halfway through enrollment at Site 2 (patients P091–P120), a secondary blood pressure device was introduced that systematically shifted SBP measurements upward by $+5.5$ to $+6.3$ mmHg.
3. **Imputation Hazards**: Standard single median imputation artificially attenuates treatment effect estimates toward the null when applied unconditionally across arms in high-missingness centers.

This repository provides an audit, an ICH E9 regulatory-aligned Analysis of Covariance (ANCOVA), a principled sensitivity battery, regression assumption diagnostics, and an isolated exploratory machine learning benchmark.

---

## 2. Key Statistical Findings (Reproduced from Ground Truth)

All reported results are deterministically reproduced by our automated test suite (`pytest tests/`) and master notebook (`main.ipynb`):

| Analysis Scenario | Population & Strategy | Sample Size ($N$) | TrX Effect Estimate | 95% Confidence Interval | $p$-value | Clinical Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Regulatory Model** | Sites 1–2 Baseline-Adjusted ANCOVA (Week 12 SBP) | **$N = 119$** | **$-7.81$ mmHg** | **$[-10.57, -5.06]$** | **$1.45 \times 10^{-7}$** | **Statistically significant SBP reduction** |
| **Change-Score ANCOVA** | Sites 1–2 Change Score ($\Delta\text{SBP} = \text{SBP}_0 - \text{SBP}_{12}$) | $N = 119$ | $+7.81$ mmHg | $[+5.06, +10.57]$ | $1.45 \times 10^{-7}$ | Mathematically isomorphic to primary model |
| **Complete-Case Analysis** | Sites 1–2 Complete Cases (No Imputation) | $N = 107$ | $+9.65$ mmHg | $[+7.10, +12.19]$ | $2.43 \times 10^{-11}$ | Unaffected by median shrinkage bias |
| **All-Sites Complete-Case** | All Sites Complete Cases | $N = 120$ | $+9.31$ mmHg | $[+6.94, +11.69]$ | $4.63 \times 10^{-12}$ | Robust effect including Site 3 completers |
| **All-Sites Imputed** | All Sites Median Imputed | $N = 177$ | $+5.93$ mmHg | $[+3.05, +8.81]$ | $6.83 \times 10^{-5}$ | Attenuated by 77% median imputation at Site 3 |
| **Robust Covariance (HC3)** | Sites 1–2 Heteroscedasticity-Consistent | $N = 119$ | $+7.81$ mmHg | $[+5.32, +10.30]$ | $2.39 \times 10^{-8}$ | Robust against minor residual non-normality |
| **Influence Sensitivity** | Sites 1–2 Excluding Cook's $D > 4/N$ ($n=4$) | $N = 115$ | $+7.91$ mmHg | $[+5.57, +10.25]$ | $1.42 \times 10^{-9}$ | Extreme stability against influential points |
| **Minimal ANCOVA** | Sites 1–2 (Treatment + Baseline SBP only) | $N = 119$ | $+7.71$ mmHg | $[+4.95, +10.47]$ | $1.50 \times 10^{-7}$ | Robust to covariate omission |

```
                       TrX Treatment Effect Estimate (mmHg)
Scenario                [95% Confidence Interval]
-------------------------------------------------------------------------
1. Primary ANCOVA (S1-2)    |-------[====*====]-------|  7.81 mmHg (p=1.5e-7)
2. Complete Case (S1-2)     |---------[====*====]-----|  9.65 mmHg (p=2.4e-11)
3. Complete Case (All)      |---------[====*====]-----|  9.31 mmHg (p=4.6e-12)
4. All Sites (Imputed)      |---[====*====]-----------|  5.93 mmHg (p=6.8e-5)
5. Robust HC3 SE (S1-2)     |-------[====*====]-------|  7.81 mmHg (p=2.4e-8)
6. Excl. Influential (S1-2) |-------[====*====]-------|  7.91 mmHg (p=1.4e-9)
-------------------------------------------------------------------------
                     0 mmHg (Null)   5 mmHg        10 mmHg
```

---

## 3. Methodological Breakthroughs & Audit Insights

### 3.1 Mathematical Equivalence: Change Score vs. Baseline-Adjusted ANCOVA
A common point of interview debate is whether to model post-treatment SBP ($\text{SBP}_{12}$) or the change score ($\Delta\text{SBP} = \text{SBP}_0 - \text{SBP}_{12}$).
Our audit provides the analytical proof that when baseline SBP ($\text{SBP}_0$) is included as a linear covariate, **both specifications are mathematically identical**:
$$\text{SBP}_{12} = \alpha_1 + \beta_{\text{TrX}} T + \gamma_1 \text{SBP}_0 + \epsilon_1 \iff \Delta\text{SBP} = -\alpha_1 - \beta_{\text{TrX}} T + (1 - \gamma_1) \text{SBP}_0 - \epsilon_1$$
- Point estimate magnitude: Exactly $7.81$ mmHg
- Standard error: Exactly $1.391$ mmHg
- $t$-statistic magnitude: Exactly $5.617$
- $p$-value: Exactly $1.45 \times 10^{-7}$
- Residual variance, AIC ($825.0$), and log-likelihood ($-404.51$) are identical.
- Model 1 (Follow-up SBP12) is preferred per ICH E9 guidelines as it directly models the post-treatment clinical state.

### 3.2 Elimination of the Design Matrix Singularity Artifact
In the original repository notebook, `deviation_flag` was included in the formula for Sites 1–2. However, protocol deviations occurred **exclusively at Site 3** (`deviation_flag == 0` for all 120 patients in Sites 1–2).
- Including an all-zero column created matrix rank-deficiency (eigenvalue $2.75 \times 10^{-31}$, condition number $3.33 \times 10^{18}$).
- Statsmodels produced a spurious numerical artifact: coefficient $2.258 \times 10^{-14}$ ($p=0.001$).
- **Solution**: Dropping the invariant term eliminated matrix singularity (condition number dropped to $2.09 \times 10^3$) while preserving all substantive model parameters.

### 3.3 Site 3 Exclusion & The Median Imputation Shrinkage Bias
Site-level median imputation imputed 76.7% ($n=46/60$) of Site 3 patients with the site median without conditioning on treatment arm.
- Imputing identical post-treatment blood pressures for both Placebo and TrX patients forced the treatment difference among those 46 patients to exactly zero.
- This artificially collapsed Site 3's estimated effect from **$+5.73$ mmHg** (in observed complete cases) to $+0.33$ mmHg ($p=0.86$), creating a false impression of site-level treatment failure.
- In complete cases ($N=120$), the treatment effect across all sites is $+9.31$ mmHg ($p = 4.63 \times 10^{-12}$).

### 3.4 Nested Device Bias at Site 2
At Site 2, Device 2 introduced an additive measurement bias of $+5.5$ to $+6.3$ mmHg across post-baseline visits.
- Because Device 2 was never utilized at Sites 1 or 3, a global device effect is unidentifiable from a Site 2 cohort effect.
- However, within Site 2, randomization was balanced across devices (15 Placebo, 15 TrX).
- The treatment $\times$ device interaction test yielded $p = 0.698$, confirming that device bias is an additive calibration offset that does not bias the treatment contrast.

---

## 4. Supplementary Machine Learning vs. Causal Biostatistics

The repository benchmarks predictive machine learning algorithms strictly as an **exploratory predictive exercise**, maintaining clear boundaries from causal inference:

| Model | Holdout Test $R^2$ (70/30 Split) | Holdout Test MAE (mmHg) | 5-Fold CV $R^2$ (Mean $\pm$ SD) | 5-Fold CV MAE (mmHg) | Beats Mean Baseline? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Mean Baseline)** | $-0.021$ | $8.41$ | $-0.035 \pm 0.032$ | $8.58 \pm 0.77$ | Baseline |
| **Ridge Regression** | $+0.239$ | $7.85$ | $+0.177 \pm 0.101$ | $7.91 \pm 1.34$ | **Yes** ($+0.56$ mmHg) |
| **Random Forest** | $+0.217$ | $7.83$ | $+0.270 \pm 0.166$ | $7.22 \pm 0.90$ | **Yes** ($+0.58$ mmHg) |
| **Gradient Boosting** | $+0.147$ | $7.78$ | $+0.272 \pm 0.148$ | $7.03 \pm 0.90$ | **Yes** ($+0.63$ mmHg) |

> **Critical Data Science Demarcation**:
> 1. ML models achieve test MAE $\approx 7.8$ mmHg, which is roughly equal to the entire treatment effect size ($7.81$ mmHg).
> 2. On small sub-cohorts (Sites 1–2, $N=119$), Gradient Boosting overfits and yields negative cross-validated $R^2$ ($-0.13$).
> 3. Predictive accuracy ($R^2 \approx 0.22$) does not establish causal efficacy or replace regulatory ANCOVA modeling.

---

## 5. Visualizations Gallery

All publication figures are exported at 300 DPI to the `figures/` directory:

| Figure | Description | File |
| :--- | :--- | :--- |
| **Figure 1** | Missing SBP measurements heatmap by site & visit, plus Week 12 attrition by arm | `figures/fig1_missingness_overview.png` |
| **Figure 2** | Baseline vs Follow-up SBP distributions by treatment arm and site | `figures/fig2_sbp_distributions.png` |
| **Figure 3** | Sensitivity Analysis Forest Plot across all 8 analytical scenarios | `figures/fig3_sensitivity_forest.png` |
| **Figure 4** | 4-Panel OLS Diagnostics (Residuals vs Fitted, Normal Q-Q, Scale-Location, Cook's D) | `figures/fig4_model_diagnostics.png` |
| **Figure 5** | Site 2 calibration shift between Device 1 and Device 2 across visits | `figures/fig5_site2_device_shift.png` |
| **Figure 6** | Holdout test $R^2$ and MAE benchmarks against Dummy Baseline | `figures/fig6_ml_benchmarks.png` |

---

## 6. Repository Architecture

```
clinical-trial-analysis/
├── README.md                      # Executive project presentation and verified findings
├── requirements.txt              # Pinned dependency manifest
├── trialx_data.csv               # UNTOUCHED simulated Phase II clinical trial dataset
├── main.ipynb                    # Master executable analytical notebook
├── original_main.ipynb           # Untouched reference backup of original hackathon work
├── src/                          # Modular production analytical library
│   ├── __init__.py               # Package metadata
│   ├── data_quality.py           # Ingestion, validation, baseline balance, and IQR checks
│   ├── missing_data.py           # Dropout modeling, MICE, and tipping-point MNAR analysis
│   ├── statistical_analysis.py   # ANCOVA (both formulations), diagnostics, multiplicity
│   ├── sensitivity_analysis.py   # Multi-scenario sensitivity battery
│   ├── predictive_models.py      # Pipelines, CV benchmarks, dummy baselines, importance
│   └── visualizations.py         # Publication-grade figures (forest plots, diagnostics)
├── tests/                        # Automated unit & integration tests (100% pass rate)
│   ├── __init__.py
│   ├── test_data_quality.py      # Ingestion, schema validation, and balance tests
│   ├── test_statistical_models.py# ANCOVA isomorphism, diagnostics, device tests
│   └── test_reproducibility.py   # Deterministic reproducibility and tipping point tests
├── figures/                      # High-resolution publication figures (300 DPI)
└── docs/                         # In-depth technical documentation
    ├── data_dictionary.md        # Comprehensive variable specifications
    └── statistical_methodology.md# Biostatistical monograph and mathematical proofs
```

---

## 7. Installation & Reproducibility Guide

### Environment Setup
Clone the repository and install dependencies in Python 3.11+:

```bash
git clone https://github.com/ShreyanshDas24/clinical-trial-analysis.git
cd clinical-trial-analysis
pip install -r requirements.txt
```

### Run Automated Unit Tests
Verify all statistical calculations, isomorphism proofs, and schema assertions:

```bash
pytest -v tests
```
*Expected Output: `16 passed in ~9.8s`*

### Execute Master Notebook
Run the end-to-end analytical pipeline and regenerate all figures:

```bash
jupyter execute main.ipynb
# Or launch interactively:
jupyter notebook main.ipynb
```

---

## 8. Limitations & Clinical Assumptions

1. **Synthetic Nature**: All patient profiles and hemodynamic measures are simulated. No inference should be drawn regarding real-world medications.
2. **Untestable MNAR**: While logistic regression confirmed that dropout is strongly predicted by site, post-baseline attrition at Site 3 cannot be proven to be MAR. However, our tipping-point sensitivity analysis demonstrated that TrX efficacy remains statistically significant even if TrX dropouts suffered a penalty of up to $\delta = 10$ mmHg.
3. **Site 2 Device Generalizability**: Because Device 2 was only deployed at Site 2, device effects cannot be distinguished from site sub-cohort effects. However, randomized 1:1 allocation within device cohorts prevented confounding of the treatment contrast.

---

## 9. Contributors & Portfolio Attribution

- **Shreyansh Kumar Das** — Lead Author, Data Science & Biostatistics Portfolio  
- Collaborative contributions: Sanjana Sarkar, Subhodeep Bhattacharjee, Atrijo Roy
