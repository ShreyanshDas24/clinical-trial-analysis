# Statistical Methodology & Theoretical Foundations

This monograph documents the methodological choices, mathematical proofs, and biostatistical principles governing the analysis of the Phase II clinical trial.

---

## 1. Primary Efficacy Analysis: ANCOVA Theory & Isomorphism

### 1.1 Why ANCOVA is Superior to Raw Change Scores
In randomized controlled trials (RCTs), investigators frequently deliberate between analyzing:
1. Post-treatment score unadjusted ($Y_1$)
2. Simple change score unadjusted ($\Delta Y = Y_0 - Y_1$)
3. Analysis of Covariance (ANCOVA) adjusting post-treatment for baseline ($Y_1 \sim \text{Treatment} + Y_0$)

Under baseline randomization, both the unadjusted change score and unadjusted follow-up score are unbiased estimators of the average treatment effect. However, **ANCOVA is asymptotically more efficient** than either:
$$\text{Var}(\hat{\beta}_{\text{ANCOVA}}) \approx \text{Var}(\hat{\beta}_{\text{Change}}) \cdot \left(1 - \frac{(1 - \rho)^2}{2(1 - \rho)}\right) \le \text{Var}(\hat{\beta}_{\text{Post}})$$
where $\rho$ is the correlation between baseline and post-treatment measurements. Whenever $\rho \neq 0.5$, ANCOVA provides greater statistical power and tighter confidence intervals by conditioning on baseline variance.

### 1.2 Mathematical Proof of Model Isomorphism
Consider the two baseline-adjusted specifications:

**Model A (Change-Score Formulation):**
$$\Delta Y_i = \alpha_A + \beta_{\text{TrX}, A} T_i + \gamma_A Y_{0, i} + \mathbf{X}_i^\top \boldsymbol{\theta}_A + \epsilon_{A, i}$$
where $\Delta Y_i = Y_{0, i} - Y_{1, i}$ (positive = blood pressure reduction).

**Model B (Follow-Up Formulation):**
$$Y_{1, i} = \alpha_B + \beta_{\text{TrX}, B} T_i + \gamma_B Y_{0, i} + \mathbf{X}_i^\top \boldsymbol{\theta}_B + \epsilon_{B, i}$$

Because $\Delta Y_i = Y_{0, i} - Y_{1, i}$, we can substitute $Y_{1, i} = Y_{0, i} - \Delta Y_i$:
$$Y_{0, i} - \Delta Y_i = \alpha_B + \beta_{\text{TrX}, B} T_i + \gamma_B Y_{0, i} + \mathbf{X}_i^\top \boldsymbol{\theta}_B + \epsilon_{B, i}$$
$$\implies \Delta Y_i = -\alpha_B - \beta_{\text{TrX}, B} T_i + (1 - \gamma_B) Y_{0, i} - \mathbf{X}_i^\top \boldsymbol{\theta}_B - \epsilon_{B, i}$$

Equating coefficients:
$$\beta_{\text{TrX}, A} = -\beta_{\text{TrX}, B}$$
$$\gamma_A = 1 - \gamma_B$$
$$\text{Var}(\epsilon_A) = \text{Var}(\epsilon_B)$$

**Consequences**:
- The treatment effect estimates are **numerically identical in magnitude** ($|\hat{\beta}_{\text{TrX}}| = 7.81$ mmHg) and standard error ($\text{SE} = 1.391$ mmHg).
- The $t$-statistic magnitude ($|t| = 5.617$) and $p$-value ($p = 1.45 \times 10^{-7}$) are identical.
- In Model A, $+7.81$ indicates TrX produces **$7.81$ mmHg greater reduction** than Placebo.
- In Model B, $-7.81$ indicates TrX achieves a **$7.81$ mmHg lower final blood pressure** than Placebo.
- Model B is pre-specified as the primary model per ICH E9 guidelines.

---

## 2. Resolving the Numerical Singularity Artifact

In `main.ipynb` (Cell 39), the original formula was:
$$\Delta\text{SBP} \sim \text{treatment} + \text{sbp\_0} + \text{site} + \text{device\_flag} + \mathbf{deviation\_flag} + \text{age} + \text{bmi} + \text{sex}$$
fitted on Sites 1–2.

### Root Cause
Protocol deviations occurred solely at Site 3:
- Site 1: 0 deviations
- Site 2: 0 deviations
- Site 3: 15 deviations

In the Sites 1–2 subset, the column `deviation_flag` contains $119$ zeros and $0$ ones ($Var = 0$). Including an unvarying column alongside the model intercept creates an exact linear dependency:
$$\mathbf{X}_{\text{deviation}} = 0 \cdot \mathbf{1}$$

In floating-point matrix inversion, this generated:
- Condition Number: $3.33 \times 10^{18}$ (design matrix numerically singular)
- Smallest Eigenvalue: $2.75 \times 10^{-31}$
- Spurious coefficient: $2.258 \times 10^{-14}$ ($p=0.001$)

**Resolution**: Omission of zero-variance columns in sub-populations reduces the condition number to $2.09 \times 10^3$, eliminating the warning while preserving all substantive regression parameters.

---

## 3. Missing Data Taxonomy & Regulatory Considerations

### 3.1 Rubin's Missing Data Taxonomy
1. **Missing Completely at Random (MCAR)**: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M)$. Missingness is independent of both observed and unobserved data. Example: Sporadic age/BMI recording omissions ($< 3\%$).
2. **Missing at Random (MAR)**: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M \mid Y_{\text{obs}})$. Given observed covariates, missingness does not depend on unobserved outcomes. Example: Site 3 administrative trial closure where attrition is explained by site.
3. **Missing Not at Random (MNAR)**: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}})$ depends on the unobserved value $Y_{\text{mis}}$. Example: Patients drop out because their true blood pressure remained dangerously elevated.

### 3.2 The Impossibility of Proving MNAR from Observed Data
Logistic regression of dropout status on baseline covariates tests $P(M \mid X)$. While it can reject MCAR (showing missingness depends on site), **it cannot verify or rule out MNAR**. Whether unobserved outcomes differ from observed outcomes is untestable without external tracking data.

### 3.3 Flaws of Single Median Imputation
Imputing missing values with the overall site median without conditioning on treatment arm mechanically forces the treatment difference among missing patients to zero:
$$\hat{\Delta}_{\text{imputed}} = \text{Median}(\text{Site 3}) - \text{Median}(\text{Site 3}) = 0$$
At Site 3 (76.7% missing), this shrunk the observed complete-case treatment effect of $+5.73$ mmHg down to $+0.33$ mmHg ($p=0.86$).

---

## 4. Nested Device Bias at Site 2

At Site 2, patients P061–P090 were measured with Device 1, while patients P091–P120 were measured with Device 2.
- Device 2 introduced an additive measurement shift of $+5.52$ mmHg at Week 4, $+6.25$ mmHg at Week 8, and $+4.00$ mmHg at Week 12.
- Because Device 2 was never used at Site 1 or Site 3, a global device effect is structurally unidentifiable from a Site 2 cohort effect.
- However, randomization within Site 2 was balanced (15 Placebo, 15 TrX on Device 1; 15 Placebo, 15 TrX on Device 2).
- The treatment $\times$ device interaction test yielded $p = 0.698$, confirming the device represents an additive calibration offset that does not bias the treatment contrast.

---

## 5. Machine Learning vs. Causal Biostatistics

| Dimension | Confirmatory Biostatistical Analysis (ANCOVA) | Supplementary Machine Learning (RF / GB / Ridge) |
| :--- | :--- | :--- |
| **Objective** | Unbiased estimation of average treatment effect ($\beta_{\text{TrX}}$) | Minimization of out-of-sample prediction error ($\text{MAE}, \text{RMSE}$) |
| **Primary Metric** | Treatment effect estimate, 95% CI, $p$-value | Out-of-sample $R^2$, MAE |
| **Error Control** | Strict Family-Wise Type I error control ($\alpha = 0.05$) | Regularization and cross-validation loss |
| **Small-Sample Behavior** | Well-calibrated $t$-distribution inference | Boosted trees overfit ($N=119 \implies \text{CV } R^2 < 0$) |
| **Regulatory Standing** | ICH E9 standard for drug approval | Exploratory prognostic biomarker discovery |
