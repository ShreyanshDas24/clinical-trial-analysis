# Clinical Trial Data Dictionary

## Study Overview
- **Protocol**: Simulated Phase II Randomized Controlled Trial of Antihypertensive Investigational Product TrX versus Placebo
- **Design**: Multi-center, parallel-group, double-blind, 1:1 randomization
- **Planned Sample Size**: $N = 180$ (60 patients across 3 sites: Site 1, Site 2, Site 3)
- **Primary Endpoint**: Change in Systolic Blood Pressure from Baseline (Week 0) to Week 12 ($\Delta\text{SBP} = \text{SBP}_0 - \text{SBP}_{12}$, in mmHg)

---

## Variable Specifications

| Column Name | Data Type | Units / Format | Clinical Description | Valid Range / Categories | Missingness Mechanism & Frequency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `patient_id` | String | Identifier (`P001` to `P180`) | Unique trial subject identifier | Exactly 180 unique IDs | Complete (0% missing) |
| `site` | Integer | Categorical ($1, 2, 3$) | Clinical trial site location | $1$ (Site 1), $2$ (Site 2), $3$ (Site 3); 60 patients/site | Complete (0% missing) |
| `treatment` | String | Categorical | Randomized treatment assignment | `Placebo` ($n=90$), `TrX` ($n=90$) | Complete (0% missing) |
| `age` | Float | Years | Subject age at study enrollment | Plausible range: $18.0\text{--}100.0$ years (Observed: $30\text{--}79$) | Sporadic MCAR: $3$ patients ($1.67\%$: P109, P164, P167) |
| `sex` | String | Categorical | Biological sex | `M` ($n=93$), `F` ($n=87$) | Complete (0% missing) |
| `bmi` | Float | $\text{kg/m}^2$ | Body Mass Index at screening | Plausible range: $10.0\text{--}60.0$ (Observed: $18.0\text{--}37.7$) | Sporadic MCAR: $6$ patients ($3.33\%$) |
| `sbp_0` | Float | mmHg | Baseline Systolic Blood Pressure (Week 0) | Plausible range: $70.0\text{--}250.0$ (Observed: $110\text{--}187$) | Complete (0% missing) |
| `sbp_4` | Float | mmHg | Post-baseline SBP at Visit 2 (Week 4) | Plausible range: $70.0\text{--}250.0$ (Observed: $105\text{--}190$) | Monotone Attrition: $26$ patients ($14.44\%$; 21 at Site 3) |
| `sbp_8` | Float | mmHg | Post-baseline SBP at Visit 3 (Week 8) | Plausible range: $70.0\text{--}250.0$ (Observed: $100\text{--}197$) | Cumulative Dropout: $54$ patients ($30.00\%$; 46 at Site 3) |
| `sbp_12` | Float | mmHg | Primary endpoint SBP at Visit 4 (Week 12) | Plausible range: $70.0\text{--}250.0$ (Observed: $102\text{--}186$) | Primary Endpoint Missingness: $54$ patients ($30.00\%$; 46 at Site 3) |
| `dropout_week`| Float | Weeks ($4.0, 8.0, \text{NaN}$) | Week of trial attrition | $4.0$ ($n=26$), $8.0$ ($n=28$), $\text{NaN}$ ($n=126$ completers) | Monotone dropout tracker |
| `device_flag` | Integer | Categorical ($1, 2$) | Blood pressure measurement device model | $1$ (Standard), $2$ (Secondary unit introduced at Site 2) | Complete; $2$ exists only at Site 2 for P091–P120 ($n=30$) |
| `deviation_flag`| Integer | Binary indicator ($0, 1$) | Major Good Clinical Practice (GCP) protocol deviation | $0$ (Compliant), $1$ (Major protocol deviation) | Complete; $1$ exists only at Site 3 ($n=15$) |

---

## Derived Endpoints

| Derived Column | Formula | Interpretation |
| :--- | :--- | :--- |
| `delta_sbp` | $\text{sbp\_0} - \text{sbp\_12}$ | Reduction in systolic blood pressure from baseline (mmHg). **Positive values indicate clinical improvement** (lowered blood pressure). Negative values indicate increased blood pressure. |
| `dropout` | $\mathbb{I}(\text{dropout\_week is not NaN})$ | Binary indicator of participant study discontinuation ($1 = \text{dropped out}, 0 = \text{completed}$). |
