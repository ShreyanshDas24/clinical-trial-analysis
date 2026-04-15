# Clinical Trial Analysis — Contaminated Multi-Site Study

This project presents a complete statistical and machine learning analysis of a simulated **Phase II antihypertensive randomized controlled trial (RCT)**. The objective is to evaluate the effectiveness of a treatment compared to placebo across multiple clinical sites while accounting for data quality issues and site-level variability.

---

## Repository Structure

| File               | Description                                    |
| ------------------ | ---------------------------------------------- |
| `main.ipynb`       | Jupyter notebook containing full analysis      |
| `trialx_data.csv`  | Dataset with patient-level clinical trial data |
| `requirements.txt` | List of required Python libraries              |

---

##  Project Overview

###  Data Integrity & Exploration

* Analysis of missing data patterns across sites
* Outlier detection using the IQR method
* Investigation of site-specific inconsistencies
* Dropout mechanism assessment (MCAR, MAR, MNAR)
* Site-level reliability evaluation

---

### Statistical Modeling

* ANCOVA model:

  `delta_sbp ~ treatment + site + baseline + age + bmi + device_flag`

* Estimation of treatment effect

* Hypothesis testing and significance analysis

* Multiple testing correction (Holm / Bonferroni)

---

### Machine Learning Models

* Ridge Regression
* Random Forest
* Gradient Boosting

These models are used to validate findings and compare predictive performance.

---

## Key Results

* The treatment shows a statistically significant reduction in systolic blood pressure
* Site-level variability influences inference and requires careful handling
* Results remain consistent under sensitivity analysis
* Machine learning models support the statistical conclusions

---

## Requirements

Install all dependencies using:

```bash id="r1qf9v"
pip install -r requirements.txt
```

---

## How to Run

```bash id="f3m8c2"
jupyter notebook main.ipynb
```

---

## 👥 Contributors

This project was developed as a collaborative effort.

* **Shreyansh Kumar Das**
* **Sanjana Sarkar**
* **Subhodeep Bhattacharjee**
* **Atrijo Roy**

**Shreyansh Kumar Das**
