# Systematic Hyperparameter Sensitivity Analysis of Random Forest

**Author:** Pirapong Singsathid  
**Affiliation:** Department of Mathematics, Faculty of Science, Khon Kaen University, Thailand  

---

## Overview

This repository contains the complete experimental code and results for a systematic sensitivity analysis of five key Random Forest hyperparameters using a **full factorial design** (4⁵ = 1,024 configurations) evaluated via **10-fold cross-validation** across **20 benchmark datasets** (12 classification, 8 regression).

### Hyperparameters Studied

| Hyperparameter | Levels |
|---|---|
| `n_estimators` | 50, 100, 200, 500 |
| `max_depth` | None, 5, 10, 20 |
| `max_features` | sqrt, log2, 0.5, 1.0 |
| `min_samples_split` | 2, 5, 10, 20 |
| `min_samples_leaf` | 1, 2, 5, 10 |

### Statistical Methods

- **η² (eta-squared):** ANOVA-style effect size measuring the proportion of performance variance attributable to each hyperparameter
- **Friedman test:** Non-parametric significance test across parameter levels (α = 0.05)
- **Wilcoxon signed-rank post-hoc:** Pairwise comparisons between levels within each parameter

---

## Repository Structure

```
rf-hyperparameter-sensitivity/
├── code/
│   └── rf_sensitivity_analysis_robust.py   # Main experiment script
├── results/
│   ├── iris/
│   │   ├── results_iris.csv        # Raw results: all 1,024 configs × 10-fold CV
│   │   ├── stats_iris.csv          # η², Friedman stat/p, significance per parameter
│   │   ├── summary_iris.csv        # Best configuration + best metric value
│   │   └── wilcoxon_iris.csv       # Pairwise Wilcoxon post-hoc results
│   ├── wine/
│   ├── ionosphere/
│   ├── vehicle/
│   ├── breast_cancer/
│   ├── credit-g/
│   ├── banknote/
│   ├── digits/
│   ├── segment/
│   ├── phoneme/
│   ├── covertype/
│   ├── pol/
│   ├── diabetes/
│   ├── abalone/
│   ├── kin8nm/
│   ├── elevators/
│   ├── bike_sharing/
│   ├── houses/
│   ├── california_housing/
│   └── house_sales/
├── requirements.txt
└── README.md
```

---

## Datasets

### Classification (12 datasets)

| Dataset | Source | Samples | Features | Classes |
|---|---|---|---|---|
| Iris | sklearn | 150 | 4 | 3 |
| Wine | sklearn | 178 | 13 | 3 |
| Ionosphere | OpenML 59 | 351 | 34 | 2 |
| Vehicle | OpenML 54 | 846 | 18 | 4 |
| Breast Cancer | sklearn | 569 | 30 | 2 |
| Credit-G | OpenML 31 | 1,000 | 20 | 2 |
| Banknote | OpenML 1462 | 1,372 | 4 | 2 |
| Digits | sklearn | 1,797 | 64 | 10 |
| Segment | OpenML 36 | 2,310 | 19 | 7 |
| Phoneme | OpenML 1489 | 5,404 | 5 | 2 |
| Covertype | OpenML 180 | 10,000† | 54 | 7 |
| Pol | OpenML 722 | 15,000 | 26 | 2 |

†Stratified random sample from 581,012 instances (random_state=42)

### Regression (8 datasets)

| Dataset | Source | Samples | Features |
|---|---|---|---|
| Diabetes | sklearn | 442 | 10 |
| Abalone | OpenML 183 | 4,177 | 8 |
| Kin8nm | OpenML 189 | 8,192 | 8 |
| Elevators | OpenML 216 | 16,599 | 18 |
| Bike Sharing | OpenML 42712 | 17,389 | 12 |
| Houses | OpenML 537 | 20,640 | 8 |
| California Housing | sklearn | 20,640 | 8 |
| House Sales | OpenML 42165 | 21,613 | 17 |

---

## How to Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the experiment

```bash
python code/rf_sensitivity_analysis_robust.py
```

The script will:
- Download datasets automatically via `sklearn` and `fetch_openml`
- Run all 1,024 configurations × 10-fold CV for each dataset
- Save results to `results_rf_sensitivity/<dataset>/`
- Compute η², Friedman test, and Wilcoxon post-hoc statistics

**Estimated runtime:** 20–40 hours total (n_jobs=-1, parallel execution). A checkpoint/resume mechanism is built in — safe to interrupt and restart.

### 3. Output files per dataset

| File | Description |
|---|---|
| `results_<dataset>.csv` | Raw CV scores for all 1,024 configurations |
| `stats_<dataset>.csv` | η², Friedman statistic, p-value, significance flag |
| `summary_<dataset>.csv` | Best hyperparameter configuration and metric value |
| `wilcoxon_<dataset>.csv` | Pairwise Wilcoxon signed-rank test results |

---

## Key Findings

- **`max_depth`** is the dominant hyperparameter across both task types (mean η² = 0.39 classification, 0.73 regression)
- **`max_features`** ranks second (mean η² = 0.27 classification, 0.12 regression)
- **`n_estimators`** exhibits diminishing returns beyond 100–200 trees (mean η² < 0.01)
- **`min_samples_split`** is largely redundant with `min_samples_leaf` (mean η² < 0.03)
- The rank ordering `max_depth > max_features > min_samples_leaf > min_samples_split > n_estimators` holds consistently across all 20 datasets

---

## Environment

- Python 3.13
- scikit-learn 1.7.2
- SciPy 1.16.3
- NumPy, pandas, matplotlib, seaborn


---

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.
