# Systematic Hyperparameter Sensitivity Analysis of Random Forest

**Authors:** Thapakorn Sinlapakorn, Pirapong Singsathid  
**Affiliation:** Department of Mathematics, Faculty of Science, Khon Kaen University, Thailand  
**Corresponding author:** Pirapong Singsathid — pirasi@kku.ac.th

---

## Overview

This repository contains the complete experimental code and results for a systematic sensitivity analysis of five Random Forest hyperparameters using a **full factorial design** (4^5 = 1,024 configurations) evaluated via **10-fold cross-validation** across **20 benchmark datasets** (12 classification, 8 regression) — 204,800 model evaluations in total.

The accompanying manuscript is in [`paper/`](paper/).

### Hyperparameters studied

| Hyperparameter | Levels | scikit-learn default |
|---|---|---|
| `n_estimators` | 50, 100, 200, 500 | 100 |
| `max_depth` | None, 5, 10, 20 | None |
| `max_features` | sqrt, log2, 0.5, 1.0 | sqrt |
| `min_samples_split` | 2, 5, 10, 20 | 2 |
| `min_samples_leaf` | 1, 2, 5, 10 | 1 |

Every other argument of `RandomForestClassifier` / `RandomForestRegressor` is held at its scikit-learn default, including `bootstrap`, `criterion`, `max_samples`, `ccp_alpha`, `max_leaf_nodes` and `class_weight`. Section 3.1 of the manuscript gives the selection criteria and the reason for each exclusion.

### Statistical methods

- **eta-squared** — ANOVA-style effect size: the proportion of performance variance attributable to each hyperparameter. Because the factorial is complete, the sum-of-squares decomposition is exact and needs no surrogate model.
- **Friedman test** — non-parametric test of level differences within each parameter (alpha = 0.05).
- **Wilcoxon signed-rank post-hoc** — all 6 pairwise level comparisons within each parameter.
- **Shaffer's static procedure** — family-wise error control over each parameter's 6 comparisons.
- **Percentile bootstrap** — 95% confidence intervals for the cross-dataset mean eta-squared (B = 10,000, seed 42).

---

## Repository structure

```
rf-hyperparameter-sensitivity/
├── code/
│   ├── rf_sensitivity_analysis.py    # main experiment
│   └── bootstrap_and_shaffer.py      # bootstrap CIs + Shaffer correction
├── results/
│   └── <dataset>/
│       ├── results_<dataset>.csv       # all 1,024 configs x 10-fold CV, with per-config timing
│       ├── stats_<dataset>.csv         # eta-squared, Friedman statistic, p-value, significance
│       ├── summary_<dataset>.csv       # best configuration and metric value
│       ├── best_configs_<dataset>.csv  # top-ranked configurations
│       └── wilcoxon_<dataset>.csv      # pairwise Wilcoxon post-hoc results
├── paper/
│   └── rf_sensitivity_manuscript_fcds_v8.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

The 20 dataset folders are `iris`, `wine`, `ionosphere`, `vehicle`, `breast_cancer`, `credit-g`, `banknote`, `digits`, `segment`, `phoneme`, `covertype`, `pol` (classification) and `diabetes`, `abalone`, `kin8nm`, `elevators`, `bike_sharing`, `houses`, `california_housing`, `house_sales` (regression).

---

## Datasets

### Classification (12)

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
| Covertype | OpenML 180 | 10,000* | 54 | 7 |
| Pol | OpenML 722 | 15,000 | 26 | 2† |

*Stratified random sample from 581,012 instances (`random_state=42`).

†OpenML 722 is a binarized version of a regression dataset: the numeric target is split into two classes at its mean. It is retained in the reported figures, so every classification mean above includes one binarized regression problem. Section 4.1 of the manuscript gives the robustness check (excluding it moves the classification `max_depth` mean from 0.389 to 0.355 and leaves the ranking unchanged).

### Regression (8)

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

## Reproducing the results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Re-derive the published statistics from the stored results

```bash
python code/bootstrap_and_shaffer.py
```

Runs in seconds and needs only the Python standard library — no scikit-learn, SciPy, NumPy or pandas. It reads `results/<dataset>/{stats,wilcoxon}_<dataset>.csv` and prints:

- the mean eta-squared per hyperparameter per task type (reproduces Tables 4 and 6 of the manuscript);
- 95% percentile bootstrap intervals for those means (the "95% CI" rows);
- the Wilcoxon significance counts, uncorrected and after Shaffer's static procedure (both columns of Table 9).

The script deliberately re-derives the already-published means and uncorrected counts first. Both reproduce exactly, which is what validates the corrected figures printed alongside them.

### 3. Re-run the full experiment (optional)

```bash
python code/rf_sensitivity_analysis.py
```

Datasets are downloaded automatically through `sklearn.datasets` and `fetch_openml`. The script writes to `results_rf_sensitivity/<dataset>/` and has a checkpoint/resume mechanism, so it is safe to interrupt and restart.

**Measured cost:** summing the per-configuration `time_s` column over all 20,480 rows gives **95.5 hours** of compute (`n_jobs=-1`). It is very unevenly distributed — `pol` alone accounts for 16.0 h and `elevators` for 12.6 h, while `iris` takes 42 minutes.

---

## Key findings

- **`max_depth`** is the dominant hyperparameter in both task types: mean eta-squared = 0.389 [0.200, 0.588] for classification and 0.729 [0.496, 0.902] for regression.
- **`max_features`** ranks second, and matters proportionally more for classification (0.268 [0.146, 0.403]) than for regression (0.119 [0.053, 0.196]).
- **`n_estimators`** and **`min_samples_split`** are negligible in both task types (mean eta-squared < 0.025; the bootstrap upper bound for `n_estimators` is 0.008 and 0.013). Both can be left at their defaults.
- `min_samples_split` is largely redundant with `min_samples_leaf`: in the grid used here, 9 of the 16 level combinations leave the split constraint inert because the leaf constraint binds first.
- The ranking `max_depth > max_features > min_samples_leaf > min_samples_split > n_estimators` holds for the **cross-dataset means**, not for every individual dataset. Ranked by their own dominant parameter, the 20 datasets fall into three regimes: `max_depth`-dominant (13), `max_features`-dominant (6, including Iris and Wine, where the `max_depth` effect is essentially zero) and `min_samples_leaf`-dominant (1, Vehicle).
- **Largest effect is not the same as best tuning target.** `max_depth` carries the largest effect size, but `max_depth=None` is the scikit-learn default *and* the best setting on 17 of 20 datasets, so the practical instruction is to leave it unconstrained rather than to sweep it. `max_features` is where a sweep pays: its `sqrt` default is best on only 5 of 20. `min_samples_leaf` is worth tuning on small datasets (n < 1,000) and wherever depth is left unconstrained. Section 5.3 of the manuscript gives the full ordering.

---

## Environment

Python 3.13, scikit-learn 1.7.2, SciPy 1.16.3, NumPy, pandas, matplotlib, seaborn.
`random_state=42` and `n_jobs=-1` throughout.

---

## License

MIT — see [LICENSE](LICENSE).
