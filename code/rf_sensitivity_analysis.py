# =============================================================================
# Systematic Hyperparameter Sensitivity Analysis: Random Forest
# Tasks: Classification + Regression
# Author: Pirapong Jitsatha, Dept. of Mathematics, Khon Kaen University
# =============================================================================
# - Single-dataset mode: set DATASET_NAME below, run once per dataset (12 phases)
# - Statistical tests: Friedman, Wilcoxon post-hoc (within-parameter pairs), η²
# - Metrics: Weighted F1 + Accuracy + Std (classification)
#            R² + RMSE + Std (regression)
# - Checkpoint: CSV append + txt index (no pickle, no RAM accumulation)
# =============================================================================

import os
import time
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    load_diabetes, fetch_california_housing
)
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

warnings.filterwarnings('ignore')

# =============================================================================
# *** SET DATASET NAME HERE BEFORE RUNNING ***
# Classification (12): 'iris', 'wine', 'breast_cancer', 'digits',
#                      'ionosphere', 'vehicle', 'banknote', 'segment',
#                      'phoneme', 'credit-g', 'covertype', 'pol'
# Regression     (8) : 'diabetes', 'abalone', 'kin8nm', 'elevators',
#                      'bike_sharing', 'houses', 'california_housing',
#                      'house_sales'
# =============================================================================

DATASET_NAME = 'house_sales'   # <--- change this for each phase

# =============================================================================
# CONFIGURATION
# =============================================================================

WORKING_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(WORKING_DIR, 'results_rf_sensitivity', DATASET_NAME)
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'checkpoint_rf_sensitivity')
PLOTS_DIR      = os.path.join(RESULTS_DIR, 'plots')

for d in [RESULTS_DIR, CHECKPOINT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

N_FOLDS          = 10
CHECKPOINT_EVERY = 50
RANDOM_STATE     = 42
COVERTYPE_SUBSET = 10_000   # stratified sample size for covertype

# =============================================================================
# HYPERPARAMETER GRID  (4^5 = 1,024 configs)
# =============================================================================

PARAM_GRID = {
    'n_estimators':      [50, 100, 200, 500],
    'max_depth':         [None, 5, 10, 20],
    'max_features':      ['sqrt', 'log2', 0.5, 1.0],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 5, 10],
}

PARAM_NAMES  = list(PARAM_GRID.keys())
PARAM_LEVELS = [PARAM_GRID[k] for k in PARAM_NAMES]
ALL_CONFIGS  = list(itertools.product(*PARAM_LEVELS))
N_CONFIGS    = len(ALL_CONFIGS)   # 1,024

# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASET_INFO = {
    # ==================== CLASSIFICATION (13 datasets) ====================
    # Small
    'iris'               : ('classification', 'sklearn', None),    # 150s, 4f, 3c
    'wine'               : ('classification', 'sklearn', None),    # 178s, 13f, 3c
    # Medium
    'breast_cancer'      : ('classification', 'sklearn', None),    # 569s, 30f, 2c
    'digits'             : ('classification', 'sklearn', None),    # 1.8Ks, 64f, 10c — high-dim small
    'ionosphere'         : ('classification', 'openml',  59),      # 351s, 34f, 2c — noisy
    'vehicle'            : ('classification', 'openml',  54),      # 846s, 18f, 4c — multi-class
    'banknote'           : ('classification', 'openml',  1462),    # 1.4Ks, 4f, 2c — low-dim
    'segment'            : ('classification', 'openml',  36),      # 2.3Ks, 19f, 7c — many classes
    'phoneme'            : ('classification', 'openml',  1489),    # 5.4Ks, 5f, 2c — medium binary
    'credit-g'           : ('classification', 'openml',  31),      # 1Ks, 20f, 2c — imbalanced+categorical
    # Large
    'covertype'          : ('classification', 'openml',  180),     # 581Ks→10Ks subset, 54f, 7c
    'pol'                : ('classification', 'openml',  722),     # 15Ks, 26f — binary cls

    # ==================== REGRESSION (8 datasets) ====================
    # Small
    'diabetes'           : ('regression',     'sklearn', None),    # 442s, 10f
    # Medium
    'abalone'            : ('regression',     'openml',  183),     # 4.2Ks, 8f — categorical feature
    'kin8nm'             : ('regression',     'openml',  189),     # 8.2Ks, 8f — nonlinear
    # Large
    'elevators'          : ('regression',     'openml',  216),     # 16.6Ks, 18f
    'bike_sharing'       : ('regression',     'openml',  42712),   # 17.4Ks, 12f — temporal
    'houses'             : ('regression',     'openml',  537),     # 20.6Ks, 8f — real estate
    'california_housing' : ('regression',     'sklearn', None),    # 20.6Ks, 8f — geographic
    'house_sales'        : ('regression',     'openml',  42165),   # 21.6Ks, 17f — mixed types
}


def load_dataset(name):
    """Load a single dataset by name. Returns (X, y, task)."""
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {list(DATASET_INFO.keys())}")

    task, source, openml_id = DATASET_INFO[name]

    # --- sklearn built-ins ---
    if source == 'sklearn':
        if name == 'iris':
            d = load_iris();            return d.data, d.target, task
        if name == 'wine':
            d = load_wine();            return d.data, d.target, task
        if name == 'breast_cancer':
            d = load_breast_cancer();   return d.data, d.target, task
        if name == 'diabetes':
            d = load_diabetes();        return d.data, d.target, task
        if name == 'california_housing':
            d = fetch_california_housing(); return d.data, d.target, task
        if name == 'digits':
            from sklearn.datasets import load_digits
            d = load_digits();          return d.data, d.target, task

    # --- OpenML ---
    from sklearn.datasets import fetch_openml
    print(f"Fetching '{name}' from OpenML (id={openml_id}) ...")
    d = fetch_openml(data_id=openml_id, as_frame=True, parser='auto')

    # Handle mixed-type features: encode categoricals, keep numerics
    X_df = d.data.copy()
    for col in X_df.select_dtypes(include=['object', 'category']).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
    X = X_df.values.astype(float)

    # Process target
    if task == 'classification':
        y = LabelEncoder().fit_transform(d.target.astype(str))
    else:
        # Regression: handle categorical/object target (e.g. abalone rings stored as string)
        if d.target.dtype == 'object' or d.target.dtype.name == 'category':
            y = pd.to_numeric(d.target, errors='coerce').values.astype(float)
            valid = ~np.isnan(y)
            if valid.sum() == 0:
                raise ValueError(f"All target values non-numeric for '{name}'")
            if valid.sum() < len(y):
                print(f"  Removed {(~valid).sum()} non-numeric target rows")
                X, y = X[valid], y[valid]
        else:
            y = d.target.values.astype(float)

    # Stratified subset for covertype only
    if name == 'covertype' and len(y) > COVERTYPE_SUBSET:
        sss = StratifiedShuffleSplit(n_splits=1,
                                     train_size=COVERTYPE_SUBSET,
                                     random_state=RANDOM_STATE)
        idx, _ = next(sss.split(X, y))
        X, y   = X[idx], y[idx]
        print(f"  Stratified subset: {COVERTYPE_SUBSET} samples "
              f"(random_state={RANDOM_STATE})")

    return X, y, task

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_classification(config, X, y):
    """10-fold stratified CV → mean/std Accuracy and Weighted F1."""
    params = dict(zip(PARAM_NAMES, config))
    model  = RandomForestClassifier(
        n_estimators      = params['n_estimators'],
        max_depth         = params['max_depth'],
        max_features      = params['max_features'],
        min_samples_split = params['min_samples_split'],
        min_samples_leaf  = params['min_samples_leaf'],
        random_state      = RANDOM_STATE,
        n_jobs            = -1
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    acc_list, f1_list = [], []
    for tr, te in skf.split(X, y):
        model.fit(X[tr], y[tr])
        yp = model.predict(X[te])
        acc_list.append(accuracy_score(y[te], yp))
        f1_list.append(f1_score(y[te], yp, average='weighted', zero_division=0))
    return (np.mean(f1_list),  np.std(f1_list),
            np.mean(acc_list), np.std(acc_list))


def evaluate_regression(config, X, y):
    """10-fold CV → mean/std R² and RMSE."""
    params = dict(zip(PARAM_NAMES, config))
    model  = RandomForestRegressor(
        n_estimators      = params['n_estimators'],
        max_depth         = params['max_depth'],
        max_features      = params['max_features'],
        min_samples_split = params['min_samples_split'],
        min_samples_leaf  = params['min_samples_leaf'],
        random_state      = RANDOM_STATE,
        n_jobs            = -1
    )
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r2_list, rmse_list = [], []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        yp = model.predict(X[te])
        r2_list.append(r2_score(y[te], yp))
        rmse_list.append(np.sqrt(mean_squared_error(y[te], yp)))
    return (np.mean(r2_list),   np.std(r2_list),
            np.mean(rmse_list), np.std(rmse_list))

# =============================================================================
# EXPERIMENT LOOP  (CSV append + txt checkpoint — no RAM accumulation)
# =============================================================================

def run_experiment(X, y, task):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'ckpt_{DATASET_NAME}.txt')
    csv_path  = os.path.join(RESULTS_DIR,    f'results_{DATASET_NAME}.csv')

    if task == 'classification':
        header           = (PARAM_NAMES +
                            ['mean_f1', 'std_f1',
                             'mean_accuracy', 'std_accuracy', 'time_s'])
        primary_metric   = 'mean_f1'
        secondary_metric = 'mean_accuracy'
    else:
        header           = (PARAM_NAMES +
                            ['mean_r2', 'std_r2',
                             'mean_rmse', 'std_rmse', 'time_s'])
        primary_metric   = 'mean_r2'
        secondary_metric = 'mean_rmse'

    # Resume or start fresh
    if os.path.exists(ckpt_path) and os.path.exists(csv_path):
        with open(ckpt_path, 'r') as f:
            start_idx = int(f.read().strip()) + 1
        print(f"Resuming from config {start_idx}/{N_CONFIGS}")
    else:
        start_idx = 0
        pd.DataFrame(columns=header).to_csv(csv_path, index=False)
        print(f"Starting fresh — {N_CONFIGS} configs × {N_FOLDS}-fold CV")

    print(f"{'='*60}")
    print(f"Dataset        : {DATASET_NAME} | Task: {task} | Shape: {X.shape}")
    print(f"Primary metric : {primary_metric}")
    print(f"{'='*60}")

    for i in range(start_idx, N_CONFIGS):
        config = ALL_CONFIGS[i]
        t0     = time.time()

        if task == 'classification':
            m1, m2, m3, m4 = evaluate_classification(config, X, y)
        else:
            m1, m2, m3, m4 = evaluate_regression(config, X, y)

        row = list(config) + [m1, m2, m3, m4, round(time.time() - t0, 4)]
        pd.DataFrame([row], columns=header).to_csv(
            csv_path, mode='a', header=False, index=False)

        if (i + 1) % 50 == 0 or i == N_CONFIGS - 1:
            print(f"  [{i+1:>4}/{N_CONFIGS}]  {primary_metric}={m1:.4f}  "
                  f"elapsed={time.time()-t0:.2f}s")

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == N_CONFIGS - 1:
            with open(ckpt_path, 'w') as f:
                f.write(str(i))

    # Single CSV read after all configs are done
    df = pd.read_csv(csv_path)
    print(f"\nDone. Best {primary_metric}: {df[primary_metric].max():.4f}")
    return df, task, primary_metric, secondary_metric

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_eta_squared(df, param, metric):
    # dropna=False ensures max_depth=None (stored as NaN) is included as a group
    grand_mean = df[metric].mean()
    groups     = [g[metric].values for _, g in df.groupby(param, dropna=False)]
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total   = ((df[metric] - grand_mean)**2).sum()
    return ss_between / ss_total if ss_total > 0 else 0.0


def run_friedman(df, param, metric):
    # dropna=False ensures max_depth=None (stored as NaN) is included as a group
    groups  = [g[metric].values for _, g in df.groupby(param, dropna=False)]
    min_len = min(len(g) for g in groups)
    if len(groups) < 2 or min_len < 2:
        return np.nan, np.nan
    stat, p = friedmanchisquare(*[g[:min_len] for g in groups])
    return float(stat), float(p)


def run_wilcoxon_posthoc(df, param, metric):
    """Pairwise Wilcoxon between levels of a single parameter (within-parameter only)."""
    # dropna=False ensures max_depth=None (stored as NaN) is included as a group
    level_data = {k: g[metric].values for k, g in df.groupby(param, dropna=False)}
    levels     = list(level_data.keys())
    results    = {}
    for l1, l2 in combinations(levels, 2):
        g1, g2  = level_data[l1], level_data[l2]
        min_len = min(len(g1), len(g2))
        try:
            stat, p = wilcoxon(g1[:min_len], g2[:min_len])
            stat, p = float(stat), float(p)
        except Exception:
            stat, p = np.nan, np.nan
        results[(str(l1), str(l2))] = {
            'stat'       : round(stat, 4) if not np.isnan(stat) else np.nan,
            'p_value'    : round(p, 4)    if not np.isnan(p)    else np.nan,
            'significant': bool(p < 0.05) if not np.isnan(p)    else False
        }
    return results


def run_statistical_analysis(df, task, primary_metric):
    print(f"\n{'='*60}")
    print(f"Statistical Analysis — {DATASET_NAME} ({primary_metric})")
    print(f"{'='*60}")

    stat_summary = {}
    eta_scores   = {}

    for param in PARAM_NAMES:
        eta2      = compute_eta_squared(df, param, primary_metric)
        fstat, fp = run_friedman(df, param, primary_metric)
        posthoc   = run_wilcoxon_posthoc(df, param, primary_metric)

        eta_scores[param]   = eta2
        stat_summary[param] = {
            'eta_squared'      : round(eta2, 4),
            'friedman_stat'    : round(fstat, 4) if not np.isnan(fstat) else np.nan,
            'friedman_p'       : round(fp, 4)    if not np.isnan(fp)    else np.nan,
            'significant'      : bool(fp < 0.05) if not np.isnan(fp)    else False,
            'posthoc_wilcoxon' : posthoc
        }

        sig = '*' if stat_summary[param]['significant'] else ' '
        fp_str = f"{fp:.4f}" if not np.isnan(fp) else "N/A"
        print(f"  {param:22s} | η²={eta2:.4f} | Friedman p={fp_str:>8} {sig}")

    # Ranking
    ranking = sorted(eta_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Sensitivity Ranking:")
    for rank, (param, eta2) in enumerate(ranking, 1):
        print(f"    {rank}. {param}: η²={eta2:.4f}")

    # Save stats CSV
    rows = [{'dataset': DATASET_NAME, 'task': task, 'parameter': param,
              'eta_squared': s['eta_squared'],
              'friedman_stat': s['friedman_stat'],
              'friedman_p': s['friedman_p'],
              'significant': s['significant']}
            for param, s in stat_summary.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, f'stats_{DATASET_NAME}.csv'), index=False)

    # Save Wilcoxon post-hoc CSV
    ph_rows = [{'parameter': param, 'level_1': l1, 'level_2': l2,
                'stat': res['stat'], 'p_value': res['p_value'],
                'significant': res['significant']}
               for param, s in stat_summary.items()
               for (l1, l2), res in s['posthoc_wilcoxon'].items()]
    pd.DataFrame(ph_rows).to_csv(
        os.path.join(RESULTS_DIR, f'wilcoxon_{DATASET_NAME}.csv'), index=False)

    print(f"\nSaved stats_{DATASET_NAME}.csv and wilcoxon_{DATASET_NAME}.csv")
    return stat_summary

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_main_effects(df, primary_metric, stat_summary):
    """Mean ± std of primary metric per parameter level (with η² and significance)."""
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(22, 4))
    fig.suptitle(f'Main Effect Plots — {DATASET_NAME} ({primary_metric})', fontsize=13)

    for ax, param in zip(axes, PARAM_NAMES):
        grp   = df.groupby(param)[primary_metric]
        means = grp.mean()
        stds  = grp.std()
        ax.errorbar(range(len(means)), means.values, yerr=stds.values,
                    marker='o', capsize=4, linewidth=2, color='steelblue')
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels([str(v) for v in means.index], rotation=30, fontsize=8)
        eta2 = stat_summary[param]['eta_squared']
        sig  = '*' if stat_summary[param]['significant'] else ''
        ax.set_title(f'{param}\nη²={eta2:.3f}{sig}', fontsize=9)
        ax.set_ylabel(primary_metric if ax == axes[0] else '')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'main_effects_{DATASET_NAME}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved main_effects_{DATASET_NAME}.png")


def plot_boxplots(df, primary_metric):
    """Distribution of primary metric per parameter level."""
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(22, 5))
    fig.suptitle(
        f'Distribution per Parameter Level — {DATASET_NAME} ({primary_metric})',
        fontsize=13)

    for ax, param in zip(axes, PARAM_NAMES):
        levels = sorted(df[param].unique(), key=lambda x: (x is None, str(x)))
        data   = [df[df[param] == lv][primary_metric].values for lv in levels]
        bp     = ax.boxplot(data, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightsteelblue')
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels([str(lv) for lv in levels], rotation=30, fontsize=7)
        ax.set_title(param, fontsize=9)
        ax.set_ylabel(primary_metric if ax == axes[0] else '')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'boxplot_{DATASET_NAME}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot_{DATASET_NAME}.png")


def plot_interaction_heatmaps(df, primary_metric):
    """Pairwise interaction heatmaps: mean primary metric for param_i × param_j."""
    pairs = list(combinations(PARAM_NAMES, 2))
    ncols = 3
    nrows = (len(pairs) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()
    fig.suptitle(f'Pairwise Interaction Heatmaps — {DATASET_NAME}', fontsize=13)

    for ax, (p1, p2) in zip(axes, pairs):
        pivot         = df.groupby([p1, p2])[primary_metric].mean().unstack()
        pivot.index   = [str(v) for v in pivot.index]
        pivot.columns = [str(v) for v in pivot.columns]
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='coolwarm',
                    ax=ax, linewidths=0.3, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{p1} × {p2}', fontsize=9)
        ax.set_xlabel(p2, fontsize=8)
        ax.set_ylabel(p1, fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'interaction_{DATASET_NAME}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved interaction_{DATASET_NAME}.png")


def plot_sensitivity_bar(stat_summary):
    """Horizontal bar: η² ranking — red = Friedman significant, blue = not."""
    params = sorted(stat_summary, key=lambda p: stat_summary[p]['eta_squared'])
    values = [stat_summary[p]['eta_squared'] for p in params]
    colors = ['tomato' if stat_summary[p]['significant'] else 'lightsteelblue'
              for p in params]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(params, values, color=colors, edgecolor='white')
    ax.set_xlabel('η² (variance explained)')
    ax.set_title(f'Parameter Sensitivity Ranking — {DATASET_NAME}\n'
                 f'(red = Friedman p < 0.05)', fontsize=11)
    for bar, val in zip(bars, values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'sensitivity_bar_{DATASET_NAME}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sensitivity_bar_{DATASET_NAME}.png")


def plot_top_configs(df, primary_metric, secondary_metric, top_n=10):
    """Bar chart of top-N configs ranked by primary metric."""
    top    = df.nlargest(top_n, primary_metric).reset_index(drop=True)
    labels = [f"C{i+1}" for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, top[primary_metric], color='steelblue', edgecolor='white')
    ax.set_title(f'Top {top_n} Configs — {DATASET_NAME} ({primary_metric})', fontsize=12)
    ax.set_ylabel(primary_metric)
    ax.set_ylim(top[primary_metric].min() * 0.98,
                top[primary_metric].max() * 1.005)
    ax.grid(True, axis='y', alpha=0.3)

    best = top.iloc[0]
    note = " | ".join([f"{p}={best[p]}" for p in PARAM_NAMES])
    ax.text(0.5, -0.2, f"Best: {note}", transform=ax.transAxes,
            ha='center', fontsize=7, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'top_configs_{DATASET_NAME}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top_configs_{DATASET_NAME}.png")


def export_best_config(df, primary_metric, secondary_metric, task):
    """Save top-10 configs and single-row summary to CSV."""
    top10 = df.nlargest(10, primary_metric).reset_index(drop=True)
    top10.insert(0, 'rank', range(1, len(top10) + 1))
    top10.to_csv(
        os.path.join(RESULTS_DIR, f'best_configs_{DATASET_NAME}.csv'), index=False)

    best    = df.loc[df[primary_metric].idxmax()]
    summary = {'dataset': DATASET_NAME, 'task': task,
               'n_configs': N_CONFIGS, 'n_folds': N_FOLDS,
               'primary_metric': primary_metric,
               f'best_{primary_metric}' : round(best[primary_metric],  4),
               f'best_{secondary_metric}': round(best[secondary_metric], 4)}
    for p in PARAM_NAMES:
        summary[f'best_{p}'] = best[p]

    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS_DIR, f'summary_{DATASET_NAME}.csv'), index=False)

    print(f"\nSaved best_configs_{DATASET_NAME}.csv and summary_{DATASET_NAME}.csv")
    print(f"\nBest config ({primary_metric} = {best[primary_metric]:.4f}):")
    for p in PARAM_NAMES:
        print(f"  {p:22s} = {best[p]}")
    print(f"  {secondary_metric:22s} = {best[secondary_metric]:.4f}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(f"RF Hyperparameter Sensitivity Analysis")
    print(f"Dataset     : {DATASET_NAME}")
    print(f"Configs     : {N_CONFIGS}  |  Folds: {N_FOLDS}")
    print(f"Results dir : {RESULTS_DIR}")
    print("=" * 60)

    # 1. Load single dataset
    X, y, task = load_dataset(DATASET_NAME)
    if task == 'classification':
        desc = f"{len(np.unique(y))} classes"
    else:
        desc = f"target range [{y.min():.2f}, {y.max():.2f}]"
    print(f"Loaded: shape={X.shape}, task={task}, {desc}")

    # 2. Run experiment (checkpoint/resume)
    df, task, primary_metric, secondary_metric = run_experiment(X, y, task)

    # 3. Statistical analysis
    stat_summary = run_statistical_analysis(df, task, primary_metric)

    # 4. Visualizations
    print(f"\nGenerating plots → {PLOTS_DIR}")
    plot_main_effects(df, primary_metric, stat_summary)
    plot_boxplots(df, primary_metric)
    plot_interaction_heatmaps(df, primary_metric)
    plot_sensitivity_bar(stat_summary)
    plot_top_configs(df, primary_metric, secondary_metric)

    # 5. Export best config + summary
    export_best_config(df, primary_metric, secondary_metric, task)

    print(f"\nAll done! Results saved to: {RESULTS_DIR}")                                                                                                                                                                                                                                                                                                                                                                          