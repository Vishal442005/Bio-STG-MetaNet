"""
Statistical Significance Testing Module.
Performs Paired t-tests, Wilcoxon signed-rank tests, Cohen's d effect sizes,
and 95% Confidence Intervals across 10-fold cross-validation folds.
"""

import numpy as np
import scipy.stats as stats
import pandas as pd


def compute_cohens_d(x, y):
    """Computes Cohen's d effect size for paired samples."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(mean_diff / std_diff)


def compute_confidence_interval(data, confidence=0.95):
    """Computes mean and two-sided confidence interval."""
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return float(mean), float(mean - h), float(mean + h)


def compare_models_statistically(proposed_folds, baseline_folds, metric="accuracy"):
    """
    Compares proposed model vs a baseline model over 10 cross-validation folds.
    Returns: paired t-test p-value, Wilcoxon p-value, Cohen's d, and 95% CIs.
    """
    x = np.asarray(proposed_folds[metric])
    y = np.asarray(baseline_folds[metric])
    
    # Paired t-test
    t_stat, p_val_ttest = stats.ttest_rel(x, y)
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_val_wilcoxon = stats.wilcoxon(x, y)
    except Exception:
        w_stat, p_val_wilcoxon = 0.0, 1.0
        
    d = compute_cohens_d(x, y)
    
    mean_p, ci_low_p, ci_high_p = compute_confidence_interval(x)
    mean_b, ci_low_b, ci_high_b = compute_confidence_interval(y)
    
    return {
        "metric": metric,
        "proposed_mean": mean_p,
        "proposed_ci95": f"[{ci_low_p:.2f}, {ci_high_p:.2f}]",
        "baseline_mean": mean_b,
        "baseline_ci95": f"[{ci_low_b:.2f}, {ci_high_b:.2f}]",
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_val_ttest),
        "wilcoxon_stat": float(w_stat),
        "p_value_wilcoxon": float(p_val_wilcoxon),
        "cohens_d": d,
        "statistically_significant": bool(p_val_ttest < 0.05)
    }
