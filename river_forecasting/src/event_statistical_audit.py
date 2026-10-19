"""
river_forecasting/src/event_statistical_audit.py
=================================================
Independent, rigorous statistical audit and inference module for
evaluating forecast performance differences between models on event series.

Implements:
1. Loss differential calculation (absolute error and squared error)
2. Classic Diebold-Mariano (1995) with horizon-dependent rectangular truncation
3. Newey-West / Bartlett long-run covariance estimation
4. Harvey-Leybourne-Newbold (HLN 1997) small-sample correction with Student-t df
5. Moving-Block Bootstrap across multiple block lengths (b in [1..6])
6. Multiple testing corrections (Bonferroni, Holm-Bonferroni) across lead times
7. Effective sample size estimation accounting for temporal autocorrelation
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

def compute_loss_differential(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_candidate: np.ndarray,
    loss: str = "mae"
) -> Dict[str, Union[float, np.ndarray, int]]:
    """
    Computes paired loss differential d_t = Loss(baseline) - Loss(candidate).
    Under this convention:
      d_t > 0: candidate model has smaller error than baseline (candidate wins)
      d_t < 0: baseline model has smaller error than candidate (baseline wins)
    
    Also provides delta_mae = Loss(candidate) - Loss(baseline) = -d_t.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_base = np.asarray(y_pred_baseline, dtype=float)
    y_cand = np.asarray(y_pred_candidate, dtype=float)

    if loss == "mae":
        loss_base = np.abs(y_base - y_true)
        loss_cand = np.abs(y_cand - y_true)
    elif loss == "mse":
        loss_base = (y_base - y_true) ** 2
        loss_cand = (y_cand - y_true) ** 2
    else:
        raise ValueError(f"Unsupported loss function: {loss}")

    # d_t > 0 means candidate (multimodal) is better
    d = loss_base - loss_cand
    delta = loss_cand - loss_base

    mae_base = float(np.mean(loss_base))
    mae_cand = float(np.mean(loss_cand))
    mean_d = float(np.mean(d))
    delta_mae = float(np.mean(delta))
    pct_gain = (delta_mae / mae_base) * 100.0 if mae_base > 0 else 0.0

    wins_cand = int(np.sum(loss_cand < loss_base))
    wins_base = int(np.sum(loss_base < loss_cand))
    ties = int(np.sum(loss_base == loss_cand))
    n_total = len(y_true)

    return {
        "d_series": d,
        "delta_series": delta,
        "loss_baseline": loss_base,
        "loss_candidate": loss_cand,
        "mae_baseline": mae_base,
        "mae_candidate": mae_cand,
        "mean_d": mean_d,
        "delta_mae": delta_mae,
        "pct_gain": pct_gain,
        "candidate_wins": wins_cand,
        "candidate_win_pct": (wins_cand / n_total) * 100.0,
        "baseline_wins": wins_base,
        "baseline_win_pct": (wins_base / n_total) * 100.0,
        "ties": ties,
        "n_samples": n_total,
    }


def compute_sample_autocovariances(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Computes sample autocovariances gamma(0..max_lag) with denominator T."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_centered = x - np.mean(x)
    gamma = np.zeros(max_lag + 1, dtype=float)
    for k in range(max_lag + 1):
        if k == 0:
            gamma[0] = np.mean(x_centered ** 2)
        else:
            gamma[k] = np.sum(x_centered[k:] * x_centered[:-k]) / n
    return gamma


def compute_diebold_mariano(
    d: np.ndarray,
    h: int = 1,
    kernel: str = "rectangular",
    bandwidth: Optional[int] = None,
    use_hln: bool = True
) -> Dict[str, float]:
    """
    Computes the Diebold-Mariano (1995) test statistic for loss differential d.
    
    Parameters:
      d: array of loss differentials d_t = Loss(baseline) - Loss(candidate)
      h: forecast horizon (steps ahead). Under optimal forecast theory,
         errors are at most (h - 1) correlated.
      kernel: 'rectangular' (classic DM) or 'bartlett' (Newey-West guaranteed PSD)
      bandwidth: max lag for autocovariances. Defaults to (h - 1) for rectangular,
                 or max(h - 1, 1) for bartlett if not specified.
      use_hln: whether to apply Harvey, Leybourne, and Newbold (1997) small-sample
               correction factor and use Student-t distribution with (T - 1) df.
    """
    d = np.asarray(d, dtype=float)
    t_len = len(d)
    d_bar = float(np.mean(d))

    if bandwidth is None:
        if kernel == "rectangular":
            bandwidth = max(0, h - 1)
        elif kernel == "bartlett":
            bandwidth = max(1, h - 1)
        else:
            bandwidth = max(0, h - 1)

    gamma = compute_sample_autocovariances(d, max_lag=bandwidth)

    if kernel == "rectangular":
        if bandwidth == 0:
            v_long_run = gamma[0]
        else:
            v_long_run = gamma[0] + 2.0 * float(np.sum(gamma[1 : bandwidth + 1]))
    elif kernel == "bartlett":
        if bandwidth == 0:
            v_long_run = gamma[0]
        else:
            weights = np.array([1.0 - k / (bandwidth + 1) for k in range(1, bandwidth + 1)])
            v_long_run = gamma[0] + 2.0 * float(np.sum(weights * gamma[1 : bandwidth + 1]))
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    # Guard against non-positive variance estimate
    v_long_run = max(v_long_run, 1e-12)
    se_d_bar = np.sqrt(v_long_run / t_len)
    dm_stat = d_bar / se_d_bar

    # HLN small sample correction (Harvey, Leybourne, Newbold 1997)
    if use_hln:
        hln_factor = np.sqrt(max(1e-12, (t_len + 1 - 2 * h + (h * (h - 1)) / t_len) / t_len))
        stat_final = dm_stat * hln_factor
        df = t_len - 1
        p_two_sided = float(2.0 * (1.0 - stats.t.cdf(np.abs(stat_final), df=df)))
        p_one_sided = float(1.0 - stats.t.cdf(stat_final, df=df))
    else:
        hln_factor = 1.0
        stat_final = dm_stat
        df = None
        p_two_sided = float(2.0 * (1.0 - stats.norm.cdf(np.abs(stat_final))))
        p_one_sided = float(1.0 - stats.norm.cdf(stat_final))

    rho_1 = float(gamma[1] / gamma[0]) if len(gamma) > 1 and gamma[0] > 0 else 0.0

    return {
        "mean_d": d_bar,
        "long_run_var": float(v_long_run),
        "se_d_bar": float(se_d_bar),
        "dm_stat_raw": float(dm_stat),
        "hln_factor": float(hln_factor),
        "dm_stat_hln": float(stat_final),
        "df": df,
        "p_two_sided": p_two_sided,
        "p_one_sided": p_one_sided,
        "bandwidth": bandwidth,
        "kernel": kernel,
        "rho_1": rho_1,
    }


def compute_moving_block_bootstrap(
    delta_series: np.ndarray,
    block_lengths: List[int] = [1, 2, 3, 4, 5, 6],
    n_boot: int = 5000,
    seed: int = 42
) -> Dict[int, Dict[str, float]]:
    """
    Performs moving-block bootstrap (Kunsch 1989) on delta_series (e_cand - e_base).
    Evaluates across multiple block lengths to test sensitivity to serial correlation.
    
    Returns standard error, 95% percentile CI, 95% basic CI, and empirical p-value.
    """
    delta_series = np.asarray(delta_series, dtype=float)
    t_len = len(delta_series)
    orig_mean = float(np.mean(delta_series))
    rng = np.random.default_rng(seed)

    results = {}
    for b in block_lengths:
        num_blocks = t_len - b + 1
        if num_blocks <= 0:
            continue

        blocks = [delta_series[i : i + b] for i in range(num_blocks)]
        blocks_needed = int(np.ceil(t_len / b))

        boot_means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sampled_idx = rng.integers(0, num_blocks, size=blocks_needed)
            sampled_series = np.concatenate([blocks[idx] for idx in sampled_idx])[:t_len]
            boot_means[i] = np.mean(sampled_series)

        se_boot = float(np.std(boot_means, ddof=1))
        ci_perc_low, ci_perc_high = [float(q) for q in np.percentile(boot_means, [2.5, 97.5])]
        # Basic bootstrap CI: [2*orig - q_97.5, 2*orig - q_2.5]
        ci_basic_low = float(2.0 * orig_mean - ci_perc_high)
        ci_basic_high = float(2.0 * orig_mean - ci_perc_low)
        # Probability of no improvement (delta >= 0)
        p_no_improv = float(np.mean(boot_means >= 0.0))

        results[b] = {
            "block_length": b,
            "bootstrap_se": se_boot,
            "ci_percentile_95": [ci_perc_low, ci_perc_high],
            "ci_basic_95": [ci_basic_low, ci_basic_high],
            "p_value_empirical": p_no_improv,
        }

    return results


def compute_effective_sample_size(series: np.ndarray) -> Dict[str, float]:
    """
    Estimates effective sample size accounting for AR(1) autocorrelation:
      N_eff = T * (1 - rho_1) / (1 + rho_1)
    """
    series = np.asarray(series, dtype=float)
    t_len = len(series)
    gamma = compute_sample_autocovariances(series, max_lag=1)
    rho_1 = float(gamma[1] / gamma[0]) if gamma[0] > 0 else 0.0
    rho_1_clipped = float(np.clip(rho_1, -0.99, 0.99))
    n_eff = float(t_len * (1.0 - rho_1_clipped) / (1.0 + rho_1_clipped))

    return {
        "n_nominal": t_len,
        "rho_1": rho_1,
        "n_eff": round(n_eff, 2),
    }


def compute_multiple_testing_adjustments(
    p_values: Dict[str, float],
    alpha: float = 0.05
) -> Dict[str, Dict[str, Union[float, bool]]]:
    """
    Applies Bonferroni and Holm-Bonferroni corrections across multiple horizons.
    """
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_items)

    results = {}
    # Bonferroni threshold
    bonf_alpha = alpha / m

    for rank, (name, p_val) in enumerate(sorted_items):
        # Holm threshold: alpha / (m - rank)
        holm_alpha = alpha / (m - rank)
        p_bonf_adj = min(1.0, p_val * m)
        p_holm_adj = min(1.0, p_val * (m - rank))

        results[name] = {
            "p_raw": p_val,
            "rank": rank + 1,
            "bonferroni_thresh": bonf_alpha,
            "bonferroni_p_adj": p_bonf_adj,
            "bonferroni_significant": bool(p_val <= bonf_alpha),
            "holm_thresh": holm_alpha,
            "holm_p_adj": p_holm_adj,
            "holm_significant": bool(p_val <= holm_alpha),
        }

    return results
