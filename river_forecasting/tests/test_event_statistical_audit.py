"""
river_forecasting/tests/test_event_statistical_audit.py
========================================================
Regression test suite for the independent statistical audit module
(Diebold-Mariano, HLN correction, moving block bootstrap, multiple testing,
and effective sample size estimation).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.event_statistical_audit import (
    compute_loss_differential,
    compute_diebold_mariano,
    compute_sample_autocovariances,
    compute_moving_block_bootstrap,
    compute_effective_sample_size,
    compute_multiple_testing_adjustments,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_CSV = REPO_ROOT / "data" / "historical" / "event_july2026_predictions.csv"


def test_loss_differential_sign_and_properties():
    """Verify sign convention and win rate calculation."""
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_base = np.array([12.0, 12.0, 12.0, 12.0])  # error = 2.0
    y_cand = np.array([11.0, 11.0, 11.0, 11.0])  # error = 1.0 (better)

    res = compute_loss_differential(y_true, y_base, y_cand)
    # d_t = |e_base| - |e_cand| = 2 - 1 = +1.0
    assert np.allclose(res["d_series"], np.array([1.0, 1.0, 1.0, 1.0]))
    assert res["mean_d"] == 1.0
    # delta = |e_cand| - |e_base| = -1.0
    assert res["delta_mae"] == -1.0
    assert res["pct_gain"] == -50.0  # 50% reduction in MAE
    assert res["candidate_wins"] == 4
    assert res["candidate_win_pct"] == 100.0


def test_diebold_mariano_hln_small_sample_factor():
    """Verify exact Harvey-Leybourne-Newbold (1997) factor calculation."""
    # HLN factor = sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    t_len = 30
    for h, expected_factor in [
        (1, np.sqrt((30 + 1 - 2 + 0) / 30)),
        (2, np.sqrt((30 + 1 - 4 + 2 / 30) / 30)),
        (3, np.sqrt((30 + 1 - 6 + 6 / 30) / 30)),
    ]:
        d = np.ones(t_len)
        res = compute_diebold_mariano(d, h=h, kernel="rectangular", use_hln=True)
        assert abs(res["hln_factor"] - expected_factor) < 1e-6
        assert res["df"] == t_len - 1


def test_diebold_mariano_horizon_dependent_lags():
    """Verify that rectangular DM uses bandwidth = h - 1."""
    d = np.random.RandomState(42).randn(30)
    for h in [1, 2, 3]:
        res = compute_diebold_mariano(d, h=h, kernel="rectangular", use_hln=True)
        assert res["bandwidth"] == h - 1


def test_moving_block_bootstrap_se_expands_with_block_length():
    """Verify that moving-block bootstrap SE expands with block length under positive autocorrelation."""
    df = pd.read_csv(PREDICTIONS_CSV)
    y_true = df["y_true_1h"].values
    y_st = df["pred_stage_ridge_1h"].values
    y_mm = df["pred_mm_ridge_1h"].values

    diff_res = compute_loss_differential(y_true, y_st, y_mm)
    delta = diff_res["delta_series"]

    boot_res = compute_moving_block_bootstrap(
        delta, block_lengths=[1, 2, 4, 6], n_boot=2000, seed=42
    )

    # Standard error should increase as block length accounts for serial dependence
    se_b1 = boot_res[1]["bootstrap_se"]
    se_b2 = boot_res[2]["bootstrap_se"]
    se_b4 = boot_res[4]["bootstrap_se"]
    se_b6 = boot_res[6]["bootstrap_se"]

    assert se_b1 < se_b2 < se_b4 <= se_b6 * 1.05
    # 95% CI upper bound should strictly remain below zero (negative delta MAE)
    for b in [1, 2, 4, 6]:
        assert boot_res[b]["ci_percentile_95"][1] < 0.0
        assert boot_res[b]["p_value_empirical"] < 0.01


def test_effective_sample_size_accounts_for_ar1():
    """Verify N_eff calculation for positively correlated series."""
    df = pd.read_csv(PREDICTIONS_CSV)
    y_true = df["y_true_1h"].values
    y_st = df["pred_stage_ridge_1h"].values
    y_mm = df["pred_mm_ridge_1h"].values
    d = compute_loss_differential(y_true, y_st, y_mm)["d_series"]

    ess = compute_effective_sample_size(d)
    assert ess["n_nominal"] == 30
    assert ess["rho_1"] > 0.6  # Highly autocorrelated
    assert ess["n_eff"] < 10.0  # Effective independent samples drastically reduced


def test_multiple_testing_adjustment():
    """Verify Bonferroni and Holm-Bonferroni thresholds and ordering."""
    p_vals = {"h1": 0.0001, "h2": 0.010, "h3": 0.040}
    adj = compute_multiple_testing_adjustments(p_vals, alpha=0.05)

    assert adj["h1"]["bonferroni_significant"] is True
    assert adj["h2"]["bonferroni_significant"] is True
    # 0.040 * 3 = 0.120 > 0.05 -> NOT significant under Bonferroni
    assert adj["h3"]["bonferroni_significant"] is False
    # But 0.040 <= 0.05 / 1 -> significant under Holm step-down
    assert adj["h3"]["holm_significant"] is True


def test_event_predictions_full_audit_validation():
    """Verify all statistical properties against actual saved event predictions."""
    assert PREDICTIONS_CSV.exists()
    df = pd.read_csv(PREDICTIONS_CSV)
    assert len(df) == 30

    for h_str, h in [("1h", 1), ("2h", 2), ("3h", 3)]:
        y_true = df[f"y_true_{h_str}"].values
        y_pers = df[f"pred_pers_{h_str}"].values
        y_st = df[f"pred_stage_ridge_{h_str}"].values
        y_mm = df[f"pred_mm_ridge_{h_str}"].values

        diff_mm_st = compute_loss_differential(y_true, y_st, y_mm)
        diff_p_mm = compute_loss_differential(y_true, y_mm, y_pers)

        # 1. Multimodal Ridge beats Stage Ridge
        assert diff_mm_st["delta_mae"] < 0
        assert diff_mm_st["candidate_wins"] >= 25  # At least 83% wins

        # 2. Persistence beats Multimodal Ridge
        assert diff_p_mm["delta_mae"] < 0
        assert diff_p_mm["candidate_wins"] >= 22  # Persistence beats MM in >= 73% cases
        assert diff_p_mm["mae_candidate"] < diff_mm_st["mae_candidate"]
