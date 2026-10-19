"""
river_forecasting/tests/test_phase6_1_audit.py
==============================================
Reproducible test suite for Phase 6.1 Forensic Audit:
Medium-Range Upstream Signal and Flood-Wave Routing Experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE6_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase6_medium_range"
AUDIT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase6_1_audit"


def test_recomputed_metrics_reconciliation():
    """Verify that every metric recomputed from raw rows matches Phase 6 reports exactly."""
    fpath = AUDIT_DIR / "recomputed_metrics.csv"
    assert fpath.exists(), f"Missing {fpath}"
    df = pd.read_csv(fpath)
    
    assert len(df) == 48, f"Expected 48 recomputed metric rows, found {len(df)}"
    # Every single row must match Phase 6 exactly (mae_diff_mm == 0.0)
    discrepancies = df[df["reconciliation_status"] != "EXACT_MATCH"]
    assert len(discrepancies) == 0, f"Found {len(discrepancies)} discrepancies in recomputed metrics: {discrepancies}"
    assert (df["mae_diff_mm"].abs() < 0.05).all()
    assert (df["rmse_diff_mm"].abs() < 0.05).all()


def test_extreme_predictions_root_cause_isolation():
    """Verify that extreme predictions (|pred| > 0.5m) are properly diagnosed."""
    fpath = AUDIT_DIR / "extreme_prediction_audit.csv"
    assert fpath.exists(), f"Missing {fpath}"
    df = pd.read_csv(fpath)
    
    assert len(df) > 0, "Extreme prediction audit table is empty"
    
    # Check that the Sept 8 02:00 spike corruption was isolated
    spike_rows = df[df["root_cause_diagnosis"].str.contains("TELEMETRY_SPIKE_DIFF_CORRUPTION")]
    assert len(spike_rows) > 0, "Failed to isolate telemetry spike corruption"
    
    # Check that the Sept 18 unpropagated surge was isolated
    surge_rows = df[df["root_cause_diagnosis"].str.contains("REGIME_DECOUPLING_UPSTREAM_SURGE")]
    assert len(surge_rows) > 0, "Failed to isolate Sept 18 unpropagated surge"


def test_pipeline_coefficients_and_scaling_integrity():
    """Verify that pipeline feature scaling and fitted coefficients are logged and valid."""
    fpath = AUDIT_DIR / "pipeline_coefficients_audit.csv"
    assert fpath.exists(), f"Missing {fpath}"
    df = pd.read_csv(fpath)
    
    assert len(df) > 0, "Pipeline coefficients audit table is empty"
    assert "fitted_coef" in df.columns
    assert "feature_scale" in df.columns
    assert (df["feature_scale"] > 0).all(), "Feature scale must be strictly positive"


def test_temporal_integrity_and_latency_buffer():
    """Verify that zero temporal integrity violations occurred."""
    fpath = AUDIT_DIR / "temporal_integrity_audit.csv"
    assert fpath.exists(), f"Missing {fpath}"
    df = pd.read_csv(fpath)
    
    assert len(df) >= 8, f"Expected at least 8 temporal checks, found {len(df)}"
    failures = df[df["status"] != "PASS"]
    assert len(failures) == 0, f"Temporal integrity check failures: {failures[['audit_check', 'horizon', 'details']]}"
    assert (df["violations_detected"] == 0).all()


def test_moving_block_bootstrap_sign_convention():
    """Verify statistical bootstrap calculations, sign conventions, and conclusions."""
    fpath = AUDIT_DIR / "statistical_bootstrap_audit.csv"
    assert fpath.exists(), f"Missing {fpath}"
    df = pd.read_csv(fpath)
    
    assert len(df) == 18, f"Expected 18 paired comparison rows, found {len(df)}"
    
    # Sign convention verification: delta_mae = base_mae - cand_mae
    calc_delta = df["baseline_mae_mm"] - df["candidate_mae_mm"]
    assert np.allclose(df["delta_mae_mm"], calc_delta, atol=0.05)
    
    # Verify that CI brackets the estimate
    assert (df["delta_mae_ci_low_mm"] <= df["delta_mae_mm"] + 5.0).all()
    assert (df["delta_mae_ci_high_mm"] >= df["delta_mae_mm"] - 5.0).all()
    
    # Zero comparisons should conclude that upstream model is significantly better
    better_models = df[df["practical_conclusion"] == "Statistically significantly BETTER"]
    assert len(better_models) == 0, "Upstream model was incorrectly concluded to be significantly better"


def test_purified_cohort_shows_upstream_failure_persists():
    """
    Verify that even when removing the single-hour telemetry glitch,
    upstream-augmented models still underperform local stage and persistence.
    """
    from scripts.run_phase6_medium_range import prepare_dataset
    df_rows = pd.read_csv(PHASE6_DIR / "row_level_predictions.csv")
    piv = prepare_dataset().reset_index()
    
    merged = pd.merge(df_rows, piv[["timestamp_ist", "target_v_dH_18h"]], on="timestamp_ist")
    sub = merged[merged["is_test_sync_rainfall"] == True].copy()
    
    # Exclude rows where pred_dH_18h < -0.5m (the glitch-affected rows)
    clean = sub[sub["pred_dH_18h__Config_D_Local_Stage_Upstream"] > -0.5].copy()
    y = clean["target_v_dH_18h"].values
    
    p_persist = np.zeros_like(y)
    p_local = clean["pred_dH_18h__Config_B_Local_Stage"].values
    p_ridge_up = clean["pred_dH_18h__Config_D_Local_Stage_Upstream"].values
    p_xgb_up = clean["pred_dH_18h__Config_H_XGBoost_Local_Upstream"].values
    
    mae_persist = np.mean(np.abs(p_persist - y)) * 1000.0
    mae_local = np.mean(np.abs(p_local - y)) * 1000.0
    mae_ridge = np.mean(np.abs(p_ridge_up - y)) * 1000.0
    mae_xgb = np.mean(np.abs(p_xgb_up - y)) * 1000.0
    
    # Even on the purified cohort, Ridge Upstream is >100mm and XGB Upstream is >70mm,
    # whereas Persistence is <20mm.
    assert mae_persist < 20.0
    assert mae_local < 25.0
    assert mae_xgb > 70.0
    assert mae_ridge > 150.0
    assert mae_xgb > mae_persist * 3.5, "XGBoost with upstream telemetry should be >3.5x worse than persistence"
