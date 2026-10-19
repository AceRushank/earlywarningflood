"""
river_forecasting/tests/test_phase5_upstream_telemetry.py
=========================================================
Unit and integration test suite for Phase 5:
Upstream Telemetry Integration and Out-of-Sample Surge Forecasting.

Tests cover:
  1. Strict timestamp alignment and operational latency buffer (zero lookahead).
  2. Data leakage safeguards: zero future stage or rainfall leakage into feature sets.
  3. Missing upstream telemetry handling and fallback engagement.
  4. Rainfall fallback routing under upstream integration (Stage 0 audit compliance).
  5. Deliverables existence and schema verification.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.phase1_ml_pipeline import (
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_MULTIMODAL,
)
from river_forecasting.src.phase5_upstream_telemetry import (
    FEAT_UPSTREAM,
    FEAT_STAGE_UPSTREAM,
    FEAT_ALL,
    UpstreamAwareRAAGv2,
    Phase5Hybrid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase5_upstream_telemetry"


@pytest.fixture
def synthetic_phase5_df():
    """Generates synthetic DataFrame with Vishnupuri, Rainfall, and Upstream telemetry."""
    rng = np.random.default_rng(105)
    n = 120
    dts = pd.date_range("2026-09-01 00:00:00", periods=n, freq="1h")
    
    # Upstream stage at Mudgal (~382 m MSL)
    upstream_raw = 382.0 + np.cumsum(rng.normal(0, 0.02, size=n))
    
    # Local stage at Vishnupuri (~354 m MSL)
    local_raw = 354.0 + np.cumsum(rng.normal(0, 0.01, size=n))
    
    df = pd.DataFrame({
        "dt": dts,
        "stage_vishnupuri": local_raw,
        "stage_mudgal": upstream_raw,
        "rain_depth_1h_mm": np.maximum(0, rng.normal(0.5, 2.0, size=n)),
        "rain_3h_mm": np.maximum(0, rng.normal(1.5, 3.0, size=n)),
        "rain_6h_mm": np.maximum(0, rng.normal(3.0, 5.0, size=n)),
        "rain_12h_mm": np.maximum(0, rng.normal(6.0, 8.0, size=n)),
        "rain_24h_mm": np.maximum(0, rng.normal(12.0, 10.0, size=n)),
        "api_7d_mm": np.maximum(0, rng.normal(20.0, 5.0, size=n)),
    })
    
    # Causal lag features
    df["dH_v_1h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(1)
    df["dH_v_3h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(3)
    df["dH_v_6h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(6)
    df["accel_v_1h"] = df["dH_v_1h"] - df["dH_v_1h"].shift(1)
    
    # Upstream features strictly lagged by >= 1 hour (operational latency buffer)
    df["stage_m_lag1"] = df["stage_mudgal"].shift(1)
    df["stage_m_lag2"] = df["stage_mudgal"].shift(2)
    df["stage_m_lag4"] = df["stage_mudgal"].shift(4)
    df["stage_m_lag7"] = df["stage_mudgal"].shift(7)
    df["stage_m_lag25"] = df["stage_mudgal"].shift(25)
    
    df["dH_m_1h"] = df["stage_m_lag1"] - df["stage_m_lag2"]
    df["dH_m_3h"] = df["stage_m_lag1"] - df["stage_m_lag4"]
    df["dH_m_6h"] = df["stage_m_lag1"] - df["stage_m_lag7"]
    df["dH_m_24h"] = df["stage_m_lag1"] - df["stage_m_lag25"]
    
    # Future targets
    df["target_v_dH_1h"] = 0.5 * df["dH_v_1h"] + 0.001 * df["rain_depth_1h_mm"] + rng.normal(0, 0.002, size=n)
    df["target_v_1h"] = df["stage_vishnupuri"] + df["target_v_dH_1h"]
    
    return df.dropna().reset_index(drop=True)


def test_upstream_telemetry_latency_and_alignment(synthetic_phase5_df):
    """Verify that all upstream features at forecast origin t are strictly lagged by >= 1 hour."""
    df = synthetic_phase5_df
    for feat in FEAT_UPSTREAM:
        assert feat in df.columns, f"Feature {feat} must exist in dataframe"
        
    # Verify that stage_m_lag1 corresponds to stage_mudgal at t-1
    assert np.allclose(df["stage_m_lag1"].iloc[1:], df["stage_mudgal"].iloc[:-1], atol=1e-6)
    
    # Verify that dH_m_1h is strictly computed from t-1 and t-2 observations
    expected_dh1 = df["stage_m_lag1"] - df["stage_m_lag2"]
    np.testing.assert_allclose(df["dH_m_1h"], expected_dh1, atol=1e-6)


def test_data_leakage_safeguards():
    """Verify zero lookahead: targets must never appear in any feature set."""
    target_names = ["target_v_1h", "target_v_2h", "target_v_3h", "target_v_dH_1h", "target_v_dH_2h", "target_v_dH_3h"]
    for t_name in target_names:
        assert t_name not in FEAT_STAGE_DYN
        assert t_name not in FEAT_MULTIMODAL
        assert t_name not in FEAT_UPSTREAM
        assert t_name not in FEAT_STAGE_UPSTREAM
        assert t_name not in FEAT_ALL


def test_missing_upstream_telemetry_handling(synthetic_phase5_df):
    """Verify that when upstream telemetry drops out, model routes cleanly to stage-only fallback."""
    df = synthetic_phase5_df.copy()
    y_dh = df["target_v_dH_1h"].values
    
    model = UpstreamAwareRAAGv2(horizon=1, use_upstream=True)
    model.fit(df, y_dh)
    
    # Mask upstream features on rows 10..20
    df_masked = df.copy()
    df_masked.loc[10:20, FEAT_UPSTREAM] = np.nan
    
    preds, is_fb, regimes, _ = model.predict(df_masked)
    assert len(preds) == len(df)
    assert np.all(np.isfinite(preds))
    # Rows with missing upstream telemetry must have fallback active
    assert np.all(is_fb[10:21])
    assert not np.any(is_fb[0:10])


def test_rainfall_fallback_routing_with_upstream(synthetic_phase5_df):
    """Verify that when rainfall drops out, model activates fallback without silent zero-fill."""
    df = synthetic_phase5_df.copy()
    y_dh = df["target_v_dH_1h"].values
    
    hybrid = Phase5Hybrid(horizon=1, use_upstream=True)
    hybrid.fit(df, y_dh)
    
    # Mask rainfall features on rows 15..25
    df_masked = df.copy()
    df_masked.loc[15:25, FEAT_RAIN_CONT] = np.nan
    
    preds, is_fb, _ = hybrid.predict(df_masked)
    assert len(preds) == len(df)
    assert np.all(np.isfinite(preds))
    # Fallback must be engaged on rows where rainfall is missing
    assert np.all(is_fb[15:26])


def test_phase5_deliverables_exist():
    """Verify that all mandatory deliverables exist and contain valid data."""
    required_files = [
        "data_source_audit.md",
        "station_inventory.csv",
        "timestamp_availability_audit.csv",
        "feature_ablation_results.csv",
        "horizon_metrics.csv",
        "event_metrics.csv",
        "outage_metrics.csv",
        "paired_comparisons.csv",
        "experiment_predictions.csv",
        "model_manifest.json",
    ]
    for fname in required_files:
        p = OUTPUT_DIR / fname
        assert p.exists(), f"Deliverable {fname} does not exist in {OUTPUT_DIR}"
        assert p.stat().st_size > 50, f"Deliverable {fname} appears empty or truncated"
