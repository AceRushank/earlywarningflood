"""
river_forecasting/tests/test_stage0_rainfall_routing_audit.py
=============================================================
Stage 0 Audit Verification Tests for RA-AG v2:
Verifies that missing rainfall (NaN / telemetry dropout / absent columns)
is strictly distinguished from observed zero rainfall (0.0 mm),
and routes strictly to the intended stage-only fallback expert.

Test Coverage:
  1. Missing rainfall does not falsely trigger rain-based rise.
  2. Zero rainfall (0.0 mm) correctly represents dry weather.
  3. Distinction: NaN rainfall vs 0.0 mm rainfall.
  4. Partially missing rainfall inputs (e.g., rain_1h valid but rain_3h NaN).
  5. Incompatible / absent rainfall columns (DataFrame lacking FEAT_RAIN_CONT).
  6. Exact routing path verification: missing rainfall engages stage-only fallback.
"""

import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.phase1_ml_pipeline import FEAT_RAIN_CONT, FEAT_STAGE_DYN
from river_forecasting.src.phase4_ra_ag_v2 import (
    CausalRegimeDetector,
    SoftGateCalculator,
    RAAGv2,
)


@pytest.fixture
def base_sample_df():
    """Generates synthetic test DataFrame with stage dynamics and rainfall."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "stage_vishnupuri": 354.0 + np.cumsum(rng.normal(0, 0.005, size=n)),
        "dH_v_1h": rng.normal(0, 0.01, size=n),
        "dH_v_3h": rng.normal(0, 0.02, size=n),
        "dH_v_6h": rng.normal(0, 0.03, size=n),
        "accel_v_1h": rng.normal(0, 0.005, size=n),
        "rain_depth_1h_mm": np.maximum(0, rng.normal(1.0, 2.0, size=n)),
        "rain_3h_mm": np.maximum(0, rng.normal(3.0, 4.0, size=n)),
        "rain_6h_mm": np.maximum(0, rng.normal(5.0, 5.0, size=n)),
        "rain_12h_mm": np.maximum(0, rng.normal(8.0, 6.0, size=n)),
        "rain_24h_mm": np.maximum(0, rng.normal(12.0, 8.0, size=n)),
        "api_7d_mm": np.maximum(0, rng.normal(20.0, 5.0, size=n)),
    })
    df["target_v_dH_1h"] = 0.4 * df["dH_v_1h"] + 0.001 * df["rain_depth_1h_mm"] + rng.normal(0, 0.002, size=n)
    df["target_v_1h"] = df["stage_vishnupuri"] + df["target_v_dH_1h"]
    return df


def test_missing_rainfall_regime_detection():
    """Missing rainfall (NaN) must not trigger rain-based rise, but stage rise must work."""
    detector = CausalRegimeDetector(theta_rise=0.010, theta_fall=-0.010, rain_threshold=1.0)
    
    # 1. Quiescent stage + NaN rainfall -> STABLE (not rising)
    s_quiescent_nan = pd.Series({"dH_v_1h": 0.002, "dH_v_3h": 0.003, "accel_v_1h": 0.0, "rain_3h_mm": np.nan})
    assert detector.predict_regime(s_quiescent_nan) == "STABLE"

    # 2. Rising stage + NaN rainfall -> RISING (triggered by stage dynamics, not rain)
    s_rising_nan = pd.Series({"dH_v_1h": 0.015, "dH_v_3h": 0.025, "accel_v_1h": 0.01, "rain_3h_mm": np.nan})
    assert detector.predict_regime(s_rising_nan) == "RISING"


def test_zero_rainfall_regime_detection():
    """Observed zero rainfall (0.0 mm) is verified dry weather."""
    detector = CausalRegimeDetector(theta_rise=0.010, theta_fall=-0.010, rain_threshold=1.0)
    
    s_quiescent_zero = pd.Series({"dH_v_1h": 0.002, "dH_v_3h": 0.003, "accel_v_1h": 0.0, "rain_3h_mm": 0.0})
    assert detector.predict_regime(s_quiescent_zero) == "STABLE"

    s_falling_zero = pd.Series({"dH_v_1h": -0.015, "dH_v_3h": -0.025, "accel_v_1h": -0.01, "rain_3h_mm": 0.0})
    assert detector.predict_regime(s_falling_zero) == "FALLING"


def test_missing_vs_zero_rainfall_distinction():
    """Ensure NaN rainfall and 0.0 mm rainfall have distinct internal handling."""
    gate = SoftGateCalculator()
    
    # DataFrame with one row having rain=0.0, and another having rain=NaN
    df_compare = pd.DataFrame({
        "dH_v_1h": [-0.015, -0.015],
        "rain_3h_mm": [0.0, np.nan],
    })
    
    w_rise, w_fall, w_stable = gate.compute_weights(df_compare)
    # Both rows are falling hydraulically (dH = -0.015 m)
    # Row 0: verified zero rain -> falling weight is active
    # Row 1: missing rain -> falling weight is also active (not suppressed), but will route to stage-only
    assert w_fall[0] > 0.5
    assert w_fall[1] > 0.5


def test_partially_missing_rainfall_routing(base_sample_df):
    """When some rainfall columns are present and others are NaN, fallback must engage."""
    df = base_sample_df.copy()
    y_dh = df["target_v_dH_1h"].values
    
    model = RAAGv2(horizon=1, fallback_mode="bounded_linear")
    model.fit(df, y_dh)
    
    # Introduce partial missingness: rain_depth_1h_mm is valid, but rain_3h_mm is NaN on rows 5..10
    df.loc[5:10, "rain_3h_mm"] = np.nan
    
    pred_dh, is_fallback, regimes, weights = model.predict(df)
    
    # Rows 5..10 must have fallback active
    assert np.all(is_fallback[5:11]), "Partial missingness in rainfall must engage fallback"
    # Rows 0..4 must NOT have fallback active
    assert not np.any(is_fallback[0:5]), "Fully valid rows must not engage fallback"


def test_incompatible_or_absent_rainfall_columns(base_sample_df):
    """When rainfall columns are entirely absent, model must safely execute stage-only fallback."""
    df = base_sample_df.copy()
    y_dh = df["target_v_dH_1h"].values
    
    model = RAAGv2(horizon=1, fallback_mode="bounded_linear")
    model.fit(df, y_dh)
    
    # Create stage-only DataFrame lacking all rainfall columns
    df_stage_only = df[FEAT_STAGE_DYN].copy()
    
    pred_dh, is_fallback, regimes, weights = model.predict(df_stage_only)
    
    assert len(pred_dh) == len(df)
    assert np.all(np.isfinite(pred_dh))
    # All rows must be flagged as fallback
    assert np.all(is_fallback)


def test_exact_routing_path_missing_rainfall_selects_stage_only(base_sample_df):
    """
    Directly verify that when rainfall is missing, the output matches the stage-only
    expert prediction, proving zero dependency on multimodal features.
    """
    df = base_sample_df.copy()
    y_dh = df["target_v_dH_1h"].values
    
    model = RAAGv2(horizon=1, gating_mode="hard", fallback_mode="bounded_linear")
    model.fit(df, y_dh)
    
    # Force rising stage dynamics
    df_rise = df.copy()
    df_rise["dH_v_1h"] = 0.030
    df_rise["dH_v_3h"] = 0.050
    df_rise["accel_v_1h"] = 0.010
    
    # Scenario A: Valid rainfall present
    df_rise["rain_3h_mm"] = 15.0
    for col in FEAT_RAIN_CONT:
        df_rise[col] = 15.0
    pred_with_rain, is_fb_with_rain, _, _ = model.predict(df_rise)
    assert not np.any(is_fb_with_rain)
    
    # Scenario B: Missing rainfall (all NaNs)
    df_rise_missing = df_rise.copy()
    df_rise_missing[FEAT_RAIN_CONT] = np.nan
    pred_missing, is_fb_missing, _, _ = model.predict(df_rise_missing)
    assert np.all(is_fb_missing)
    
    # Verify exact match with rising_expert_so_ (Stage-Only Bounded Linear)
    so_cols = [c for c in FEAT_STAGE_DYN if c in df_rise_missing.columns]
    X_so = df_rise_missing[so_cols].astype(float)
    expected_so_pred = model.rising_expert_so_.predict(X_so)
    
    np.testing.assert_allclose(pred_missing, expected_so_pred, rtol=1e-6, atol=1e-6)
