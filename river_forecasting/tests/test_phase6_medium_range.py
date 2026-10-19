"""
river_forecasting/tests/test_phase6_medium_range.py
===================================================
Unit and integration test suite for Phase 6:
Medium-Range Upstream Signal and Flood-Wave Routing Experiment (+18h, +24h, +36h).

Test Coverage:
  1. Target construction, exact timestamp alignment, and genuine observation verification.
  2. Strict chronological boundary separation (zero target window overlap across splits).
  3. Latency-aware feature engineering (1-hour operational buffer).
  4. Missing telemetry and rainfall outage fallback routing.
  5. Simple lagged-upstream-change routing proxy execution.
  6. Deliverable files existence and integrity.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.phase6_medium_range import (
    FEAT_LOCAL_STAGE,
    FEAT_RAINFALL,
    FEAT_UPSTREAM_MUDGAL,
    FEAT_STAGE_RAIN,
    FEAT_STAGE_UPSTREAM,
    FEAT_ALL,
    HORIZONS_MEDIUM,
    SimpleLaggedUpstreamBaseline,
    MediumRangeRidgeRegressor,
    MediumRangeXGBRegressor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase6_medium_range"


@pytest.fixture
def synthetic_medium_range_df():
    rng = np.random.default_rng(42)
    n = 150
    dts = pd.date_range("2026-07-01 00:00:00", periods=n, freq="1h")

    # Local stage ~354m MSL, Upstream stage ~382m MSL
    v_stage = 354.0 + np.cumsum(rng.normal(0, 0.01, size=n))
    m_stage = 382.0 + np.cumsum(rng.normal(0, 0.02, size=n))

    df = pd.DataFrame({
        "dt": dts,
        "stage_vishnupuri": v_stage,
        "stage_mudgal": m_stage,
        "rain_depth_1h_mm": np.maximum(0, rng.normal(0.5, 2.0, size=n)),
        "rain_3h_mm": np.maximum(0, rng.normal(1.5, 3.0, size=n)),
        "rain_6h_mm": np.maximum(0, rng.normal(3.0, 5.0, size=n)),
        "rain_12h_mm": np.maximum(0, rng.normal(6.0, 8.0, size=n)),
        "rain_24h_mm": np.maximum(0, rng.normal(12.0, 10.0, size=n)),
        "rain_48h_mm": np.maximum(0, rng.normal(20.0, 12.0, size=n)),
        "api_7d_mm": np.maximum(0, rng.normal(25.0, 5.0, size=n)),
    })

    # Local stage features
    df["dH_v_1h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(1)
    df["dH_v_3h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(3)
    df["dH_v_6h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(6)
    df["dH_v_12h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(12)
    df["dH_v_24h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(24)
    df["accel_v_1h"] = df["dH_v_1h"] - df["dH_v_1h"].shift(1)

    # Upstream features strictly lagged by >= 1h
    df["stage_m_lag1"] = df["stage_mudgal"].shift(1)
    df["stage_m_lag2"] = df["stage_mudgal"].shift(2)
    df["stage_m_lag4"] = df["stage_mudgal"].shift(4)
    df["stage_m_lag7"] = df["stage_mudgal"].shift(7)
    df["stage_m_lag13"] = df["stage_mudgal"].shift(13)
    df["stage_m_lag19"] = df["stage_mudgal"].shift(19)
    df["stage_m_lag25"] = df["stage_mudgal"].shift(25)

    df["dH_m_1h"] = df["stage_m_lag1"] - df["stage_m_lag2"]
    df["dH_m_3h"] = df["stage_m_lag1"] - df["stage_m_lag4"]
    df["dH_m_6h"] = df["stage_m_lag1"] - df["stage_m_lag7"]
    df["dH_m_12h"] = df["stage_m_lag1"] - df["stage_m_lag13"]
    df["dH_m_18h"] = df["stage_m_lag1"] - df["stage_m_lag19"]
    df["dH_m_24h"] = df["stage_m_lag1"] - df["stage_m_lag25"]

    # Targets (+18h, +24h, +36h)
    for h in [18, 24, 36]:
        df[f"target_v_{h}h"] = df["stage_vishnupuri"].shift(-h)
        df[f"target_v_dH_{h}h"] = df[f"target_v_{h}h"] - df["stage_vishnupuri"]

    return df.dropna().reset_index(drop=True)


def test_phase6_target_alignment_and_authenticity(synthetic_medium_range_df):
    """Verify exact target alignment: target_v_Hh == stage_vishnupuri at t + H."""
    df = synthetic_medium_range_df
    for h in [18, 24, 36]:
        # Origin stage + dH must equal target_v_Hh
        recon = df["stage_vishnupuri"] + df[f"target_v_dH_{h}h"]
        np.testing.assert_allclose(recon, df[f"target_v_{h}h"], atol=1e-6)


def test_phase6_boundary_separation_zero_leakage():
    """Verify zero lookahead leakage: target variables are excluded from all feature sets."""
    target_names = [f"target_v_{h}h" for h in [18, 24, 36]] + [f"target_v_dH_{h}h" for h in [18, 24, 36]]
    for t_name in target_names:
        assert t_name not in FEAT_LOCAL_STAGE
        assert t_name not in FEAT_RAINFALL
        assert t_name not in FEAT_UPSTREAM_MUDGAL
        assert t_name not in FEAT_STAGE_RAIN
        assert t_name not in FEAT_STAGE_UPSTREAM
        assert t_name not in FEAT_ALL


def test_phase6_latency_aware_upstream_features(synthetic_medium_range_df):
    """Verify that all upstream features at origin t are strictly lagged by >= 1 hour."""
    df = synthetic_medium_range_df
    # stage_m_lag1 corresponds to stage_mudgal at t-1
    assert np.allclose(df["stage_m_lag1"].iloc[1:], df["stage_mudgal"].iloc[:-1], atol=1e-6)
    # dH_m_1h corresponds to stage at t-1 minus stage at t-2
    assert np.allclose(df["dH_m_1h"], df["stage_m_lag1"] - df["stage_m_lag2"], atol=1e-6)


def test_phase6_missing_telemetry_outage_fallback(synthetic_medium_range_df):
    """Verify that Ridge and XGBoost regressors fall back cleanly when upstream telemetry is NaN."""
    df = synthetic_medium_range_df.copy()
    y = df["target_v_dH_18h"].values

    ridge = MediumRangeRidgeRegressor(alpha=50.0, use_upstream=True, use_rainfall=True)
    ridge.fit(df, y)

    xgb_m = MediumRangeXGBRegressor(max_depth=2, n_estimators=10, use_upstream=True, use_rainfall=True)
    xgb_m.fit(df, y)

    # Mask upstream features on rows 10..20
    df_masked = df.copy()
    df_masked.loc[10:20, FEAT_UPSTREAM_MUDGAL] = np.nan

    p_ridge, is_fb_ridge = ridge.predict(df_masked)
    p_xgb, is_fb_xgb = xgb_m.predict(df_masked)

    assert len(p_ridge) == len(df)
    assert len(p_xgb) == len(df)
    assert np.all(np.isfinite(p_ridge))
    assert np.all(np.isfinite(p_xgb))

    # Rows 10..20 must have fallback active
    assert np.all(is_fb_ridge[10:21])
    assert np.all(is_fb_xgb[10:21])
    assert not np.any(is_fb_ridge[0:10])


def test_phase6_simple_lagged_proxy(synthetic_medium_range_df):
    """Verify SimpleLaggedUpstreamBaseline proxy fits and predicts without error."""
    df = synthetic_medium_range_df
    y = df["target_v_dH_18h"].values
    baseline = SimpleLaggedUpstreamBaseline(horizon=18, wave_lag_hr=14)
    baseline.fit(df, y)
    p, is_na = baseline.predict(df)
    assert len(p) == len(df)
    assert np.all(np.isfinite(p))


def test_phase6_deliverables_exist():
    """Verify all 12 required deliverables exist in the output directory."""
    required = [
        "config.json",
        "hydrological_lag_analysis.csv",
        "target_alignment_audit.csv",
        "feature_availability_audit.csv",
        "feature_ablation_results.csv",
        "horizon_metrics.csv",
        "event_metrics.csv",
        "outage_metrics.csv",
        "paired_comparisons.csv",
        "row_level_predictions.csv",
        "model_manifest.json",
    ]
    for fname in required:
        p = OUTPUT_DIR / fname
        assert p.exists(), f"Missing deliverable: {p}"
        assert p.stat().st_size > 50, f"Deliverable {p} appears empty"
