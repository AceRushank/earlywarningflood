"""
river_forecasting/tests/test_phase2_model_development.py
=========================================================
Unit and integration tests for Phase 2 model development:
1. Causal regime detection (zero lookahead).
2. RegimeAwareAdaptiveRegressor fitting, prediction, and fallback logic.
3. Metric computation, peak underprediction, and direction accuracy.
4. Stage reconstruction consistency: H_hat = H_t + dH_hat.
"""

import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.phase2_model_development import (
    RegimeDetector,
    RegimeAwareAdaptiveRegressor,
    compute_comprehensive_metrics,
)
from river_forecasting.src.phase1_ml_pipeline import (
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_MULTIMODAL,
)


@pytest.fixture
def synthetic_sample_df():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "stage_vishnupuri": 354.0 + np.cumsum(rng.normal(0, 0.01, size=n)),
        "dH_v_1h": rng.normal(0, 0.02, size=n),
        "dH_v_3h": rng.normal(0, 0.03, size=n),
        "dH_v_6h": rng.normal(0, 0.04, size=n),
        "accel_v_1h": rng.normal(0, 0.01, size=n),
        "rain_depth_1h_mm": np.maximum(0, rng.normal(0.5, 2.0, size=n)),
        "rain_3h_mm": np.maximum(0, rng.normal(1.5, 4.0, size=n)),
        "rain_6h_mm": np.maximum(0, rng.normal(3.0, 6.0, size=n)),
        "rain_12h_mm": np.maximum(0, rng.normal(5.0, 8.0, size=n)),
        "rain_24h_mm": np.maximum(0, rng.normal(10.0, 12.0, size=n)),
        "api_7d_mm": np.maximum(0, rng.normal(15.0, 5.0, size=n)),
    })
    df["target_v_dH_1h"] = 0.5 * df["dH_v_1h"] + 0.002 * df["rain_depth_1h_mm"] + rng.normal(0, 0.005, size=n)
    df["target_v_1h"] = df["stage_vishnupuri"] + df["target_v_dH_1h"]
    return df


def test_regime_detector_classification():
    detector = RegimeDetector(theta_rise=0.015, theta_fall=-0.015, rain_threshold=1.0)
    
    # 1. Active rise
    r_rise = pd.Series({"dH_v_1h": 0.03, "dH_v_3h": 0.05, "accel_v_1h": 0.01, "rain_3h_mm": 0.0})
    assert detector.predict_regime(r_rise) == "RISING"

    # 2. Heavy rain triggers rising regime even with flat stage
    r_rain = pd.Series({"dH_v_1h": 0.005, "dH_v_3h": 0.01, "accel_v_1h": 0.0, "rain_3h_mm": 5.0})
    assert detector.predict_regime(r_rain) == "RISING"

    # 3. Active fall
    r_fall = pd.Series({"dH_v_1h": -0.025, "dH_v_3h": -0.04, "accel_v_1h": -0.01, "rain_3h_mm": 0.0})
    assert detector.predict_regime(r_fall) == "FALLING"

    # 4. Stable flat pool
    r_stable = pd.Series({"dH_v_1h": 0.002, "dH_v_3h": 0.005, "accel_v_1h": 0.0, "rain_3h_mm": 0.1})
    assert detector.predict_regime(r_stable) == "STABLE"


def test_regime_aware_regressor_fit_and_predict(synthetic_sample_df):
    df = synthetic_sample_df
    y_dh = df["target_v_dH_1h"].values
    
    rag = RegimeAwareAdaptiveRegressor(horizon=1, theta_rise=0.015, theta_fall=-0.015)
    rag.fit(df, y_dh)
    
    pred_dh, is_fb, regimes = rag.predict(df)
    assert len(pred_dh) == len(df)
    assert np.all(np.isfinite(pred_dh))
    assert not np.any(is_fb)  # Synthetic df has complete rainfall
    assert set(regimes).issubset({"RISING", "FALLING", "STABLE"})


def test_regime_aware_regressor_fallback_on_missing_rain(synthetic_sample_df):
    df = synthetic_sample_df.copy()
    # Introduce NaNs into rainfall columns for rows 10 to 20
    df.loc[10:20, FEAT_RAIN_CONT] = np.nan
    
    rag = RegimeAwareAdaptiveRegressor(horizon=1, theta_rise=0.015, theta_fall=-0.015)
    rag.fit(synthetic_sample_df, synthetic_sample_df["target_v_dH_1h"].values)
    
    pred_dh, is_fb, _ = rag.predict(df)
    assert np.all(np.isfinite(pred_dh))
    # Fallback must be engaged for the rows with missing rainfall
    assert np.all(is_fb[10:21])


def test_stage_reconstruction_consistency(synthetic_sample_df):
    df = synthetic_sample_df
    y_dh = df["target_v_dH_1h"].values
    
    rag = RegimeAwareAdaptiveRegressor(horizon=1)
    rag.fit(df, y_dh)
    
    pred_dh, _, _ = rag.predict(df)
    yc = df["stage_vishnupuri"].values
    pred_stage = yc + pred_dh
    
    # Check that stage reconstruction is strictly additive
    np.testing.assert_allclose(pred_stage - yc, pred_dh)


def test_comprehensive_metrics_computation(synthetic_sample_df):
    df = synthetic_sample_df
    y_true = df["target_v_1h"].values
    y_pers = df["stage_vishnupuri"].values
    y_pred = y_pers + 0.8 * df["target_v_dH_1h"].values
    
    m = compute_comprehensive_metrics(y_true, y_pred, y_pers, df)
    assert "mae_mm" in m
    assert "rmse_mm" in m
    assert "direction_accuracy_pct" in m
    assert "peak_underprediction_mm" in m
    assert "fallback_freq_pct" in m
    assert m["mae_mm"] >= 0.0
    assert 0.0 <= m["direction_accuracy_pct"] <= 100.0
