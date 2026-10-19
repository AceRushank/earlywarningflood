"""
river_forecasting/tests/test_live_inference.py
==============================================
Unit tests verifying the live-data probe, official NWIC station identity,
observation timestamp, and Model D inference validation.
"""

from pathlib import Path
import numpy as np
import pytest

from river_forecasting.src.live_inference import (
    probe_nwic_live_endpoint,
    get_latest_official_observation,
    predict_multimodal_dh,
    get_live_system_status,
    MODEL_D_FEATURE_NAMES,
    MODEL_D_WEIGHTS
)


def test_official_station_identity_and_latest_observation():
    """Verify that the official NWIC source returns Vishnupuri and the exact latest timestamp."""
    obs = get_latest_official_observation()
    assert "error" not in obs, f"Observation check failed: {obs}"

    # 1. Exact Station Identity
    assert obs["station_name"] == "Vishnupuri"
    assert obs["agency"] == "Maharashtra SW"
    assert obs["basin"] == "Godavari"
    assert obs["district"] == "Nanded"
    assert abs(obs["latitude"] - 19.12694) < 0.001
    assert abs(obs["longitude"] - 77.28083) < 0.001

    # 2. Datum & FRL thresholds
    assert obs["mean_sea_level_datum_m"] == 366.50
    assert obs["zero_gauge_rl_m"] == 0.0
    assert obs["frl_threshold_m_msl"] == 355.00

    # 3. Observation value and timestamp
    assert "2026-10-09 23:00" in obs["latest_observation_time_ist"]
    assert abs(obs["latest_stage_m_msl"] - 353.953) < 0.001

    # 4. Data age check: Stale data (>60 hours) cannot support sub-hourly live forecast
    assert obs["data_age_hours"] > 60.0
    assert obs["is_sufficiently_recent_for_live"] is False


def test_predict_multimodal_dh_refuses_when_rainfall_missing():
    """Verify strict prohibition of zero-imputation or prediction without rainfall."""
    stage_dyn = {
        "dH_v_1h": 0.05,
        "dH_v_3h": 0.12,
        "dH_v_6h": 0.20,
        "accel_v_1h": 0.02
    }

    # Case 1: Rainfall is None
    res_none = predict_multimodal_dh(current_stage=353.953, stage_dyn_features=stage_dyn, rainfall_features=None)
    assert res_none["forecast_valid"] is False
    assert res_none["forecasts"]["+1h"] is None
    assert "Rainfall inputs unavailable" in res_none["reason"]

    # Case 2: Rainfall has NaN
    rain_nan = {
        "rain_depth_1h_mm": np.nan,
        "rain_3h_mm": 5.0,
        "rain_6h_mm": 10.0,
        "rain_12h_mm": 15.0,
        "rain_24h_mm": 20.0,
        "api_7d_mm": 35.0
    }
    res_nan = predict_multimodal_dh(current_stage=353.953, stage_dyn_features=stage_dyn, rainfall_features=rain_nan)
    assert res_nan["forecast_valid"] is False
    assert res_nan["forecasts"]["+1h"] is None

    # Case 3: Missing rainfall key
    rain_incomplete = {
        "rain_depth_1h_mm": 2.0,
        "rain_3h_mm": 5.0
    }
    res_inc = predict_multimodal_dh(current_stage=353.953, stage_dyn_features=stage_dyn, rainfall_features=rain_incomplete)
    assert res_inc["forecast_valid"] is False


def test_predict_multimodal_dh_computes_consistent_predictions_when_valid():
    """Verify that Model D produces consistent +1h, +2h, +3h forecasts when inputs are valid."""
    stage_dyn = {
        "dH_v_1h": 0.02,
        "dH_v_3h": 0.05,
        "dH_v_6h": 0.10,
        "accel_v_1h": 0.01
    }
    rain_valid = {
        "rain_depth_1h_mm": 3.5,
        "rain_3h_mm": 8.2,
        "rain_6h_mm": 14.0,
        "rain_12h_mm": 22.5,
        "rain_24h_mm": 31.0,
        "api_7d_mm": 45.0
    }

    res = predict_multimodal_dh(current_stage=353.953, stage_dyn_features=stage_dyn, rainfall_features=rain_valid)
    assert res["forecast_valid"] is True
    assert "+1h" in res["forecasts"]
    assert "+2h" in res["forecasts"]
    assert "+3h" in res["forecasts"]

    # Forecast stage should be within reasonable bounds of current stage
    for h in ["+1h", "+2h", "+3h"]:
        pred_stage = res["forecasts"][h]["predicted_stage_m"]
        assert 350.0 < pred_stage < 356.0
        # Predicted dH should be a small reasonable increment (<0.5m)
        pred_dh = res["forecasts"][h]["predicted_dh_m"]
        assert abs(pred_dh) < 0.5


def test_live_system_status_reports_accurate_diagnostic():
    """Verify that the full diagnostic report accurately exposes feed unavailability."""
    status = get_live_system_status()
    assert status["live_telemetry_feed"]["status"] == "UNAVAILABLE"
    assert status["rainfall_data_feed"]["status"] == "UNAVAILABLE"
    assert status["multimodal_model_status"]["status"] == "INACTIVE"
    assert status["multimodal_model_status"]["forecasts"]["+1h"] is None
    assert status["station_identity"]["name"] == "Vishnupuri"
