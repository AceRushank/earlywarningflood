"""
river_forecasting/tests/test_delayed_observation_forecasting.py
==============================================================
Validates the delayed-observation forecasting mode for Vishnupuri Barrage:
1. Verifies strictly causal feature construction at t_gauge (zero future leakage).
2. Verifies alignment of target timestamps to t_gauge + 1h, +2h, +3h.
3. Tests refusal on missing rainfall / satellite outages (no zero-imputation).
4. Tests behavior when gauge observation is ~48h+ old.
5. Verifies comparison with persistence baseline on identical origins.
6. Verifies separation of Live Operational Mode (3h freshness safeguard)
   from Delayed-Observation Mode (research demonstration only).
7. Verifies scientific rationale against repurposing 1-3h model as 48h nowcast.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.live_inference import (
    generate_delayed_origin_forecast,
    get_live_system_status,
    predict_multimodal_dh,
    MODEL_D_WEIGHTS
)
from river_forecasting.src.multivariable_pipeline import prepare_multivariable_dataset


@pytest.fixture(scope="module")
def prepared_df():
    """Loads and caches multivariable dataset for fast testing."""
    return prepare_multivariable_dataset()


def test_delayed_origin_forecast_aligned_historical_success(prepared_df):
    """
    Verifies that for an authentic historical origin with valid gauge and rainfall,
    the model generates a valid delayed-origin forecast with exact timestamp alignments.
    """
    t_gauge_str = "2026-09-20 12:00:00"
    simulated_now = pd.to_datetime("2026-09-22 12:00:00")  # Exactly 48 hours later
    
    res = generate_delayed_origin_forecast(
        t_gauge=t_gauge_str,
        df_dataset=prepared_df,
        current_time_ist=simulated_now
    )

    assert res["forecast_valid"] is True
    assert res["forecast_mode"] == "DELAYED_OBSERVATION"
    assert "2026-09-20 12:00:00" in res["gauge_observation_time_ist"]
    assert "2026-09-20 12:00:00" in res["forecast_origin_ist"]
    assert res["is_live_current_prediction"] is False
    assert "DELAYED-ORIGIN FORECAST" in res["scientific_label"]
    assert "RESEARCH DEMONSTRATION ONLY" in res["disclaimer"]
    assert abs(res["gauge_age_hours"] - 48.0) < 0.1

    # Check target alignments: strictly t_gauge + 1h, +2h, +3h
    forecasts = res["forecasts"]
    assert "+1h" in forecasts and "+2h" in forecasts and "+3h" in forecasts
    
    assert "2026-09-20 13:00:00" in forecasts["+1h"]["target_timestamp_ist"]
    assert "2026-09-20 14:00:00" in forecasts["+2h"]["target_timestamp_ist"]
    assert "2026-09-20 15:00:00" in forecasts["+3h"]["target_timestamp_ist"]

    # Verify physical plausibility
    for h in ["+1h", "+2h", "+3h"]:
        fc = forecasts[h]
        assert 350.0 < fc["predicted_stage_m"] < 356.0
        assert abs(fc["predicted_dh_m"]) < 0.5
        assert fc["persistence_stage_m"] == pytest.approx(354.471, abs=0.001)
        # Model D vs Persistence difference must equal predicted dH
        expected_diff = fc["predicted_stage_m"] - fc["persistence_stage_m"]
        assert fc["model_d_vs_persistence_diff_m"] == pytest.approx(expected_diff, abs=0.0001)


def test_delayed_origin_strictly_no_future_leakage(prepared_df):
    """
    Verifies that all features used at origin t_gauge were derived strictly
    from information available at or before t_gauge, with zero future target leakage.
    """
    t_gauge = pd.to_datetime("2026-07-30 18:00:00")
    res = generate_delayed_origin_forecast(t_gauge=t_gauge, df_dataset=prepared_df)
    assert res["forecast_valid"] is True

    features_used = res["features_at_origin"]
    row = prepared_df[prepared_df["dt"] == t_gauge].iloc[0]

    # Verify features strictly match historical row at t_gauge
    assert features_used["stage_observed_m"] == row["stage_vishnupuri"]
    assert features_used["stage_dynamics"]["dH_v_1h"] == row["dH_v_1h"]
    assert features_used["stage_dynamics"]["dH_v_3h"] == row["dH_v_3h"]
    assert features_used["stage_dynamics"]["dH_v_6h"] == row["dH_v_6h"]
    assert features_used["stage_dynamics"]["accel_v_1h"] == row["accel_v_1h"]
    assert features_used["rainfall_depth_1h_mm"] == row["rain_depth_1h_mm"]
    assert features_used["api_7d_mm"] == row["api_7d_mm"]


def test_delayed_origin_satellite_outage_refusal(prepared_df):
    """
    Verifies that when t_gauge falls in a satellite outage,
    the model strictly refuses to predict and prohibits zero-imputation.
    """
    outage_t_gauge = "2026-08-01 12:00:00"  # Confirmed MOSDAC satellite outage
    res = generate_delayed_origin_forecast(t_gauge=outage_t_gauge, df_dataset=prepared_df)

    assert res["forecast_valid"] is False
    assert res["forecasts"]["+1h"] is None
    assert res["forecasts"]["+2h"] is None
    assert res["forecasts"]["+3h"] is None
    assert "missing observations / satellite outage" in res["reason"]
    assert "Zero-imputation" in res["reason"]


def test_delayed_origin_at_unarchived_gauge_observation():
    """
    Verifies that when gauge telemetry is ~48h+ old (e.g. October 10) but satellite
    rainfall is not in the local archive, the model refuses to substitute present-day rainfall.
    """
    # October 10 is outside the 2026 monsoon archive (which ends Sept 30)
    res = generate_delayed_origin_forecast(t_gauge="2026-10-10 23:00:00")
    
    assert res["forecast_valid"] is False
    assert res["forecasts"]["+1h"] is None
    assert "not present in the local archive" in res["reason"]
    assert "Cross-temporal substitution" in res["reason"]
    assert res["is_live_current_prediction"] is False


def test_live_system_status_preserves_operational_freshness_safeguard():
    """
    Verifies that get_live_system_status() maintains strict separation:
    - Real-time operational forecasts remain BLOCKED when telemetry is stale (>3h).
    - Delayed-observation mode is provided with clear demonstration labelling.
    """
    status = get_live_system_status()

    # 1. Operational Mode: LIVE_OPERATIONAL with safeguard
    assert status["operational_mode"] == "LIVE_OPERATIONAL"
    safeguard = status["operational_freshness_safeguard"]
    assert safeguard["rule"] == "Stage telemetry must be <= 3.0 hours old for real-time operational forecasting."
    assert safeguard["status"] == "BLOCKED"  # Because telemetry age is >50h

    # 2. Delayed-observation mode separation
    assert "delayed_observation_mode" in status
    assert "delayed_demonstration_sample" in status
    
    demo = status["delayed_demonstration_sample"]
    assert demo["forecast_valid"] is True
    assert demo["forecast_mode"] == "DELAYED_OBSERVATION"
    assert demo["is_live_current_prediction"] is False


def test_current_stage_estimation_infeasibility_documented():
    """
    Verifies that the system explicitly documents why a +1h/+2h/+3h model
    cannot be repurposed as a 48h nowcast estimator.
    """
    res = generate_delayed_origin_forecast()
    infeasibility = res.get("current_stage_estimation_infeasibility", {})
    
    assert infeasibility.get("is_possible_with_current_model") is False
    assert "requires a 48h horizon" in infeasibility.get("scientific_rationale")
    assert "unobserved radial gate operations" in infeasibility.get("scientific_rationale")
    assert "autoregressive rollout drift" in infeasibility.get("scientific_rationale")
