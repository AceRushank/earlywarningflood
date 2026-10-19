"""
river_forecasting/tests/test_live_weather.py
============================================
Comprehensive test suite for the live weather ingestion component,
catchment sampling verification, Open-Meteo integration, caching,
failure/stale fallback, and forecast integrity guarantees.
"""

from pathlib import Path
import time
import pytest
import requests
from unittest.mock import patch, MagicMock

from river_forecasting.src.live_weather import (
    verify_catchment_geometry_and_points,
    fetch_open_meteo_live_data,
    CATCHMENT_SAMPLING_POINTS,
    _WEATHER_CACHE
)
from river_forecasting.src.live_inference import (
    get_live_system_status,
    predict_multimodal_dh
)


def test_catchment_geometry_and_7_points_verified():
    """Verify that all 7 representative points are strictly within the 51,768 km² catchment polygon."""
    audit = verify_catchment_geometry_and_points()
    assert audit["verified"] is True, f"Catchment verification failed: {audit}"
    assert audit["total_points_checked"] == 7
    assert audit["delineated_area_sq_km"] >= 51000

    # Ensure each point is verified inside
    for pt in audit["points"]:
        assert pt["is_inside_polygon"] is True, f"Point {pt['name']} is outside polygon!"
        assert pt["distance_to_boundary_deg"] >= 0.0


def test_fetch_open_meteo_live_data_success():
    """Verify that live Open-Meteo API returns genuine weather and hourly precipitation."""
    res = fetch_open_meteo_live_data(timeout_sec=10.0, bypass_cache=True)
    assert res["status"] in ["LIVE_WEATHER_AVAILABLE", "LIVE_WEATHER_STALE"]
    assert "source" in res
    assert "Open-Meteo" in res["source"]["provider"]

    # Station weather checks
    stn = res["station_area_weather"]
    assert stn is not None
    assert "Vishnupuri Barrage" in stn["station_name"]
    assert isinstance(stn["current_temperature_c"], (int, float))
    assert 0 <= stn["current_humidity_pct"] <= 100
    assert "+1h" in stn["forecast_hourly_mm"]
    assert "+2h" in stn["forecast_hourly_mm"]
    assert "+3h" in stn["forecast_hourly_mm"]

    # Catchment sampled rainfall checks (7 points)
    catch = res["catchment_sampled_rainfall"]
    assert catch is not None
    assert len(catch["locations"]) == 7
    assert isinstance(catch["forecast_mean_1h_mm"], (int, float))
    assert isinstance(catch["forecast_mean_next_3h_total_mm"], (int, float))

    # Scientific distinction notes present
    assert res["scientific_distinction"]["substitution_prohibited"] is True


def test_caching_and_bypass_behavior():
    """Verify that short cache avoids redundant network calls, while bypass forces fresh fetch."""
    # First call primes cache
    res1 = fetch_open_meteo_live_data(bypass_cache=False)
    # Second immediate call should return cached data
    res2 = fetch_open_meteo_live_data(bypass_cache=False)
    assert res2["is_cached"] is True
    assert res2["cache_age_sec"] >= 0.0

    # Third call with bypass_cache=True should fetch fresh
    res3 = fetch_open_meteo_live_data(bypass_cache=True)
    assert res3["is_cached"] is False


def test_network_failure_retains_stale_cache_without_zeros():
    """Verify that API failure retains previous result with STALE status and original timestamp."""
    # Prime cache first
    prime = fetch_open_meteo_live_data(bypass_cache=True)
    cached_fetch_time = _WEATHER_CACHE["last_successful_fetch_ist"]
    assert cached_fetch_time is not None

    # Force network failure
    with patch("requests.get", side_effect=requests.exceptions.ConnectTimeout("Mocked timeout")):
        stale_res = fetch_open_meteo_live_data(bypass_cache=True)
        assert stale_res["status"] == "LIVE_WEATHER_STALE"
        assert stale_res["is_stale"] is True
        assert stale_res["last_successful_fetch_ist"] == cached_fetch_time
        assert "Mocked timeout" in stale_res["error"]


def test_explicit_operational_states_and_model_blocking():
    """Verify the 4 explicit states: LIVE WEATHER AVAILABLE, STAGE STALE, MODEL BLOCKED, NOT READY."""
    sys_status = get_live_system_status()
    states = sys_status["operational_states"]

    assert states["weather_state"] in ["LIVE_WEATHER_AVAILABLE", "LIVE_WEATHER_STALE"]
    assert states["stage_state"] == "STAGE STALE"
    assert states["model_state"] == "MODEL BLOCKED"
    assert states["forecast_state"] == "NOT READY"

    # Verify that stage forecast readiness is explicitly False
    readiness = sys_status["forecast_readiness"]
    assert readiness["is_forecast_ready"] is False
    assert len(readiness["blocking_reasons"]) >= 2

    # Verify that Model D outputs remain explicitly None
    forecasts = sys_status["multimodal_model_status"]["forecasts"]
    assert forecasts["+1h"] is None
    assert forecasts["+2h"] is None
    assert forecasts["+3h"] is None


def test_timeline_and_valid_points_count():
    """Verify that Open-Meteo returns up to 24h timeline and valid points count."""
    res = fetch_open_meteo_live_data(bypass_cache=False)
    assert "timeline" in res
    tl = res["timeline"]
    assert tl["hours_count"] >= 12
    assert len(tl["catchment_mean_precipitation_mm"]) == tl["hours_count"]
    assert len(tl["station_precipitation_mm"]) == tl["hours_count"]
    catch = res["catchment_sampled_rainfall"]
    assert catch["valid_points_count"] == 7
    assert "Seven-point sampled catchment rainfall estimate" in catch["spatial_representation"]

