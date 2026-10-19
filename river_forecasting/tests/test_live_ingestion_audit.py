"""
river_forecasting/tests/test_live_ingestion_audit.py
===================================================
Tests verifying genuine live data ingestion pipelines:
1. National Water Data Portal (NWDP/NWIC) CWC River Water Level Telemetry Datastore API.
2. Open-Meteo live weather freshness, issue/valid times, HTTP 200, cache age, and model interval.
3. ISRO MOSDAC INSAT-3DS operational catalog authentication and scan verification.
4. Strict forecast safety: Model D blocked on stale telemetry origin or incompatible inputs.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import datetime
import pytest

from river_forecasting.src.live_inference import (
    probe_nwic_live_endpoint,
    fetch_nwdp_live_telemetry,
    audit_mosdac_live_status,
    get_live_system_status,
    predict_multimodal_dh
)
from river_forecasting.src.live_weather import fetch_open_meteo_live_data


def test_nwdp_live_fetch_returns_vishnupuri_record():
    """Verify that fetch_nwdp_live_telemetry accesses the official NWDP CKAN API and returns Vishnupuri."""
    res = fetch_nwdp_live_telemetry(timeout_sec=6.0)
    assert res["station_name"] == "Vishnupuri"
    assert res["agency"] == "Maharashtra SW"
    assert res["station_code"] == "006-SW-MAHA"
    assert res["basin"] == "Godavari"
    assert res["mean_sea_level_datum_m"] == 366.50
    assert res["zero_gauge_rl_m"] == 0.0
    assert isinstance(res["latest_stage_m_msl"], (int, float))
    assert 350.0 < res["latest_stage_m_msl"] < 356.0
    
    # Must explicitly identify status as STALE when observation age > 3h
    assert res["data_age_hours"] > 3.0
    assert res["is_sufficiently_recent_for_live"] is False
    assert res["status"] == "STAGE STALE"


def test_nwdp_timeout_fallback_preserves_stale_safeguard():
    """Verify that when NWDP network times out, it gracefully falls back to local archive marked STALE."""
    with patch("requests.get", side_effect=Exception("Connection timed out")):
        res = fetch_nwdp_live_telemetry(timeout_sec=1.0)
        assert res["station_name"] == "Vishnupuri"
        assert res["status"] == "STAGE STALE"
        assert res["is_sufficiently_recent_for_live"] is False
        assert res["feed_quality"] == "LOCAL_OFFICIAL_ARCHIVE_FALLBACK"


def test_open_meteo_freshness_metadata():
    """Verify Open-Meteo returns request time, valid time, HTTP status, and NWP model interval."""
    weather = fetch_open_meteo_live_data(timeout_sec=8.0, bypass_cache=True)
    assert weather["status"] in ["LIVE_WEATHER_AVAILABLE", "LIVE_WEATHER_STALE"]
    assert "request_time_ist" in weather
    assert "forecast_valid_time_ist" in weather
    assert "http_status" in weather
    assert weather["http_status"] == 200
    assert "model_run_interval_notes" in weather
    assert "polls every 5 min" in weather["model_run_interval_notes"]


def test_mosdac_catalog_authentication_and_scans():
    """Verify MOSDAC catalog authenticates and verifies current operational scans."""
    mosdac = audit_mosdac_live_status(timeout_sec=6.0)
    assert mosdac["authenticated"] is True
    assert mosdac["status"] == "AUTHENTICATED_ONLINE"
    assert mosdac["dataset_id"] == "3SIMG_L2B_HEM"
    assert "V01R00" in mosdac["product_version"]
    assert mosdac["today_scans_count"] > 0
    assert "3SIMG_" in mosdac["latest_scan_identifier"]


def test_forecast_safety_strictly_blocks_model_d_on_stale_telemetry():
    """Verify that Model D forecasts are blocked and explicit reasons are provided."""
    status = get_live_system_status()
    assert status["forecast_readiness"]["is_forecast_ready"] is False
    assert status["operational_states"]["stage_state"] == "STAGE STALE"
    assert status["operational_states"]["model_state"] == "MODEL BLOCKED"
    assert status["multimodal_model_status"]["forecasts"]["+1h"] is None
    assert status["multimodal_model_status"]["forecasts"]["+2h"] is None
    assert status["multimodal_model_status"]["forecasts"]["+3h"] is None

    reasons = " ".join(status["forecast_readiness"]["blocking_reasons"])
    assert "latency limit" in reasons or "stale origin" in reasons
    assert "NWP" in reasons or "satellite" in reasons


def test_telemetry_agency_distinction_maharashtra_sw_vs_cwc():
    """Verify that station 006-SW-MAHA is explicitly attributed to Maharashtra SW and not confused with CWC."""
    status = get_live_system_status()
    stn = status["station_identity"]
    assert stn["agency"] == "Maharashtra SW"
    assert "Maharashtra" in status["live_telemetry_feed"]["official_source"]
    assert "agency_provenance" in status["live_telemetry_feed"]
    assert "Maharashtra Surface Water Department" in status["live_telemetry_feed"]["agency_provenance"]
    assert "NOT Central Water Commission" in status["live_telemetry_feed"]["agency_provenance"]

