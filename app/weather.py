"""
app/weather.py

Fetches forward hourly rainfall forecasts from Open-Meteo (free, no API key).
API docs: https://open-meteo.com/en/docs

Extracted multi-horizon metrics per ward centroid:
  - next_6h_rainfall_mm  : sum of the next 6 hourly precipitation values
  - next_12h_rainfall_mm : sum of the next 12 hourly precipitation values
  - next_24h_rainfall_mm : sum of the next 24 hourly precipitation values
  - peak_hourly_mm       : maximum single-hour precipitation in the next 24 hours
  - forecast_timestamp   : ISO 8601 UTC timestamp of forecast extraction
  - source_model         : meteorological forecast model identifier
  - weather_available    : boolean flag indicating if forecast was retrieved

Results are cached per ward centroid with a 15-minute TTL to prevent
redundant calls.

FAILURE HANDLING:
  If the Open-Meteo API fails (timeout, 5xx, network error), weather_available is
  set to False and rainfall values are returned as None (NEVER silently converted to 0 mm).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

# In-memory cache: (lat_rounded, lon_rounded) -> (timestamp, payload)
_cache: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, 3), round(lon, 3))


def fetch_rainfall(lat: float, lon: float) -> dict[str, Any]:
    """
    Fetches forward precipitation forecast for coordinate (lat, lon).
    Returns dictionary with multi-horizon totals and metadata.
    On failure, returns weather_available=False with None rainfall values.
    """
    key = _cache_key(lat, lon)
    now = time.monotonic()

    # Return cached result if still fresh
    if key in _cache:
        ts, payload = _cache[key]
        if now - ts < CACHE_TTL_SECONDS:
            return payload

    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation",
                "forecast_days": 2,
                "timezone": "UTC",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        hourly_precip: list[float] = data.get("hourly", {}).get("precipitation", [])
        if not hourly_precip:
            raise ValueError("Malformed Open-Meteo response: 'hourly.precipitation' is missing or empty")

        # Extraction across horizons (hourly_precip[0] is the current forecast hour)
        next_6h = float(sum(hourly_precip[:6]))
        next_12h = float(sum(hourly_precip[:12]))
        next_24h = float(sum(hourly_precip[:24]))
        peak_24h = float(max(hourly_precip[:24])) if hourly_precip[:24] else 0.0

        utc_now = dt.datetime.now(dt.timezone.utc).isoformat()
        source_model = data.get("generationtime_ms", None)
        model_name = "ECMWF / GFS Seamless (Open-Meteo NWP)"

        result = {
            "weather_available": True,
            "next_6h_rainfall_mm": round(next_6h, 2),
            "next_12h_rainfall_mm": round(next_12h, 2),
            "next_24h_rainfall_mm": round(next_24h, 2),
            "peak_hourly_mm": round(peak_24h, 2),
            "forecast_timestamp": utc_now,
            "source_model": model_name,
            "error_message": None,
        }

        # Cache successful responses
        _cache[key] = (now, result)
        return result

    except Exception as exc:
        log.warning("Open-Meteo API call failed for (%.3f, %.3f): %s", lat, lon, exc)
        failure_result = {
            "weather_available": False,
            "next_6h_rainfall_mm": None,
            "next_12h_rainfall_mm": None,
            "next_24h_rainfall_mm": None,
            "peak_hourly_mm": None,
            "forecast_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_model": "Unavailable",
            "error_message": str(exc),
        }
        # Do not cache failures for long, but return safe object
        return failure_result
