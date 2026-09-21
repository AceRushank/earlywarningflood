"""
app/main.py

Mumbai Ward-Level Flood Early-Warning System — FastAPI Application.

SCIENTIFIC ARCHITECTURE:
  Combines:
    1. Spatial Susceptibility (Ridge Regression trained on 2018-2025 Sentinel-1 SAR
       inundation ratio and physiographic features).
    2. Real-Time Forward NWP Forecasts (Open-Meteo multi-window hourly precipitation).
    3. Transparent Hydrometeorological Decision Layer (IMD classifications and
       BRIMSTOWAD urban drainage benchmark).

Exposes:
  GET /wards/risk  -> Ward-level hybrid flood risk scoring and simulated early warnings.

DISCLAIMER:
  The numerical output 'hybrid_risk_index' is a relative decision ranking in [0.0, 1.0].
  It is NOT a calibrated probability of flooding. Alert channels are simulated text
  renderings for research prototype evaluation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.data_loader import load_ward_data
from app.scoring import compute_risk_scores, generate_alert_channels
from app.weather import fetch_rainfall

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger(__name__)

_ward_base: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ward_base
    log.info("Loading ward boundary, geometric area, and baseline data…")
    _ward_base = load_ward_data()
    log.info("Ward data initialized. %d wards ready.", len(_ward_base))
    yield
    log.info("Shutting down service.")


app = FastAPI(
    title="Mumbai Flood Early-Warning API (Hybrid Framework)",
    description=(
        "Hybrid flood early-warning microservice for Mumbai's 24 BMC administrative wards. "
        "Couples an ML spatial susceptibility model (Ridge regression on historical SAR inundation ratio) "
        "with live Open-Meteo forward rainfall forecasts. Outputs a relative 'hybrid_risk_index' and "
        "internal decision categories (LOW, MODERATE, HIGH, CRITICAL). "
        "This is an uncalibrated research prototype; it is NOT an official IMD warning."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get(
    "/wards/risk",
    summary="Ward-level hybrid flood risk scores and early warning advisories",
    response_description="List of ward assessment objects, sorted by hybrid_risk_index descending.",
)
def get_wards_risk(
    horizon: str = Query(
        "next_24h",
        enum=["next_6h", "next_12h", "next_24h"],
        description="Forecast accumulation window to evaluate.",
    ),
    simulate_rain_mm: float | None = Query(
        None,
        description="TEST OVERRIDE ONLY. Overrides rainfall across all wards for deterministic testing.",
    ),
) -> list[dict[str, Any]]:
    if _ward_base is None:
        raise HTTPException(status_code=503, detail="Ward baseline data not loaded.")

    df = _ward_base.copy()

    # Ingest live rainfall per centroid (cached 15 min per rounded coordinate)
    weather_results = []
    for _, row in df.iterrows():
        w_data = fetch_rainfall(row["lat"], row["lon"])
        weather_results.append(w_data)

    weather_df = pd.DataFrame(weather_results, index=df.index)
    df = pd.concat([df, weather_df], axis=1)

    # Determine data mode
    if simulate_rain_mm is not None:
        data_mode = "simulation"
        df["weather_available"] = True
        df["next_6h_rainfall_mm"] = float(simulate_rain_mm)
        df["next_12h_rainfall_mm"] = float(simulate_rain_mm)
        df["next_24h_rainfall_mm"] = float(simulate_rain_mm)
        df["peak_hourly_mm"] = round(float(simulate_rain_mm) / 3.0, 2)
        df["source_model"] = "Deterministic Simulation Test Override"
    else:
        data_mode = "live"

    # Compute risk metrics under chosen horizon
    df = compute_risk_scores(df, horizon=horizon)

    # Sort: put valid indices descending, followed by weather-unavailable records
    df["sort_key"] = df["hybrid_risk_index"].fillna(-1.0)
    df = df.sort_values("sort_key", ascending=False).drop(columns=["sort_key"])

    response_items = []
    for _, row in df.iterrows():
        alerts = generate_alert_channels(row)
        response_items.append(
            {
                "ward_name":                  row["ward_name"],
                "bmc_ward_code":              row["bmc_ward_code"],
                "ward_area_km2":              float(row["ward_area_km2"]),
                "lat":                         round(float(row["lat"]), 6),
                "lon":                         round(float(row["lon"]), 6),
                "historical_flood_score":     round(float(row["historical_flood_score"]), 4),
                "inundation_ratio":           round(float(row["inundation_ratio"]), 6),
                "forecast_horizon":           row["forecast_horizon"],
                "forecast_rainfall_mm":       row["forecast_rainfall_mm"],
                "next_6h_rainfall_mm":        row["next_6h_rainfall_mm"],
                "next_12h_rainfall_mm":       row["next_12h_rainfall_mm"],
                "next_24h_rainfall_mm":       row["next_24h_rainfall_mm"],
                "peak_hourly_mm":             row["peak_hourly_mm"],
                "imd_24h_rainfall_category":  row["imd_24h_rainfall_category"],
                "imd_intensity_category":     row["imd_intensity_category"],
                "ml_susceptibility":          row["ml_susceptibility"],
                "susceptibility_norm":        row["susceptibility_norm"],
                "drainage_intensity_ratio":   row["drainage_intensity_ratio"],
                "rainfall_hazard":            row["rainfall_hazard"],
                "hybrid_risk_index":          row["hybrid_risk_index"],
                "risk_level":                 row["risk_level"],
                "early_warning":              bool(row["early_warning"]),
                "weather_available":          bool(row["weather_available"]),
                "forecast_timestamp":         row["forecast_timestamp"],
                "source_model":               row["source_model"],
                "data_mode":                  data_mode,
                "alert_channels":             alerts,
                # Backward-compatibility aliases
                "predicted_susceptibility":   row["ml_susceptibility"],
                "risk_score":                 row["hybrid_risk_index"],
                "alert":                      bool(row["early_warning"]),
            }
        )

    return response_items


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Mumbai Flood Early-Warning API (Hybrid Framework)",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoint": "/wards/risk",
        "supported_horizons": ["next_6h", "next_12h", "next_24h"],
    }
