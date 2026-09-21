"""
app/scoring.py

Hybrid risk scoring and multi-channel alert message generation for the
Mumbai Flood Early-Warning System.

SCIENTIFIC & ARCHITECTURAL FOUNDATION:
  This module combines:
    1. Spatial Susceptibility (ML Model):
       Predicted historical inundation ratio (flooded_area_km2 / ward_area_km2)
       from a regularized Ridge regression pipeline trained on static physiographic features.
       Represents relative intrinsic vulnerability; does NOT predict event probabilities.
    2. Real-Time Meteorological Forcing:
       Forecast precipitation accumulation (next 6h, 12h, 24h) and peak hourly intensity
       from Open-Meteo Numerical Weather Predictions (NWP).
    3. Meteorological & Engineering Benchmarks:
       - IMD 24-hour rainfall classification (accumulation reference)
       - IMD hourly rainfall intensity classification (short-duration spell reference)
       - Mumbai BRIMSTOWAD 50 mm/hr storm drainage design benchmark

OUTPUT SPECIFICATION:
  - hybrid_risk_index: Dimensionless relative decision index in [0.0, 1.0].
    IT IS NOT A PROBABILITY (e.g. 0.73 does NOT mean 73% chance of flooding).
  - risk_level: Internal prototype decision categories:
      LOW       [0.00, 0.25)
      MODERATE  [0.25, 0.50)
      HIGH      [0.50, 0.75)
      CRITICAL  [0.75, 1.00]
    These are internal system decision states, NOT official IMD warnings.

ZERO-RAIN SAFETY GUARANTEE:
  If forecast rainfall and peak intensity are zero, hybrid_risk_index is strictly 0.0
  and early_warning is strictly False, regardless of spatial susceptibility.

WEATHER FAILURE SAFETY:
  If weather_available is False, hybrid_risk_index is None and risk_level is
  set to "WEATHER_UNAVAILABLE" (never silently converted to 0 mm).

DISCLAIMER:
  The formula hybrid_risk_index = H_rain * (0.50 + 0.50 * S_norm) is an
  uncalibrated prototype decision heuristic. It is NOT an empirically fitted
  hydraulic model, calibrated probability model, or flood-damage function.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "ward_susceptibility_model.pkl"

GEO_FEATURES = [
    "elevation_mean",
    "slope_mean",
    "builtup_pct",
    "water_pct",
    "vegetation_pct",
    "dist_to_water_mean",
]

# IMD 24-hour rainfall classification anchor (in mm)
IMD_EXTREMELY_HEAVY_24H_MM = 204.4

# Mumbai BRIMSTOWAD urban storm-water drainage design intensity benchmark (in mm/hr)
BRIMSTOWAD_DRAINAGE_BENCHMARK_MM_PER_HR = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model() -> dict[str, Any] | None:
    if not _MODEL_PATH.exists():
        log.warning(
            "ward_susceptibility_model.pkl not found at %s. "
            "Run train_ward_susceptibility.py first.",
            _MODEL_PATH,
        )
        return None
    try:
        with open(_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        log.info(
            "Loaded spatial susceptibility model: %s (target: %s, training wards: %d)",
            bundle.get("model_type", "unknown"),
            bundle.get("target", "inundation_ratio"),
            bundle.get("training_ward_count", 0),
        )
        return bundle
    except Exception as exc:
        log.error("Failed to load model bundle: %s", exc)
        return None


_MODEL_BUNDLE: dict[str, Any] | None = _load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Classifications & Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def classify_imd_24h_rainfall(rainfall_mm: float | None) -> str:
    """
    Official India Meteorological Department (IMD) 24-hour rainfall categories:
      0.0 mm            -> NO_RAIN
      0.1 - 15.5 mm     -> VERY_LIGHT_TO_LIGHT
      15.6 - 64.4 mm    -> MODERATE
      64.5 - 115.5 mm   -> HEAVY
      115.6 - 204.4 mm  -> VERY_HEAVY
      >= 204.5 mm       -> EXTREMELY_HEAVY

    These are meteorological accumulation categories, NOT flood occurrence thresholds.
    """
    if rainfall_mm is None or np.isnan(rainfall_mm):
        return "UNKNOWN"
    if rainfall_mm <= 0.05:
        return "NO_RAIN"
    elif rainfall_mm <= 15.5:
        return "VERY_LIGHT_TO_LIGHT"
    elif rainfall_mm <= 64.4:
        return "MODERATE"
    elif rainfall_mm <= 115.5:
        return "HEAVY"
    elif rainfall_mm <= 204.4:
        return "VERY_HEAVY"
    else:
        return "EXTREMELY_HEAVY"


def classify_imd_intensity(peak_hourly_mm: float | None) -> str:
    """
    Official IMD short-duration rainfall intensity spell categories:
      0.0 mm/hr         -> NO_RAIN
      0.1 - 10.0 mm/hr  -> LIGHT_SPELL
      10.1 - 20.0 mm/hr -> MODERATE_SPELL
      20.1 - 30.0 mm/hr -> INTENSE_SPELL
      30.1 - 50.0 mm/hr -> VERY_INTENSE_SPELL
      50.1 - 100.0 mm/hr-> EXTREMELY_INTENSE_SPELL
      > 100.0 mm/hr     -> EXCEPTIONALLY_HEAVY_SPELL
    """
    if peak_hourly_mm is None or np.isnan(peak_hourly_mm):
        return "UNKNOWN"
    if peak_hourly_mm <= 0.05:
        return "NO_RAIN"
    elif peak_hourly_mm <= 10.0:
        return "LIGHT_SPELL"
    elif peak_hourly_mm <= 20.0:
        return "MODERATE_SPELL"
    elif peak_hourly_mm <= 30.0:
        return "INTENSE_SPELL"
    elif peak_hourly_mm <= 50.0:
        return "VERY_INTENSE_SPELL"
    elif peak_hourly_mm <= 100.0:
        return "EXTREMELY_INTENSE_SPELL"
    else:
        return "EXCEPTIONALLY_HEAVY_SPELL"


def calculate_drainage_intensity_ratio(peak_hourly_mm: float | None) -> float | None:
    """
    Ratio of forecast peak hourly rainfall to Mumbai's BRIMSTOWAD storm drainage
    design capacity (50 mm/hr).
    ENGINEERING BENCHMARK COMPARISON ONLY. Not a hydraulic flood simulation.
    """
    if peak_hourly_mm is None or np.isnan(peak_hourly_mm):
        return None
    return round(float(peak_hourly_mm) / BRIMSTOWAD_DRAINAGE_BENCHMARK_MM_PER_HR, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Computation
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_risk_scores(df: pd.DataFrame, horizon: str = "next_24h") -> pd.DataFrame:
    """
    Computes spatial susceptibility, rainfall hazard, and hybrid risk index.

    Supported horizon parameters:
      'next_6h', 'next_12h', 'next_24h' (default)

    Adds columns:
      - forecast_horizon
      - forecast_rainfall_mm
      - ml_susceptibility
      - susceptibility_norm
      - drainage_intensity_ratio
      - imd_24h_rainfall_category
      - imd_intensity_category
      - rainfall_hazard
      - hybrid_risk_index
      - risk_level
      - early_warning
    """
    df = df.copy()

    # Map selected horizon column
    horizon_col_map = {
        "next_6h": "next_6h_rainfall_mm",
        "next_12h": "next_12h_rainfall_mm",
        "next_24h": "next_24h_rainfall_mm",
    }
    target_rain_col = horizon_col_map.get(horizon, "next_24h_rainfall_mm")
    df["forecast_horizon"] = horizon
    df["forecast_rainfall_mm"] = df[target_rain_col]

    # 1. Spatial susceptibility inference via ML model
    if _MODEL_BUNDLE is not None and all(f in df.columns for f in GEO_FEATURES):
        model = _MODEL_BUNDLE["model"]
        X = df[GEO_FEATURES].values.astype(float)
        raw_preds = model.predict(X)
        df["ml_susceptibility"] = np.clip(raw_preds, 0.0, None).round(6)
    else:
        log.warning("Susceptibility model unavailable; using inundation_ratio fallback.")
        df["ml_susceptibility"] = df.get("inundation_ratio", 0.0)

    df["susceptibility_norm"] = _minmax(df["ml_susceptibility"]).round(4)

    # 2. Benchmark ratios & meteorological categories
    df["drainage_intensity_ratio"] = df["peak_hourly_mm"].apply(calculate_drainage_intensity_ratio)
    df["imd_24h_rainfall_category"] = df["next_24h_rainfall_mm"].apply(classify_imd_24h_rainfall)
    df["imd_intensity_category"] = df["peak_hourly_mm"].apply(classify_imd_intensity)

    # 3. Meteorological Rainfall Hazard (H_rain)
    # - 24-hour horizon: combines 24h accumulation (anchored to IMD 204.4 mm reference)
    #   and peak hourly intensity (anchored to 50 mm/hr BRIMSTOWAD engineering drainage benchmark).
    # - Shorter horizons (next_6h, next_12h): 204.4 mm is NOT used as a denominator.
    #   Rather than inventing unverified 6h/12h accumulation thresholds or artificial
    #   scaling, the meteorological hazard basis relies transparently on the peak-hourly-intensity
    #   drainage benchmark (anchored to 50 mm/hr), while raw 6h/12h rainfall is exposed as an
    #   informational forecast quantity.
    def calc_hazard(row: pd.Series) -> float | None:
        if not row.get("weather_available", True) or pd.isna(row["forecast_rainfall_mm"]):
            return None
        rain = float(row["forecast_rainfall_mm"])
        peak = float(row["peak_hourly_mm"]) if not pd.isna(row["peak_hourly_mm"]) else 0.0

        if rain <= 0.05 and peak <= 0.05:
            return 0.0

        h_intensity = min(peak / BRIMSTOWAD_DRAINAGE_BENCHMARK_MM_PER_HR, 1.0)

        if horizon == "next_24h":
            # 204.4 mm is an official IMD 24-hour accumulation reference
            h_accum = min(rain / IMD_EXTREMELY_HEAVY_24H_MM, 1.0)
            return round(max(h_accum, h_intensity), 4)
        else:
            # For shorter horizons (next_6h, next_12h), do NOT divide by 204.4 mm
            # and do NOT invent arbitrary accumulation thresholds.
            # Hazard is driven directly by peak hourly drainage intensity.
            return round(h_intensity, 4)

    df["rainfall_hazard"] = df.apply(calc_hazard, axis=1)

    # 4. Hybrid Risk Index Formulation
    # Prototype heuristic formula: H_rain * (0.50 + 0.50 * S_norm)
    # Ensures that when H_rain = 1.0 (extreme rain), all wards have index >= 0.50 (HIGH/CRITICAL)
    # and when H_rain = 0.0, index is strictly 0.0 (LOW).
    def calc_hybrid_index(row: pd.Series) -> float | None:
        h_rain = row["rainfall_hazard"]
        if h_rain is None:
            return None
        if h_rain == 0.0:
            return 0.0
        s_norm = float(row["susceptibility_norm"])
        idx = h_rain * (0.50 + 0.50 * s_norm)
        return round(float(np.clip(idx, 0.0, 1.0)), 4)

    df["hybrid_risk_index"] = df.apply(calc_hybrid_index, axis=1)

    # 5. Internal System Decision Categories
    def assign_category(row: pd.Series) -> str:
        idx = row["hybrid_risk_index"]
        if idx is None:
            return "WEATHER_UNAVAILABLE"
        if idx < 0.25:
            return "LOW"
        elif idx < 0.50:
            return "MODERATE"
        elif idx < 0.75:
            return "HIGH"
        else:
            return "CRITICAL"

    df["risk_level"] = df.apply(assign_category, axis=1)
    df["early_warning"] = df["hybrid_risk_index"].apply(
        lambda x: bool(x is not None and x >= 0.50)
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Multi-Channel Alert Drafting
# ─────────────────────────────────────────────────────────────────────────────

def generate_alert_channels(row: pd.Series) -> dict[str, Any] | None:
    """
    Returns simulated alert messages for notified wards, or None if risk is LOW.
    ALL OUTPUTS ARE IN-PROCESS STRINGS ONLY. NO EXTERNAL DISPATCH SERVICE IS CONNECTED.
    """
    risk_level = row["risk_level"]
    if risk_level in ("LOW", "WEATHER_UNAVAILABLE"):
        return None

    ward = row["ward_name"]
    horizon = row.get("forecast_horizon", "next_24h")
    rain = row.get("forecast_rainfall_mm", 0.0)
    peak = row.get("peak_hourly_mm", 0.0)
    susc = row.get("ml_susceptibility", 0.0)
    idx = row.get("hybrid_risk_index", 0.0)
    imd_rain_cat = row.get("imd_24h_rainfall_category", "UNKNOWN")
    imd_int_cat = row.get("imd_intensity_category", "UNKNOWN")

    disclaimer = "(Prototype early-warning assessment — NOT an official IMD alert)"

    # 1. SMS draft (<160 chars)
    sms = (
        f"[Ward {ward}] SYSTEM RISK: {risk_level}. "
        f"Rain {rain:.0f}mm ({horizon}), peak {peak:.0f}mm/h. "
        f"Avoid waterlogged subways. {disclaimer}"
    )[:160]

    # 2. IVR Call Script
    ivr = (
        f"Namaste. Yeh Mumbai Flood Early Warning research prototype ka alert hai. "
        f"Ward {ward} ke liye agle {horizon} mein system risk level {risk_level} hai. "
        f"Anumanit baarish kareeb {rain:.0f} millimetre hai aur peak intensity {peak:.0f} millimetre prati ghanta. "
        f"Sachet rahein aur official civic guidelines follow karein. {disclaimer}."
    )

    # 3. Push Notification (WhatsApp / Citizen App)
    icons = {"MODERATE": "\U0001f7e1", "HIGH": "\U0001f7e0", "CRITICAL": "\U0001f534"}
    push = {
        "title": f"{icons.get(risk_level, '\u26a0\ufe0f')} Ward {ward} Flood Advisory: {risk_level}",
        "body": (
            f"Forecast: {rain:.1f}mm over {horizon} (Peak: {peak:.1f}mm/hr, {imd_int_cat}). "
            f"Spatial susceptibility: {susc:.4f}. Hybrid index: {idx:.3f}. {disclaimer}."
        ),
    }

    # 4. Ward Officer Dashboard Entry
    dashboard = (
        f"WARD={ward} | RISK={risk_level} | HYBRID_INDEX={idx:.3f} | "
        f"HORIZON={horizon} | RAIN={rain:.1f}mm ({imd_rain_cat}) | "
        f"PEAK_INTENSITY={peak:.1f}mm/hr ({imd_int_cat}) | "
        f"ML_SUSCEPTIBILITY={susc:.4f} | "
        f"STATUS=SIMULATED_DISSEMINATION"
    )

    return {
        "sms_draft": sms,
        "ivr_script": ivr,
        "push_notification": push,
        "dashboard_entry": dashboard,
    }
