"""
app/data_loader.py

Loads and joins spatial data sources:
  1. BMC ward boundary GeoJSON (CC BY 4.0, GitHub / local cache fallback)
  2. Mumbai ward flood history CSV (Jalem et al. 2026, Earth 7(3):91, CC BY)
  3. Locality -> ward-code crosswalk (ward_locality_crosswalk.csv)
  4. Static physiographic features (ward_geographic_features.csv)

Computes:
  - Exact ward area in km² using metric projection UTM Zone 43N (EPSG:32643)
  - Geometric centroids for weather query locations
  - Historical flood score (mean flooded km²/yr, 2018-2025)
  - Inundation ratio = historical_flood_score / ward_area_km2
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

log = logging.getLogger(__name__)

GEOJSON_URL = (
    "https://raw.githubusercontent.com/sanjanakrishnan/mumbai_spatial_data"
    "/main/BMC_admin_wards.geojson"
)

_ROOT = Path(__file__).resolve().parent.parent
_FLOOD_CSV     = _ROOT / "data" / "mumbai_ward_flood_history.csv"
_CROSSWALK_CSV = _ROOT / "ward_locality_crosswalk.csv"
_GEO_CSV       = _ROOT / "data" / "ward_geographic_features.csv"
_TMP_GEOJSON   = _ROOT / "data" / "_bmc_wards_tmp.geojson"

_YEAR_COLS = [f"flooded_km2_{y}" for y in range(2018, 2026)]

GEO_FEATURE_COLS = [
    "elevation_mean",
    "slope_mean",
    "builtup_pct",
    "water_pct",
    "vegetation_pct",
    "dist_to_water_mean",
]


def load_ward_data() -> pd.DataFrame:
    """
    Returns DataFrame with one row per BMC ward containing:
      ward_name, bmc_ward_code, ward_area_km2, lat, lon,
      historical_flood_score, inundation_ratio, and GEO_FEATURE_COLS.
    """
    gdf = _load_geojson()
    flood_df = _load_flood_history()
    geo_df = _load_geo_features()
    merged = _join_data(gdf, flood_df, geo_df)
    return merged


def _load_geojson() -> gpd.GeoDataFrame:
    """Loads GeoJSON boundaries with local cache fallback."""
    try:
        log.info("Fetching ward boundary GeoJSON from GitHub…")
        resp = requests.get(GEOJSON_URL, timeout=10)
        resp.raise_for_status()
        _TMP_GEOJSON.write_bytes(resp.content)
    except Exception as exc:
        if _TMP_GEOJSON.exists():
            log.warning("Could not fetch GeoJSON from GitHub (%s); using local cache.", exc)
        else:
            raise

    gdf = gpd.read_file(str(_TMP_GEOJSON))
    gdf_metric = gdf.to_crs(epsg=32643)  # UTM zone 43N for Mumbai metric area
    gdf["ward_area_km2"] = (gdf_metric.area / 1e6).round(3)
    gdf["centroid"] = gdf_metric.centroid.to_crs(epsg=4326)

    gdf["lat"] = gdf["centroid"].y
    gdf["lon"] = gdf["centroid"].x
    gdf = gdf.rename(columns={"name": "bmc_ward_code"})
    gdf["bmc_ward_code"] = gdf["bmc_ward_code"].str.strip()

    log.info("GeoJSON loaded. Ward codes: %s", sorted(gdf["bmc_ward_code"].tolist()))
    return gdf[["bmc_ward_code", "ward_area_km2", "lat", "lon", "geometry"]]


def _load_flood_history() -> pd.DataFrame:
    flood_raw = pd.read_csv(_FLOOD_CSV)
    crosswalk = pd.read_csv(_CROSSWALK_CSV)

    flood_raw["locality_name"] = flood_raw["locality_name"].str.strip()
    crosswalk["locality_name"] = crosswalk["locality_name"].str.strip()
    crosswalk["bmc_ward_code"] = crosswalk["bmc_ward_code"].str.strip()

    expanded_rows = []
    for _, row in crosswalk.iterrows():
        for wc in [w.strip() for w in row["bmc_ward_code"].split(" or ")]:
            expanded_rows.append({
                "locality_name": row["locality_name"],
                "bmc_ward_code": wc,
                "confidence": row["confidence"],
            })
    crosswalk_expanded = pd.DataFrame(expanded_rows)
    merged = crosswalk_expanded.merge(flood_raw, on="locality_name", how="left")

    ward_agg = merged.groupby("bmc_ward_code")[_YEAR_COLS].mean().reset_index()
    ward_agg["historical_flood_score"] = ward_agg[_YEAR_COLS].mean(axis=1)

    log.info("Flood history resolved for %d ward codes.", len(ward_agg))
    return ward_agg[["bmc_ward_code", "historical_flood_score"]]


def _load_geo_features() -> pd.DataFrame:
    if not _GEO_CSV.exists():
        log.warning("ward_geographic_features.csv not found at %s.", _GEO_CSV)
        return pd.DataFrame(columns=["bmc_ward_code"] + GEO_FEATURE_COLS)

    geo = pd.read_csv(_GEO_CSV, comment="#")
    geo["bmc_ward_code"] = geo["bmc_ward_code"].str.strip()
    return geo[["bmc_ward_code"] + GEO_FEATURE_COLS]


def _join_data(
    gdf: gpd.GeoDataFrame,
    flood_df: pd.DataFrame,
    geo_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = gdf.merge(flood_df, on="bmc_ward_code", how="left")

    # Compute physical inundation ratio before filling missing
    merged["inundation_ratio"] = (merged["historical_flood_score"] / merged["ward_area_km2"]).round(6)

    matched = merged[merged["historical_flood_score"].notna()]["bmc_ward_code"].tolist()
    unmatched = merged[merged["historical_flood_score"].isna()]["bmc_ward_code"].tolist()

    if unmatched:
        log.warning("Wards with NO historical flood data (defaulting to 0): %s", unmatched)

    merged["historical_flood_score"] = merged["historical_flood_score"].fillna(0.0)
    merged["inundation_ratio"] = merged["inundation_ratio"].fillna(0.0)

    if not geo_df.empty:
        merged = merged.merge(geo_df, on="bmc_ward_code", how="left")
        for col in GEO_FEATURE_COLS:
            merged[col] = merged[col].fillna(0.0)
    else:
        for col in GEO_FEATURE_COLS:
            merged[col] = 0.0

    merged["ward_name"] = merged["bmc_ward_code"]

    out_cols = [
        "ward_name", "bmc_ward_code", "ward_area_km2", "lat", "lon",
        "historical_flood_score", "inundation_ratio",
    ] + GEO_FEATURE_COLS
    return merged[out_cols].copy()
