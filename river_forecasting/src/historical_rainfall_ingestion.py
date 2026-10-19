"""
river_forecasting/src/historical_rainfall_ingestion.py
======================================================
Historical Rainfall Ingestion & Catchment Physiography Pipeline.

Ingests and processes:
1. Official IMD 0.25° Daily Gridded Rainfall Archive (1969-2024, Pune CRS).
   - Extracts daily catchment-average rainfall across 71 cells intersecting
     the 51,768.25 km² Vishnupuri upstream catchment.
   - Computes 1-day, 3-day, and 7-day antecedent accumulations.
   - Preserves source integrity: Never mixes daily with half-hourly satellite data.
2. Catchment Physiographic Attributes:
   - Delineated area, pour point coordinates, reach length, channel slope,
     and mean basin elevation.
3. Source Provenance & Coverage Metadata:
   - Explicit data source labels, resolution, missing data indicators, and units.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPATIAL_GEOJSON = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "vishnupuri_upstream_catchment.geojson"
RAW_NC_DIR = REPO_ROOT / "data" / "historical" / "raw_nc"

OUT_DAILY_CSV = REPO_ROOT / "river_forecasting" / "outputs" / "historical_catchment_rainfall_daily.csv"
OUT_PARAMS_JSON = REPO_ROOT / "river_forecasting" / "outputs" / "catchment_physiographic_parameters.json"
OUT_METADATA_JSON = REPO_ROOT / "river_forecasting" / "outputs" / "rainfall_sources_metadata.json"


def extract_catchment_imd_daily(year: int = 2024) -> pd.DataFrame:
    """
    Extracts daily catchment-average rainfall for Vishnupuri from the official
    IMD 0.25-degree NetCDF archive for the given year.
    """
    nc_path = RAW_NC_DIR / f"ind{year}_rfp25.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"IMD NetCDF file not found: {nc_path}")
    if not SPATIAL_GEOJSON.exists():
        raise FileNotFoundError(f"Catchment GeoJSON not found: {SPATIAL_GEOJSON}")

    gdf = gpd.read_file(SPATIAL_GEOJSON)
    poly = gdf.geometry.iloc[0]

    with xr.open_dataset(nc_path) as ds:
        lats = ds["LATITUDE"].values
        lons = ds["LONGITUDE"].values

        # Identify all 0.25-degree grid cells inside the catchment polygon
        lat_idx, lon_idx = [], []
        cell_coords = []
        for i, lat in enumerate(lats):
            if 18.5 <= lat <= 20.75:
                for j, lon in enumerate(lons):
                    if 73.25 <= lon <= 77.5:
                        if poly.contains(Point(lon, lat)):
                            lat_idx.append(i)
                            lon_idx.append(j)
                            cell_coords.append((float(lat), float(lon)))

        # Extract rainfall matrix: shape (TIME, N_CELLS)
        rf_raw = ds["RAINFALL"].values[:, lat_idx, lon_idx]
        # Filter invalid fill values (IMD uses -999.0 or values > 1000 for missing)
        rf_clean = np.where((rf_raw < 0) | (rf_raw > 1000), np.nan, rf_raw)

        # Spatial average across contributing grid cells
        daily_mean = np.nanmean(rf_clean, axis=1)
        valid_cells_count = np.sum(~np.isnan(rf_clean), axis=1)

        times = pd.to_datetime(ds["TIME"].values)

    df = pd.DataFrame({
        "date": times.strftime("%Y-%m-%d"),
        "year": times.year,
        "doy": times.dayofyear,
        "rainfall_1d_mm": np.round(daily_mean, 2),
        "active_cells_observed": valid_cells_count,
        "total_catchment_cells": len(cell_coords),
        "source": "IMD_0.25DEG_DAILY_GRIDDED"
    })

    # Antecedent rolling accumulations (strictly antecedent including current day)
    df["rainfall_3d_accum_mm"] = np.round(df["rainfall_1d_mm"].rolling(3, min_periods=1).sum(), 2)
    df["rainfall_7d_accum_mm"] = np.round(df["rainfall_1d_mm"].rolling(7, min_periods=1).sum(), 2)

    return df


def generate_catchment_parameters() -> Dict[str, Any]:
    """Compiles defensible catchment physiography and station parameters."""
    return {
        "station_information": {
            "name": "Vishnupuri Barrage",
            "project_name": "Shankar Sagar Jalashay",
            "river": "Godavari",
            "sub_basin": "Upper Godavari Basin",
            "district": "Nanded",
            "state": "Maharashtra",
            "coordinates": {
                "latitude_deg": 19.126944,
                "longitude_deg": 77.280833,
                "pour_point_crs": "EPSG:4326"
            },
            "hydraulic_levels": {
                "gauge_zero_rl_m": 0.0,
                "mean_sea_level_m": 366.50,
                "full_reservoir_level_m": 355.00,
                "warning_level_m": 354.00,
                "danger_level_m": 354.80,
                "lowest_observed_event1_m": 353.388,
                "peak_observed_event1_m": 354.798
            }
        },
        "catchment_physiography": {
            "drainage_area_hydrobasins_sq_km": 51768.25,
            "drainage_area_cwc_benchmark_sq_km": 51390.78,
            "area_discrepancy_pct": 0.73,
            "delineation_source": "HydroBASINS Asia Level 08 (WWF/USGS)",
            "dem_foundation": "HydroSHEDS 3-arc-second / 90m SRTM conditioned DEM",
            "main_channel_length_km": 650.0,
            "headwater_origin": "Trimbakeshwar, Western Ghats, Nashik",
            "headwater_elevation_m_msl": 1067.0,
            "outlet_elevation_m_msl": 350.5,
            "average_river_channel_slope_m_per_km": 1.10,
            "average_river_channel_slope_pct": 0.11,
            "estimated_mean_basin_elevation_m_msl": 540.0,
            "estimated_mean_basin_slope_pct": 2.80,
            "contributing_subbasins_count": 67
        },
        "meteorological_coverage": {
            "imd_gridded_resolution_deg": 0.25,
            "imd_grid_cells_inside_catchment": 71,
            "insat3ds_hem_curvilinear_pixels": 2869,
            "insat3ds_spatial_resolution_approx_km": 4.0
        }
    }


def generate_source_metadata() -> Dict[str, Any]:
    """Generates explicit source metadata distinguishing IMD daily from satellite rainfall."""
    return {
        "sources": {
            "source_1_imd_daily": {
                "name": "India Meteorological Department (IMD) Gridded Daily Rainfall",
                "center": "Climate Research and Services (CRS), IMD Pune",
                "spatial_resolution": "0.25 degree x 0.25 degree (~27.5 km)",
                "temporal_resolution": "Daily (08:30 IST to 08:30 IST / 24-hour total)",
                "catchment_cells": 71,
                "units": "millimeters per day (mm/day)",
                "historical_period_available": "1969 to 2024",
                "missing_data_policy": "Flagged as NaN; excluded from spatial mean; no synthetic zero-imputation",
                "usage_role": "Medium-range antecedent wetness context (1-day, 3-day, 7-day catchment accumulation)"
            },
            "source_2_mosdac_insat3ds_hem": {
                "name": "ISRO/MOSDAC INSAT-3DS Hydro-Estimator Method (HEM)",
                "product_id": "3SIMG_L2B_HEM_V01R00",
                "center": "Space Applications Centre (SAC / ISRO), MOSDAC",
                "spatial_resolution": "Native 4 km geostationary curvilinear projection",
                "temporal_resolution": "Half-hourly (aggregated to hourly)",
                "catchment_curvilinear_pixels": 2869,
                "units": "Instantaneous rate (mm/hr) and hourly depth (mm)",
                "event1_coverage_window": "2024-07-31 05:00 IST to 2024-08-05 23:00 IST",
                "event1_catalog_accounting": {
                    "total_slots": 288,
                    "valid_readable_granules": 275,
                    "catalog_gaps": 9,
                    "server_truncated_corrupted": 4
                },
                "missing_data_policy": "Strict NaN preservation; partial hours explicitly flagged; zero forward/backward filling",
                "usage_role": "High-frequency convective rain tracking for Event 1 rising-limb and lag correlation analysis"
            },
            "source_3_cwc_stage_telemetry": {
                "name": "Central Water Commission (CWC) / Maharashtra WRD Telemetry",
                "station": "Vishnupuri Barrage (Nanded)",
                "sensor_type": "Radar / Pressure Transducer River Stage Telemetry",
                "temporal_resolution": "Hourly",
                "units": "Meters above mean sea level (m MSL)",
                "quality_flags": "VALID, ESTIMATED, MISSING",
                "usage_role": "Hydrological forecast target and autoregressive stage feature foundation"
            }
        }
    }


def run_ingestion():
    print("=" * 80)
    print("RUNNING HISTORICAL RAINFALL INGESTION & PARAMETER EXPORT")
    print("=" * 80)

    # 1. Extract IMD 2024 daily rainfall
    print("\n[STEP 1] Extracting IMD 0.25-degree daily catchment rainfall for 2024...")
    df_imd = extract_catchment_imd_daily(2024)
    df_imd.to_csv(OUT_DAILY_CSV, index=False)
    print(f"  -> Saved {len(df_imd)} daily rows to: {OUT_DAILY_CSV}")
    jul_aug = df_imd[(df_imd['date'] >= '2024-07-01') & (df_imd['date'] <= '2024-08-31')]
    print(f"  -> July-August 2024 total rainfall: {jul_aug['rainfall_1d_mm'].sum():.2f} mm across 62 days")

    # 2. Export physiographic parameters
    print("\n[STEP 2] Exporting catchment physiographic parameters...")
    params = generate_catchment_parameters()
    with open(OUT_PARAMS_JSON, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  -> Saved parameters JSON: {OUT_PARAMS_JSON}")

    # 3. Export metadata
    print("\n[STEP 3] Exporting source metadata & coverage accounting...")
    metadata = generate_source_metadata()
    with open(OUT_METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> Saved metadata JSON: {OUT_METADATA_JSON}")

    print("\n[COMPLETED] Historical rainfall and catchment parameters successfully generated.")


if __name__ == "__main__":
    run_ingestion()
