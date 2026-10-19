"""
scripts/extract_sep2026_rainfall.py
===================================
Extracts catchment-averaged precipitation for September 8-22, 2026 INSAT-3DS HEM granules
using the verified HydroBASINS Level 08 upstream polygon for Vishnupuri Barrage.
"""

import os
import sys
import glob
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.catchment_extractor import CatchmentRainfallExtractor

POLYGON_PATH = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "vishnupuri_upstream_catchment.geojson"
HDF_DIR = REPO_ROOT / "data" / "mosdac" / "hdf_archive"
OUT_SCAN_CSV = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_scans_sep2026.csv"
OUT_HOURLY_CSV = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_hourly_sep2026.csv"

def run_extraction():
    print("=" * 70)
    print("STARTING INSAT-3DS HEM EXTRACTION FOR SEPTEMBER 8-22, 2026")
    print("=" * 70)
    
    if not POLYGON_PATH.exists():
        raise FileNotFoundError(f"Polygon not found: {POLYGON_PATH}")
    
    files = sorted(glob.glob(str(HDF_DIR / "*.h5")))
    print(f"Total HDF5 files found: {len(files)}")
    if len(files) == 0:
        raise RuntimeError("No HDF5 files found in archive!")

    # 1. Initialize Extractor & Cache Mask
    t0 = time.time()
    extractor = CatchmentRainfallExtractor(POLYGON_PATH)
    sample_h5 = Path(files[0])
    n_pix = extractor.build_catchment_pixel_mask(sample_h5)
    print(f"Curvilinear Catchment Mask Built: {n_pix} pixels")
    print(f"Catchment Area: {extractor.catchment_area_sq_km:.2f} km^2")
    print(f"Mask Initialization Time: {time.time() - t0:.2f}s\n")

    # 2. Extract each file
    scan_records = []
    t_start = time.time()
    
    for idx, fpath in enumerate(files, 1):
        fp = Path(fpath)
        res = extractor.extract_single_timestep(fp)
        if res["success"]:
            scan_records.append({
                "source_file": fp.name,
                "timestamp_utc": res["timestamp_utc"],
                "timestamp_ist": res["timestamp_ist"],
                "total_catchment_pixels": res["total_catchment_pixels"],
                "valid_rainfall_pixels": res["valid_rainfall_pixels"],
                "fill_pixels_excluded": res["fill_pixels_excluded"],
                "coverage_ratio": res["coverage_ratio"],
                "mean_rainfall_rate_mm_hr": res["mean_rainfall_mm_hr"],
                "min_rainfall_rate_mm_hr": res["min_rainfall_mm_hr"],
                "max_rainfall_rate_mm_hr": res["max_rainfall_mm_hr"],
                "std_rainfall_rate_mm_hr": res["std_rainfall_mm_hr"],
                "zero_rainfall_pixels": res["zero_rainfall_pixels"],
                "positive_rainfall_pixels": res["positive_rainfall_pixels"],
                "qc_flag": res["qc_flag"]
            })
        else:
            print(f"Warning: Extraction failed for {fp.name}: {res.get('error')}")

        if idx % 100 == 0 or idx == len(files):
            elapsed = time.time() - t_start
            print(f"  Processed {idx}/{len(files)} files ({elapsed:.1f}s, {idx/elapsed:.1f} fps)...")

    df_scans = pd.DataFrame(scan_records)
    print(f"\nExtraction complete in {time.time() - t_start:.2f}s!")
    print(f"Extracted {len(df_scans)} valid scan records.")

    # Deduplicate if necessary
    dups = df_scans.duplicated(subset=["timestamp_utc"]).sum()
    print(f"Duplicate scan timestamps: {dups}")
    if dups > 0:
        df_scans.sort_values(["timestamp_utc", "coverage_ratio"], ascending=[True, False], inplace=True)
        df_scans.drop_duplicates(subset=["timestamp_utc"], keep="first", inplace=True)

    df_scans.sort_values("timestamp_utc", inplace=True)
    df_scans.reset_index(drop=True, inplace=True)
    
    OUT_SCAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_scans.to_csv(OUT_SCAN_CSV, index=False)
    print(f"Saved scan-level records to: {OUT_SCAN_CSV} ({len(df_scans)} rows)")

    # 3. Construct Hourly Aggregation
    # Scan IST times
    df_scans["dt_ist"] = pd.to_datetime(df_scans["timestamp_ist"])
    df_scans["dt_utc"] = pd.to_datetime(df_scans["timestamp_utc"])

    # Hourly grid in IST
    # Hourly telemetry is stamped at the top of the hour: 2026-09-08 06:00:00 to 2026-09-22 19:00:00
    # For a stage report at time T (e.g., 06:00 IST), the preceding 1-hour window is (T - 1h, T],
    # containing scans at T - 30min and T.
    min_ist = df_scans["dt_ist"].min()  # 2026-09-08 05:30:00
    max_ist = df_scans["dt_ist"].max()  # 2026-09-22 19:00:00

    # Hourly grid starting at 06:00 IST (first hour with full 1h preceding window) through 19:00 IST
    start_hour = min_ist.ceil("1h")
    end_hour = max_ist.floor("1h")
    hourly_grid = pd.date_range(start_hour, end_hour, freq="1h")

    print(f"\nConstructing hourly rainfall series from {start_hour} to {end_hour} ({len(hourly_grid)} expected hours)...")

    hourly_rows = []
    for h_dt in hourly_grid:
        h_prev = h_dt - timedelta(hours=1)
        # Preceding 1-hour window: (h_prev, h_dt]
        window_scans = df_scans[(df_scans["dt_ist"] > h_prev) & (df_scans["dt_ist"] <= h_dt)]
        
        n_scans = len(window_scans)
        granules = ";".join(window_scans["source_file"].tolist())
        
        if n_scans >= 2:
            # Standard 30-min cadence: two scans in 1 hour
            rates = window_scans["mean_rainfall_rate_mm_hr"].values
            mean_rate = float(np.mean(rates))
            # Hourly accumulated depth = sum of 30-min depths (rate * 0.5h)
            hourly_depth = float(np.sum(rates * 0.5))
            qc = "COMPLETE"
        elif n_scans == 1:
            # Single scan observed in 1-hour window
            rate = float(window_scans["mean_rainfall_rate_mm_hr"].iloc[0])
            mean_rate = rate
            # If 1 scan represents 30 min, depth is rate * 0.5h, or rate * 1.0h if assumed steady
            # We document conservative 0.5h depth for the observed scan, and report rate
            hourly_depth = float(rate * 0.5)
            qc = "PARTIAL_1_OF_2"
        else:
            mean_rate = np.nan
            hourly_depth = np.nan
            qc = "MISSING"

        hourly_rows.append({
            "timestamp_ist": h_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_utc": (h_dt - timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "scan_count": n_scans,
            "mean_rainfall_rate_mm_hr": round(mean_rate, 4) if not np.isnan(mean_rate) else np.nan,
            "hourly_rainfall_depth_mm": round(hourly_depth, 4) if not np.isnan(hourly_depth) else np.nan,
            "completeness": qc,
            "source_granules": granules
        })

    df_hourly = pd.DataFrame(hourly_rows)
    df_hourly.to_csv(OUT_HOURLY_CSV, index=False)
    print(f"Saved hourly aggregated rainfall to: {OUT_HOURLY_CSV} ({len(df_hourly)} hours)")
    print("\nCompleteness breakdown:")
    print(df_hourly["completeness"].value_counts())
    print("\nRainfall summary:")
    print(df_hourly[["mean_rainfall_rate_mm_hr", "hourly_rainfall_depth_mm"]].describe())

if __name__ == "__main__":
    run_extraction()
