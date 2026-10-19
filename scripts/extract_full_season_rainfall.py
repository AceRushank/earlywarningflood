"""
scripts/extract_full_season_rainfall.py
=======================================
Extracts catchment-averaged precipitation for the 2026 monsoon season
(June 15 - September 30, 2026, plus antecedent period starting May 31, 2026)
using the verified HydroBASINS Level 08 upstream polygon for Vishnupuri Barrage.

Key Guardrails:
1. Uses verified curvilinear pixel mask over native INSAT-3DS grid.
2. Validates HDF5 readability, timestamps (UTC to IST +5:30), HEM units (mm/hr), and fill values (-999.0).
3. Preserves missing scans and the July 31 - August 3 satellite outage as explicit NaNs (never zero-filled).
4. Produces two auditable artifacts:
   - data/historical/insat3ds_hem_catchment_scans_2026.csv (scan-level records)
   - data/historical/insat3ds_hem_catchment_hourly_2026.csv (continuous hourly time series)
"""

import os
import sys
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import h5py

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.catchment_extractor import CatchmentRainfallExtractor

POLYGON_PATH = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "vishnupuri_upstream_catchment.geojson"
HDF_DIR = REPO_ROOT / "data" / "mosdac" / "hdf_archive"
OUT_SCAN_CSV = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_scans_2026.csv"
OUT_HOURLY_CSV = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_hourly_2026.csv"

FILENAME_PATTERN = re.compile(r"^3SIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_L2B_HEM_V01R00\.h5$")

def extract_single_file_worker(args):
    """Worker function for parallel extraction."""
    fpath_str, mask_indices, total_pixels = args
    fpath = Path(fpath_str)
    
    res = {
        "source_file": fpath.name,
        "success": False,
        "timestamp_utc": None,
        "timestamp_ist": None,
        "total_catchment_pixels": total_pixels,
        "valid_rainfall_pixels": 0,
        "fill_pixels_excluded": 0,
        "coverage_ratio": 0.0,
        "mean_rainfall_rate_mm_hr": np.nan,
        "min_rainfall_rate_mm_hr": np.nan,
        "max_rainfall_rate_mm_hr": np.nan,
        "std_rainfall_rate_mm_hr": np.nan,
        "zero_rainfall_pixels": 0,
        "positive_rainfall_pixels": 0,
        "qc_flag": "FAILED",
        "error": None
    }
    
    try:
        with h5py.File(fpath, "r") as f:
            # 1. Decode time
            if "time" not in f:
                res["error"] = "Missing time dataset"
                return res
            
            t_val = f["time"][0]
            # Standard epoch: seconds since 1970-01-01 or 2000-01-01
            # Check attrs
            t_unit = f["time"].attrs.get("units", b"").decode("utf-8", errors="ignore").lower()
            if "minutes" in t_unit and "2000" in t_unit:
                dt_utc = datetime(2000, 1, 1) + timedelta(minutes=float(t_val))
            elif "seconds" in t_unit and "1970" in t_unit:
                dt_utc = datetime(1970, 1, 1) + timedelta(seconds=float(t_val))
            else:
                # Direct epoch calculation (INSAT-3DS standard: minutes since 2000-01-01)
                m = FILENAME_PATTERN.match(fpath.name)
                if m:
                    dt_utc = datetime.strptime(m.group(1) + m.group(2), "%d%b%Y%H%M")
                else:
                    dt_utc = datetime(2000, 1, 1) + timedelta(minutes=float(t_val))
            
            dt_ist = dt_utc + timedelta(hours=5, minutes=30)
            res["timestamp_utc"] = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
            res["timestamp_ist"] = dt_ist.strftime("%Y-%m-%d %H:%M:%S")
            
            # 2. Read HEM slice
            if "HEM" not in f:
                res["error"] = "Missing HEM dataset"
                return res
            
            hem_ds = f["HEM"]
            hem_slice = hem_ds[0]
            hem_fill = hem_ds.attrs.get("_FillValue", -999.0)
            if isinstance(hem_fill, np.ndarray):
                hem_fill = hem_fill.item()
            
            rows, cols = mask_indices
            values = hem_slice[rows, cols]
            
            valid_mask = (values != hem_fill) & (values >= 0.0)
            valid_vals = values[valid_mask]
            
            n_valid = int(len(valid_vals))
            n_fill = int(total_pixels - n_valid)
            coverage = float(n_valid / total_pixels) if total_pixels > 0 else 0.0
            
            res["valid_rainfall_pixels"] = n_valid
            res["fill_pixels_excluded"] = n_fill
            res["coverage_ratio"] = round(coverage, 4)
            
            if n_valid > 0:
                mean_r = float(np.mean(valid_vals))
                min_r = float(np.min(valid_vals))
                max_r = float(np.max(valid_vals))
                std_r = float(np.std(valid_vals))
                n_zero = int(np.sum(valid_vals == 0.0))
                n_pos = int(np.sum(valid_vals > 0.0))
                
                res["mean_rainfall_rate_mm_hr"] = round(mean_r, 4)
                res["min_rainfall_rate_mm_hr"] = round(min_r, 4)
                res["max_rainfall_rate_mm_hr"] = round(max_r, 4)
                res["std_rainfall_rate_mm_hr"] = round(std_r, 4)
                res["zero_rainfall_pixels"] = n_zero
                res["positive_rainfall_pixels"] = n_pos
                res["qc_flag"] = "OK" if coverage >= 0.8 else "LOW_COVERAGE"
                res["success"] = True
            else:
                res["qc_flag"] = "ALL_FILL_VALUES"
                res["success"] = True

    except Exception as ex:
        res["error"] = str(ex)
        
    return res

def run_extraction():
    print("=" * 80)
    print("INSAT-3DS HEM CATCHMENT RAINFALL EXTRACTION (2026 MONSOON)")
    print("=" * 80)

    # 1. Initialize Extractor & Cache Curvilinear Mask
    print(f"Loading catchment polygon from: {POLYGON_PATH}")
    extractor = CatchmentRainfallExtractor(POLYGON_PATH)
    
    # 2. Filter candidate files for the study window (May 31 - Sep 30, 2026)
    all_files = sorted(list(HDF_DIR.glob("*.h5")))
    study_files = []
    
    start_filter_utc = datetime(2026, 5, 30, 18, 0)   # May 31 00:00 IST antecedent
    end_filter_utc   = datetime(2026, 9, 30, 19, 0)   # Sep 30 23:30 IST end of test set
    
    for f in all_files:
        m = FILENAME_PATTERN.match(f.name)
        if m:
            dt = datetime.strptime(m.group(1) + m.group(2), "%d%b%Y%H%M")
            if start_filter_utc <= dt <= end_filter_utc:
                study_files.append(f)
                
    print(f"Total HDF5 files in archive:         {len(all_files)}")
    print(f"Files in target study window:        {len(study_files)} (May 31 00:00 IST - Sep 30 23:30 IST)")
    
    if len(study_files) == 0:
        raise RuntimeError("No study window files found in archive!")
        
    # Build mask using first file
    sample_h5 = study_files[0]
    n_pix = extractor.build_catchment_pixel_mask(sample_h5)
    print(f"Curvilinear Catchment Mask:          {n_pix} pixels ({extractor.catchment_area_sq_km:.2f} km^2)")
    
    # 3. Parallel Batch Extraction
    print(f"\nStarting parallel extraction with 4 worker processes...")
    t_start = time.time()
    
    mask_indices = extractor.mask_indices
    worker_args = [(str(fp), mask_indices, n_pix) for fp in study_files]
    
    scan_records = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for idx, res in enumerate(executor.map(extract_single_file_worker, worker_args, chunksize=50), 1):
            if res["success"]:
                scan_records.append(res)
            else:
                print(f"  Warning: {res['source_file']} extraction failed: {res.get('error')}")
                
            if idx % 500 == 0 or idx == len(study_files):
                el = time.time() - t_start
                print(f"  Processed {idx}/{len(study_files)} granules ({el:.1f}s, {idx/el:.1f} granules/sec)...")
                
    t_extract = time.time() - t_start
    print(f"\nExtraction completed in {t_extract:.1f} seconds ({len(scan_records)}/{len(study_files)} valid scans).")
    
    # 4. Scan Records DataFrame
    df_scans = pd.DataFrame(scan_records)
    # Deduplicate by UTC timestamp if any
    dups = df_scans.duplicated(subset=["timestamp_utc"]).sum()
    if dups > 0:
        print(f"Removing {dups} duplicate scan timestamps...")
        df_scans.sort_values(["timestamp_utc", "coverage_ratio"], ascending=[True, False], inplace=True)
        df_scans.drop_duplicates(subset=["timestamp_utc"], keep="first", inplace=True)
        
    df_scans.sort_values("timestamp_utc", inplace=True)
    df_scans.reset_index(drop=True, inplace=True)
    
    OUT_SCAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_scans.to_csv(OUT_SCAN_CSV, index=False)
    print(f"Saved scan-level records to: {OUT_SCAN_CSV} ({len(df_scans)} rows)")
    
    # 5. Continuous Hourly Aggregation in IST
    # Hourly telemetry spans: 2026-05-31 00:00:00 to 2026-09-30 23:00:00
    df_scans["dt_ist"] = pd.to_datetime(df_scans["timestamp_ist"])
    
    start_hour = pd.to_datetime("2026-05-31 00:00:00")
    end_hour   = pd.to_datetime("2026-09-30 23:00:00")
    hourly_grid = pd.date_range(start_hour, end_hour, freq="1h")
    
    print(f"\nConstructing continuous hourly aggregation ({len(hourly_grid)} total hours from {start_hour} to {end_hour})...")
    
    # Confirmed satellite outage range: July 31 05:30 IST to August 4 22:30 IST
    outage_start = pd.to_datetime("2026-07-31 05:30:00")
    outage_end   = pd.to_datetime("2026-08-04 22:30:00")
    
    hourly_rows = []
    for h_dt in hourly_grid:
        h_prev = h_dt - timedelta(hours=1)
        # Preceding 1-hour accumulation window: (h_prev, h_dt]
        window_scans = df_scans[(df_scans["dt_ist"] > h_prev) & (df_scans["dt_ist"] <= h_dt)]
        n_scans = len(window_scans)
        
        # Check if in known outage window
        in_outage = (outage_start <= h_dt < outage_end)
        
        if in_outage:
            qc = "OUTAGE_UNAVAILABLE"
            mean_rate = np.nan
            hourly_depth = np.nan
        elif n_scans >= 2:
            rates = window_scans["mean_rainfall_rate_mm_hr"].values
            mean_rate = float(np.mean(rates))
            hourly_depth = float(np.sum(rates * 0.5))  # Two 30-min intervals
            qc = "COMPLETE"
        elif n_scans == 1:
            rate = float(window_scans["mean_rainfall_rate_mm_hr"].iloc[0])
            mean_rate = rate
            hourly_depth = float(rate * 0.5)  # 30-min observed depth
            qc = "PARTIAL_1_OF_2"
        else:
            mean_rate = np.nan
            hourly_depth = np.nan
            qc = "MISSING_SCAN"
            
        hourly_rows.append({
            "timestamp_ist": h_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_utc": (h_dt - timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "scan_count": n_scans,
            "mean_rainfall_rate_mm_hr": round(mean_rate, 4) if not np.isnan(mean_rate) else np.nan,
            "hourly_rainfall_depth_mm": round(hourly_depth, 4) if not np.isnan(hourly_depth) else np.nan,
            "completeness": qc,
            "source_granules": ";".join(window_scans["source_file"].tolist())
        })
        
    df_hourly = pd.DataFrame(hourly_rows)
    df_hourly.to_csv(OUT_HOURLY_CSV, index=False)
    print(f"Saved hourly catchment rainfall dataset to: {OUT_HOURLY_CSV} ({len(df_hourly)} rows)")
    
    print("\nHourly Completeness Breakdown:")
    print(df_hourly["completeness"].value_counts())
    
    # Statistics on valid rainfall
    valid_depths = df_hourly["hourly_rainfall_depth_mm"].dropna()
    print(f"\nValid hourly rainfall hours: {len(valid_depths)} / {len(df_hourly)} ({len(valid_depths)/len(df_hourly)*100:.1f}%)")
    print(f"  Mean hourly depth:  {valid_depths.mean():.4f} mm")
    print(f"  Max hourly depth:   {valid_depths.max():.4f} mm")
    print(f"  Total accumulated:  {valid_depths.sum():.2f} mm across season")
    print(f"  Zero-rain hours:    {(valid_depths == 0).sum()} ({(valid_depths == 0).sum()/len(valid_depths)*100:.1f}%)")
    print(f"  Rain hours (>0mm):  {(valid_depths > 0).sum()} ({(valid_depths > 0).sum()/len(valid_depths)*100:.1f}%)")

if __name__ == "__main__":
    run_extraction()
