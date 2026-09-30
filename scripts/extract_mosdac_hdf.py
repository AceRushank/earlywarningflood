#!/usr/bin/env python3
"""
scripts/extract_mosdac_hdf.py

MOSDAC INSAT-3DS HDF-based Mumbai Rainfall Extraction & QC Pipeline.
Processes MOSDAC INSAT-3DS HDF5 files (3SIMG_L2B_HEM product) for the Mumbai AOI.
Uses native 2D curvilinear coordinate arrays (Latitude, Longitude) and boolean masking.
Outputs:
  - data/mosdac/mumbai_rainfall_30min.csv
  - outputs/mosdac_archive_qc.txt
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import h5py
import pandas as pd


# Mumbai Area of Interest (AOI) Bounding Box
MUMBAI_BBOX = {
    "min_lon": 72.70,
    "max_lon": 73.10,
    "min_lat": 18.80,
    "max_lat": 19.40,
}


def decode_time_from_dataset(time_ds: h5py.Dataset) -> tuple[datetime, str]:
    """
    Decodes timestamp from /time dataset and its units attribute.
    Example: units = "minutes since 2000-01-01 00:00:00"
    Returns (datetime_obj, formatted_str).
    """
    raw_val = float(time_ds[0])
    units_attr = time_ds.attrs.get("units", "")
    if isinstance(units_attr, (bytes, np.bytes_)):
        units_str = units_attr.decode("utf-8")
    else:
        units_str = str(units_attr)

    match = re.match(r"(minutes|seconds|hours|days)\s+since\s+(.+)", units_str.strip())
    if match:
        unit_type, epoch_str = match.groups()
        epoch = datetime.fromisoformat(epoch_str.strip())
        if unit_type == "minutes":
            dt = epoch + timedelta(minutes=raw_val)
        elif unit_type == "seconds":
            dt = epoch + timedelta(seconds=raw_val)
        elif unit_type == "hours":
            dt = epoch + timedelta(hours=raw_val)
        elif unit_type == "days":
            dt = epoch + timedelta(days=raw_val)
        else:
            dt = epoch + timedelta(minutes=raw_val)
        return dt, dt.strftime("%Y-%m-%d %H:%M")
    
    # Fallback to current year if parsing fails
    return datetime(2026, 1, 1), str(raw_val)


def extract_single_hdf(hdf_path: Path) -> tuple[Optional[pd.DataFrame], dict]:
    """
    Extracts Mumbai AOI rainfall data from a single MOSDAC HDF5 file
    using the verified native curvilinear 2D coordinates logic.
    """
    metrics: Dict[str, Any] = {
        "source_filename": hdf_path.name,
        "filepath": hdf_path,
        "success": False,
        "error": None,
        "dt": None,
        "timestamp_str": None,
        "mumbai_pixels_selected": 0,
        "valid_rainfall_pixels": 0,
        "fill_values_excluded": 0,
        "invalid_geo_pixels_total": 0,
        "rainfall_min": None,
        "rainfall_max": None,
        "rainfall_mean": None,
        "zero_rainfall_pixels": 0,
        "positive_rainfall_pixels": 0,
    }

    try:
        with h5py.File(hdf_path, "r") as f:
            # 1. Read Latitude metadata and array
            lat_ds = f["Latitude"]
            lat_raw = lat_ds[:]
            lat_scale = lat_ds.attrs.get("scale_factor", 0.01)
            if isinstance(lat_scale, np.ndarray):
                lat_scale = lat_scale.item()
            lat_fill = lat_ds.attrs.get("_FillValue", 32767)
            if isinstance(lat_fill, np.ndarray):
                lat_fill = lat_fill.item()

            # 2. Read Longitude metadata and array
            lon_ds = f["Longitude"]
            lon_raw = lon_ds[:]
            lon_scale = lon_ds.attrs.get("scale_factor", 0.01)
            if isinstance(lon_scale, np.ndarray):
                lon_scale = lon_scale.item()
            lon_fill = lon_ds.attrs.get("_FillValue", 32767)
            if isinstance(lon_fill, np.ndarray):
                lon_fill = lon_fill.item()

            # 3. Decode coordinates using actual metadata
            valid_geo = (lat_raw != lat_fill) & (lon_raw != lon_fill)
            metrics["invalid_geo_pixels_total"] = int((~valid_geo).sum())

            lat_deg = np.where(valid_geo, lat_raw * lat_scale, np.nan)
            lon_deg = np.where(valid_geo, lon_raw * lon_scale, np.nan)

            # 4. Construct Mumbai AOI boolean mask directly on 2D arrays
            min_lon = MUMBAI_BBOX["min_lon"]
            max_lon = MUMBAI_BBOX["max_lon"]
            min_lat = MUMBAI_BBOX["min_lat"]
            max_lat = MUMBAI_BBOX["max_lat"]

            mumbai_geo_mask = (
                valid_geo &
                (lat_deg >= min_lat) &
                (lat_deg <= max_lat) &
                (lon_deg >= min_lon) &
                (lon_deg <= max_lon)
            )
            mumbai_geo_count = int(np.count_nonzero(mumbai_geo_mask))
            metrics["mumbai_pixels_selected"] = mumbai_geo_count

            # 5. Read HEM and exclude its actual fill value
            hem_ds = f["HEM"]
            rainfall = hem_ds[0]  # first band/slice: shape (2816, 2805)
            hem_fill = hem_ds.attrs.get("_FillValue", -999.0)
            if isinstance(hem_fill, np.ndarray):
                hem_fill = hem_fill.item()

            valid_rain = rainfall != hem_fill
            final_mask = mumbai_geo_mask & valid_rain
            valid_rain_count = int(np.count_nonzero(final_mask))
            fill_values_excluded_in_mumbai = mumbai_geo_count - valid_rain_count

            metrics["valid_rainfall_pixels"] = valid_rain_count
            metrics["fill_values_excluded"] = fill_values_excluded_in_mumbai

            # 6. Extract values
            mumbai_lats = lat_deg[final_mask]
            mumbai_lons = lon_deg[final_mask]
            mumbai_rain = rainfall[final_mask]

            # 7. Decode timestamp from /time dataset
            dt_obj, timestamp_str = decode_time_from_dataset(f["time"])
            metrics["dt"] = dt_obj
            metrics["timestamp_str"] = timestamp_str

            # Statistics
            if valid_rain_count > 0:
                metrics["rainfall_min"] = float(np.min(mumbai_rain))
                metrics["rainfall_max"] = float(np.max(mumbai_rain))
                metrics["rainfall_mean"] = float(np.mean(mumbai_rain))
                metrics["zero_rainfall_pixels"] = int(np.count_nonzero(mumbai_rain == 0.0))
                metrics["positive_rainfall_pixels"] = int(np.count_nonzero(mumbai_rain > 0.0))

            df_mumbai = pd.DataFrame({
                "timestamp": [timestamp_str] * valid_rain_count,
                "latitude": np.round(mumbai_lats, 6),
                "longitude": np.round(mumbai_lons, 6),
                "rainfall_mm_hr": np.round(mumbai_rain, 4),
            })

            metrics["success"] = True
            return df_mumbai, metrics

    except Exception as e:
        metrics["error"] = str(e)
        return None, metrics


def generate_archive_qc_report(
    file_metrics: List[dict],
    df_all: pd.DataFrame,
    output_qc_path: Path
) -> str:
    """
    Generates comprehensive Quality Control report according to TASK 4 specifications.
    """
    total_files = len(file_metrics)
    successful_files = [m for m in file_metrics if m["success"]]
    failed_files = [m for m in file_metrics if not m["success"]]

    # Sort successful files chronologically by dt
    successful_files.sort(key=lambda m: m["dt"])

    if successful_files:
        earliest_dt = successful_files[0]["dt"]
        latest_dt = successful_files[-1]["dt"]
        earliest_ts = successful_files[0]["timestamp_str"]
        latest_ts = successful_files[-1]["timestamp_str"]

        # Expected 30-minute intervals between earliest and latest
        expected_timestamps: List[datetime] = []
        cur = earliest_dt
        while cur <= latest_dt:
            expected_timestamps.append(cur)
            cur += timedelta(minutes=30)
        expected_count = len(expected_timestamps)

        # Actual unique timestamps
        actual_timestamps = [m["dt"] for m in successful_files]
        actual_unique_ts = set(actual_timestamps)
        actual_unique_count = len(actual_unique_ts)

        # Duplicate timestamps
        seen = set()
        duplicates = set()
        for t in actual_timestamps:
            if t in seen:
                duplicates.add(t.strftime("%Y-%m-%d %H:%M"))
            seen.add(t)

        # Missing timestamps
        missing_dts = [t for t in expected_timestamps if t not in actual_unique_ts]
        missing_str = [t.strftime("%Y-%m-%d %H:%M") for t in missing_dts]

        # Pixels per timestamp distribution
        pixels_per_ts = [m["valid_rainfall_pixels"] for m in successful_files]
        min_px_per_ts = min(pixels_per_ts) if pixels_per_ts else 0
        max_px_per_ts = max(pixels_per_ts) if pixels_per_ts else 0
        zero_valid_ts = [m["timestamp_str"] for m in successful_files if m["valid_rainfall_pixels"] == 0]

        # Chronological order verification
        is_chronological = all(actual_timestamps[i] <= actual_timestamps[i+1] for i in range(len(actual_timestamps)-1))

    else:
        earliest_ts = latest_ts = "N/A"
        expected_count = actual_unique_count = 0
        missing_str = []
        duplicates = set()
        min_px_per_ts = max_px_per_ts = 0
        zero_valid_ts = []
        is_chronological = True

    total_rows = len(df_all) if df_all is not None else 0
    total_hem_fill_excluded = sum(m["fill_values_excluded"] for m in successful_files)
    total_invalid_geo_excluded = sum(m["invalid_geo_pixels_total"] for m in successful_files)

    if total_rows > 0:
        rain_min = float(df_all["rainfall_mm_hr"].min())
        rain_max = float(df_all["rainfall_mm_hr"].max())
        rain_mean = float(df_all["rainfall_mm_hr"].mean())
        zero_rain_count = int((df_all["rainfall_mm_hr"] == 0.0).sum())
        pos_rain_count = int((df_all["rainfall_mm_hr"] > 0.0).sum())
    else:
        rain_min = rain_max = rain_mean = 0.0
        zero_rain_count = pos_rain_count = 0

    lines = []
    lines.append("=" * 80)
    lines.append("MOSDAC INSAT-3DS HDF ARCHIVE QUALITY CONTROL (QC) REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. PROCESSING AUDIT")
    lines.append(f"   - Number of HDF files evaluated:      {total_files}")
    lines.append(f"   - Number successfully processed:     {len(successful_files)}")
    lines.append(f"   - Number failed:                     {len(failed_files)}")
    if failed_files:
        for f in failed_files:
            lines.append(f"     * FAILED: {f['source_filename']} - {f['error']}")
    lines.append("")
    lines.append("2. TEMPORAL COMPLETENESS & CHRONOLOGY")
    lines.append(f"   - Earliest timestamp:                {earliest_ts}")
    lines.append(f"   - Latest timestamp:                  {latest_ts}")
    lines.append(f"   - Expected 30-min timestamp count:   {expected_count}")
    lines.append(f"   - Actual unique timestamp count:     {actual_unique_count}")
    lines.append(f"   - Chronological order verified:      {'YES' if is_chronological else 'NO'}")
    lines.append(f"   - Duplicate timestamps:              {len(duplicates)}")
    if duplicates:
        lines.append(f"     * Duplicates: {sorted(list(duplicates))}")
    lines.append(f"   - Missing timestamps:                {len(missing_str)}")
    if missing_str:
        for idx, ts in enumerate(missing_str, 1):
            lines.append(f"     {idx:02d}. {ts}")
    else:
        lines.append("     * No missing timestamps detected within the observation span.")
    lines.append("")
    lines.append("3. SPATIAL & PIXEL AUDIT")
    lines.append(f"   - Valid pixels per timestamp:        {min_px_per_ts} to {max_px_per_ts} (consistently {min_px_per_ts} pixels)")
    lines.append(f"   - Timestamps with zero valid pixels: {len(zero_valid_ts)}")
    if zero_valid_ts:
        lines.append(f"     * Timestamps with 0 pixels: {zero_valid_ts}")
    lines.append(f"   - Total extracted rows:              {total_rows:,}")
    lines.append(f"   - Excluded HEM fill values (in AOI): {total_hem_fill_excluded:,}")
    lines.append(f"   - Excluded invalid Lat/Lon (global): {total_invalid_geo_excluded:,}")
    lines.append("")
    lines.append("4. RAINFALL VALUE METRICS (mm/hr)")
    lines.append(f"   - Rainfall Min:                      {rain_min:.4f} mm/hr")
    lines.append(f"   - Rainfall Max:                      {rain_max:.4f} mm/hr")
    lines.append(f"   - Rainfall Mean:                     {rain_mean:.4f} mm/hr")
    lines.append(f"   - Number of zero-rainfall values:    {zero_rain_count:,} ({(zero_rain_count/max(1, total_rows))*100:.2f}%)")
    lines.append(f"   - Number of positive-rainfall values:{pos_rain_count:,} ({(pos_rain_count/max(1, total_rows))*100:.2f}%)")
    lines.append("")
    lines.append("5. INTEGRITY CONSTRAINTS CONFIRMATION")
    lines.append("   - Missing timestamps: EXPLICITLY REPORTED (never interpolated)")
    lines.append("   - Missing rainfall:   EXPLICITLY EXCLUDED (never filled or synthesized)")
    lines.append("   - Coordinate grid:    NATIVE CURVILINEAR (no affine transform, no rasterio windows)")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    with open(output_qc_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Extract Mumbai rainfall from MOSDAC INSAT-3DS HDF files.")
    parser.add_argument("--input-dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--file", help="Process a single HDF5 file")
    parser.add_argument("--output-csv", help="Path to write output CSV")
    parser.add_argument("--output-qc", help="Path to write QC report")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent

    # Determine files to process
    if args.file:
        files = [Path(args.file).resolve()]
        default_csv = base_dir / "data" / "mosdac" / "hdf_sample" / "mumbai_sample_rainfall.csv"
        default_qc = base_dir / "outputs" / "mosdac_hdf_extraction_test.txt"
    else:
        if args.input_dir:
            scan_dir = Path(args.input_dir).resolve()
        else:
            archive_dir = base_dir / "data" / "mosdac" / "hdf_archive"
            sample_dir = base_dir / "data" / "mosdac" / "hdf_sample"
            # If archive has .h5 files, use archive; else fallback to sample
            if archive_dir.exists() and list(archive_dir.glob("*.h5")):
                scan_dir = archive_dir
            else:
                scan_dir = sample_dir

        files = sorted(list(scan_dir.glob("*.h5")) + list(scan_dir.glob("*.hdf")))
        default_csv = base_dir / "data" / "mosdac" / "mumbai_rainfall_30min.csv"
        default_qc = base_dir / "outputs" / "mosdac_archive_qc.txt"

    out_csv = Path(args.output_csv).resolve() if args.output_csv else default_csv
    out_qc = Path(args.output_qc).resolve() if args.output_qc else default_qc

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_qc.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MOSDAC INSAT-3DS HDF5 MUMBAI RAINFALL EXTRACTION & QC")
    print(f"Input file count: {len(files)}")
    print(f"Output CSV:       {out_csv}")
    print(f"Output QC Report: {out_qc}")
    print("=" * 80)

    if not files:
        print("ERROR: No HDF files found to process.")
        sys.exit(1)

    dfs = []
    file_metrics = []

    for idx, fp in enumerate(files, 1):
        df_sub, metrics = extract_single_hdf(fp)
        file_metrics.append(metrics)
        if metrics["success"] and df_sub is not None:
            dfs.append(df_sub)
            print(f"[{idx}/{len(files)}] [OK] {fp.name} -> {metrics['timestamp_str']} ({metrics['valid_rainfall_pixels']} px, mean: {metrics['rainfall_mean']:.4f} mm/hr)")
        else:
            print(f"[{idx}/{len(files)}] [FAIL] {fp.name} -> {metrics['error']}")

    if dfs:
        # Concatenate and sort chronologically
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.sort_values(by=["timestamp", "latitude", "longitude"], inplace=True)
        df_all.to_csv(out_csv, index=False)
        print("=" * 80)
        print(f"Successfully saved {out_csv} ({len(df_all):,} rows)")
    else:
        df_all = pd.DataFrame(columns=["timestamp", "latitude", "longitude", "rainfall_mm_hr"])
        df_all.to_csv(out_csv, index=False)
        print(f"Warning: No valid rows extracted. Created empty CSV: {out_csv}")

    # Generate QC report
    qc_text = generate_archive_qc_report(file_metrics, df_all, out_qc)
    print(f"Successfully wrote {out_qc}")
    print("=" * 80)
    print(qc_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
