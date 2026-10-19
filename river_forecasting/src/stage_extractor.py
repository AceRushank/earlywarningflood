"""
river_forecasting/src/stage_extractor.py
========================================
Stage 1 Telemetry Audit and Processing Module for Vishnupuri Barrage, Godavari River.

Performs:
1. Locates and filters real Maharashtra river telemetry CSV.
2. Identifies Vishnupuri Barrage station (District: Nanded, Lat: 19.126944, Lon: 77.280833).
3. Extracts the July 4, 2024 to August 22, 2024 peak monsoon window.
4. Parses timestamps, sorts chronologically, verifies zero duplicates, identifies missing hours,
   audits invalid/sentinel values, and flags stage discontinuities without modifying raw observations.
5. Saves clean processed dataset to river_forecasting/outputs/stage_observations.csv.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd


# Constants for Vishnupuri Barrage Station
VISHNUPURI_METADATA = {
    "station_name": "Vishnupuri",
    "full_name": "Vishnupuri Barrage, Nanded, Maharashtra",
    "district": "Nanded",
    "tehsil": "NANDED",
    "river": "Godavari",
    "basin": "Godavari",
    "latitude": 19.12694444,
    "longitude": 77.28083333,
    "agency": "Maharashtra SW",
    "state": "Maharashtra",
    "rl_of_zero_gauge": 0.0,
    "mean_sea_level_m": 366.50,
}

WINDOW_START = datetime(2024, 7, 4, 0, 0)
WINDOW_END = datetime(2024, 8, 22, 23, 59, 59)


def extract_and_audit_stage(
    raw_csv_path: Path,
    output_csv_path: Optional[Path] = None,
    discontinuity_threshold_m: float = 0.50
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts Vishnupuri Barrage telemetry for 2024-07-04 to 2024-08-22,
    performs comprehensive QC, and saves the verified processed CSV.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw telemetry CSV not found: {raw_csv_path}")

    # Read raw dataset
    usecols = [
        "Station", "District", "River", "Basin", "Latitude", "Longitude",
        "RL_of_zeroGauge", "MeanSeaLevel", "Data Acquisition Time",
        "River Water Level Telemetry Hourly (meter)"
    ]
    df_raw = pd.read_csv(raw_csv_path, usecols=usecols, low_memory=False)

    # 1. Station identification
    st_mask = df_raw["Station"].astype(str).str.strip() == VISHNUPURI_METADATA["station_name"]
    df_vishnu = df_raw[st_mask].copy()

    if df_vishnu.empty:
        raise ValueError(f"No records found for station {VISHNUPURI_METADATA['station_name']}")

    total_station_records = len(df_vishnu)

    # 2. Timestamp parsing
    df_vishnu["dt"] = pd.to_datetime(
        df_vishnu["Data Acquisition Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    unparsed_count = int(df_vishnu["dt"].isna().sum())
    if unparsed_count > 0:
        raise ValueError(f"Failed to parse {unparsed_count} timestamps for Vishnupuri")

    # 3. Filter target window
    mask_window = (df_vishnu["dt"] >= WINDOW_START) & (df_vishnu["dt"] <= WINDOW_END)
    df_win = df_vishnu[mask_window].copy()

    if df_win.empty:
        raise ValueError(f"No records found in window {WINDOW_START} to {WINDOW_END}")

    # 4. Chronological sorting
    df_win.sort_values(by="dt", inplace=True)
    df_win.reset_index(drop=True, inplace=True)

    # 5. Check duplicate timestamps
    duplicate_mask = df_win["dt"].duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    duplicates_list = df_win.loc[duplicate_mask, "dt"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    # 6. Parse stage observations
    val_col = "River Water Level Telemetry Hourly (meter)"
    df_win["stage_m"] = pd.to_numeric(df_win[val_col], errors="coerce")
    nan_stage_count = int(df_win["stage_m"].isna().sum())

    # Sentinel / negative checks (e.g., -99.999 or <= 0)
    invalid_sentinel_count = int((df_win["stage_m"] < 0).sum())

    # 7. Check missing hours
    min_observed_dt = df_win["dt"].min()
    max_observed_dt = df_win["dt"].max()

    expected_hourly_range = pd.date_range(start=min_observed_dt, end=max_observed_dt, freq="1h")
    total_expected_hours = len(expected_hourly_range)
    total_observed_hours = len(df_win)

    missing_hourly_dts = expected_hourly_range.difference(df_win["dt"])
    missing_hours_count = len(missing_hourly_dts)
    completeness_percentage = (total_observed_hours / total_expected_hours) * 100.0

    # Gap intervals analysis
    df_win["time_diff_hours"] = df_win["dt"].diff().dt.total_seconds() / 3600.0
    gaps_df = df_win[df_win["time_diff_hours"] > 1.0].copy()
    gaps_count = len(gaps_df)

    gaps_summary: List[Dict[str, Any]] = []
    for _, gap_row in gaps_df.iterrows():
        gap_hours = float(gap_row["time_diff_hours"])
        end_time = gap_row["dt"]
        start_time = end_time - pd.Timedelta(hours=gap_hours)
        gaps_summary.append({
            "gap_start": start_time.strftime("%Y-%m-%d %H:%M"),
            "gap_end": end_time.strftime("%Y-%m-%d %H:%M"),
            "duration_hours": gap_hours,
            "missing_observations_count": int(gap_hours - 1),
        })

    # 8. Stage dynamics & Discontinuity analysis
    df_win["stage_change_m"] = df_win["stage_m"].diff()
    df_win["rate_of_change_m_per_hr"] = df_win["stage_change_m"] / df_win["time_diff_hours"]

    # Identify potential discontinuities / spikes
    discontinuities: List[Dict[str, Any]] = []
    flags = ["VALID"] * len(df_win)

    for i in range(1, len(df_win)):
        d_stage = df_win.loc[i, "stage_change_m"]
        dt_hr = df_win.loc[i, "time_diff_hours"]
        abs_change = abs(d_stage)

        # Flag jump if instantaneous 1-hour jump exceeds threshold
        if dt_hr == 1.0 and abs_change >= discontinuity_threshold_m:
            timestamp_str = df_win.loc[i, "dt"].strftime("%Y-%m-%d %H:%M")
            prev_stage = float(df_win.loc[i-1, "stage_m"])
            curr_stage = float(df_win.loc[i, "stage_m"])
            
            # Check if this is an isolated 1-hour spike that returns on the next hour
            is_transient = False
            if i + 1 < len(df_win):
                next_dt_hr = df_win.loc[i+1, "time_diff_hours"]
                next_d_stage = df_win.loc[i+1, "stage_change_m"]
                if next_dt_hr == 1.0 and abs(next_d_stage + d_stage) < 0.05:
                    is_transient = True

            flag_label = "TRANSIENT_1M_SPIKE" if is_transient else "DISCONTINUITY_STEP"
            flags[i] = flag_label

            discontinuities.append({
                "timestamp": timestamp_str,
                "previous_stage_m": prev_stage,
                "current_stage_m": curr_stage,
                "stage_change_m": float(d_stage),
                "is_transient_spike": is_transient,
                "description": (
                    f"Isolated 1-hour jump of {d_stage:+.3f}m returning to baseline next hour"
                    if is_transient else f"Step change of {d_stage:+.3f}m"
                )
            })

    df_win["quality_flag"] = flags

    # Construct clean output dataframe
    output_df = pd.DataFrame({
        "timestamp_ist": df_win["dt"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "station_name": VISHNUPURI_METADATA["station_name"],
        "river": VISHNUPURI_METADATA["river"],
        "basin": VISHNUPURI_METADATA["basin"],
        "district": VISHNUPURI_METADATA["district"],
        "latitude": float(df_win["Latitude"].iloc[0]),
        "longitude": float(df_win["Longitude"].iloc[0]),
        "rl_of_zero_gauge_m": float(df_win["RL_of_zeroGauge"].iloc[0]),
        "mean_sea_level_m": float(df_win["MeanSeaLevel"].iloc[0]),
        "stage_m": df_win["stage_m"].round(4),
        "stage_change_m": df_win["stage_change_m"].round(4),
        "time_diff_hours": df_win["time_diff_hours"].round(2),
        "rate_of_change_m_per_hr": df_win["rate_of_change_m_per_hr"].round(4),
        "quality_flag": df_win["quality_flag"],
        "raw_acquisition_time": df_win["Data Acquisition Time"]
    })

    if output_csv_path:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_csv_path, index=False)

    # Compute comprehensive audit metrics
    metrics = {
        "station_metadata": {
            "station_name": VISHNUPURI_METADATA["station_name"],
            "full_name": VISHNUPURI_METADATA["full_name"],
            "district": VISHNUPURI_METADATA["district"],
            "river": VISHNUPURI_METADATA["river"],
            "basin": VISHNUPURI_METADATA["basin"],
            "latitude": float(df_win["Latitude"].iloc[0]),
            "longitude": float(df_win["Longitude"].iloc[0]),
            "rl_of_zero_gauge_m": float(df_win["RL_of_zeroGauge"].iloc[0]),
            "mean_sea_level_m": float(df_win["MeanSeaLevel"].iloc[0]),
        },
        "window_specifications": {
            "target_start_iso": WINDOW_START.strftime("%Y-%m-%d %H:%M:%S"),
            "target_end_iso": WINDOW_END.strftime("%Y-%m-%d %H:%M:%S"),
            "earliest_observed_timestamp": min_observed_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "latest_observed_timestamp": max_observed_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "total_days_span": round((max_observed_dt - min_observed_dt).total_seconds() / 86400.0, 2),
        },
        "record_counts": {
            "total_raw_station_records": total_station_records,
            "window_observed_records": total_observed_hours,
            "window_expected_hourly_slots": total_expected_hours,
            "missing_hourly_slots": missing_hours_count,
            "completeness_percentage": round(completeness_percentage, 2),
            "duplicate_timestamps_count": duplicate_count,
            "nan_or_unparsed_values_count": nan_stage_count,
            "sentinel_or_negative_values_count": invalid_sentinel_count,
        },
        "stage_statistics_meters": {
            "min_stage_m": round(float(df_win["stage_m"].min()), 3),
            "max_stage_m": round(float(df_win["stage_m"].max()), 3),
            "mean_stage_m": round(float(df_win["stage_m"].mean()), 3),
            "std_stage_m": round(float(df_win["stage_m"].std()), 3),
            "stage_range_m": round(float(df_win["stage_m"].max() - df_win["stage_m"].min()), 3),
        },
        "temporal_gaps": {
            "total_gaps_exceeding_1_hour": gaps_count,
            "max_gap_duration_hours": float(gaps_df["time_diff_hours"].max()) if not gaps_df.empty else 1.0,
            "gap_duration_counts": {
                f"{int(k)}h": int(v) for k, v in gaps_df["time_diff_hours"].value_counts().to_dict().items()
            },
            "gaps_list": gaps_summary,
        },
        "stage_discontinuities": {
            "threshold_meters": discontinuity_threshold_m,
            "detected_count": len(discontinuities),
            "events": discontinuities,
        }
    }

    return output_df, metrics
