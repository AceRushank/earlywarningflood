"""
scripts/audit_telemetry_synchronization.py
=========================================
Audits timestamp alignment, missingness, telemetry gap profiles,
and realistic transmission latency for upstream candidate stations.
Outputs:
  - river_forecasting/outputs/phase5_upstream_telemetry/station_inventory.csv
  - river_forecasting/outputs/phase5_upstream_telemetry/timestamp_availability_audit.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase5_upstream_telemetry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_2026 = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"

def run_audit():
    print("=" * 80)
    print("STAGE 1 & 2: TELEMETRY & SYNCHRONIZATION AUDIT")
    print("=" * 80)

    # 1. Candidate Stations Metadata
    candidates_meta = [
        {
            "station_name": "Vishnupuri",
            "station_type": "Target Barrage Pool",
            "agency": "Maharashtra WRD / NWIC",
            "river": "Godavari",
            "tributary": "Mainstem",
            "district": "Nanded",
            "latitude": 19.1269,
            "longitude": 77.2808,
            "elevation_datum_m": "Mean Sea Level",
            "hydrological_relation": "Target Forecast Point",
            "river_distance_km": 0.0,
            "discharge_available": "No (Stage only)",
        },
        {
            "station_name": "Mudgal Barrage",
            "station_type": "Barrage Gauge",
            "agency": "Maharashtra WRD / NWIC",
            "river": "Godavari",
            "tributary": "Mainstem",
            "district": "Parbhani",
            "latitude": 19.0889,
            "longitude": 76.4917,
            "elevation_datum_m": "Mean Sea Level",
            "hydrological_relation": "Direct Upstream Mainstem (~105 km)",
            "river_distance_km": 105.0,
            "discharge_available": "No (Stage only)",
        },
        {
            "station_name": "Siddeshwar Dam",
            "station_type": "Storage Dam Reservoir",
            "agency": "Maharashtra WRD / NWIC",
            "river": "Godavari",
            "tributary": "Purna River",
            "district": "Hingoli",
            "latitude": 19.6128,
            "longitude": 76.9719,
            "elevation_datum_m": "Mean Sea Level",
            "hydrological_relation": "Upstream Tributary (Purna Confluence)",
            "river_distance_km": 95.0,
            "discharge_available": "No (Stage only)",
        },
        {
            "station_name": "Majalgaon",
            "station_type": "Dam Reservoir",
            "agency": "Maharashtra WRD / NWIC",
            "river": "Godavari",
            "tributary": "Sindphana River",
            "district": "Beed",
            "latitude": 19.1508,
            "longitude": 76.1869,
            "elevation_datum_m": "Mean Sea Level",
            "hydrological_relation": "Upstream Tributary (~140 km)",
            "river_distance_km": 140.0,
            "discharge_available": "No (Stage only)",
        },
        {
            "station_name": "Lower Dudhana Dam",
            "station_type": "Storage Dam Reservoir",
            "agency": "Maharashtra WRD / NWIC",
            "river": "Godavari",
            "tributary": "Dudhana River",
            "district": "Parbhani",
            "latitude": 19.5308,
            "longitude": 76.3928,
            "elevation_datum_m": "Mean Sea Level",
            "hydrological_relation": "Upstream Tributary (~120 km)",
            "river_distance_km": 120.0,
            "discharge_available": "No (Stage only)",
        },
        {
            "station_name": "Raher",
            "station_type": "River Gauge",
            "agency": "Maharashtra WRD / NWIC",
            "river": "Godavari",
            "tributary": "Mainstem",
            "district": "Nanded",
            "latitude": 18.8969,
            "longitude": 77.6772,
            "elevation_datum_m": "Mean Sea Level",
            "hydrological_relation": "Downstream Mainstem (~55 km)",
            "river_distance_km": -55.0,
            "discharge_available": "No (Stage only)",
        },
        {
            "station_name": "Babhali Barrage",
            "station_type": "Barrage Gauge",
            "agency": "Maharashtra WRD / NWIC",
            "river": "Godavari",
            "tributary": "Mainstem",
            "district": "Nanded",
            "latitude": 18.8536,
            "longitude": 76.7794,
            "elevation_datum_m": "Mean Sea Level",
            "hydrological_relation": "Downstream Mainstem (~70 km)",
            "river_distance_km": -70.0,
            "discharge_available": "No (Stage only)",
        },
    ]

    df_inventory = pd.DataFrame(candidates_meta)
    inv_path = OUTPUT_DIR / "station_inventory.csv"
    df_inventory.to_csv(inv_path, index=False)
    print(f"Saved station inventory -> {inv_path}")

    # 2. Detailed Telemetry Synchronization Analysis
    print("Reading telemetry data for synchronization audit...")
    station_names = [m["station_name"] for m in candidates_meta]
    chunks = []
    for chunk in pd.read_csv(CSV_2026, usecols=["Station", "Data Acquisition Time", "River Water Level Telemetry Hourly (meter)"], chunksize=100000):
        sub = chunk[chunk["Station"].isin(station_names)]
        if len(sub) > 0:
            chunks.append(sub)
    df_raw = pd.concat(chunks, ignore_index=True)
    df_raw["dt"] = pd.to_datetime(df_raw["Data Acquisition Time"], format="%d-%m-%Y %H:%M")
    df_raw["stage"] = pd.to_numeric(df_raw["River Water Level Telemetry Hourly (meter)"], errors="coerce")

    # Timeframe: June 15 to Sept 30, 2026
    start_dt = pd.to_datetime("2026-06-15 00:00:00")
    end_dt = pd.to_datetime("2026-09-30 23:00:00")
    df_monsoon = df_raw[(df_raw["dt"] >= start_dt) & (df_raw["dt"] <= end_dt)].copy()

    # QC filters
    # Vishnupuri sensor dropout < 345
    df_monsoon.loc[(df_monsoon["Station"] == "Vishnupuri") & (df_monsoon["stage"] < 345.0), "stage"] = np.nan
    # Sensor dropouts < -90 or > 2000
    df_monsoon.loc[(df_monsoon["stage"] <= -90.0) | (df_monsoon["stage"] > 2000.0), "stage"] = np.nan

    piv = df_monsoon.pivot_table(index="dt", columns="Station", values="stage", aggfunc="first")
    full_grid = pd.date_range(start_dt, end_dt, freq="1h")
    piv = piv.reindex(full_grid)
    total_hours = len(full_grid)

    splits = {
        "WHOLE_MONSOON": (start_dt, end_dt),
        "TRAIN": (pd.to_datetime("2026-06-15 00:00:00"), pd.to_datetime("2026-08-10 23:00:00")),
        "VAL": (pd.to_datetime("2026-08-11 00:00:00"), pd.to_datetime("2026-08-31 23:00:00")),
        "TEST_WHOLE_SEP": (pd.to_datetime("2026-09-01 00:00:00"), pd.to_datetime("2026-09-30 23:00:00")),
        "TEST_SYNC_RAINFALL": (pd.to_datetime("2026-09-08 06:00:00"), pd.to_datetime("2026-09-22 19:00:00")),
    }

    audit_rows = []
    for split_name, (s_dt, e_dt) in splits.items():
        sub = piv.loc[s_dt:e_dt]
        n_sub = len(sub)
        for st in station_names:
            if st not in sub.columns:
                continue
            s_series = sub[st]
            valid_cnt = s_series.notna().sum()
            missing_cnt = n_sub - valid_cnt
            avail_pct = (valid_cnt / n_sub) * 100.0

            # Gaps analysis
            is_na = s_series.isna()
            gap_lengths = []
            cur_len = 0
            for val in is_na:
                if val:
                    cur_len += 1
                else:
                    if cur_len > 0:
                        gap_lengths.append(cur_len)
                        cur_len = 0
            if cur_len > 0:
                gap_lengths.append(cur_len)

            max_gap = max(gap_lengths) if gap_lengths else 0
            num_gaps = len(gap_lengths)

            # Telemetry latency note:
            # Operational acquisition is hourly at HH:00.
            # Real-world transmission delay to state data center is 1 to 2 hours.
            # To ensure zero lookahead and account for latency, upstream features at forecast time t
            # must use observations recorded at t - latency (where latency >= 1h).
            audit_rows.append({
                "split": split_name,
                "station": st,
                "total_hours": n_sub,
                "valid_hours": valid_cnt,
                "missing_hours": missing_cnt,
                "availability_pct": round(avail_pct, 2),
                "num_gap_episodes": num_gaps,
                "max_gap_hours": max_gap,
                "nominal_frequency": "1 hour",
                "operational_latency_buffer_hr": 1,
                "mean_stage_m": round(s_series.mean(), 3) if valid_cnt > 0 else np.nan,
                "min_stage_m": round(s_series.min(), 3) if valid_cnt > 0 else np.nan,
                "max_stage_m": round(s_series.max(), 3) if valid_cnt > 0 else np.nan,
                "std_stage_m": round(s_series.std(), 3) if valid_cnt > 0 else np.nan,
            })

    df_audit = pd.DataFrame(audit_rows)
    audit_path = OUTPUT_DIR / "timestamp_availability_audit.csv"
    df_audit.to_csv(audit_path, index=False)
    print(f"Saved timestamp availability audit -> {audit_path}")

    # Print summary table for WHOLE_MONSOON and TEST_SYNC_RAINFALL
    print("\n--- Availability Summary (Whole Monsoon 2026: 2592 Hours) ---")
    monsoon_sum = df_audit[df_audit["split"] == "WHOLE_MONSOON"]
    for _, r in monsoon_sum.iterrows():
        print(f"Station: {r['station']:25s} | Valid: {r['valid_hours']}/{r['total_hours']} ({r['availability_pct']}%) | Max Gap: {r['max_gap_hours']}h | Range: {r['min_stage_m']:.2f} - {r['max_stage_m']:.2f}m")

    print("\n--- Availability Summary (Synchronized Test Period: Sept 8-22, 350 Hours) ---")
    sync_sum = df_audit[df_audit["split"] == "TEST_SYNC_RAINFALL"]
    for _, r in sync_sum.iterrows():
        print(f"Station: {r['station']:25s} | Valid: {r['valid_hours']}/{r['total_hours']} ({r['availability_pct']}%) | Max Gap: {r['max_gap_hours']}h | Range: {r['min_stage_m']:.2f} - {r['max_stage_m']:.2f}m")

if __name__ == "__main__":
    run_audit()
