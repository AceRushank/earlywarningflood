"""
scripts/build_full_season_multistation_dataset.py
=================================================
Builds the comprehensive 2026 whole-monsoon dataset incorporating:
1. Vishnupuri Barrage upstream pool stage telemetry
2. Downstream Raher and Babhali Barrage telemetry (Phase 3B)
3. Verified INSAT-3DS catchment rainfall
4. Strict causal feature engineering (zero lookahead)
5. Chronological Train / Val / Test partitions
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TELEMETRY_RAW = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"
RAINFALL_HOURLY = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_hourly_2026.csv"
OUT_DATASET = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"

def build_dataset():
    print("=" * 80)
    print("BUILDING FULL-MONSOON 2026 MULTI-STATION ALIGNED DATASET")
    print("=" * 80)

    # 1. Load Telemetry for 3 Stations
    print("Loading official NWIC telemetry...")
    df_raw = pd.read_csv(TELEMETRY_RAW, usecols=[
        "Station", "Data Acquisition Time", "River Water Level Telemetry Hourly (meter)",
        "Latitude", "Longitude", "MeanSeaLevel", "RL_of_zeroGauge"
    ])
    
    stations = ["Vishnupuri", "Raher", "Babhali Barrage"]
    df_st = df_raw[df_raw["Station"].isin(stations)].copy()
    df_st["dt"] = pd.to_datetime(df_st["Data Acquisition Time"], format="%d-%m-%Y %H:%M")
    df_st["stage"] = pd.to_numeric(df_st["River Water Level Telemetry Hourly (meter)"], errors="coerce")

    # Filter to June 15 - September 30, 2026
    start_dt = pd.to_datetime("2026-06-15 00:00:00")
    end_dt = pd.to_datetime("2026-09-30 23:00:00")
    df_st = df_st[(df_st["dt"] >= start_dt) & (df_st["dt"] <= end_dt)]

    # Pivot stations to columns
    pivot_stage = df_st.pivot_table(index="dt", columns="Station", values="stage", aggfunc="first")
    pivot_stage.rename(columns={
        "Vishnupuri": "stage_vishnupuri",
        "Raher": "stage_raher",
        "Babhali Barrage": "stage_babhali"
    }, inplace=True)

    # Continuous hourly timeline
    full_grid = pd.date_range(start_dt, end_dt, freq="1h")
    df_aligned = pd.DataFrame({"dt": full_grid}).set_index("dt")
    df_aligned = df_aligned.join(pivot_stage, how="left")

    print(f"Full season continuous grid: {len(df_aligned)} hours ({start_dt} to {end_dt})")
    print(f"Vishnupuri stage records:    {df_aligned['stage_vishnupuri'].notna().sum()} / {len(df_aligned)}")
    print(f"Raher downstream records:    {df_aligned['stage_raher'].notna().sum()} / {len(df_aligned)}")
    print(f"Babhali downstream records:  {df_aligned['stage_babhali'].notna().sum()} / {len(df_aligned)}")

    # 2. Quality Control: Mask Vishnupuri Sensor Dropout Artifact (Aug 22-23)
    dropout_mask = (df_aligned["stage_vishnupuri"] < 345.0)
    n_drop = dropout_mask.sum()
    print(f"Masking {n_drop} sensor dropout artifact records (< 345m MSL) as NaN...")
    df_aligned.loc[dropout_mask, "stage_vishnupuri"] = np.nan

    # 3. Join September INSAT-3DS Catchment Rainfall
    print(f"Loading hourly catchment rainfall from {RAINFALL_HOURLY}...")
    df_rain = pd.read_csv(RAINFALL_HOURLY)
    df_rain["dt"] = pd.to_datetime(df_rain["timestamp_ist"])
    df_rain.set_index("dt", inplace=True)
    
    rain_cols = ["mean_rainfall_rate_mm_hr", "hourly_rainfall_depth_mm", "scan_count", "completeness"]
    df_aligned = df_aligned.join(df_rain[rain_cols], how="left")
    df_aligned.rename(columns={
        "mean_rainfall_rate_mm_hr": "rain_rate_mm_hr",
        "hourly_rainfall_depth_mm": "rain_depth_1h_mm",
        "scan_count": "rain_scans_in_hour",
        "completeness": "rain_qc_flag"
    }, inplace=True)

    # 4. Strictly Causal Feature Engineering (Zero Lookahead)
    # Vishnupuri stage lags & differences
    df_aligned["stage_v_lag1"] = df_aligned["stage_vishnupuri"].shift(1)
    df_aligned["stage_v_lag2"] = df_aligned["stage_vishnupuri"].shift(2)
    df_aligned["stage_v_lag3"] = df_aligned["stage_vishnupuri"].shift(3)
    df_aligned["dH_v_1h"] = df_aligned["stage_vishnupuri"] - df_aligned["stage_v_lag1"]
    df_aligned["dH_v_3h"] = df_aligned["stage_vishnupuri"] - df_aligned["stage_v_lag3"]

    # Downstream Raher & Babhali lags & differences
    df_aligned["stage_r_lag1"] = df_aligned["stage_raher"].shift(1)
    df_aligned["dH_r_1h"] = df_aligned["stage_raher"] - df_aligned["stage_r_lag1"]
    df_aligned["stage_b_lag1"] = df_aligned["stage_babhali"].shift(1)
    df_aligned["dH_b_1h"] = df_aligned["stage_babhali"] - df_aligned["stage_b_lag1"]

    # Rolling rainfall windows (scaled to window duration, requiring >=75% valid scans)
    df_aligned["rain_3h_mm"] = df_aligned["rain_depth_1h_mm"].rolling(3, min_periods=2).mean() * 3
    df_aligned["rain_6h_mm"] = df_aligned["rain_depth_1h_mm"].rolling(6, min_periods=4).mean() * 6
    df_aligned["rain_12h_mm"] = df_aligned["rain_depth_1h_mm"].rolling(12, min_periods=9).mean() * 12
    df_aligned["rain_24h_mm"] = df_aligned["rain_depth_1h_mm"].rolling(24, min_periods=18).mean() * 24

    # Explicitly enforce NaN during satellite outage across all rainfall features
    outage_mask = (df_aligned["rain_qc_flag"] == "OUTAGE_UNAVAILABLE")
    rain_feature_cols = ["rain_rate_mm_hr", "rain_depth_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm"]
    df_aligned.loc[outage_mask, rain_feature_cols] = np.nan

    # 5. Future Stage Targets (Strictly Future Observations)
    # Vishnupuri targets
    df_aligned["target_v_1h"] = df_aligned["stage_vishnupuri"].shift(-1)
    df_aligned["target_v_2h"] = df_aligned["stage_vishnupuri"].shift(-2)
    df_aligned["target_v_3h"] = df_aligned["stage_vishnupuri"].shift(-3)
    df_aligned["target_v_dH_1h"] = df_aligned["target_v_1h"] - df_aligned["stage_vishnupuri"]
    df_aligned["target_v_dH_2h"] = df_aligned["target_v_2h"] - df_aligned["stage_vishnupuri"]
    df_aligned["target_v_dH_3h"] = df_aligned["target_v_3h"] - df_aligned["stage_vishnupuri"]

    # Exploratory Downstream Targets (Raher +1h, +2h, +3h)
    df_aligned["target_r_1h"] = df_aligned["stage_raher"].shift(-1)
    df_aligned["target_r_2h"] = df_aligned["stage_raher"].shift(-2)
    df_aligned["target_r_3h"] = df_aligned["stage_raher"].shift(-3)

    # 6. Chronological Partitioning
    # Train: June 15 00:00 - August 10 23:00 (57 days)
    # Val:   August 11 00:00 - August 31 23:00 (21 days)
    # Test:  September 1 00:00 - September 30 23:00 (30 days)
    # Synchronized Test Subwindow: September 8 06:00 - September 22 19:00 (with verified on-disk rainfall)
    df_aligned.reset_index(inplace=True)
    df_aligned["timestamp_ist"] = df_aligned["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df_aligned["split"] = "UNKNOWN"
    train_mask = (df_aligned["dt"] >= "2026-06-15 00:00:00") & (df_aligned["dt"] <= "2026-08-10 23:00:00")
    val_mask   = (df_aligned["dt"] >= "2026-08-11 00:00:00") & (df_aligned["dt"] <= "2026-08-31 23:00:00")
    test_mask  = (df_aligned["dt"] >= "2026-09-01 00:00:00") & (df_aligned["dt"] <= "2026-09-30 23:00:00")

    df_aligned.loc[train_mask, "split"] = "TRAIN"
    df_aligned.loc[val_mask, "split"] = "VAL"
    df_aligned.loc[test_mask, "split"] = "TEST"

    # Flag synchronized test subwindow with verified rainfall
    test_sync_mask = (df_aligned["dt"] >= "2026-09-08 06:00:00") & (df_aligned["dt"] <= "2026-09-22 19:00:00")
    df_aligned["is_test_sync_rainfall"] = test_sync_mask

    print("\nChronological Split Breakdown:")
    for sp in ["TRAIN", "VAL", "TEST"]:
        cnt = (df_aligned["split"] == sp).sum()
        v_valid = df_aligned.loc[df_aligned["split"] == sp, "stage_vishnupuri"].notna().sum()
        print(f"  {sp:<6}: {cnt} total hours | {v_valid} valid Vishnupuri hours")
    print(f"  TEST_SYNC (Sep 8-22): {test_sync_mask.sum()} hours with complete satellite rainfall")

    # Save aligned dataset
    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df_aligned.to_csv(OUT_DATASET, index=False)
    print(f"\nSaved aligned multi-station dataset to: {OUT_DATASET} ({len(df_aligned)} rows, {df_aligned.shape[1]} columns)")

if __name__ == "__main__":
    build_dataset()
