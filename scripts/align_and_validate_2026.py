"""
scripts/align_and_validate_2026.py
==================================
Aligns September 8-22, 2026 hourly INSAT-3DS catchment rainfall with official
Vishnupuri stage telemetry, generates diagnostic plots, and evaluates forecasting eligibility.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TELEMETRY_CSV = REPO_ROOT / "data" / "historical" / "vishnupuri_telemetry_2026_probe.csv"
RAINFALL_HOURLY_CSV = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_hourly_sep2026.csv"
OUTPUT_ALIGNED_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_september_aligned.csv"
OUTPUT_PLOT_PATH = REPO_ROOT / "data" / "historical" / "diagnostic_plots_sep2026.png"

def align_and_validate():
    print("=" * 70)
    print("PHASE 2: ALIGNING 2026 STAGE TELEMETRY & SATELLITE RAINFALL")
    print("=" * 70)

    # 1. Load Telemetry
    print(f"Loading stage telemetry from {TELEMETRY_CSV}...")
    df_stage = pd.read_csv(TELEMETRY_CSV)
    df_stage["dt_ist"] = pd.to_datetime(df_stage["datetime"])
    
    # 2. Load Hourly Rainfall
    print(f"Loading hourly rainfall from {RAINFALL_HOURLY_CSV}...")
    df_rain = pd.read_csv(RAINFALL_HOURLY_CSV)
    df_rain["dt_ist"] = pd.to_datetime(df_rain["timestamp_ist"])

    # 3. Establish Common Hourly Grid
    start_dt = df_rain["dt_ist"].min()  # 2026-09-08 06:00:00
    end_dt = df_rain["dt_ist"].max()    # 2026-09-22 19:00:00
    expected_grid = pd.date_range(start_dt, end_dt, freq="1h")
    n_expected = len(expected_grid)
    print(f"\nExpected continuous hourly interval: {start_dt} to {end_dt}")
    print(f"Total expected timesteps: {n_expected} hours")

    # Filter telemetry to this window
    stage_window = df_stage[(df_stage["dt_ist"] >= start_dt) & (df_stage["dt_ist"] <= end_dt)].copy()
    stage_window = stage_window.sort_values("dt_ist").drop_duplicates(subset=["dt_ist"]).reset_index(drop=True)
    
    print(f"Observed stage records in window: {len(stage_window)} / {n_expected} ({len(stage_window)/n_expected*100:.2f}%)")
    print(f"Observed rain records in window:  {len(df_rain)} / {n_expected} ({len(df_rain)/n_expected*100:.2f}%)")

    # Missing stage timestamps
    observed_stage_dts = set(stage_window["dt_ist"])
    missing_stage_dts = sorted([dt for dt in expected_grid if dt not in observed_stage_dts])
    print(f"Missing stage hours count: {len(missing_stage_dts)}")
    if missing_stage_dts:
        print(f"  Missing stage hours sample: {[str(d) for d in missing_stage_dts[:10]]}")

    # Build continuous skeleton dataframe
    df_aligned = pd.DataFrame({"dt_ist": expected_grid})
    df_aligned["timestamp_ist"] = df_aligned["dt_ist"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_aligned["timestamp_utc"] = (df_aligned["dt_ist"] - timedelta(hours=5, minutes=30)).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Merge stage
    stage_cols = ["dt_ist", "stage_m", "MeanSeaLevel", "RL_of_zeroGauge", "Station"]
    df_aligned = pd.merge(df_aligned, stage_window[stage_cols], on="dt_ist", how="left")
    
    # Merge rain
    rain_cols = ["dt_ist", "mean_rainfall_rate_mm_hr", "hourly_rainfall_depth_mm", "scan_count", "completeness", "source_granules"]
    df_aligned = pd.merge(df_aligned, df_rain[rain_cols], on="dt_ist", how="left")

    # Rename & clean columns
    df_aligned.rename(columns={
        "mean_rainfall_rate_mm_hr": "rain_rate_mm_hr",
        "hourly_rainfall_depth_mm": "rain_depth_1h_mm",
        "scan_count": "rain_scans_in_hour",
        "completeness": "rain_qc_flag"
    }, inplace=True)

    # 4. Feature Engineering without Data Leakage (Strictly Past and Present)
    # Stage derivatives (only valid if previous timestamp was strictly 1 hour ago)
    # To avoid silent errors over missing hours, calculate difference against exact 1-hour lag
    df_aligned.set_index("dt_ist", inplace=True)

    # Past stage lags
    df_aligned["stage_lag_1h"] = df_aligned["stage_m"].shift(1)
    df_aligned["stage_lag_2h"] = df_aligned["stage_m"].shift(2)
    df_aligned["stage_lag_3h"] = df_aligned["stage_m"].shift(3)
    df_aligned["dH_dt_1h"] = df_aligned["stage_m"] - df_aligned["stage_lag_1h"]
    df_aligned["dH_dt_3h"] = df_aligned["stage_m"] - df_aligned["stage_lag_3h"]

    # Antecedent rainfall windows (sum of past depths up to current hour t)
    # Note: min_periods ensures we don't fabricate rainfall if too many slots are missing
    df_aligned["rain_3h_mm"] = df_aligned["rain_depth_1h_mm"].rolling(3, min_periods=3).sum()
    df_aligned["rain_6h_mm"] = df_aligned["rain_depth_1h_mm"].rolling(6, min_periods=6).sum()
    df_aligned["rain_12h_mm"] = df_aligned["rain_depth_1h_mm"].rolling(12, min_periods=12).sum()
    df_aligned["rain_24h_mm"] = df_aligned["rain_depth_1h_mm"].rolling(24, min_periods=24).sum()

    # Targets (Future Stage without leakage into features)
    df_aligned["target_stage_1h"] = df_aligned["stage_m"].shift(-1)
    df_aligned["target_stage_2h"] = df_aligned["stage_m"].shift(-2)
    df_aligned["target_stage_3h"] = df_aligned["stage_m"].shift(-3)
    df_aligned["target_dH_1h"] = df_aligned["target_stage_1h"] - df_aligned["stage_m"]
    df_aligned["target_dH_2h"] = df_aligned["target_stage_2h"] - df_aligned["stage_m"]
    df_aligned["target_dH_3h"] = df_aligned["target_stage_3h"] - df_aligned["stage_m"]

    df_aligned.reset_index(inplace=True)

    # 5. Evaluate Eligible Sample Counts
    # For a forecast at time t:
    # Requires: valid stage_m, valid stage_lag_1h (for dH/dt), valid rain_depth_1h_mm, valid rain_24h_mm
    feature_cols = ["stage_m", "dH_dt_1h", "rain_depth_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm"]
    valid_features = df_aligned[feature_cols].notna().all(axis=1)

    eligible_t1 = df_aligned[valid_features & df_aligned["target_stage_1h"].notna()]
    eligible_t2 = df_aligned[valid_features & df_aligned["target_stage_2h"].notna()]
    eligible_t3 = df_aligned[valid_features & df_aligned["target_stage_3h"].notna()]

    print("\n--- ELIGIBLE SAMPLE COUNTS FOR FORECASTING ---")
    print(f"Total timesteps:             {len(df_aligned)}")
    print(f"Timesteps with valid features: {valid_features.sum()} (burn-in for 24h lag reduces first 24h)")
    print(f"Eligible for +1h lead time:   {len(eligible_t1)}")
    print(f"Eligible for +2h lead time:   {len(eligible_t2)}")
    print(f"Eligible for +3h lead time:   {len(eligible_t3)}")

    # 6. Save Aligned Dataset
    df_aligned.to_csv(OUTPUT_ALIGNED_CSV, index=False)
    print(f"\nSaved aligned 2026 dataset to: {OUTPUT_ALIGNED_CSV} ({len(df_aligned)} rows)")

    # 7. Summary Statistics of Stage & Rainfall Dynamics
    print("\n--- STAGE DYNAMICS SUMMARY (Sept 8-22, 2026) ---")
    st = df_aligned["stage_m"].dropna()
    print(f"Stage min:     {st.min():.3f} m MSL")
    print(f"Stage max:     {st.max():.3f} m MSL")
    print(f"Stage mean:    {st.mean():.3f} m MSL")
    print(f"Stage std:     {st.std():.3f} m")
    print(f"Stage delta:   {st.max() - st.min():.3f} m")

    dh1 = df_aligned["dH_dt_1h"].dropna()
    print(f"Max 1h rise:   +{dh1.max():.4f} m/hr (at {df_aligned.loc[dh1.idxmax(), 'timestamp_ist']})")
    print(f"Max 1h fall:   {dh1.min():.4f} m/hr (at {df_aligned.loc[dh1.idxmin(), 'timestamp_ist']})")
    
    # Non-zero rain hours
    rain = df_aligned["rain_depth_1h_mm"].dropna()
    rain_pos = rain[rain > 0.0]
    print("\n--- RAINFALL DYNAMICS SUMMARY ---")
    print(f"Rainfall hours > 0 mm: {len(rain_pos)} / {len(rain)} ({len(rain_pos)/len(rain)*100:.1f}%)")
    print(f"Max hourly rain depth: {rain.max():.4f} mm")
    print(f"Total event rainfall:  {rain.sum():.2f} mm")
    print(f"Max 24h rainfall:      {df_aligned['rain_24h_mm'].max():.2f} mm")

    # 8. Create Diagnostic Plots
    print("\nGenerating diagnostic plots...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1.2, 1.2]})

    # Plot 1: River Stage & MSL
    ax1 = axes[0]
    ax1.plot(df_aligned["dt_ist"], df_aligned["stage_m"], color="#1e3a8a", lw=2, label="Vishnupuri Stage (Observed Telemetry)")
    ax1.axhline(355.0, color="#dc2626", ls="--", lw=1.2, label="Full Reservoir Level (~355.0 m MSL)")
    ax1.axhline(354.0, color="#f59e0b", ls=":", lw=1.2, label="High Stage Threshold (~354.0 m MSL)")
    ax1.set_ylabel("Water Level (m MSL)", fontsize=11, fontweight="bold")
    ax1.set_title("Vishnupuri Barrage — September 8–22, 2026 Hydro-Meteorological Alignment", fontsize=14, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.9)
    ax1.set_ylim(354.2, 355.2)

    # Plot 2: Catchment Rainfall (Inverted Hyetograph & 6h Moving Sum)
    ax2 = axes[1]
    ax2.bar(df_aligned["dt_ist"], df_aligned["rain_depth_1h_mm"], width=0.035, color="#0284c7", alpha=0.8, label="Hourly Catchment Rain (mm)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df_aligned["dt_ist"], df_aligned["rain_6h_mm"], color="#7c3aed", lw=1.8, label="Antecedent 6h Rain (mm)")
    ax2.set_ylabel("1h Rain Depth (mm)", fontsize=10, fontweight="bold", color="#0284c7")
    ax2_twin.set_ylabel("6h Rain (mm)", fontsize=10, fontweight="bold", color="#7c3aed")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # Combined legend for ax2
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2t, labels_2 + labels_2t, loc="upper right", framealpha=0.9)

    # Plot 3: Stage Rate of Change (dH/dt) & Missing Data Indicators
    ax3 = axes[2]
    pos_mask = df_aligned["dH_dt_1h"] >= 0
    neg_mask = df_aligned["dH_dt_1h"] < 0
    ax3.bar(df_aligned.loc[pos_mask, "dt_ist"], df_aligned.loc[pos_mask, "dH_dt_1h"], width=0.035, color="#10b981", label="Rise (dH/dt > 0)")
    ax3.bar(df_aligned.loc[neg_mask, "dt_ist"], df_aligned.loc[neg_mask, "dH_dt_1h"], width=0.035, color="#ef4444", label="Fall (dH/dt < 0)")
    
    # Highlight missing stage points
    if missing_stage_dts:
        ax3.scatter(missing_stage_dts, [0]*len(missing_stage_dts), color="#6b7280", marker="x", s=40, zorder=5, label=f"Missing Stage ({len(missing_stage_dts)}h)")

    ax3.axhline(0, color="#9ca3af", lw=0.8)
    ax3.set_ylabel("dH/dt (m / hr)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Time (IST)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", framealpha=0.9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=180)
    plt.close()
    print(f"Saved diagnostic plots to: {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    align_and_validate()
