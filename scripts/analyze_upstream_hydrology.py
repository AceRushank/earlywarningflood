"""
scripts/analyze_upstream_hydrology.py
====================================
Analyzes the hydrological lag, cross-correlation, and telemetry synchronization
between upstream Mudgal Barrage and local Vishnupuri Barrage for Monsoon 2026.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_CSV = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"

def analyze():
    print("Loading telemetry for Vishnupuri and Mudgal Barrage...")
    df_raw = pd.read_csv(TELEMETRY_CSV, usecols=[
        "Station", "Data Acquisition Time", "River Water Level Telemetry Hourly (meter)"
    ])
    
    stations = ["Vishnupuri", "Mudgal Barrage", "Majalgaon", "Lower Dudhana Dam", "Siddeshwar Dam"]
    df = df_raw[df_raw["Station"].isin(stations)].copy()
    df["dt"] = pd.to_datetime(df["Data Acquisition Time"], format="%d-%m-%Y %H:%M")
    df["stage"] = pd.to_numeric(df["River Water Level Telemetry Hourly (meter)"], errors="coerce")
    
    start_dt = pd.to_datetime("2026-06-15 00:00:00")
    end_dt = pd.to_datetime("2026-09-30 23:00:00")
    df = df[(df["dt"] >= start_dt) & (df["dt"] <= end_dt)]
    
    # Pivot
    piv = df.pivot_table(index="dt", columns="Station", values="stage", aggfunc="first")
    
    # Continuous grid
    grid = pd.date_range(start_dt, end_dt, freq="1h")
    piv = piv.reindex(grid)
    
    # Mask Vishnupuri dropout artifact (< 345m)
    piv.loc[piv["Vishnupuri"] < 345.0, "Vishnupuri"] = np.nan
    # Mask any Mudgal dropout artifact (< 370m)
    piv.loc[piv["Mudgal Barrage"] < 370.0, "Mudgal Barrage"] = np.nan
    
    # Compute 1h differences
    dh = piv.diff(1)
    
    print("\n--- Summary Statistics (Monsoon 2026: June 15 - Sept 30) ---")
    for col in piv.columns:
        valid_cnt = piv[col].notna().sum()
        pct = valid_cnt / len(piv) * 100
        print(f"Station: {col:25s} | Valid: {valid_cnt}/{len(piv)} ({pct:.1f}%) | Mean: {piv[col].mean():.2f}m | Min: {piv[col].min():.2f}m | Max: {piv[col].max():.2f}m")
        
    print("\n--- Cross-Correlation: Upstream Stations vs Vishnupuri Stage (H) ---")
    print(f"{'Lag (hr)':<10} | {'Mudgal':<10} | {'Majalgaon':<10} | {'Lower Dudhana':<15} | {'Siddeshwar':<12}")
    for lag in [0, 1, 2, 3, 6, 12, 18, 24, 30, 36, 48]:
        corrs = {}
        for st in ["Mudgal Barrage", "Majalgaon", "Lower Dudhana Dam", "Siddeshwar Dam"]:
            # Shift upstream station backward (upstream at t-lag vs vishnupuri at t)
            shifted = piv[st].shift(lag)
            corr = piv["Vishnupuri"].corr(shifted)
            corrs[st] = corr
        print(f"t - {lag:02d}h     | {corrs['Mudgal Barrage']:<10.4f} | {corrs['Majalgaon']:<10.4f} | {corrs['Lower Dudhana Dam']:<15.4f} | {corrs['Siddeshwar Dam']:<12.4f}")

    print("\n--- Cross-Correlation: Upstream dH_1h vs Vishnupuri dH_1h ---")
    print(f"{'Lag (hr)':<10} | {'Mudgal dH':<10} | {'Majalgaon dH':<12} | {'Lower Dudhana dH':<18} | {'Siddeshwar dH':<15}")
    for lag in [0, 1, 2, 3, 6, 12, 18, 24, 30, 36, 48]:
        corrs_dh = {}
        for st in ["Mudgal Barrage", "Majalgaon", "Lower Dudhana Dam", "Siddeshwar Dam"]:
            shifted = dh[st].shift(lag)
            corr = dh["Vishnupuri"].corr(shifted)
            corrs_dh[st] = corr
        print(f"t - {lag:02d}h     | {corrs_dh['Mudgal Barrage']:<10.4f} | {corrs_dh['Majalgaon']:<12.4f} | {corrs_dh['Lower Dudhana Dam']:<18.4f} | {corrs_dh['Siddeshwar Dam']:<15.4f}")

if __name__ == "__main__":
    analyze()
