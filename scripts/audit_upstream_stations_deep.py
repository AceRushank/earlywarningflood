"""
scripts/audit_upstream_stations_deep.py
======================================
Deep hydrological and telemetry audit of upstream candidate stations
in the Godavari basin relative to Vishnupuri Barrage (Nanded).
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_2026 = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"
CSV_2021_2025 = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2021_2025.csv"

CANDIDATES = [
    "Vishnupuri",
    "Mudgal Barrage",
    "Lower Dudhana Dam",
    "Majalgaon",
    "Siddeshwar Dam",
    "Upper Penganga (Isapur Dam)",
    "Upper Manar Dam",
    "Manar Dam",
    "Manjra Dam",
    "Gangapur Dam",
    "Darna Dam",
    "Raher",
    "Babhali Barrage",
]

def audit_dataset(csv_path, label):
    print("=" * 80)
    print(f"AUDITING FILE: {csv_path.name} ({label})")
    print("=" * 80)
    if not csv_path.exists():
        print(f"File {csv_path} does not exist!")
        return {}

    # Read candidate rows
    chunks = []
    usecols = [
        "Station", "Agency", "River", "Basin", "Tributary", "District",
        "Latitude", "Longitude", "Is_DischargeDataAvailable",
        "RL_of_zeroGauge", "MeanSeaLevel", "Data Acquisition Time",
        "River Water Level Telemetry Hourly (meter)"
    ]
    
    print("Scanning CSV in chunks...")
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=100000, low_memory=False):
        sub = chunk[chunk["Station"].isin(CANDIDATES)]
        if len(sub) > 0:
            chunks.append(sub)
            
    if not chunks:
        print("No candidate records found!")
        return {}
        
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total candidate rows matched: {len(df)}")
    
    df["dt"] = pd.to_datetime(df["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["stage"] = pd.to_numeric(df["River Water Level Telemetry Hourly (meter)"], errors="coerce")
    
    results = {}
    for st in CANDIDATES:
        st_df = df[df["Station"] == st].copy()
        if len(st_df) == 0:
            results[st] = {"status": "NOT_FOUND"}
            continue
            
        lat = st_df["Latitude"].iloc[0]
        lon = st_df["Longitude"].iloc[0]
        river = st_df["River"].iloc[0]
        trib = st_df["Tributary"].iloc[0]
        dist = st_df["District"].iloc[0]
        agency = st_df["Agency"].iloc[0]
        q_avail = st_df["Is_DischargeDataAvailable"].iloc[0]
        rl_zero = st_df["RL_of_zeroGauge"].iloc[0]
        msl = st_df["MeanSeaLevel"].iloc[0]
        
        # Valid stage filter
        valid_stage = st_df[st_df["stage"].notna() & (st_df["stage"] > -90) & (st_df["stage"] < 2500)]
        min_dt = st_df["dt"].min()
        max_dt = st_df["dt"].max()
        
        # 2026 Monsoon focus: June 15 to Sept 30
        monsoon_df = st_df[(st_df["dt"] >= "2026-06-15") & (st_df["dt"] <= "2026-09-30 23:00")]
        monsoon_valid = monsoon_df[monsoon_df["stage"].notna() & (monsoon_df["stage"] > -90) & (monsoon_df["stage"] < 2500)]
        
        results[st] = {
            "status": "FOUND",
            "agency": agency,
            "district": dist,
            "river": river,
            "tributary": trib,
            "latitude": lat,
            "longitude": lon,
            "discharge_avail": q_avail,
            "rl_zero": rl_zero,
            "msl": msl,
            "total_records": len(st_df),
            "valid_records": len(valid_stage),
            "min_date": str(min_dt),
            "max_date": str(max_dt),
            "monsoon_2026_total": len(monsoon_df),
            "monsoon_2026_valid": len(monsoon_valid),
            "stage_min": monsoon_valid["stage"].min() if len(monsoon_valid) > 0 else np.nan,
            "stage_max": monsoon_valid["stage"].max() if len(monsoon_valid) > 0 else np.nan,
            "stage_std": monsoon_valid["stage"].std() if len(monsoon_valid) > 0 else np.nan,
        }
        
    return results

if __name__ == "__main__":
    res_2026 = audit_dataset(CSV_2026, "2026-2030 Dataset")
    for st, d in res_2026.items():
        if d.get("status") == "FOUND":
            print(f"\nSTATION: {st}")
            print(f"  Coords: ({d['latitude']}, {d['longitude']}) | River: {d['river']} | Trib: {d['tributary']} | District: {d['district']}")
            print(f"  Discharge Avail: {d['discharge_avail']} | Agency: {d['agency']}")
            print(f"  Coverage: {d['min_date']} to {d['max_date']} | Total rows: {d['total_records']}, Valid: {d['valid_records']}")
            print(f"  Monsoon 2026 Valid (June 15 - Sept 30): {d['monsoon_2026_valid']} / {d['monsoon_2026_total']} ({d['monsoon_2026_valid']/2592*100:.1f}%)")
            print(f"  Stage Range: {d['stage_min']:.3f} to {d['stage_max']:.3f} m (Std: {d['stage_std']:.3f} m)")
        else:
            print(f"\nSTATION: {st} -> NOT FOUND")
