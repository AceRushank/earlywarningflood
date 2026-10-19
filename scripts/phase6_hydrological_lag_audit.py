"""
scripts/phase6_hydrological_lag_audit.py
========================================
Stage 2: Hydrological Hypothesis Validation and Telemetry Audit for Phase 6.
Analyzes the lagged correlation and flood-wave routing relationship between
upstream Mudgal Barrage and downstream Vishnupuri Barrage across lead times
from 0h to 48h, evaluated strictly on training partitions to avoid leakage.

Outputs:
  - river_forecasting/outputs/phase6_medium_range/hydrological_lag_analysis.csv
  - river_forecasting/outputs/phase6_medium_range/feature_availability_audit.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase6_medium_range"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_RAW = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"


def run_lag_and_availability_audit():
    print("=" * 80)
    print("PHASE 6: HYDROLOGICAL LAG AUDIT & TELEMETRY SYNCHRONIZATION")
    print("=" * 80)

    # 1. Load authentic telemetry
    print(f"Reading raw telemetry from {CSV_RAW}...")
    usecols = [
        "Station", "Data Acquisition Time", "River Water Level Telemetry Hourly (meter)",
        "Latitude", "Longitude", "District", "River", "Basin"
    ]
    chunks = []
    for chunk in pd.read_csv(CSV_RAW, usecols=usecols, chunksize=100000):
        sub = chunk[chunk["Station"].isin(["Vishnupuri", "Mudgal Barrage"])]
        if len(sub) > 0:
            chunks.append(sub)
    df_raw = pd.concat(chunks, ignore_index=True)
    df_raw["dt"] = pd.to_datetime(df_raw["Data Acquisition Time"], format="%d-%m-%Y %H:%M")
    df_raw["stage"] = pd.to_numeric(df_raw["River Water Level Telemetry Hourly (meter)"], errors="coerce")

    # QC filters
    df_raw.loc[(df_raw["Station"] == "Vishnupuri") & (df_raw["stage"] < 345.0), "stage"] = np.nan
    df_raw.loc[(df_raw["Station"] == "Mudgal Barrage") & (df_raw["stage"] < 370.0), "stage"] = np.nan

    # Pivot to continuous hourly grid
    start_dt = pd.to_datetime("2026-06-15 00:00:00")
    end_dt = pd.to_datetime("2026-09-30 23:00:00")
    grid = pd.date_range(start_dt, end_dt, freq="1h")

    piv = df_raw.pivot_table(index="dt", columns="Station", values="stage", aggfunc="first")
    piv = piv.reindex(grid)
    piv.rename(columns={"Vishnupuri": "stage_vishnupuri", "Mudgal Barrage": "stage_mudgal"}, inplace=True)

    # Calculate rate of changes
    piv["dH_v_1h"] = piv["stage_vishnupuri"].diff(1)
    piv["dH_v_3h"] = piv["stage_vishnupuri"].diff(3)
    piv["dH_v_6h"] = piv["stage_vishnupuri"].diff(6)
    piv["dH_v_12h"] = piv["stage_vishnupuri"].diff(12)
    piv["dH_v_18h"] = piv["stage_vishnupuri"].diff(18)
    piv["dH_v_24h"] = piv["stage_vishnupuri"].diff(24)

    piv["dH_m_1h"] = piv["stage_mudgal"].diff(1)
    piv["dH_m_3h"] = piv["stage_mudgal"].diff(3)
    piv["dH_m_6h"] = piv["stage_mudgal"].diff(6)
    piv["dH_m_12h"] = piv["stage_mudgal"].diff(12)
    piv["dH_m_18h"] = piv["stage_mudgal"].diff(18)
    piv["dH_m_24h"] = piv["stage_mudgal"].diff(24)

    # 2. Chronological Splits
    train_mask = (piv.index >= "2026-06-15 00:00:00") & (piv.index <= "2026-08-10 23:00:00")
    val_mask   = (piv.index >= "2026-08-11 00:00:00") & (piv.index <= "2026-08-31 23:00:00")
    test_mask  = (piv.index >= "2026-09-01 00:00:00") & (piv.index <= "2026-09-30 23:00:00")
    
    # Specific major July monsoon surge episode on train (July 24 00:00 to July 31 23:00)
    surge_july_mask = (piv.index >= "2026-07-24 00:00:00") & (piv.index <= "2026-07-31 23:00:00")
    
    # Rising limb mask on train (dH_v > 0.010 m or dH_m > 0.020 m)
    train_rising_mask = train_mask & ((piv["dH_v_1h"] > 0.010) | (piv["dH_m_1h"] > 0.020))

    # 3. Lagged Cross-Correlation Analysis
    # Treat 18-24h as initial hypothesis, search lags 0h to 48h
    lags = list(range(0, 49))
    lag_records = []

    cohorts = {
        "Train_Full": train_mask,
        "Train_July_Surge": surge_july_mask,
        "Train_Rising_Limbs": train_rising_mask,
        "Val_Full": val_mask,
        "Test_Full": test_mask,
    }

    for c_name, mask in cohorts.items():
        sub = piv[mask].copy()
        for tau in lags:
            # Upstream shifted backward by tau hours (Mudgal at t - tau vs Vishnupuri at t)
            shifted_m_stage = piv["stage_mudgal"].shift(tau).loc[mask]
            shifted_m_dh1 = piv["dH_m_1h"].shift(tau).loc[mask]
            shifted_m_dh6 = piv["dH_m_6h"].shift(tau).loc[mask]
            shifted_m_dh12 = piv["dH_m_12h"].shift(tau).loc[mask]
            shifted_m_dh24 = piv["dH_m_24h"].shift(tau).loc[mask]

            r_stage = sub["stage_vishnupuri"].corr(shifted_m_stage)
            r_dh1 = sub["dH_v_1h"].corr(shifted_m_dh1)
            r_dh6 = sub["dH_v_6h"].corr(shifted_m_dh6)
            r_dh12 = sub["dH_v_12h"].corr(shifted_m_dh12)
            r_dh24 = sub["dH_v_24h"].corr(shifted_m_dh24)

            lag_records.append({
                "cohort": c_name,
                "lag_hours": tau,
                "corr_stage_H": round(r_stage, 4) if pd.notna(r_stage) else np.nan,
                "corr_dH_1h": round(r_dh1, 4) if pd.notna(r_dh1) else np.nan,
                "corr_dH_6h": round(r_dh6, 4) if pd.notna(r_dh6) else np.nan,
                "corr_dH_12h": round(r_dh12, 4) if pd.notna(r_dh12) else np.nan,
                "corr_dH_24h": round(r_dh24, 4) if pd.notna(r_dh24) else np.nan,
            })

    df_lags = pd.DataFrame(lag_records)
    lag_csv_path = OUTPUT_DIR / "hydrological_lag_analysis.csv"
    df_lags.to_csv(lag_csv_path, index=False)
    print(f"Saved hydrological lag analysis -> {lag_csv_path}")

    # Inspect peak lag on Train_Full
    df_train_lags = df_lags[df_lags["cohort"] == "Train_Full"]
    peak_dh1_row = df_train_lags.loc[df_train_lags["corr_dH_1h"].idxmax()]
    peak_dh6_row = df_train_lags.loc[df_train_lags["corr_dH_6h"].idxmax()]
    peak_dh24_row = df_train_lags.loc[df_train_lags["corr_dH_24h"].idxmax()]

    print(f"\n--- Peak Cross-Correlation on Train Split (June 15 - August 10, 2026) ---")
    print(f"  Peak 1-hour change correlation:  lag tau = {int(peak_dh1_row['lag_hours'])}h (r = {peak_dh1_row['corr_dH_1h']:.4f})")
    print(f"  Peak 6-hour change correlation:  lag tau = {int(peak_dh6_row['lag_hours'])}h (r = {peak_dh6_row['corr_dH_6h']:.4f})")
    print(f"  Peak 24-hour change correlation: lag tau = {int(peak_dh24_row['lag_hours'])}h (r = {peak_dh24_row['corr_dH_24h']:.4f})")

    # 4. Feature Availability and Latency Audit
    print("\nAuditing feature availability, reporting delays, and missingness...")
    splits = {
        "Train (June 15 - Aug 10)": train_mask,
        "Val (Aug 11 - Aug 31)": val_mask,
        "Test (Sept 1 - Sept 30)": test_mask,
        "Whole Monsoon (June 15 - Sept 30)": pd.Series(True, index=piv.index),
    }

    avail_rows = []
    for sp_name, s_mask in splits.items():
        sub_piv = piv[s_mask]
        n_hours = len(sub_piv)

        for st_col, st_name in [("stage_vishnupuri", "Vishnupuri (Downstream)"), ("stage_mudgal", "Mudgal Barrage (Upstream)")]:
            valid_cnt = sub_piv[st_col].notna().sum()
            missing_cnt = n_hours - valid_cnt
            pct = (valid_cnt / n_hours) * 100.0

            # Gap lengths
            is_na = sub_piv[st_col].isna()
            gaps = []
            c_gap = 0
            for val in is_na:
                if val:
                    c_gap += 1
                else:
                    if c_gap > 0:
                        gaps.append(c_gap)
                        c_gap = 0
            if c_gap > 0:
                gaps.append(c_gap)
            max_gap = max(gaps) if gaps else 0

            avail_rows.append({
                "period": sp_name,
                "station": st_name,
                "total_hours": n_hours,
                "valid_hours": int(valid_cnt),
                "missing_hours": int(missing_cnt),
                "availability_pct": round(pct, 2),
                "max_consecutive_gap_hr": int(max_gap),
                "reporting_latency_buffer_hr": 0 if "Vishnupuri" in st_name else 1,
                "provenance": "MWRD / CWC NWIC Surface Water Telemetry",
            })

    df_avail = pd.DataFrame(avail_rows)
    avail_csv_path = OUTPUT_DIR / "feature_availability_audit.csv"
    df_avail.to_csv(avail_csv_path, index=False)
    print(f"Saved feature availability audit -> {avail_csv_path}")

    for _, r in df_avail.iterrows():
        print(f"  [{r['period']}] {r['station']:28s} | Valid: {r['valid_hours']}/{r['total_hours']} ({r['availability_pct']}%) | Max Gap: {r['max_consecutive_gap_hr']}h")


if __name__ == "__main__":
    run_lag_and_availability_audit()
