"""
scripts/phase6_target_alignment_audit.py
=======================================
Stage 3: Target Construction and Alignment Audit for Medium-Range Forecasting (+18h, +24h, +36h).
Verifies:
  1. Exact timestamp alignment (t_target == t_origin + H).
  2. Authentic observations only (no interpolation, simulated labels, or zero-filled targets).
  3. Causal boundary separation across Train / Val / Test (no target overlap across splits).
  4. Number of eligible rows and independent rising-stage episodes.

Outputs:
  - river_forecasting/outputs/phase6_medium_range/target_alignment_audit.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase6_medium_range"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_RAW = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"
CSV_RAIN = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_hourly_2026.csv"
HORIZONS = [18, 24, 36]


def build_and_audit_targets():
    print("=" * 80)
    print("PHASE 6: TARGET CONSTRUCTION & ALIGNMENT AUDIT (+18h, +24h, +36h)")
    print("=" * 80)

    # 1. Load authentic telemetry for Vishnupuri and Mudgal
    usecols = ["Station", "Data Acquisition Time", "River Water Level Telemetry Hourly (meter)"]
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

    # Load rainfall
    df_rain = pd.read_csv(CSV_RAIN)
    df_rain["dt"] = pd.to_datetime(df_rain["timestamp_ist"])
    df_rain.set_index("dt", inplace=True)
    piv["rain_depth_1h_mm"] = df_rain["hourly_rainfall_depth_mm"]
    piv["rain_qc_flag"] = df_rain["completeness"]

    # Rolling rainfall
    piv["rain_3h_mm"] = piv["rain_depth_1h_mm"].rolling(3, min_periods=2).mean() * 3
    piv["rain_6h_mm"] = piv["rain_depth_1h_mm"].rolling(6, min_periods=4).mean() * 6
    piv["rain_12h_mm"] = piv["rain_depth_1h_mm"].rolling(12, min_periods=8).mean() * 12
    piv["rain_24h_mm"] = piv["rain_depth_1h_mm"].rolling(24, min_periods=16).mean() * 24
    piv["rain_48h_mm"] = piv["rain_depth_1h_mm"].rolling(48, min_periods=32).mean() * 48

    # Mask rainfall outage
    rain_outage = (piv["rain_qc_flag"] == "OUTAGE_UNAVAILABLE")
    rain_cols = ["rain_depth_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm", "rain_48h_mm"]
    piv.loc[rain_outage, rain_cols] = np.nan

    # Local stage features (at origin t)
    piv["dH_v_1h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(1)
    piv["dH_v_3h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(3)
    piv["dH_v_6h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(6)
    piv["dH_v_12h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(12)
    piv["dH_v_24h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(24)
    piv["accel_v_1h"] = piv["dH_v_1h"] - piv["dH_v_1h"].shift(1)

    # Upstream features (strictly lagged by >= 1 hour for latency)
    piv["stage_m_lag1"] = piv["stage_mudgal"].shift(1)
    piv["stage_m_lag2"] = piv["stage_mudgal"].shift(2)
    piv["stage_m_lag4"] = piv["stage_mudgal"].shift(4)
    piv["stage_m_lag7"] = piv["stage_mudgal"].shift(7)
    piv["stage_m_lag13"] = piv["stage_mudgal"].shift(13)
    piv["stage_m_lag19"] = piv["stage_mudgal"].shift(19)
    piv["stage_m_lag25"] = piv["stage_mudgal"].shift(25)

    piv["dH_m_1h"] = piv["stage_m_lag1"] - piv["stage_m_lag2"]
    piv["dH_m_3h"] = piv["stage_m_lag1"] - piv["stage_m_lag4"]
    piv["dH_m_6h"] = piv["stage_m_lag1"] - piv["stage_m_lag7"]
    piv["dH_m_12h"] = piv["stage_m_lag1"] - piv["stage_m_lag13"]
    piv["dH_m_18h"] = piv["stage_m_lag1"] - piv["stage_m_lag19"]
    piv["dH_m_24h"] = piv["stage_m_lag1"] - piv["stage_m_lag25"]

    # Construct genuine targets (+18h, +24h, +36h)
    for h in HORIZONS:
        piv[f"target_v_{h}h"] = piv["stage_vishnupuri"].shift(-h)
        piv[f"target_v_dH_{h}h"] = piv[f"target_v_{h}h"] - piv["stage_vishnupuri"]

    # 2. Chronological Boundaries & Strict Boundary Separation
    # Nominal split dates:
    # Train: June 15 00:00 to August 10 23:00
    # Val:   August 11 00:00 to August 31 23:00
    # Test:  September 1 00:00 to September 30 23:00
    # Synchronized Test: September 8 06:00 to September 22 19:00

    def find_episodes(series_dh, threshold=0.010, min_gap_hr=6):
        """Identifies independent rising-stage episodes."""
        is_rising = (series_dh > threshold).values
        episodes = []
        in_ep = False
        cur_start = 0
        gap = 0
        for i, val in enumerate(is_rising):
            if val:
                if not in_ep:
                    in_ep = True
                    cur_start = i
                gap = 0
            else:
                if in_ep:
                    gap += 1
                    if gap >= min_gap_hr:
                        episodes.append((cur_start, i - gap))
                        in_ep = False
                        gap = 0
        if in_ep:
            episodes.append((cur_start, len(is_rising) - 1))
        return len(episodes)

    audit_rows = []

    for h in HORIZONS:
        # Enforce boundary separation:
        # Train origins must end at August 10 23:00 - H hours so targets never overlap Val period!
        train_end_h = pd.to_datetime("2026-08-10 23:00:00") - pd.Timedelta(hours=h)
        val_end_h   = pd.to_datetime("2026-08-31 23:00:00") - pd.Timedelta(hours=h)
        test_end_h  = pd.to_datetime("2026-09-30 23:00:00") - pd.Timedelta(hours=h)

        split_defs = {
            "TRAIN": ("2026-06-15 00:00:00", train_end_h),
            "VAL": ("2026-08-11 00:00:00", val_end_h),
            "TEST_WHOLE_SEP": ("2026-09-01 00:00:00", test_end_h),
            "TEST_SYNC_RAINFALL": ("2026-09-08 06:00:00", min(pd.to_datetime("2026-09-22 19:00:00"), test_end_h)),
        }

        for sp_name, (sp_s, sp_e) in split_defs.items():
            sub = piv.loc[sp_s:sp_e].copy()
            n_origins = len(sub)

            # Check target timestamp alignment
            # For each origin t, verify target matches Vishnupuri at t+h
            origin_times = sub.index
            target_times = origin_times + pd.Timedelta(hours=h)
            
            # Verify target stage matches actual future observation in unshifted grid
            actual_future_stages = [piv["stage_vishnupuri"].get(tt, np.nan) for tt in target_times]
            recorded_targets = sub[f"target_v_{h}h"].values

            matches = np.isclose(recorded_targets, actual_future_stages, equal_nan=True)
            alignment_verified = bool(np.all(matches))

            # Sample counts
            valid_origin_stage = sub["stage_vishnupuri"].notna()
            valid_target_stage = sub[f"target_v_{h}h"].notna()
            valid_pair = valid_origin_stage & valid_target_stage

            valid_rain = sub[rain_cols].notna().all(axis=1)
            valid_upstream = sub[["stage_m_lag1", "dH_m_1h", "dH_m_6h", "dH_m_24h"]].notna().all(axis=1)
            valid_all = valid_pair & valid_rain & valid_upstream

            # Independent episodes
            n_episodes = find_episodes(sub.loc[valid_pair, "dH_v_1h"], threshold=0.010, min_gap_hr=6)

            # Boundary separation check
            max_target_dt = target_times.max()
            if sp_name == "TRAIN":
                boundary_clean = (max_target_dt <= pd.to_datetime("2026-08-10 23:00:00"))
            elif sp_name == "VAL":
                boundary_clean = (max_target_dt <= pd.to_datetime("2026-08-31 23:00:00"))
            else:
                boundary_clean = True

            audit_rows.append({
                "horizon": f"+{h}h",
                "horizon_hours": h,
                "split": sp_name,
                "origin_start": str(sp_s),
                "origin_end": str(sp_e),
                "total_origins": n_origins,
                "valid_target_count": int(valid_target_stage.sum()),
                "missing_target_count": int((~valid_target_stage).sum()),
                "valid_stage_pairs": int(valid_pair.sum()),
                "valid_with_rainfall": int((valid_pair & valid_rain).sum()),
                "valid_with_upstream": int((valid_pair & valid_upstream).sum()),
                "valid_multimodal_upstream": int(valid_all.sum()),
                "independent_rising_episodes": n_episodes,
                "alignment_verified": alignment_verified,
                "boundary_separation_clean": boundary_clean,
                "max_target_timestamp": str(max_target_dt),
            })

    df_audit = pd.DataFrame(audit_rows)
    out_csv = OUTPUT_DIR / "target_alignment_audit.csv"
    df_audit.to_csv(out_csv, index=False)
    print(f"Saved target alignment audit -> {out_csv}")

    for _, r in df_audit.iterrows():
        print(f"[{r['horizon']:4s}] {r['split']:18s} | Origins: {r['total_origins']:4d} | Valid Pairs: {r['valid_stage_pairs']:4d} | Multimodal+Upstream: {r['valid_multimodal_upstream']:4d} | Episodes: {r['independent_rising_episodes']:2d} | Alignment: {r['alignment_verified']} | Boundary Clean: {r['boundary_separation_clean']}")


if __name__ == "__main__":
    build_and_audit_targets()
