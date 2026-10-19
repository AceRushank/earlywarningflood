"""
river_forecasting/src/audit_event1_lags.py
==========================================
Independent audit and verification of rainfall-stage lag cross-correlations.
"""

import pandas as pd
import numpy as np

def run_lag_audit():
    df = pd.read_csv("river_forecasting/outputs/event1_aligned_rainfall_stage.csv")
    clean_ds = df["clean_stage_m"].diff()
    rain_s = df["hourly_depth_mm"]

    print("=" * 115)
    print("INDEPENDENT LAG CROSS-CORRELATION AUDIT (0 to 72 hours, step 3h)")
    print("=" * 115)
    header = f"{'Lag (h)':<8} | {'N':<4} | {'Corr (r)':<9} | {'Stage Window (IST)':<42} | {'Rainfall Window (IST)':<42}"
    print(header)
    print("-" * 115)

    results = []
    for lag in range(0, 73, 3):
        shifted_rain = rain_s.shift(lag)
        mask = shifted_rain.notna() & clean_ds.notna()
        n = int(mask.sum())
        if n > 10:
            r = float(np.corrcoef(shifted_rain[mask], clean_ds[mask])[0, 1])
            s_start = df.loc[mask, "timestamp_ist"].iloc[0]
            s_end = df.loc[mask, "timestamp_ist"].iloc[-1]
            
            orig_idx = df.index[mask] - lag
            r_start = df.loc[orig_idx, "timestamp_ist"].iloc[0]
            r_end = df.loc[orig_idx, "timestamp_ist"].iloc[-1]
            
            row = {
                "lag_hours": lag,
                "n_samples": n,
                "correlation": r,
                "stage_start": s_start,
                "stage_end": s_end,
                "rain_start": r_start,
                "rain_end": r_end
            }
            results.append(row)
            print(f"{lag:<8} | {n:<4} | {r:<+9.3f} | {s_start} to {s_end} | {r_start} to {r_end}")

    df_res = pd.DataFrame(results)
    
    # Analyze the lag = 66h correlation in detail
    print("\n" + "=" * 115)
    print("DETAILED DECONSTRUCTION OF THE 66-HOUR LAG CORRELATION")
    print("=" * 115)
    lag = 66
    shifted_rain = rain_s.shift(lag)
    mask = shifted_rain.notna() & clean_ds.notna()
    
    stage_indices = df.index[mask]
    rain_indices = stage_indices - lag
    
    sub = pd.DataFrame({
        "timestamp_ist_stage": df.loc[stage_indices, "timestamp_ist"].values,
        "timestamp_ist_rain": df.loc[rain_indices, "timestamp_ist"].values,
        "rain_hourly_mm": rain_s.loc[rain_indices].values,
        "stage_change_m": clean_ds.loc[stage_indices].values,
        "clean_stage_m": df.loc[stage_indices, "clean_stage_m"].values
    })

    print(f"Top 5 largest rainfall values at lag 66 and their aligned stage changes:")
    top_rain = sub.sort_values("rain_hourly_mm", ascending=False).head(5)
    for _, row in top_rain.iterrows():
        print(f"  Rain at {row['timestamp_ist_rain']} ({row['rain_hourly_mm']:.3f} mm) -> Stage change at {row['timestamp_ist_stage']} ({row['stage_change_m']:+.3f} m, stage={row['clean_stage_m']:.3f} m)")

    print(f"\nTop 5 largest stage rises at lag 66 and their aligned rainfalls:")
    top_stage = sub.sort_values("stage_change_m", ascending=False).head(5)
    for _, row in top_stage.iterrows():
        print(f"  Stage rise at {row['timestamp_ist_stage']} ({row['stage_change_m']:+.3f} m, stage={row['clean_stage_m']:.3f} m) <- Rain at {row['timestamp_ist_rain']} ({row['rain_hourly_mm']:.3f} mm)")

    # Check leverage / influence: What happens if the single highest rain hour (2024-07-31 22:00) is removed?
    max_rain_idx = sub["rain_hourly_mm"].idxmax()
    sub_no_max = sub.drop(max_rain_idx)
    r_no_max = np.corrcoef(sub_no_max["rain_hourly_mm"], sub_no_max["stage_change_m"])[0, 1]
    print(f"\nInfluence Test:")
    print(f"  Full correlation (N=56):                r = {df_res[df_res['lag_hours']==66]['correlation'].iloc[0]:.3f}")
    print(f"  Without single highest rain point (N=55): r = {r_no_max:.3f}")
    
    # Check autocorrelation of stage differences and rain
    ac_rain = sub["rain_hourly_mm"].autocorr(lag=1)
    ac_stage = sub["stage_change_m"].autocorr(lag=1)
    print(f"\nAutocorrelation (AR1):")
    print(f"  Rainfall lag-1 autocorrelation:     {ac_rain:.3f}")
    print(f"  Stage-change lag-1 autocorrelation: {ac_stage:.3f}")

if __name__ == "__main__":
    run_lag_audit()
