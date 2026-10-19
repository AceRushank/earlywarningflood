import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

TELEMETRY_CSV = r"data/historical/vishnupuri_telemetry_2026_probe.csv"

def rank_events():
    df = pd.read_csv(TELEMETRY_CSV)
    df['dt'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('dt').reset_index(drop=True)
    df['stage'] = pd.to_numeric(df['stage_m'], errors='coerce')

    # Focus window: June 15 to August 31, 2026
    sub = df[(df['dt'] >= '2026-06-15 00:00:00') & (df['dt'] <= '2026-08-31 23:00:00')].copy()
    sub = sub.sort_values('dt').reset_index(drop=True)

    # Detect coherent multi-day events
    # Let's inspect all major candidate windows
    candidate_windows = [
        # (Name, Start, End, Rationale)
        ("Event A (Major Monsoonal Flood Surge)", "2026-07-28 00:00:00", "2026-08-06 23:00:00", "Massive continuous stage rise from baseflow to FRL"),
        ("Event B (Pre-Surge Monsoon Inflow)", "2026-07-15 00:00:00", "2026-07-22 23:00:00", "Mid-July steady baseflow fluctuations prior to main flood"),
        ("Event C (Late-August Pool Fluctuation)", "2026-08-07 00:00:00", "2026-08-16 23:00:00", "Regulated post-peak crest holding and high releases"),
        ("Event D (Early Monsoon Bed Inflow)", "2026-06-15 00:00:00", "2026-06-25 23:00:00", "First monsoon arrival following maintenance outage"),
        ("Outage/Artifact (Sensor Dropout)", "2026-08-22 00:00:00", "2026-08-24 23:00:00", "Known 334.810m telemetry baseline dropout artifact")
    ]

    event_table = []

    for name, t_start, t_end, note in candidate_windows:
        dt_start = pd.to_datetime(t_start)
        dt_end = pd.to_datetime(t_end)
        w_df = sub[(sub['dt'] >= dt_start) & (sub['dt'] <= dt_end)].copy()
        
        expected_hrs = int((dt_end - dt_start).total_seconds() / 3600) + 1
        obs_hrs = len(w_df)
        missing_hrs = expected_hrs - obs_hrs
        pct = (obs_hrs / expected_hrs) * 100.0 if expected_hrs > 0 else 0.0

        if not w_df.empty:
            w_df['dh_1h'] = w_df['stage'].diff()
            min_st = w_df['stage'].min()
            max_st = w_df['stage'].max()
            delta_st = max_st - min_st
            max_rise = w_df['dh_1h'].max()
            max_rise_time = w_df.loc[w_df['dh_1h'].idxmax(), 'dt'] if not w_df['dh_1h'].isna().all() else None
        else:
            min_st, max_st, delta_st, max_rise, max_rise_time = np.nan, np.nan, np.nan, np.nan, None

        event_table.append({
            "name": name,
            "start_ist": t_start,
            "end_ist": t_end,
            "duration_h": expected_hrs,
            "observed_h": obs_hrs,
            "missing_h": missing_hrs,
            "coverage_pct": round(pct, 2),
            "min_stage_m": round(min_st, 3),
            "max_stage_m": round(max_st, 3),
            "total_rise_m": round(delta_st, 3),
            "max_1h_rise_m": round(max_rise, 3) if not np.isnan(max_rise) else np.nan,
            "max_rise_time": str(max_rise_time),
            "note": note
        })

    ev_df = pd.DataFrame(event_table)
    # Sort by total rise excluding the artifact
    ev_df['is_artifact'] = ev_df['name'].str.contains('Artifact')
    ev_df = ev_df.sort_values(by=['is_artifact', 'total_rise_m'], ascending=[True, False]).reset_index(drop=True)

    print("=" * 90)
    print("RANKED 2026 MONSOON RISING LIMB EVENTS (JUNE 15 - AUGUST 31, 2026)")
    print("=" * 90)
    for idx, r in ev_df.iterrows():
        rank = f"Rank {idx+1}" if not r['is_artifact'] else "DISQUALIFIED"
        print(f"\n{rank}: {r['name']}")
        print(f"  Window:          {r['start_ist']} to {r['end_ist']} ({r['duration_h']} hours)")
        print(f"  Stage Range:     {r['min_stage_m']:.3f} m -> {r['max_stage_m']:.3f} m (Rise: +{r['total_rise_m']:.3f} m)")
        print(f"  Max 1h Rise:     +{r['max_1h_rise_m']:.3f} m/hr at {r['max_rise_time']}")
        print(f"  Data Quality:    {r['observed_h']}/{r['duration_h']} hrs ({r['coverage_pct']}%), Missing: {r['missing_h']} hrs")
        print(f"  Hydrology:       {r['note']}")

    return ev_df

if __name__ == '__main__':
    rank_events()
