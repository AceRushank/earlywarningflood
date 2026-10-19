import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TELEMETRY_CSV = r"data/historical/vishnupuri_telemetry_2026_probe.csv"
OUT_PLOT = r"data/historical/full_monsoon_2026_hydrograph.png"
ARTIFACT_PLOT = r"C:\Users\aceru\.gemini\antigravity-ide\brain\4c89c7ca-48e8-4bf2-bb58-32d093d7e52d\full_monsoon_2026_hydrograph.png"

def audit_full_season():
    print("=" * 80)
    print("PHASE 1: FULL MONSOON 2026 VISHNUPURI TELEMETRY AUDIT")
    print("=" * 80)

    df = pd.read_csv(TELEMETRY_CSV)
    df['dt'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('dt').reset_index(drop=True)
    df['stage'] = pd.to_numeric(df['stage_m'], errors='coerce')

    # Full season window: June 1, 2026 00:00 to October 10, 2026 23:00
    start_dt = pd.to_datetime('2026-06-01 00:00:00')
    end_dt = pd.to_datetime('2026-10-10 23:00:00')
    sub = df[(df['dt'] >= start_dt) & (df['dt'] <= end_dt)].copy().reset_index(drop=True)

    expected_range = pd.date_range(start_dt, end_dt, freq='h')
    n_expected = len(expected_range)
    n_obs = len(sub)
    print(f"Study Window:       {start_dt} to {end_dt}")
    print(f"Expected hours:     {n_expected} hours")
    print(f"Observed records:   {n_obs} hours ({n_obs / n_expected * 100:.2f}% raw coverage)")
    print(f"Missing hours:      {n_expected - n_obs} hours")

    # Metadata & Datum
    sample = sub.iloc[0]
    print(f"\nMetadata Verification:")
    print(f"  Station:         {sample['Station']}")
    print(f"  Agency:          {sample['Agency']}")
    print(f"  River / Basin:   {sample['River']} / {sample['Basin']}")
    print(f"  Latitude / Lon:  {sample['Latitude']} N, {sample['Longitude']} E")
    print(f"  Gauge Datum:     MeanSeaLevel={sample['MeanSeaLevel']} m, RL_of_zeroGauge={sample['RL_of_zeroGauge']}")
    print(f"  Measurement:     Water Level in meters above Mean Sea Level (MSL)")

    # Data Quality: Duplicates, NaNs, Dropouts
    dups = sub['dt'].duplicated().sum()
    nans = sub['stage'].isna().sum()
    print(f"\nQuality Checks:")
    print(f"  Duplicate Timestamps: {dups}")
    print(f"  NaN / Null values:    {nans}")

    # Detect sensor dropouts
    # We know 334.810 m is the bottom baseline dropout
    dropouts = sub[sub['stage'] < 345.0]
    print(f"  Sensor Dropouts (< 345m MSL): {len(dropouts)} records")
    if not dropouts.empty:
        print(f"    Dropout window: {dropouts['dt'].min()} to {dropouts['dt'].max()}, values: {dropouts['stage'].unique()}")

    # Valid hydrological stage range (excluding dropouts)
    valid_hydro = sub[sub['stage'] >= 345.0]
    print(f"\nValid Hydrological Stage Range (excluding dropouts):")
    print(f"  Min Stage:  {valid_hydro['stage'].min():.3f} m MSL")
    print(f"  Max Stage:  {valid_hydro['stage'].max():.3f} m MSL")
    print(f"  Mean Stage: {valid_hydro['stage'].mean():.3f} m MSL")
    print(f"  Total Dynamic Range: {valid_hydro['stage'].max() - valid_hydro['stage'].min():.3f} m")

    # Reindex to continuous grid to map all gaps
    full_df = pd.DataFrame({'dt': expected_range})
    merged = pd.merge(full_df, sub[['dt', 'stage']], on='dt', how='left')
    merged['is_missing'] = merged['stage'].isna()
    merged['is_dropout'] = (merged['stage'] < 345.0)
    merged['is_valid'] = (~merged['is_missing']) & (~merged['is_dropout'])

    # Gaps analysis
    missing_blocks = []
    in_gap = False
    gap_start = None
    for idx, row in merged.iterrows():
        if row['is_missing'] and not in_gap:
            in_gap = True
            gap_start = row['dt']
        elif not row['is_missing'] and in_gap:
            in_gap = False
            gap_end = merged.loc[idx - 1, 'dt']
            gap_len = int((gap_end - gap_start).total_seconds() / 3600) + 1
            missing_blocks.append((gap_start, gap_end, gap_len))
    if in_gap:
        gap_end = merged.iloc[-1]['dt']
        gap_len = int((gap_end - gap_start).total_seconds() / 3600) + 1
        missing_blocks.append((gap_start, gap_end, gap_len))

    print(f"\nMissing Time Intervals ({len(missing_blocks)} distinct blocks):")
    for s, e, l in sorted(missing_blocks, key=lambda x: x[2], reverse=True)[:8]:
        print(f"  {s} to {e} ({l} hours)")

    # Identify all Season Episodes
    episodes = [
        {"name": "Pre-Monsoon Storage", "start": "2026-06-01 00:00", "end": "2026-06-03 23:00", "type": "Holding", "color": "#94a3b8"},
        {"name": "Maintenance Outage", "start": "2026-06-04 00:00", "end": "2026-06-14 23:00", "type": "Missing Data", "color": "#f1f5f9"},
        {"name": "Early Monsoon Inflow", "start": "2026-06-15 00:00", "end": "2026-07-10 23:00", "type": "Base Inflow", "color": "#38bdf8"},
        {"name": "Mid-July Baseflow", "start": "2026-07-11 00:00", "end": "2026-07-28 23:00", "type": "Steady Baseflow", "color": "#60a5fa"},
        {"name": "Major Flood Surge (Event A)", "start": "2026-07-29 00:00", "end": "2026-08-04 23:00", "type": "Rapid Flood Wave (+4.54m)", "color": "#2563eb"},
        {"name": "August Regulated Spills", "start": "2026-08-05 00:00", "end": "2026-08-21 23:00", "type": "High Pool Surcharge Release", "color": "#10b981"},
        {"name": "Telemetry Dropout", "start": "2026-08-22 00:00", "end": "2026-08-23 23:00", "type": "Sensor Artifact (334.8m)", "color": "#ef4444"},
        {"name": "Late-August High Pool", "start": "2026-08-24 00:00", "end": "2026-09-07 23:00", "type": "Full Pool Holding", "color": "#059669"},
        {"name": "September Convective Events", "start": "2026-09-08 00:00", "end": "2026-09-22 23:00", "type": "Audited Satellite Window", "color": "#d97706"},
        {"name": "Monsoon Recession", "start": "2026-09-23 00:00", "end": "2026-10-09 23:00", "type": "Post-Monsoon Recession", "color": "#64748b"},
    ]

    print("\nSeason-Wide Episodes Breakdown:")
    for ep in episodes:
        print(f"  {ep['name']:<30} | {ep['start']} to {ep['end']} | {ep['type']}")

    # Generate Full-Season Hydrograph Plot
    print("\nGenerating Full-Season Stage Hydrograph Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    # Mask dropouts for clean hydrological line
    clean_stage = merged['stage'].copy()
    clean_stage[clean_stage < 345.0] = np.nan

    # Upper panel: Stage MSL
    ax1.plot(merged['dt'], clean_stage, color='#1e3a8a', lw=1.8, label='Vishnupuri River Stage (Valid Telemetry)')
    # Plot dropout artifact in red dots
    dropout_rows = merged[merged['is_dropout']]
    if not dropout_rows.empty:
        ax1.scatter(dropout_rows['dt'], [349.5]*len(dropout_rows), color='#dc2626', marker='v', s=25, label='Sensor Dropout Artifact (334.8m, plotted at baseline)')

    # Operational thresholds
    ax1.axhline(355.0, color='#dc2626', ls='--', lw=1.5, label='Full Reservoir Level (FRL = 355.00 m MSL)')
    ax1.axhline(355.5, color='#991b1b', ls=':', lw=1.2, label='Maximum Water Level (MWL = 355.50 m MSL)')
    ax1.axhline(354.0, color='#f59e0b', ls='--', lw=1.2, label='High Stage / Warning Level (~354.00 m MSL)')
    ax1.axhline(347.0, color='#64748b', ls='-.', lw=1.0, label='Crest / Sill Level of Gates (347.00 m MSL)')

    # Highlight major flood surge
    ax1.axvspan(pd.to_datetime('2026-07-29'), pd.to_datetime('2026-08-04'), color='#38bdf8', alpha=0.25, label='Event A: Major Flood Surge (+4.54m)')
    # Highlight audited September window
    ax1.axvspan(pd.to_datetime('2026-09-08'), pd.to_datetime('2026-09-22'), color='#fef08a', alpha=0.35, label='September Audited Satellite Window (669 granules)')

    ax1.set_ylabel('Water Level (m MSL)', fontsize=11, fontweight='bold')
    ax1.set_title('Vishnupuri Barrage (Godavari Basin) — Whole-Monsoon 2026 Stage Hydrograph & Episode Audit', fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylim(346.5, 356.0)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=9)

    # Lower panel: Data Availability & QC Flag Status
    # Categorical indicator: 1 = Valid Telemetry, 0 = Missing / Outage, -1 = Dropout Artifact
    qc_stat = np.where(merged['is_valid'], 1.0, np.where(merged['is_dropout'], -0.5, 0.0))
    ax2.plot(merged['dt'], qc_stat, color='#0284c7', lw=1.2, label='Telemetry Status')
    ax2.fill_between(merged['dt'], 0, qc_stat, where=(qc_stat == 1.0), color='#10b981', alpha=0.4, label='Valid Hourly Telemetry (2,688 hrs)')
    ax2.fill_between(merged['dt'], 0, 1.0, where=(qc_stat == 0.0), color='#f97316', alpha=0.3, label='Missing Hours (460 hrs)')
    ax2.fill_between(merged['dt'], -0.5, 0, where=(qc_stat == -0.5), color='#ef4444', alpha=0.6, label='Sensor Dropout (20 hrs)')

    ax2.set_yticks([-0.5, 0.0, 1.0])
    ax2.set_yticklabels(['Dropout', 'Missing', 'Valid'], fontsize=9, fontweight='bold')
    ax2.set_ylabel('QC Status', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Date (IST)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', framealpha=0.9, fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=180)
    import shutil
    shutil.copy(OUT_PLOT, ARTIFACT_PLOT)
    plt.close()
    print(f"Saved Full-Season Hydrograph Plot to: {OUT_PLOT}")
    print(f"Copied to artifact directory: {ARTIFACT_PLOT}")

if __name__ == '__main__':
    audit_full_season()
