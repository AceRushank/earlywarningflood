import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_event_a():
    df = pd.read_csv('data/events_2026/vishnupuri_telemetry_event_a_2026.csv')
    df['dt'] = pd.to_datetime(df['dt'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Stage Panel
    ax1.plot(df['dt'], df['stage_m'], color='#1e3a8a', lw=2.2, label='Vishnupuri Stage (Observed Telemetry)')
    ax1.axhline(355.0, color='#dc2626', ls='--', lw=1.2, label='Full Reservoir Level (355.0 m MSL)')
    ax1.axhline(354.0, color='#f59e0b', ls=':', lw=1.2, label='High Stage Threshold (354.0 m MSL)')
    
    # Annotate peak and rising limb
    t_rise_start = pd.to_datetime('2026-07-30 12:00:00')
    t_peak = pd.to_datetime('2026-08-04 01:00:00')
    ax1.axvspan(t_rise_start, t_peak, color='#38bdf8', alpha=0.15, label='Main Rising Limb (+4.54m surge)')

    ax1.set_ylabel('Water Level (m MSL)', fontsize=11, fontweight='bold')
    ax1.set_title('Vishnupuri Barrage — Event A: Major 2026 Monsoon Flood Surge (July 27 – August 6, 2026)', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_ylim(350.0, 355.5)

    # dH/dt Panel
    pos_mask = df['dh_1h'] >= 0
    neg_mask = df['dh_1h'] < 0
    ax2.bar(df.loc[pos_mask, 'dt'], df.loc[pos_mask, 'dh_1h'], width=0.035, color='#10b981', label='Rise (dH/dt > 0)')
    ax2.bar(df.loc[neg_mask, 'dt'], df.loc[neg_mask, 'dh_1h'], width=0.035, color='#ef4444', label='Fall (dH/dt < 0)')
    ax2.axhline(0, color='#9ca3af', lw=0.8)
    ax2.set_ylabel('dH/dt (m / hr)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Time (IST)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
    fig.autofmt_xdate()

    plt.tight_layout()
    out_img = 'data/events_2026/event_a_telemetry_hydrograph.png'
    plt.savefig(out_img, dpi=180)
    plt.close()
    print(f'Saved Event A hydrograph to: {out_img}')

    # Copy to artifact dir
    import shutil
    artifact_img = r'C:\Users\aceru\.gemini\antigravity-ide\brain\4c89c7ca-48e8-4bf2-bb58-32d093d7e52d\event_a_telemetry_hydrograph.png'
    shutil.copy(out_img, artifact_img)
    print(f'Copied to artifact dir: {artifact_img}')

if __name__ == '__main__':
    plot_event_a()
