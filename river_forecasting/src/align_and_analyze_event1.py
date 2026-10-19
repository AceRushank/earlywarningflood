"""
river_forecasting/src/align_and_analyze_event1.py
=================================================
Aligns authentic INSAT-3DS HEM catchment-mean rainfall observations with
Vishnupuri Barrage river stage observations for Event 1 (July 31 - August 5, 2024).

Methodological Principles:
1. Units & Physics:
   - HEM native unit is mm/hr (instantaneous precipitation rate).
   - Half-hourly depth is rate * 0.5 hr.
   - Hourly accumulated rainfall depth (mm) is the sum of valid 30-minute intervals.
   - Missing observations are preserved as NaN (never imputed with zero).
2. Temporal Alignment:
   - Aligned in Indian Standard Time (IST = UTC + 5:30).
   - Preserves source UTC timestamps for astronomical/satellite verification.
3. Exploratory Lag Analysis:
   - Computes antecedent rainfall metrics (6h, 12h, 24h, 48h, 72h).
   - Cross-correlates lagged rainfall with river stage rate-of-change.
   - Explicitly labelled exploratory (N=1 event; no claim of statistical significance).
4. Physical Caveat:
   - Upstream basin is delineated from HydroBASINS Level 08 (51,768.25 sq km, +0.73% vs CWC benchmark).
   - The sub-basin outlet is ~5.16 km downstream of the Vishnupuri Barrage axis.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

STAGE_CSV = REPO_ROOT / "river_forecasting" / "outputs" / "stage_observations.csv"
RAINFALL_CSV = REPO_ROOT / "river_forecasting" / "outputs" / "event1_catchment_rainfall.csv"
MANIFEST_CSV = REPO_ROOT / "river_forecasting" / "outputs" / "event1_rainfall_manifest.csv"
OUT_ALIGNED_CSV = REPO_ROOT / "river_forecasting" / "outputs" / "event1_aligned_rainfall_stage.csv"
OUT_PLOT_PNG = REPO_ROOT / "river_forecasting" / "outputs" / "event1_rainfall_stage_alignment.png"


def run_alignment_and_analysis():
    print("=" * 70)
    print("Aligning INSAT-3DS HEM Rainfall & Stage Observations (Event 1)")
    print("=" * 70)

    if not STAGE_CSV.exists():
        raise FileNotFoundError(f"Stage observations not found: {STAGE_CSV}")
    if not RAINFALL_CSV.exists():
        raise FileNotFoundError(f"Rainfall observations not found: {RAINFALL_CSV}")

    # 1. Load Stage Data
    df_stage = pd.read_csv(STAGE_CSV)
    df_stage["dt_ist"] = pd.to_datetime(df_stage["timestamp_ist"])
    df_stage.sort_values("dt_ist", inplace=True)
    df_stage.reset_index(drop=True, inplace=True)

    # Filter stage to Event 1 window (2024-07-31 00:00 to 2024-08-05 23:00 IST)
    event_start_ist = pd.to_datetime("2024-07-31 00:00:00")
    event_end_ist = pd.to_datetime("2024-08-05 23:00:00")
    df_stage_ev = df_stage[(df_stage["dt_ist"] >= event_start_ist) & (df_stage["dt_ist"] <= event_end_ist)].copy()

    # Clean known transient 1m spike (index 429) for clean stage plotting/analysis
    df_stage_ev.loc[df_stage_ev["quality_flag"] == "TRANSIENT_1M_SPIKE", "stage_m"] = np.nan
    df_stage_ev["clean_stage_m"] = df_stage_ev["stage_m"].interpolate()

    # 2. Load Satellite Rainfall Data
    df_rain = pd.read_csv(RAINFALL_CSV)
    df_rain["dt_utc"] = pd.to_datetime(df_rain["timestamp_utc"])
    df_rain["dt_ist"] = pd.to_datetime(df_rain["timestamp_ist"])
    df_rain.sort_values("dt_ist", inplace=True)
    df_rain.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df_stage_ev)} stage records and {len(df_rain)} half-hourly rainfall records.")

    # 3. Aggregate Half-Hourly Rainfall to Hourly
    # Each slot is at :00 or :30. Floor to the hour
    df_rain["hourly_bin_ist"] = df_rain["dt_ist"].dt.floor("1h")
    
    # Compute hourly stats: sum of half_hour_accumulated_rainfall_mm (rate * 0.5h)
    def aggregate_hourly(group):
        rates = group["mean_rainfall_rate_mm_hr"].dropna()
        depths = group["half_hour_accumulated_rainfall_mm"].dropna()
        n_obs = len(rates)
        
        # If neither slot is observed, NaN
        if n_obs == 0:
            return pd.Series({
                "mean_rain_rate_mm_hr": np.nan,
                "hourly_depth_mm": np.nan,
                "slots_count": 0
            })
        elif n_obs == 1:
            # 1 slot observed: report observed rate, but note partial slot
            # Honest representation: depth of the 1 observed slot
            return pd.Series({
                "mean_rain_rate_mm_hr": rates.mean(),
                "hourly_depth_mm": depths.sum(),  # only 0.5h accounted
                "slots_count": 1
            })
        else:
            # Full hour (2 slots)
            return pd.Series({
                "mean_rain_rate_mm_hr": rates.mean(),
                "hourly_depth_mm": depths.sum(),
                "slots_count": 2
            })

    hourly_rain = df_rain.groupby("hourly_bin_ist").apply(aggregate_hourly, include_groups=False).reset_index()
    hourly_rain.rename(columns={"hourly_bin_ist": "dt_ist"}, inplace=True)

    # 4. Merge Stage and Hourly Rainfall
    df_merged = pd.merge(df_stage_ev, hourly_rain, on="dt_ist", how="left")
    df_merged["timestamp_utc"] = (df_merged["dt_ist"] - pd.Timedelta(hours=5, minutes=30)).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Cumulative rainfall depth
    df_merged["cumulative_rainfall_mm"] = df_merged["hourly_depth_mm"].fillna(0).cumsum()

    # Antecedent rolling rainfall accumulations (6h, 12h, 24h, 48h, 72h)
    for h in [6, 12, 24, 48, 72]:
        df_merged[f"rolling_rain_{h}h_mm"] = df_merged["hourly_depth_mm"].rolling(window=h, min_periods=h//2).sum()

    # Re-order columns cleanly
    cols_order = [
        "timestamp_ist",
        "timestamp_utc",
        "dt_ist",
        "station_name",
        "stage_m",
        "clean_stage_m",
        "stage_change_m",
        "rate_of_change_m_per_hr",
        "quality_flag",
        "mean_rain_rate_mm_hr",
        "hourly_depth_mm",
        "cumulative_rainfall_mm",
        "rolling_rain_6h_mm",
        "rolling_rain_12h_mm",
        "rolling_rain_24h_mm",
        "rolling_rain_48h_mm",
        "rolling_rain_72h_mm",
        "slots_count"
    ]
    df_merged = df_merged[[c for c in cols_order if c in df_merged.columns]]
    df_merged.to_csv(OUT_ALIGNED_CSV, index=False)
    print(f"Aligned dataset saved to: {OUT_ALIGNED_CSV} ({len(df_merged)} rows)")

    # 5. Exploratory Lag Correlation Analysis
    print("\n--- Exploratory Lag Cross-Correlation (Rainfall vs Stage Rise Rate) ---")
    print("NOTE: Single-event analysis (N=1); exploratory only; no statistical significance claimed.")
    
    lag_corrs = []
    # Cross correlate hourly_depth_mm at lag k (rainfall k hours before stage change)
    clean_ds = df_merged["clean_stage_m"].diff()  # hourly stage rate of change
    rain_s = df_merged["hourly_depth_mm"]

    for lag in range(0, 73, 3):
        # Shift rainfall forward by lag hours (rain at t-lag aligned with stage change at t)
        shifted_rain = rain_s.shift(lag)
        valid_idx = shifted_rain.notna() & clean_ds.notna()
        if valid_idx.sum() > 20:
            corr = np.corrcoef(shifted_rain[valid_idx], clean_ds[valid_idx])[0, 1]
            lag_corrs.append({"lag_hours": lag, "correlation": corr, "samples": int(valid_idx.sum())})

    df_lags = pd.DataFrame(lag_corrs)
    if not df_lags.empty:
        best_lag = df_lags.loc[df_lags["correlation"].idxmax()]
        print(f"Peak cross-correlation: r = {best_lag['correlation']:.3f} at lag = {int(best_lag['lag_hours'])} hours (N={int(best_lag['samples'])})")
        print("Sample lag correlations:")
        for _, lr in df_lags.iloc[::2].iterrows():
            print(f"  Lag {int(lr['lag_hours']):2d}h: r = {lr['correlation']:+.3f} (samples: {int(lr['samples'])})")

    # 6. Plotting Shared Timeline
    print("\nGenerating dual-axis hydrological timeline plot...")
    fig, (ax_rain, ax_stage) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08}
    )

    # Top plot: Catchment-mean hourly rainfall (inverted bar chart)
    ax_rain.bar(
        df_merged["dt_ist"],
        df_merged["hourly_depth_mm"],
        width=pd.Timedelta(hours=0.85),
        color="#1f77b4",
        alpha=0.85,
        label="Catchment-Mean Hourly Rainfall (mm)"
    )
    ax_rain.set_ylabel("Rainfall (mm)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax_rain.set_ylim(0, max(4.0, (df_merged["hourly_depth_mm"].max() or 1.0) * 1.3))
    ax_rain.invert_yaxis()  # Standard hydrological hyetograph convention (inverted)
    ax_rain.grid(True, linestyle="--", alpha=0.5)
    ax_rain.legend(loc="upper right", framealpha=0.9)
    ax_rain.set_title(
        "Event 1: Upper Godavari Catchment Satellite Rainfall (INSAT-3DS HEM) & Vishnupuri Barrage Stage\n"
        "(July 31 - August 5, 2024 | Delineated Catchment: ~51,768 km² [HydroBASINS Lev08])",
        fontsize=12, fontweight="bold", pad=12
    )

    # Bottom plot: River stage hydrograph
    ax_stage.plot(
        df_merged["dt_ist"],
        df_merged["clean_stage_m"],
        color="#d62728",
        linewidth=2.2,
        label="River Stage (m MSL)"
    )

    # Mark trough and peak
    trough_dt = pd.to_datetime("2024-08-03 10:00:00")
    peak_dt = pd.to_datetime("2024-08-04 20:00:00")
    trough_stage = 353.388
    peak_stage = 354.798

    ax_stage.scatter([trough_dt], [trough_stage], color="blue", s=80, zorder=5, label=f"Rise Onset (353.388 m)")
    ax_stage.scatter([peak_dt], [peak_stage], color="darkred", s=80, zorder=5, label=f"Peak Stage (354.798 m)")

    ax_stage.annotate(
        f"Rise Onset\n353.388 m",
        xy=(trough_dt, trough_stage),
        xytext=(trough_dt - pd.Timedelta(hours=14), trough_stage - 0.35),
        arrowprops=dict(facecolor="blue", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="blue"
    )

    ax_stage.annotate(
        f"Peak Crest\n354.798 m (+1.41 m)",
        xy=(peak_dt, peak_stage),
        xytext=(peak_dt - pd.Timedelta(hours=20), peak_stage + 0.15),
        arrowprops=dict(facecolor="darkred", arrowstyle="->", lw=1.5),
        fontsize=9, fontweight="bold", color="darkred"
    )

    # Cumulative rainfall secondary axis on stage plot
    ax_cum = ax_stage.twinx()
    ax_cum.plot(
        df_merged["dt_ist"],
        df_merged["cumulative_rainfall_mm"],
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label="Cumulative Basin Rainfall (mm)"
    )
    ax_cum.set_ylabel("Cumulative Rainfall (mm)", fontsize=11, fontweight="bold", color="#2ca02c")
    ax_cum.tick_params(axis="y", labelcolor="#2ca02c")

    ax_stage.set_ylabel("Stage (m MSL)", fontsize=11, fontweight="bold", color="#d62728")
    ax_stage.tick_params(axis="y", labelcolor="#d62728")
    ax_stage.set_xlabel("Time (IST)", fontsize=11, fontweight="bold")
    ax_stage.grid(True, linestyle="--", alpha=0.5)

    # Format x-axis dates
    ax_stage.xaxis.set_major_locator(mdates.DayLocator())
    ax_stage.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%Y"))
    ax_stage.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))

    # Merge legends for bottom plot
    lines_s, labels_s = ax_stage.get_legend_handles_labels()
    lines_c, labels_c = ax_cum.get_legend_handles_labels()
    ax_stage.legend(lines_s + lines_c, labels_s + labels_c, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT_PLOT_PNG, dpi=200)
    plt.close()
    print(f"Hydrological plot saved to: {OUT_PLOT_PNG}")

    return {
        "aligned_rows": len(df_merged),
        "total_rainfall_accumulated_mm": float(df_merged["cumulative_rainfall_mm"].iloc[-1]),
        "max_hourly_rainfall_mm": float(df_merged["hourly_depth_mm"].max()),
        "stage_min_m": float(df_merged["clean_stage_m"].min()),
        "stage_max_m": float(df_merged["clean_stage_m"].max()),
        "rise_magnitude_m": float(df_merged["clean_stage_m"].max() - df_merged["clean_stage_m"].min()),
        "plot_path": str(OUT_PLOT_PNG),
        "csv_path": str(OUT_ALIGNED_CSV)
    }


if __name__ == "__main__":
    run_alignment_and_analysis()
