"""
audit_event_dataset.py

Audits the derived 500m event-based Mumbai flood dataset:
- Data completeness & schema validation
- Duplicate & missing value checks
- Label distribution overall & per event
- Sanity validation on historical hotspots (Kurla, Hindmata, Chembur, Malad, Byculla)
- Visual distribution plot export -> data/events/event_flood_distribution.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EVENTS_DIR = DATA / "events"
DATASET_CSV = EVENTS_DIR / "mumbai_500m_events_dataset.csv"
CHART_PNG = EVENTS_DIR / "event_flood_distribution.png"

def run_audit():
    print("=" * 80)
    print("DATASET QUALITY AUDIT & SANITY VALIDATION")
    print("=" * 80)

    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)

    n_rows = len(df)
    n_events = df["event_id"].nunique()
    n_cells = df["cell_id"].nunique()
    events = df["event_id"].unique()

    print(f"1. DATASET DIMENSIONS:")
    print(f"   Total rows         : {n_rows}")
    print(f"   Unique grid cells  : {n_cells}")
    print(f"   Unique events      : {n_events}")
    print(f"   Grid cell size     : 500 m x 500 m (0.25 km2)")
    print(f"   Total spatial area : {n_cells * 0.25:.2f} km2 (Greater Mumbai land area)")

    # Missing & duplicate check
    missing = df.isna().sum()
    dups = df.duplicated(subset=["cell_id", "event_id"]).sum()
    print(f"\n2. DATA INTEGRITY:")
    print(f"   Duplicate (cell_id, event_id) rows: {dups}")
    print(f"   Missing values across all columns : {missing.sum()}")
    if missing.sum() > 0:
        print(missing[missing > 0])

    # Class distribution
    pos = (df["observed_flooded"] == 1).sum()
    neg = (df["observed_flooded"] == 0).sum()
    pos_pct = pos / n_rows * 100.0
    neg_pct = neg / n_rows * 100.0
    print(f"\n3. TARGET CLASS DISTRIBUTION (observed_flooded):")
    print(f"   Observed Flooded (1): {pos:,} ({pos_pct:.2f}%)")
    print(f"   Non-Flooded      (0): {neg:,} ({neg_pct:.2f}%)")

    # Per-event breakdown
    print(f"\n4. PER-EVENT BREAKDOWN:")
    print(f"{'Event ID':<18} | {'Event Date':<10} | {'Rain 24h':<8} | {'Peak Hr':<7} | {'Flooded Cells':<13} | {'Flooded %':<9} | Status")
    print("-" * 80)

    per_event = []
    for eid in events:
        sub = df[df["event_id"] == eid]
        e_date = sub["event_date"].iloc[0]
        r24 = sub["rainfall_24h_mm"].iloc[0]
        p_hr = sub["peak_hourly_rainfall_mm"].iloc[0]
        f_count = (sub["observed_flooded"] == 1).sum()
        f_pct = f_count / len(sub) * 100.0
        status = "DRY_CONTROL" if r24 == 0.0 else "FLOOD_EVENT"
        print(f"{eid:<18} | {e_date:<10} | {r24:<8.1f} | {p_hr:<7.1f} | {f_count:<13} | {f_pct:<8.2f}% | {status}")
        per_event.append({
            "event_id": eid,
            "event_date": e_date,
            "r24": r24,
            "flooded_cells": f_count,
            "flooded_pct": f_pct
        })
    print("-" * 80)

    # 5. Sanity Checks on Known Flood-Prone Hotspots
    print(f"\n5. SANITY VALIDATION ON KNOWN HISTORICAL HOTSPOTS:")
    hotspots = [
        ("Kurla (Ward L)", "L", 274500, 2108500),
        ("Hindmata / Dadar (Ward F/N)", "F/N", 273000, 2104000),
        ("Byculla (Ward E)", "E", 272500, 2099500),
        ("Chembur (Ward M/W)", "M/W", 277000, 2106000),
        ("Malad Subway (Ward P/N)", "P/N", 271500, 2121500),
        ("Colaba Baseline (Ward A)", "A", 271500, 2092500),
    ]

    print(f"{'Location':<26} | {'Ward':<5} | {'EV_2019 (375mm)':<15} | {'EV_2018 (115mm)':<15} | {'Dry Control (0mm)'}")
    print("-" * 80)

    for name, wc, target_x, target_y in hotspots:
        ward_cells = df[df["bmc_ward_code"] == wc]
        if len(ward_cells) == 0:
            continue
        # Get flooded state for prime deluge vs moderate vs dry
        ev_2019_flooded = ward_cells[ward_cells["event_id"] == "EV_2019_07_02"]["observed_flooded"].mean() * 100.0
        ev_2018_flooded = ward_cells[ward_cells["event_id"] == "EV_2018_07_19"]["observed_flooded"].mean() * 100.0
        ctrl_flooded = ward_cells[ward_cells["event_id"] == "CTRL_2024_03_12"]["observed_flooded"].mean() * 100.0
        print(f"{name:<26} | {wc:<5} | {ev_2019_flooded:>13.1f}% | {ev_2018_flooded:>13.1f}% | {ctrl_flooded:>15.1f}%")
    print("-" * 80)

    # 6. Physical Parameter Summary
    print(f"\n6. PHYSICAL PREDICTOR SUMMARY:")
    num_cols = [
        "rainfall_24h_mm", "antecedent_rainfall_3d_mm", "antecedent_rainfall_7d_mm", "peak_hourly_rainfall_mm",
        "elevation_mean_m", "slope_mean_deg", "builtup_fraction", "dist_to_drainage_m", "flood_fraction"
    ]
    summary = df[num_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    summary.columns = ["Mean", "StdDev", "Min", "Median", "Max"]
    print(summary.to_string())

    # 7. Visual Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of flooded cells per event
    pe_df = pd.DataFrame(per_event)
    axes[0].bar(pe_df["event_id"], pe_df["flooded_cells"], color="#2b5c8f")
    axes[0].set_title("Observed Flooded Grid Cells per Event (N=2,019)")
    axes[0].set_ylabel("Count of Flooded 500m Cells (Y=1)")
    axes[0].set_xticklabels(pe_df["event_id"], rotation=45, ha="right")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Scatter plot: 24h rain vs total flooded cells
    axes[1].scatter(pe_df["r24"], pe_df["flooded_pct"], color="#c0392b", s=70, zorder=3)
    axes[1].set_title("Rainfall Forcing vs. Observed Inundation Extent")
    axes[1].set_xlabel("Event 24-Hour Rainfall (mm)")
    axes[1].set_ylabel("Percentage of Mumbai Land Flooded (%)")
    axes[1].set_ylim(-2, 100)
    axes[1].grid(linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(CHART_PNG, dpi=200)
    plt.close()
    print(f"\nSaved audit summary visualization -> {CHART_PNG}")
    print("=" * 80)

if __name__ == "__main__":
    run_audit()
