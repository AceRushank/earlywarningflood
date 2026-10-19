"""
Detailed audit of Godavari candidate stations during 2024-2025.
Checks:
- Duplicate timestamps
- Gaps (missing intervals)
- Datum changes (RL_of_zeroGauge, MeanSeaLevel)
- Sensor errors (-99.999, out of range, stuck values)
- Water level dynamics / flood relevance (monsoon peaks, variance)
- Clean continuous windows
"""

import sys
import os
import csv
from collections import defaultdict
from datetime import datetime, timedelta

TELEMETRY_CSV = r"c:\Users\aceru\floodsystem\data\rwl_tel_hr_maharashtra_sw_006_2021_2025.csv"

CANDIDATES = [
    "Vishnupuri",
    "Babhali Barrage",
    "Upper Penganga (Isapur Dam)",
    "Majalgaon",
    "Gangapur Dam",
    "Darna Dam",
    "Raher",
    "Upper Manar Dam"
]

def parse_time(t_str):
    if not t_str:
        return None
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(t_str, fmt)
        except ValueError:
            pass
    return None

def analyze_station(station_name, rows):
    print(f"\n=======================================================")
    print(f"AUDITING STATION: {station_name}")
    print(f"Total raw records: {len(rows)}")

    # Check metadata consistency
    datums = set()
    msls = set()
    lats = set()
    lons = set()
    
    records = []
    duplicate_count = 0
    seen_times = set()

    for r in rows:
        t_str = r.get('Data Acquisition Time', '').strip()
        dt = parse_time(t_str)
        if not dt:
            continue
        
        datums.add(r.get('RL_of_zeroGauge', '').strip())
        msls.add(r.get('MeanSeaLevel', '').strip())
        lats.add(r.get('Latitude', '').strip())
        lons.add(r.get('Longitude', '').strip())

        val_str = r.get('River Water Level Telemetry Hourly (meter)', '').strip()
        try:
            val = float(val_str)
        except ValueError:
            val = None

        if dt in seen_times:
            duplicate_count += 1
        seen_times.add(dt)

        records.append((dt, val, r))

    records.sort(key=lambda x: x[0])
    
    print(f"Unique timestamps: {len(seen_times)}, Duplicates: {duplicate_count}")
    print(f"Latitudes: {lats}, Longitudes: {lons}")
    print(f"RL_of_zeroGauge values: {datums}")
    print(f"MeanSeaLevel values: {msls}")

    # Filter to 2024-2025
    rec_24_25 = [(dt, val) for dt, val, _ in records if dt.year in (2024, 2025)]
    print(f"Records in 2024-2025: {len(rec_24_25)}")
    
    if not rec_24_25:
        print("No 2024-2025 records found!")
        return

    # De-duplicate by keeping last observation for same timestamp
    dedup_dict = {}
    for dt, val in rec_24_25:
        dedup_dict[dt] = val
    sorted_dts = sorted(dedup_dict.keys())

    # Sensor errors: -99.999 or None or extreme
    error_values = [v for dt, v in rec_24_25 if v is None or v <= -90 or v > 2000]
    valid_records = [(dt, dedup_dict[dt]) for dt in sorted_dts if dedup_dict[dt] is not None and dedup_dict[dt] > -90 and dedup_dict[dt] <= 2000]
    
    print(f"Sensor error count (-99.999 / missing): {len(error_values)} ({len(error_values)/len(rec_24_25)*100:.2f}%)")
    if valid_records:
        vals = [v for _, v in valid_records]
        min_v = min(vals)
        max_v = max(vals)
        print(f"Valid water level range: {min_v:.3f} m to {max_v:.3f} m (Delta: {max_v - min_v:.3f} m)")
        
        # Check stuck values (e.g. identical for > 24 hours)
        stuck_runs = []
        cur_v = None
        cur_run = 0
        for _, v in valid_records:
            if v == cur_v:
                cur_run += 1
            else:
                if cur_run >= 48: # stuck for >= 48 consecutive readings
                    stuck_runs.append((cur_v, cur_run))
                cur_v = v
                cur_run = 1
        if cur_run >= 48:
            stuck_runs.append((cur_v, cur_run))
        if stuck_runs:
            print(f"Found {len(stuck_runs)} instances of stuck values (>= 48 consecutive readings): {stuck_runs[:3]}")

    # Gap analysis in 2024-2025
    gaps = []
    for i in range(1, len(sorted_dts)):
        prev_dt = sorted_dts[i-1]
        cur_dt = sorted_dts[i]
        diff_hrs = (cur_dt - prev_dt).total_seconds() / 3600.0
        if diff_hrs > 1.5: # missing > 1.5 hours
            gaps.append((prev_dt, cur_dt, diff_hrs))

    print(f"Total time gaps (>1.5 hr) in 2024-2025: {len(gaps)}")
    if gaps:
        large_gaps = [g for g in gaps if g[2] >= 24.0]
        print(f"Gaps >= 24 hours: {len(large_gaps)}")
        for g in sorted(gaps, key=lambda x: x[2], reverse=True)[:5]:
            print(f"  Large gap: {g[0]} to {g[1]} ({g[2]:.1f} hours / {g[2]/24:.1f} days)")

    # Find longest continuous window with valid data and no gap > 3 hours
    # Evaluate across 2024-2025
    windows = []
    if valid_records:
        cur_win_start = valid_records[0][0]
        cur_win_end = valid_records[0][0]
        cur_win_count = 1

        for i in range(1, len(valid_records)):
            prev_t, _ = valid_records[i-1]
            cur_t, _ = valid_records[i]
            diff = (cur_t - prev_t).total_seconds() / 3600.0
            if diff <= 3.0: # allow max 3h gap
                cur_win_end = cur_t
                cur_win_count += 1
            else:
                duration_days = (cur_win_end - cur_win_start).total_seconds() / 86400.0
                if duration_days >= 7.0: # at least 1 week
                    windows.append((cur_win_start, cur_win_end, duration_days, cur_win_count))
                cur_win_start = cur_t
                cur_win_end = cur_t
                cur_win_count = 1
        
        duration_days = (cur_win_end - cur_win_start).total_seconds() / 86400.0
        if duration_days >= 7.0:
            windows.append((cur_win_start, cur_win_end, duration_days, cur_win_count))

    print(f"\nTop Continuous Windows (allowing max 3-hour gap, duration >= 7 days):")
    windows.sort(key=lambda x: x[2], reverse=True)
    for w in windows[:8]:
        # Check monsoon overlap
        is_monsoon = any(m in (6, 7, 8, 9, 10) for m in (w[0].month, w[1].month))
        # Find peak level in window
        win_vals = [dedup_dict[dt] for dt in sorted_dts if w[0] <= dt <= w[1] and dedup_dict[dt] is not None and dedup_dict[dt] > -90]
        peak = max(win_vals) if win_vals else None
        min_w = min(win_vals) if win_vals else None
        var_w = (peak - min_w) if (peak and min_w) else 0
        print(f"  Window: {w[0]} to {w[1]} | {w[2]:.1f} days | {w[3]} obs | Range: [{min_w:.2f} - {peak:.2f} m, Delta: {var_w:.2f} m] | Monsoon: {is_monsoon}")

def main():
    print("Reading rows for candidate stations...")
    station_rows = defaultdict(list)
    with open(TELEMETRY_CSV, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for r in reader:
            st = r.get('Station', '').strip()
            if st in CANDIDATES:
                station_rows[st].append(r)
    
    for c in CANDIDATES:
        if c in station_rows:
            analyze_station(c, station_rows[c])
        else:
            print(f"Station {c} not found in dataset!")

if __name__ == '__main__':
    main()
