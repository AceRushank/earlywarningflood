"""
Audit of Godavari Telemetry in data/rwl_tel_hr_maharashtra_sw_006_2021_2025.csv
Focus: 2024-2025 stations overlapping INSAT-3DS operational period.
"""

import sys
import os
import csv
from collections import defaultdict
from datetime import datetime

TELEMETRY_CSV = r"c:\Users\aceru\floodsystem\data\rwl_tel_hr_maharashtra_sw_006_2021_2025.csv"

def parse_time(t_str):
    if not t_str:
        return None
    for fmt in ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(t_str, fmt)
        except ValueError:
            pass
    return None

def main():
    if not os.path.exists(TELEMETRY_CSV):
        print(f"Error: {TELEMETRY_CSV} not found")
        return

    print(f"Scanning {TELEMETRY_CSV} for Godavari stations...")
    
    # First pass: collect station summary
    stations = defaultdict(lambda: {
        'count': 0,
        'min_dt': None,
        'max_dt': None,
        'count_2024_2025': 0,
        'river': '',
        'basin': '',
        'district': '',
        'lat': '',
        'lon': '',
        'msl': '',
        'rl_zero': ''
    })

    total_rows = 0
    with open(TELEMETRY_CSV, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            basin = row.get('Basin', '').strip()
            river = row.get('River', '').strip()
            if 'Godavari' in basin or 'Godavari' in river:
                st = row.get('Station', '').strip()
                s_info = stations[st]
                s_info['count'] += 1
                if not s_info['river']:
                    s_info['river'] = river
                    s_info['basin'] = basin
                    s_info['district'] = row.get('District', '').strip()
                    s_info['lat'] = row.get('Latitude', '').strip()
                    s_info['lon'] = row.get('Longitude', '').strip()
                    s_info['msl'] = row.get('MeanSeaLevel', '').strip()
                    s_info['rl_zero'] = row.get('RL_of_zeroGauge', '').strip()
                
                t_str = row.get('Data Acquisition Time', '').strip()
                dt = parse_time(t_str)
                if dt:
                    if s_info['min_dt'] is None or dt < s_info['min_dt']:
                        s_info['min_dt'] = dt
                    if s_info['max_dt'] is None or dt > s_info['max_dt']:
                        s_info['max_dt'] = dt
                    if dt.year in (2024, 2025):
                        s_info['count_2024_2025'] += 1

    print(f"\nTotal rows scanned: {total_rows}")
    print(f"Found {len(stations)} Godavari stations.\n")
    print(f"{'Station':<30} {'District':<15} {'Total Rows':<10} {'2024-2025 Rows':<15} {'Min Date':<18} {'Max Date':<18}")
    print("-" * 110)
    for st, info in sorted(stations.items(), key=lambda x: x[1]['count_2024_2025'], reverse=True):
        min_s = str(info['min_dt']) if info['min_dt'] else "N/A"
        max_s = str(info['max_dt']) if info['max_dt'] else "N/A"
        print(f"{st:<30} {info['district']:<15} {info['count']:<10} {info['count_2024_2025']:<15} {min_s:<18} {max_s:<18}")

if __name__ == '__main__':
    main()
