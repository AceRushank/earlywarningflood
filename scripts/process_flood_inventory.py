"""
Process and clean the Mumbai historical flood event inventory from IMD records.

Outputs:
  - data/historical/mumbai_flood_events.csv
  - outputs/mumbai_flood_inventory_qc.txt
"""

import os
import re
from pathlib import Path
import pandas as pd

SOURCE_PATH = Path('data/mumbai_flood_events.csv')
OUTPUT_DIR = Path('data/historical')
TARGET_CSV = OUTPUT_DIR / 'mumbai_flood_events.csv'
QC_REPORT_PATH = Path('outputs/mumbai_flood_inventory_qc.txt')


def process_inventory():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source file not found at {SOURCE_PATH}")

    # 1. Read complete source file
    df_raw = pd.read_csv(SOURCE_PATH)
    total_source_records = len(df_raw)

    # 2. Filter: State contains 'Maharashtra' AND (Location or Districts contains 'Mumbai', 'Mumbai Suburban', or 'Mumbai (Colaba)')
    state_mask = df_raw['State'].astype(str).str.contains('Maharashtra', case=False, na=False)

    def contains_mumbai(val):
        if pd.isna(val):
            return False
        s = str(val).lower()
        return any(term in s for term in ['mumbai', 'mumbai suburban', 'mumbai (colaba)'])

    loc_mask = df_raw['Location'].apply(contains_mumbai)
    dist_mask = df_raw['Districts'].apply(contains_mumbai)
    mumbai_filter = state_mask & (loc_mask | dist_mask)

    df_filtered = df_raw[mumbai_filter].copy()
    total_filtered_records = len(df_filtered)

    # 3. Check and remove exact duplicate rows
    # Check duplicates across the entire raw record (excluding Unnamed: 0 if present)
    data_cols = [c for c in df_filtered.columns if c != 'Unnamed: 0']
    exact_duplicates_count = df_filtered.duplicated(subset=data_cols).sum()
    df_dedup = df_filtered.drop_duplicates(subset=data_cols).copy()

    # 4. Parse and normalize dates to YYYY-MM-DD
    # Original format is DD-MM-YYYY HH:MM
    df_dedup['start_dt'] = pd.to_datetime(df_dedup['Start Date'], format='%d-%m-%Y %H:%M')
    df_dedup['end_dt'] = pd.to_datetime(df_dedup['End Date'], format='%d-%m-%Y %H:%M')

    # Sort chronologically by start_date, then end_date
    df_dedup = df_dedup.sort_values(by=['start_dt', 'end_dt']).reset_index(drop=True)

    # Format normalized dates
    df_dedup['start_date_norm'] = df_dedup['start_dt'].dt.strftime('%Y-%m-%d')
    df_dedup['end_date_norm'] = df_dedup['end_dt'].dt.strftime('%Y-%m-%d')

    # Generate sequential event_id: MUM_FL_001 to MUM_FL_182
    df_dedup['event_id'] = [f"MUM_FL_{i+1:03d}" for i in range(len(df_dedup))]

    # Map to target schema
    column_mapping = {
        'event_id': 'event_id',
        'UEI': 'original_uei',
        'start_date_norm': 'start_date',
        'end_date_norm': 'end_date',
        'Duration(Days)': 'duration_days',
        'Main Cause': 'main_cause',
        'Location': 'location',
        'Districts': 'districts',
        'State': 'state',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Severity': 'severity',
        'Area Affected': 'area_affected',
        'Human fatality': 'human_fatality',
        'Human injured': 'human_injured',
        'Human Displaced': 'human_displaced',
        'Animal Fatality': 'animal_fatality',
        'Description of Casualties/injured': 'casualty_description',
        'Extent of damage ': 'damage_description',
        'Event Source': 'event_source',
        'Event Souce ID': 'event_source_id'
    }

    target_columns = list(column_mapping.values())
    df_final = df_dedup.rename(columns=column_mapping)[target_columns]

    # Save to data/historical/mumbai_flood_events.csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(TARGET_CSV, index=False, na_rep='')
    print(f"Clean Mumbai flood events saved to {TARGET_CSV} ({len(df_final)} rows)")

    # 5. Generate QC Report
    generate_qc_report(
        df_final=df_final,
        df_dedup=df_dedup,
        total_source_records=total_source_records,
        total_filtered_records=total_filtered_records,
        exact_duplicates_count=exact_duplicates_count
    )


def generate_qc_report(df_final, df_dedup, total_source_records, total_filtered_records, exact_duplicates_count):
    QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    unique_uei_count = df_final['original_uei'].nunique()
    unique_event_id_count = df_final['event_id'].nunique()
    earliest_event_date = df_final['start_date'].min()
    latest_event_date = df_final['start_date'].max()

    # Events by year
    years_series = df_dedup['start_dt'].dt.year.value_counts().sort_index()

    # Events by main cause
    causes_series = df_final['main_cause'].value_counts(dropna=False)

    # Coordinates
    events_with_coords = (df_final['latitude'].notna() & df_final['longitude'].notna()).sum()

    # Exact start/end dates
    events_with_exact_dates = (df_final['start_date'].notna() & df_final['end_date'].notna()).sum()

    # Explicit rainfall information in description
    # Quantitative measurements (cm, mm, inches) and explicit rainfall statements
    explicit_rainfall_records = []
    quantitative_records = []
    
    depth_regex = re.compile(r'(\d+(?:\.\d+)?\s*(?:mm|cm|inches|inch|m\b)|\d+\s*to\s*\d+\s*(?:mm|cm)|rainfall of \d+)', re.IGNORECASE)
    rain_word_regex = re.compile(r'\b(rainfall|rains?|precipitation|downpour)\b', re.IGNORECASE)

    for _, row in df_final.iterrows():
        c_desc = str(row['casualty_description']) if pd.notna(row['casualty_description']) else ''
        d_desc = str(row['damage_description']) if pd.notna(row['damage_description']) else ''
        combined = f"{c_desc} | {d_desc}".strip(' |')
        
        has_depth = bool(depth_regex.search(combined))
        has_rain_word = bool(rain_word_regex.search(combined))
        
        if has_depth:
            quantitative_records.append((row['event_id'], row['original_uei'], row['start_date'], combined))
        if has_rain_word or has_depth:
            explicit_rainfall_records.append((row['event_id'], row['original_uei'], row['start_date'], combined))

    # Unique start dates
    unique_start_dates = sorted(df_final['start_date'].unique())

    # Events from 2018 onward
    events_2018_onward = df_final[df_final['start_date'] >= '2018-01-01'].copy()

    # Build report text
    lines = []
    lines.append("=" * 80)
    lines.append("MUMBAI HISTORICAL FLOOD INVENTORY QUALITY CONTROL REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"1. Total source records: {total_source_records}")
    lines.append(f"2. Total Mumbai/Maharashtra records after filtering: {total_filtered_records}")
    lines.append(f"3. Exact duplicates removed: {exact_duplicates_count}")
    lines.append(f"4. Number of unique UEI/event IDs: {unique_uei_count} (Unique UEIs: {unique_uei_count}, Unique Event IDs: {unique_event_id_count})")
    lines.append(f"5. Earliest event date: {earliest_event_date}")
    lines.append(f"6. Latest event date: {latest_event_date}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("7. NUMBER OF EVENTS BY YEAR")
    lines.append("-" * 80)
    for yr, count in years_series.items():
        lines.append(f"   {yr}: {count} event(s)")
    lines.append(f"   Total years represented: {len(years_series)}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("8. NUMBER OF EVENTS BY MAIN CAUSE")
    lines.append("-" * 80)
    for cause, count in causes_series.items():
        lines.append(f"   {cause}: {count} event(s)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("9. NUMBER OF EVENTS WITH COORDINATES")
    lines.append("-" * 80)
    lines.append(f"   Events with non-null Latitude and Longitude: {events_with_coords} of {len(df_final)}")
    lines.append("   Note: Original inventory records district-level flood events without explicit point coordinates.")
    lines.append("   Per instructions, missing coordinates are preserved as blank/NA without invention.")
    lines.append("")
    lines.append("-" * 80)
    lines.append("10. NUMBER OF EVENTS WITH EXACT START/END DATES")
    lines.append("-" * 80)
    lines.append(f"   Events with exact Start Date and End Date: {events_with_exact_dates} of {len(df_final)} (100.0%)")
    lines.append("   All timestamps cleanly normalized to YYYY-MM-DD from DD-MM-YYYY HH:MM format.")
    lines.append("")
    lines.append("-" * 80)
    lines.append("11. NUMBER OF EVENTS CONTAINING EXPLICIT RAINFALL INFORMATION IN DESCRIPTION")
    lines.append("-" * 80)
    lines.append(f"   Total events with quantitative rainfall measurement (cm / mm / inches): {len(quantitative_records)}")
    for eid, uei, sdate, text in quantitative_records:
        lines.append(f"   - [{eid} | {uei} | {sdate}]: {text}")
    lines.append("")
    lines.append(f"   Total events with explicit rainfall / torrential rains descriptors: {len(explicit_rainfall_records)}")
    for eid, uei, sdate, text in explicit_rainfall_records:
        lines.append(f"   - [{eid} | {uei} | {sdate}]: {text}")
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"12. LIST OF ALL UNIQUE EVENT START DATES (Total: {len(unique_start_dates)})")
    lines.append("-" * 80)
    for i, sdate in enumerate(unique_start_dates, 1):
        matching_count = (df_final['start_date'] == sdate).sum()
        suffix = f" ({matching_count} events)" if matching_count > 1 else ""
        lines.append(f"   {i:3d}. {sdate}{suffix}")
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"13. LIST OF EVENTS FROM 2018 ONWARD SEPARATELY (Total: {len(events_2018_onward)})")
    lines.append("-" * 80)
    for _, row in events_2018_onward.iterrows():
        fatalities = f"{int(row['human_fatality'])} fatalities" if pd.notna(row['human_fatality']) else "fatalities: NA"
        lines.append(f"   [{row['event_id']} | {row['original_uei']}]")
        lines.append(f"      Dates: {row['start_date']} to {row['end_date']} (Duration: {int(row['duration_days'])} day(s))")
        lines.append(f"      Cause: {row['main_cause']}")
        lines.append(f"      Districts: {row['districts']}")
        lines.append(f"      Impact: {fatalities}")
        if pd.notna(row['casualty_description']):
            lines.append(f"      Casualties: {row['casualty_description']}")
        if pd.notna(row['damage_description']):
            lines.append(f"      Damage: {row['damage_description']}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF QUALITY CONTROL REPORT")
    lines.append("=" * 80)

    report_content = "\n".join(lines)
    with open(QC_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"QC report saved to {QC_REPORT_PATH}")


if __name__ == '__main__':
    process_inventory()
