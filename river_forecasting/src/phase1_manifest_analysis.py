"""
Phase 1 manifest analysis - understand catalog gaps and timestamp coverage.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

df = pd.read_csv("river_forecasting/outputs/event1_rainfall_manifest.csv")
print("== MANIFEST SUMMARY ==")
print(f"Rows: {len(df)}")
print("Status counts:")
print(df["download_status"].value_counts().to_string())

ts_min = df["timestamp_utc"].min()
ts_max = df["timestamp_utc"].max()
print(f"\nTimestamp range: {ts_min} to {ts_max}")

errs = df[df["download_status"].str.startswith("ERROR", na=False)]
print(f"\n== ERROR ENTRIES ({len(errs)}) ==")
print(errs[["identifier","timestamp_utc","download_status"]].to_string())

# Expected timestamps: 2024-07-31T00:00:00Z to 2024-08-05T23:30:00Z, every 30 min
start_utc = datetime(2024, 7, 31, 0, 0, tzinfo=timezone.utc)
end_utc   = datetime(2024, 8, 5, 23, 30, tzinfo=timezone.utc)
expected_ts = []
t = start_utc
while t <= end_utc:
    expected_ts.append(t.strftime("%Y-%m-%dT%H:%M:%SZ"))
    t += timedelta(minutes=30)

print(f"\n== EXPECTED GRANULES (Jul31-Aug5 UTC) ==")
print(f"Total expected: {len(expected_ts)}")

manifest_ts = set(df["timestamp_utc"].tolist())
missing_from_manifest = [ts for ts in expected_ts if ts not in manifest_ts]
print(f"In manifest: {len(manifest_ts)}")
print(f"Missing from manifest: {len(missing_from_manifest)}")
if missing_from_manifest:
    print("First 20 missing from manifest:")
    for ts in missing_from_manifest[:20]:
        print(f"  {ts}")
    if len(missing_from_manifest) > 20:
        print(f"  ... and {len(missing_from_manifest) - 20} more")

# What's on disk
RAW_DIR = Path("river_forecasting/data/mosdac_raw/3SIMG_L2B_HEM")
disk_files = {fp.name for fp in RAW_DIR.rglob("*.h5")}
print(f"\n== DISK vs MANIFEST ==")
print(f"Disk .h5 files (all): {len(disk_files)}")

# Map manifest identifiers to disk
manifest_ids = set(df["identifier"].tolist())
on_disk_in_manifest = disk_files & manifest_ids
not_in_manifest_but_on_disk = disk_files - manifest_ids
in_manifest_not_on_disk = manifest_ids - disk_files

print(f"Manifest identifiers matching disk: {len(on_disk_in_manifest)}")
print(f"On disk but NOT in manifest: {len(not_in_manifest_but_on_disk)}")
print(f"In manifest but NOT on disk: {len(in_manifest_not_on_disk)}")
if in_manifest_not_on_disk:
    print("In manifest but missing from disk:")
    for f in sorted(in_manifest_not_on_disk):
        print(f"  {f}")
