"""
Phase 1 helper: identify all non-valid HDF5 files and decode time datasets.
Reports corrupt files, physics failures, and timestamp issues.
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
import re

RAW_DIR = Path("river_forecasting/data/mosdac_raw/3SIMG_L2B_HEM")

FNAME_RE = re.compile(
    r"3SIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_L2B_HEM_V01R00\.h5$"
)
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

def parse_fname_utc(name):
    m = FNAME_RE.match(name)
    if not m:
        return None
    day, mon, year, hhmm = int(m.group(1)), MONTH_MAP[m.group(2)], int(m.group(3)), m.group(4)
    hh, mm = int(hhmm[:2]), int(hhmm[2:])
    return datetime(year, day, hhmm, hh, mm, tzinfo=timezone.utc)

def parse_fname_utc(name):
    m = FNAME_RE.match(name)
    if not m:
        return None
    day = int(m.group(1))
    mon = MONTH_MAP[m.group(2)]
    year = int(m.group(3))
    hhmm = m.group(4)
    hh, mm = int(hhmm[:2]), int(hhmm[2:])
    return datetime(year, mon, day, hh, mm, tzinfo=timezone.utc)

def decode_time(time_ds):
    """Decode HDF5 time dataset to UTC datetime."""
    val = time_ds[()]
    if isinstance(val, np.ndarray):
        val = val.flat[0]
    # Try seconds since epoch (float or int)
    # Common: seconds since 1970-01-01, minutes since epoch, or string
    units_attr = time_ds.attrs.get("units", b"")
    if isinstance(units_attr, bytes):
        units_attr = units_attr.decode("utf-8", errors="replace")
    
    # Print raw info
    print(f"  time val={val!r}, dtype={time_ds.dtype}, units={units_attr!r}")
    
    # If units contains "seconds since", parse accordingly
    if "seconds since" in units_attr.lower():
        ref_str = units_attr.lower().split("seconds since")[-1].strip()
        try:
            ref_dt = datetime.fromisoformat(ref_str.replace(" ", "T"))
            ref_dt = ref_dt.replace(tzinfo=timezone.utc)
            return ref_dt + timedelta(seconds=float(val))
        except Exception as e:
            print(f"  seconds-since parse failed: {e}")
    
    # If units contains "minutes since"
    if "minutes since" in units_attr.lower():
        ref_str = units_attr.lower().split("minutes since")[-1].strip()
        try:
            ref_dt = datetime.fromisoformat(ref_str.replace(" ", "T"))
            ref_dt = ref_dt.replace(tzinfo=timezone.utc)
            return ref_dt + timedelta(minutes=float(val))
        except Exception as e:
            print(f"  minutes-since parse failed: {e}")
    
    # Fallback: assume seconds since 1970-01-01
    try:
        return datetime.fromtimestamp(float(val), tz=timezone.utc)
    except Exception:
        return None

records = []
part_files = list(RAW_DIR.rglob("*.part"))
print(f"=== .part files: {len(part_files)} ===")
for p in part_files:
    print(f"  {p.relative_to(RAW_DIR)}  ({p.stat().st_size} bytes)")

all_h5 = sorted(RAW_DIR.rglob("*.h5"))
print(f"\n=== Total .h5 files: {len(all_h5)} ===\n")

issues = []
ok_count = 0

for fp in all_h5:
    sz = fp.stat().st_size
    fname = fp.name
    fn_ts = parse_fname_utc(fname)
    
    rec = {"file": fname, "size_mb": round(sz/1e6, 2), "fn_utc": fn_ts, "status": "UNPROCESSED", "notes": ""}
    
    if sz == 0:
        rec["status"] = "ZERO_BYTE"
        issues.append(rec)
        continue
    
    try:
        with h5py.File(fp, "r") as hf:
            keys = set(hf.keys())
            req = {"HEM", "Latitude", "Longitude", "time", "GeoX", "GeoY"}
            missing = req - keys
            if missing:
                rec["status"] = "MISSING_DATASETS"
                rec["notes"] = f"Missing: {missing}"
                issues.append(rec)
                continue
            
            hem = hf["HEM"]
            rec["shape"] = str(hem.shape)
            rec["dtype"] = str(hem.dtype)
            
            # Shape check
            if hem.shape != (1, 2816, 2805):
                rec["status"] = "BAD_SHAPE"
                rec["notes"] = f"Shape={hem.shape}"
                issues.append(rec)
                continue
            
            # Time decode
            time_internal = decode_time(hf["time"])
            rec["time_internal_utc"] = time_internal
            
            # Timestamp match
            if fn_ts and time_internal:
                diff_min = abs((time_internal - fn_ts).total_seconds()) / 60
                rec["ts_diff_min"] = round(diff_min, 1)
                if diff_min > 60:  # More than 1 hour off = probable mismatch
                    rec["status"] = "TS_MISMATCH_MAJOR"
                    rec["notes"] = f"filename={fn_ts}, internal={time_internal}, diff={diff_min:.1f}min"
                    issues.append(rec)
                    continue
                elif diff_min > 30:
                    rec["notes"] = f"ts_diff={diff_min:.1f}min (warn)"
            
            # Rainfall stats
            fill = -999.0
            data = hem[0]
            valid_mask = (data != fill) & (data >= 0.0)
            n_valid = int(np.count_nonzero(valid_mask))
            
            if n_valid > 0:
                vals = data[valid_mask]
                mean_r = float(np.mean(vals))
                max_r = float(np.max(vals))
            else:
                mean_r = 0.0
                max_r = 0.0
            
            rec["n_valid_px"] = n_valid
            rec["mean_mm_hr"] = round(mean_r, 4)
            rec["max_mm_hr"] = round(max_r, 4)
            
            # Physics check
            if max_r > 500.0:
                rec["status"] = "PHYSICS_FAIL_MAX"
                rec["notes"] = f"max={max_r}"
                issues.append(rec)
                continue
            
            if mean_r > 100.0:
                rec["status"] = "PHYSICS_FAIL_MEAN"
                rec["notes"] = f"mean={mean_r}"
                issues.append(rec)
                continue
            
            rec["status"] = "OK"
            ok_count += 1
            
    except Exception as e:
        rec["status"] = f"CORRUPT"
        rec["notes"] = str(e)[:120]
        issues.append(rec)
        continue
    
    records.append(rec)

print(f"\n=== RESULTS ===")
print(f"OK files: {ok_count}")
print(f"Problem files: {len(issues)}")
print()
if issues:
    print("=== PROBLEM FILES ===")
    for r in issues:
        print(f"  [{r['status']}] {r['file']}  size={r['size_mb']}MB  notes={r['notes']}")
