"""
river_forecasting/src/phase2_resume_download.py
================================================
Phase 2: Resume MOSDAC ingestion for Event 1 (Jul 31 – Aug 5, 2024).

Strategy:
  1. Re-query the MOSDAC catalog for the full Jul31–Aug5 window to build a
     complete list of V01R00 granules (the authoritative source of truth).
  2. Cross-reference the catalog against disk to identify:
     - Files already downloaded and HDF5-valid → skip.
     - Files missing from disk → download.
     - Files corrupt/truncated on disk → re-download.
     - Known catalog gaps (MOSDAC never published that slot) → mark as CATALOG_GAP.
  3. Download one file at a time (MOSDAC enforces single active connection per token).
     Atomic rename via .part files.
  4. Write a new manifest covering ALL 288 expected half-hourly slots (Jul31–Aug5),
     with one row per expected slot regardless of whether the file is available.
  5. Preserve the original manifest as backup before overwriting.

Usage:
  python river_forecasting/src/phase2_resume_download.py

Flags (optional):
  --dry-run          Print plan without downloading.
  --limit N          Stop after N downloads (for testing).
  --recheck-corrupt  Re-validate all existing files before deciding to skip.
"""

import sys
import os
import re
import time
import json
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

import h5py
import numpy as np
import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.mosdac_retriever import (
    search_valid_records,
    authenticate,
    logout,
    DOWNLOAD_URL,
)
from river_forecasting.src.catchment_extractor import CatchmentRainfallExtractor

RAW_DIR      = REPO_ROOT / "river_forecasting" / "data" / "mosdac_raw" / "3SIMG_L2B_HEM"
POLYGON_PATH = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "vishnupuri_upstream_catchment.geojson"
MANIFEST_CSV = REPO_ROOT / "river_forecasting" / "outputs" / "event1_rainfall_manifest.csv"
BACKUP_DIR   = REPO_ROOT / "river_forecasting" / "outputs" / "backup_phase1"
LOG_PATH     = REPO_ROOT / "river_forecasting" / "outputs" / "phase2_download.log"

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_START = datetime(2024, 7, 31, 0, 0, tzinfo=timezone.utc)
WINDOW_END   = datetime(2024, 8, 5, 23, 30, tzinfo=timezone.utc)
DATASET_ID   = "3SIMG_L2B_HEM"
TARGET_VER   = "V01R00"

# Physics thresholds (full-disk INSAT HEM — max can reach 300 mm/hr in extreme cells)
MAX_PHYS_MEAN = 100.0   # mm/hr (mean over full disk)
MAX_PHYS_MAX  = 500.0   # mm/hr (single pixel)

MONTH_MAP = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
             7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}

FNAME_RE = re.compile(
    r"3SIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_L2B_HEM_V01R00\.h5$"
)
MONTH_RMAP = {v: k for k, v in MONTH_MAP.items()}


# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────
def ts_to_filename(dt: datetime) -> str:
    """Convert UTC datetime to expected V01R00 filename."""
    day  = dt.strftime("%d")
    mon  = MONTH_MAP[dt.month]
    year = dt.strftime("%Y")
    hhmm = dt.strftime("%H%M")
    return f"3SIMG_{day}{mon}{year}_{hhmm}_L2B_HEM_V01R00.h5"


def ts_to_folder(dt: datetime) -> Path:
    """Return the raw-data subfolder for a given UTC datetime."""
    return RAW_DIR / dt.strftime("%Y") / f"{dt.strftime('%d')}{MONTH_MAP[dt.month]}"


def expected_slots() -> List[datetime]:
    """All expected UTC half-hourly timestamps for the window."""
    slots = []
    t = WINDOW_START
    while t <= WINDOW_END:
        slots.append(t)
        t += timedelta(minutes=30)
    return slots


def validate_hdf5(path: Path) -> Dict[str, Any]:
    """
    Open an HDF5 file and validate structure, timestamps, and rainfall values.
    Returns dict with keys: valid (bool), reason (str), internal_utc (datetime|None).
    """
    result = {"valid": False, "reason": "unknown", "internal_utc": None,
              "mean_mm_hr": None, "max_mm_hr": None}
    try:
        if not path.exists() or path.stat().st_size == 0:
            result["reason"] = "FILE_MISSING_OR_ZERO"
            return result

        with h5py.File(path, "r") as hf:
            req = {"HEM", "Latitude", "Longitude", "time", "GeoX", "GeoY"}
            missing = req - set(hf.keys())
            if missing:
                result["reason"] = f"MISSING_DATASETS:{missing}"
                return result

            hem = hf["HEM"]
            if hem.shape != (1, 2816, 2805) or hem.dtype != np.float32:
                result["reason"] = f"BAD_SHAPE:{hem.shape}:{hem.dtype}"
                return result

            # Decode time
            time_ds = hf["time"]
            val = time_ds[()]
            if isinstance(val, np.ndarray):
                val = val.flat[0]
            units = time_ds.attrs.get("units", b"")
            if isinstance(units, bytes):
                units = units.decode("utf-8", errors="replace")
            ref_epoch = datetime(2000, 1, 1, 0, 0, tzinfo=timezone.utc)
            if "minutes since" in units.lower():
                internal_utc = ref_epoch + timedelta(minutes=float(val))
            elif "seconds since" in units.lower():
                internal_utc = ref_epoch + timedelta(seconds=float(val))
            else:
                internal_utc = datetime.fromtimestamp(float(val), tz=timezone.utc)
            result["internal_utc"] = internal_utc

            # Physics check
            fill = -999.0
            data = hem[0]
            valid_mask = (data != fill) & (data >= 0.0)
            n = int(np.count_nonzero(valid_mask))
            if n > 0:
                vals = data[valid_mask]
                mean_r = float(np.mean(vals))
                max_r  = float(np.max(vals))
            else:
                mean_r = 0.0
                max_r  = 0.0
            result["mean_mm_hr"] = round(mean_r, 4)
            result["max_mm_hr"]  = round(max_r,  4)

            if mean_r > MAX_PHYS_MEAN or max_r > MAX_PHYS_MAX:
                result["reason"] = f"PHYSICS_FAIL:mean={mean_r:.2f},max={max_r:.2f}"
                return result

        result["valid"] = True
        result["reason"] = "OK"
        return result

    except Exception as e:
        result["reason"] = f"HDF5_CORRUPT:{str(e)[:120]}"
        return result


def download_one(rec_id: str, identifier: str, target_path: Path,
                 token: str, timeout=(15, 240), max_retries=5) -> Dict[str, Any]:
    """
    Download a single granule using the MOSDAC download API.
    Returns dict: status (str), file_size_bytes (int), error (str|None).
    """
    part_path = target_path.with_suffix(target_path.suffix + ".part")
    folder = target_path.parent
    folder.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        headers = {"Authorization": f"Bearer {token}"}
        params  = {"id": rec_id}
        try:
            resp = requests.get(DOWNLOAD_URL, headers=headers, params=params,
                                stream=True, timeout=timeout)

            if resp.status_code == 429:
                wait = 30
                log.warning(f"  Rate limited (429). Waiting {wait}s before retry {attempt}/{max_retries}...")
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                log.warning(f"  Token expired (401). Cannot refresh mid-download.")
                return {"status": "AUTH_EXPIRED", "file_size_bytes": 0, "error": "401"}

            if resp.status_code == 404:
                return {"status": "SERVER_NOT_FOUND", "file_size_bytes": 0, "error": "404"}

            if resp.status_code == 200:
                content_disp = resp.headers.get("Content-Disposition", "")
                if "filename=" not in content_disp:
                    return {"status": "SERVER_FILE_NOT_AVAILABLE", "file_size_bytes": 0,
                            "error": "no Content-Disposition filename"}

                # Stream to .part file
                bytes_written = 0
                with open(part_path, "wb") as pf:
                    for chunk in resp.iter_content(chunk_size=1_048_576):
                        if chunk:
                            pf.write(chunk)
                            bytes_written += len(chunk)

                # Atomic rename — delete corrupt/existing target first on Windows
                if part_path.exists():
                    if target_path.exists():
                        target_path.unlink()
                    part_path.rename(target_path)

                sz = target_path.stat().st_size if target_path.exists() else 0
                return {"status": "DOWNLOADED", "file_size_bytes": sz, "error": None}

            else:
                log.warning(f"  HTTP {resp.status_code} on attempt {attempt}")
                time.sleep(5 * attempt)

        except requests.exceptions.ChunkedEncodingError as e:
            log.warning(f"  ChunkedEncoding error (attempt {attempt}): {e}")
            if part_path.exists():
                part_path.unlink(missing_ok=True)
            time.sleep(10 * attempt)

        except requests.exceptions.ReadTimeout as e:
            log.warning(f"  ReadTimeout (attempt {attempt}): {e}")
            if part_path.exists():
                part_path.unlink(missing_ok=True)
            time.sleep(15 * attempt)

        except Exception as e:
            log.warning(f"  Unexpected error (attempt {attempt}): {e}")
            if part_path.exists():
                part_path.unlink(missing_ok=True)
            time.sleep(10)

    return {"status": f"FAILED_AFTER_{max_retries}_RETRIES", "file_size_bytes": 0,
            "error": "max retries exceeded"}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recheck-corrupt", action="store_true",
                        help="Re-validate all existing .h5 files before deciding to skip")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("PHASE 2: MOSDAC Ingestion Resume — Event 1 (Jul 31–Aug 5, 2024)")
    log.info("=" * 70)

    # ── Step 1: Backup original manifest ──────────────────────────────────────
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_CSV.exists():
        backup_path = BACKUP_DIR / f"event1_rainfall_manifest_pre_phase2_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
        shutil.copy2(MANIFEST_CSV, backup_path)
        log.info(f"Original manifest backed up to: {backup_path}")

    # ── Step 2: Delete orphaned .part files ───────────────────────────────────
    part_files = list(RAW_DIR.rglob("*.part"))
    for pf in part_files:
        log.info(f"Removing orphaned .part file: {pf.name}  ({pf.stat().st_size} bytes)")
        if not args.dry_run:
            pf.unlink(missing_ok=True)

    # ── Step 3: Query full catalog ────────────────────────────────────────────
    log.info(f"Querying MOSDAC catalog for {DATASET_ID} V01R00, "
             f"2024-07-31 to 2024-08-05...")
    catalog_records = search_valid_records(
        dataset_id=DATASET_ID,
        start_time="2024-07-31",
        end_time="2024-08-05",
        target_version=TARGET_VER,
        page_size=100,
    )
    log.info(f"Catalog returned {len(catalog_records)} V01R00 records.")

    if not catalog_records:
        log.error("No catalog records found. Check MOSDAC API connectivity.")
        sys.exit(1)

    # Build catalog lookup: identifier → record
    catalog_by_id = {r["identifier"]: r for r in catalog_records}

    # ── Step 4: Build the full 288-slot work plan ─────────────────────────────
    all_slots = expected_slots()
    log.info(f"Expected slots (every 30min, Jul31–Aug5): {len(all_slots)}")

    # Validate existing files on disk
    log.info("Validating existing .h5 files on disk...")
    valid_on_disk = {}  # filename → validate result
    corrupt_on_disk = {}

    for slot_ts in all_slots:
        fname = ts_to_filename(slot_ts)
        fpath = ts_to_folder(slot_ts) / fname
        if fpath.exists():
            vres = validate_hdf5(fpath)
            if vres["valid"]:
                valid_on_disk[fname] = {"path": fpath, "vres": vres}
            else:
                corrupt_on_disk[fname] = {"path": fpath, "reason": vres["reason"]}
                log.warning(f"  CORRUPT: {fname}  reason={vres['reason']}")

    log.info(f"  Valid on disk:   {len(valid_on_disk)}")
    log.info(f"  Corrupt on disk: {len(corrupt_on_disk)}")

    # ── Step 5: Decide action for each slot ───────────────────────────────────
    plan = []  # list of dicts
    for slot_ts in all_slots:
        fname = ts_to_filename(slot_ts)
        in_catalog = fname in catalog_by_id
        on_disk_valid = fname in valid_on_disk
        on_disk_corrupt = fname in corrupt_on_disk

        if on_disk_valid:
            action = "SKIP"         # already have it
        elif on_disk_corrupt:
            action = "REDOWNLOAD"   # corrupt, must refetch
        elif in_catalog:
            action = "DOWNLOAD"     # in catalog, not on disk
        else:
            action = "CATALOG_GAP"  # MOSDAC never published this slot

        plan.append({
            "slot_utc": slot_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slot_ist": (slot_ts + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "identifier": fname,
            "action": action,
            "in_catalog": in_catalog,
            "record_id": catalog_by_id[fname]["id"] if in_catalog else None,
        })

    # Summary
    action_counts = pd.DataFrame(plan)["action"].value_counts()
    log.info("Work plan summary:")
    for act, cnt in action_counts.items():
        log.info(f"  {act}: {cnt}")

    to_download = [p for p in plan if p["action"] in ("DOWNLOAD", "REDOWNLOAD")]
    log.info(f"Files to download/re-download: {len(to_download)}")

    if args.dry_run:
        log.info("[DRY RUN] Stopping before download. Full plan:")
        df_plan = pd.DataFrame(plan)
        print(df_plan.to_string())
        return

    if not to_download:
        log.info("Nothing to download. All files valid.")
    else:
        # ── Step 6: Authenticate & download ───────────────────────────────────
        config_path = REPO_ROOT / "config.json"
        token, username = authenticate(config_path)
        if not token:
            log.error("Authentication failed. Check config.json.")
            sys.exit(1)
        log.info(f"Authenticated as: {username}")

        if args.limit:
            log.info(f"--limit {args.limit}: capping downloads.")
            to_download = to_download[:args.limit]

        n_ok = 0
        n_fail = 0
        last_success_ts = None

        for i, item in enumerate(to_download, 1):
            fname      = item["identifier"]
            slot_ts    = datetime.fromisoformat(item["slot_utc"].replace("Z", "+00:00"))
            target_path = ts_to_folder(slot_ts) / fname
            rec_id     = item["record_id"]

            log.info(f"[{i}/{len(to_download)}] {item['action']}: {fname}")

            result = download_one(rec_id, fname, target_path, token)
            status = result["status"]

            if status == "DOWNLOADED":
                # Validate immediately
                vres = validate_hdf5(target_path)
                if vres["valid"]:
                    log.info(f"  ✓ Downloaded & validated. Size={result['file_size_bytes']/1e6:.2f}MB")
                    n_ok += 1
                    last_success_ts = item["slot_utc"]
                    # Update plan entry
                    for p in plan:
                        if p["identifier"] == fname:
                            p["action"] = "SKIP"
                            break
                else:
                    log.warning(f"  ✗ Downloaded but INVALID: {vres['reason']}. Removing.")
                    target_path.unlink(missing_ok=True)
                    n_fail += 1
                    for p in plan:
                        if p["identifier"] == fname:
                            p["action"] = f"VALIDATION_FAIL:{vres['reason']}"
                            break

            elif status == "AUTH_EXPIRED":
                log.warning("  Token expired. Re-authenticating...")
                token, _ = authenticate(config_path)
                # Retry this item
                result2 = download_one(rec_id, fname, target_path, token)
                if result2["status"] == "DOWNLOADED":
                    vres = validate_hdf5(target_path)
                    if vres["valid"]:
                        log.info(f"  ✓ Downloaded after re-auth. Size={result2['file_size_bytes']/1e6:.2f}MB")
                        n_ok += 1
                        last_success_ts = item["slot_utc"]
                    else:
                        target_path.unlink(missing_ok=True)
                        n_fail += 1
                else:
                    n_fail += 1
                    log.error(f"  ✗ Failed even after re-auth: {result2['status']}")
            else:
                n_fail += 1
                log.error(f"  ✗ {status}: {result.get('error','')}")

            # Periodic manifest checkpoint every 5 downloads
            if i % 5 == 0:
                _write_manifest(plan, catalog_by_id, valid_on_disk, corrupt_on_disk)
                log.info(f"  Manifest checkpoint saved at [{i}/{len(to_download)}]")

        logout(username)
        log.info(f"Logged out. Downloads: {n_ok} ok, {n_fail} failed.")
        if last_success_ts:
            log.info(f"Last successful download: {last_success_ts}")

    # ── Step 7: Write final manifest ──────────────────────────────────────────
    _write_manifest(plan, catalog_by_id, valid_on_disk, corrupt_on_disk)
    log.info(f"Final manifest written to: {MANIFEST_CSV}")

    # ── Step 8: Summary ───────────────────────────────────────────────────────
    df_plan = pd.DataFrame(plan)
    remaining_missing = df_plan[df_plan["action"].isin(
        ["DOWNLOAD", "REDOWNLOAD"]) | df_plan["action"].str.startswith("FAILED", na=False)
                                   | df_plan["action"].str.startswith("VALIDATION", na=False)]
    log.info(f"\n{'='*60}")
    log.info(f"PHASE 2 SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Total slots in window:      {len(plan)}")
    log.info(f"Valid on disk (all sources):{len(df_plan[df_plan['action']=='SKIP'])}")
    log.info(f"Catalog gaps (MOSDAC):      {len(df_plan[df_plan['action']=='CATALOG_GAP'])}")
    log.info(f"Remaining to download:      {len(remaining_missing)}")


def _write_manifest(plan, catalog_by_id, valid_on_disk, corrupt_on_disk):
    """Write the master manifest CSV from the current plan state."""
    extractor = None
    try:
        extractor = CatchmentRainfallExtractor(POLYGON_PATH)
    except Exception:
        pass

    rows = []
    for item in plan:
        fname = item["identifier"]
        slot_ts_str = item["slot_utc"]
        slot_ts = datetime.fromisoformat(slot_ts_str.replace("Z", "+00:00"))
        target_path = ts_to_folder(slot_ts) / fname

        # Determine status for manifest
        action = item["action"]
        if action == "SKIP" and fname in valid_on_disk:
            dl_status = "VERIFIED_ON_DISK"
        elif action == "CATALOG_GAP":
            dl_status = "CATALOG_GAP"
        elif fname in corrupt_on_disk:
            dl_status = f"CORRUPT_{corrupt_on_disk[fname]['reason'][:30]}"
        elif action.startswith("FAILED"):
            dl_status = action
        elif action.startswith("VALIDATION_FAIL"):
            dl_status = action
        else:
            dl_status = action

        file_size = target_path.stat().st_size if target_path.exists() else 0
        rec = catalog_by_id.get(fname, {})

        rows.append({
            "record_id":         rec.get("id", ""),
            "identifier":        fname,
            "timestamp_utc":     slot_ts_str,
            "timestamp_ist":     item["slot_ist"],
            "download_status":   dl_status,
            "file_size_bytes":   file_size,
            "in_catalog":        item["in_catalog"],
            "extract_success":   False,
            "mean_rain_mm_hr":   None,
            "coverage_ratio":    None,
            "qc_flag":           "PENDING",
        })

    df = pd.DataFrame(rows)
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_CSV, index=False)


if __name__ == "__main__":
    main()
