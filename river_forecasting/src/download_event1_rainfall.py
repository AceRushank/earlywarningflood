"""
river_forecasting/src/download_event1_rainfall.py
=================================================
Automated, resilient retrieval and extraction of authentic INSAT-3DS HEM
rainfall for Event 1 (July 31 - August 5, 2024).

Features:
- Idempotent: Skips already downloaded and verified files.
- Resilient: Handles HTTP 429 (rate limits) and HTTP 401 (token refresh).
- Validated: Verifies HDF5 readability and extracts catchment-mean rainfall on the fly.
- Detailed manifest: Produces event1_rainfall_manifest.csv with timestamps, sizes, and QC.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import requests
import h5py
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.mosdac_retriever import (
    search_valid_records,
    authenticate,
    logout,
    TOKEN_URL,
    DOWNLOAD_URL
)
from river_forecasting.src.catchment_extractor import CatchmentRainfallExtractor


def run_event1_ingestion(
    start_date: str = "2024-07-31",
    end_date: str = "2024-08-05",
    config_path: Path = REPO_ROOT / "config.json",
    dest_dir: Path = REPO_ROOT / "river_forecasting" / "data" / "mosdac_raw" / "3SIMG_L2B_HEM",
    polygon_path: Path = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "vishnupuri_upstream_catchment.geojson",
    manifest_csv: Path = REPO_ROOT / "river_forecasting" / "outputs" / "event1_rainfall_manifest.csv",
    rainfall_csv: Path = REPO_ROOT / "river_forecasting" / "outputs" / "event1_catchment_rainfall.csv"
):
    print("=" * 70)
    print(f"MOSDAC Event 1 Ingestion: {start_date} to {end_date}")
    print("=" * 70)

    # 1. Initialize Catchment Extractor
    print(f"Loading catchment polygon from {polygon_path}...")
    extractor = CatchmentRainfallExtractor(polygon_path)
    print(f"Catchment loaded: {extractor.catchment_area_sq_km:.2f} sq km")

    # 2. Search for all released V01R00 records
    print(f"Querying MOSDAC OpenSearch API for dataset 3SIMG_L2B_HEM ({start_date} to {end_date})...")
    records = search_valid_records(
        dataset_id="3SIMG_L2B_HEM",
        start_time=start_date,
        end_time=end_date,
        target_version="V01R00",
        page_size=100
    )
    print(f"Found {len(records)} authentic V01R00 records.")
    if not records:
        print("No records found. Exiting.")
        return

    # Sort chronologically
    records.sort(key=lambda r: r["updated"])
    print(f"Earliest: {records[0]['identifier']} ({records[0]['updated']})")
    print(f"Latest:   {records[-1]['identifier']} ({records[-1]['updated']})")

    # 3. Authenticate
    token, username = authenticate(config_path)
    if not token:
        print("Authentication failed! Check config.json.")
        sys.exit(1)
    print(f"Authenticated successfully as user: {username}")

    manifest_rows = []
    rainfall_rows = []

    try:
        for idx, rec in enumerate(records, 1):
            identifier = rec["identifier"]
            rec_id = rec["id"]
            prod_date = rec.get("updated")

            # Determine folder
            dt_obj = datetime.strptime(prod_date, "%Y-%m-%dT%H:%M:%SZ")
            folder = dest_dir / dt_obj.strftime("%Y") / f"{dt_obj.strftime('%d')}{dt_obj.strftime('%b').upper()}"
            folder.mkdir(parents=True, exist_ok=True)
            target_path = folder / identifier
            part_path = folder / f"{identifier}.part"

            download_status = "UNKNOWN"
            file_size_bytes = 0

            # Check if file already exists
            if target_path.exists() and target_path.stat().st_size > 0:
                download_status = "ALREADY_EXISTS"
                file_size_bytes = target_path.stat().st_size
            else:
                # Need to download with retries
                max_retries = 5
                for attempt in range(max_retries):
                    headers = {"Authorization": f"Bearer {token}"}
                    params = {"id": rec_id}
                    try:
                        resp = requests.get(DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=(15, 180))
                        
                        if resp.status_code == 429:
                            # Rate limited
                            err_data = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else {}
                            print(f"[{idx}/{len(records)}] Rate limit 429 ({err_data.get('type')}). Sleeping 25s...")
                            time.sleep(25)
                            continue

                        if resp.status_code == 401:
                            # Token expired, re-authenticate
                            print(f"[{idx}/{len(records)}] Token expired (401). Re-authenticating...")
                            token, _ = authenticate(config_path)
                            continue

                        if resp.status_code == 200:
                            content_disp = resp.headers.get("Content-Disposition", "")
                            if "filename=" not in content_disp:
                                download_status = "SERVER_FILE_NOT_FOUND"
                                break

                            # Stream to part file
                            with open(part_path, "wb") as pf:
                                for chunk in resp.iter_content(chunk_size=1048576):
                                    if chunk:
                                        pf.write(chunk)

                            # Atomic rename
                            if part_path.exists():
                                part_path.rename(target_path)

                            download_status = "DOWNLOADED"
                            file_size_bytes = target_path.stat().st_size
                            break
                        else:
                            download_status = f"HTTP_{resp.status_code}"
                            time.sleep(5)

                    except Exception as e:
                        if part_path.exists():
                            try:
                                part_path.unlink()
                            except Exception:
                                pass
                        download_status = f"ERROR_{str(e)[:30]}"
                        time.sleep(5)

            # Extract catchment rainfall if file exists
            extract_success = False
            mean_rain = None
            max_rain = None
            coverage = None
            qc_flag = "FILE_MISSING"

            if target_path.exists() and target_path.stat().st_size > 0:
                try:
                    res = extractor.extract_single_timestep(target_path)
                    extract_success = res["success"]
                    mean_rain = res["mean_rainfall_mm_hr"]
                    max_rain = res["max_rainfall_mm_hr"]
                    coverage = res["coverage_ratio"]
                    qc_flag = res["qc_flag"]

                    rainfall_rows.append({
                        "timestamp_utc": res["timestamp_utc"],
                        "timestamp_ist": res["timestamp_ist"],
                        "source_file": identifier,
                        "mean_rainfall_rate_mm_hr": mean_rain,
                        "half_hour_accumulated_rainfall_mm": round(mean_rain * 0.5, 4) if mean_rain is not None else None,
                        "max_pixel_rainfall_mm_hr": max_rain,
                        "coverage_ratio": coverage,
                        "valid_pixels": res["valid_rainfall_pixels"],
                        "total_pixels": res["total_catchment_pixels"],
                        "qc_flag": qc_flag
                    })
                except Exception as ex:
                    qc_flag = f"EXTRACTION_ERROR: {str(ex)[:40]}"

            manifest_rows.append({
                "record_id": rec_id,
                "identifier": identifier,
                "timestamp_utc": prod_date,
                "timestamp_ist": (dt_obj + pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                "download_status": download_status,
                "file_size_bytes": file_size_bytes,
                "extract_success": extract_success,
                "mean_rain_mm_hr": mean_rain,
                "coverage_ratio": coverage,
                "qc_flag": qc_flag
            })

            if idx % 10 == 0 or idx == len(records):
                print(f"[{idx}/{len(records)}] Processed {identifier} | Status: {download_status} | Rain: {mean_rain} mm/hr | QC: {qc_flag}")
                # Save partial manifest & rainfall csv
                pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
                pd.DataFrame(rainfall_rows).to_csv(rainfall_csv, index=False)

    finally:
        logout(username)
        print("Logged out of MOSDAC session cleanly.")

    # Final save
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"\nManifest saved to: {manifest_csv} ({len(df_manifest)} entries)")

    df_rainfall = pd.DataFrame(rainfall_rows)
    df_rainfall.to_csv(rainfall_csv, index=False)
    print(f"Catchment rainfall saved to: {rainfall_csv} ({len(df_rainfall)} entries)")


if __name__ == "__main__":
    run_event1_ingestion()
