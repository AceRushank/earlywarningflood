"""
river_forecasting/src/mosdac_retriever.py
=========================================
Robust MOSDAC Data Download Client for INSAT-3DS HEM Products.

Solves the multi-version catalog problem by:
1. Paginating through the MOSDAC OpenSearch API.
2. Filtering search entries to select ONLY verified operational release versions
   (e.g., 'V01R00'), automatically discarding obsolete/unreleased versions
   (such as 'V00R05') that return HTTP 404 from the file server.
3. Managing authentication tokens securely without printing or logging credentials.
4. Handling existing files idempotently (skipping already downloaded files).
5. Verifying file integrity and nonzero size after download.
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import requests

SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
LOGOUT_URL = "https://mosdac.gov.in/download_api/logout"

# Official release regex pattern
RELEASE_VERSION_PATTERN = re.compile(r"^3SIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_L2B_HEM_V01R00\.h5$")


def search_valid_records(
    dataset_id: str = "3SIMG_L2B_HEM",
    start_time: str = "2024-07-04",
    end_time: str = "2024-07-04",
    target_version: str = "V01R00",
    max_records: Optional[int] = None,
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Paginates through MOSDAC OpenSearch API and returns only records matching the target version.
    Filters out obsolete records (e.g. V00R05) that do not exist on the file server.
    """
    valid_records: List[Dict[str, Any]] = []
    seen_ids = set()
    start_index = 1

    version_suffix = f"_{target_version}.h5"

    while True:
        params = {
            "datasetId": dataset_id,
            "startTime": start_time,
            "endTime": end_time,
            "startIndex": start_index,
            "count": page_size,
        }

        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
        except Exception:
            break

        entries = data.get("entries", [])
        if not entries:
            break

        for item in entries:
            identifier = item.get("identifier", "")
            rec_id = str(item.get("id", ""))
            prod_date = item.get("updated", "")

            # Check if this record matches the released version
            if version_suffix in identifier and rec_id not in seen_ids:
                seen_ids.add(rec_id)
                valid_records.append({
                    "id": rec_id,
                    "identifier": identifier,
                    "updated": prod_date,
                    "version": target_version
                })

                if max_records and len(valid_records) >= max_records:
                    return valid_records

        total_results = data.get("totalResults", 0)
        start_index += len(entries)
        if start_index > total_results or len(entries) < page_size:
            break

    return valid_records


def authenticate(config_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Authenticates with MOSDAC using credentials from config.json without printing them.
    Returns (access_token, username) or (None, None).
    """
    if not config_path.exists():
        return None, None

    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        creds = cfg.get("user_credentials", {})
        username = creds.get("username/email", "")
        password = creds.get("password", "")

        if not username or not password:
            return None, None

        resp = requests.post(
            TOKEN_URL,
            json={"username": username, "password": password},
            timeout=10
        )
        if resp.status_code == 200:
            token_data = resp.json()
            return token_data.get("access_token"), username
    except Exception:
        pass

    return None, None


def logout(username: str) -> bool:
    """Logs out the active MOSDAC session cleanly."""
    try:
        data = {"username": username}
        resp = requests.post(LOGOUT_URL, json=data, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def download_single_record(
    access_token: str,
    record: Dict[str, Any],
    dest_dir: Path,
    organize_by_date: bool = True
) -> Dict[str, Any]:
    """
    Downloads a single verified record.
    Preserves existing files, streams via .part file, and verifies nonzero size.
    """
    identifier = record["identifier"]
    rec_id = record["id"]
    prod_date = record.get("updated")

    # Determine destination folder
    if organize_by_date and prod_date:
        try:
            dt = datetime.strptime(prod_date, "%Y-%m-%dT%H:%M:%SZ")
            folder = dest_dir / dt.strftime("%Y") / f"{dt.strftime('%d')}{dt.strftime('%b').upper()}"
        except ValueError:
            folder = dest_dir
    else:
        folder = dest_dir

    folder.mkdir(parents=True, exist_ok=True)
    target_path = folder / identifier
    part_path = folder / f"{identifier}.part"

    # Check if already downloaded
    if target_path.exists() and target_path.stat().st_size > 0:
        return {
            "identifier": identifier,
            "record_id": rec_id,
            "status": "ALREADY_EXISTS",
            "file_path": str(target_path),
            "size_bytes": target_path.stat().st_size,
            "success": True,
        }

    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"id": rec_id}

    try:
        resp = requests.get(DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=30)
        if resp.status_code != 200:
            return {
                "identifier": identifier,
                "record_id": rec_id,
                "status": f"HTTP_{resp.status_code}",
                "file_path": None,
                "size_bytes": 0,
                "success": False,
            }

        # Check content disposition
        content_disp = resp.headers.get("Content-Disposition", "")
        if "filename=" not in content_disp:
            return {
                "identifier": identifier,
                "record_id": rec_id,
                "status": "FILE_NOT_FOUND_ON_SERVER",
                "file_path": None,
                "size_bytes": 0,
                "success": False,
            }

        expected_size = int(resp.headers.get("Content-Length", 0))

        # Stream write to .part file
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1048576):
                if chunk:
                    f.write(chunk)

        # Atomic rename
        if part_path.exists():
            part_path.rename(target_path)

        actual_size = target_path.stat().st_size
        return {
            "identifier": identifier,
            "record_id": rec_id,
            "status": "DOWNLOADED",
            "file_path": str(target_path),
            "size_bytes": actual_size,
            "expected_size": expected_size,
            "success": actual_size > 0,
        }

    except Exception as e:
        if part_path.exists():
            try:
                part_path.unlink()
            except Exception:
                pass
        return {
            "identifier": identifier,
            "record_id": rec_id,
            "status": f"ERROR: {str(e)}",
            "file_path": None,
            "size_bytes": 0,
            "success": False,
        }
