"""
scripts/crawl_mosdac_catalog_2026.py
====================================
Queries the official ISRO MOSDAC OpenSearch API day-by-day from May 31 to October 10, 2026.
Discovers available operational INSAT-3DS HEM granules (V01R00 release), extracts record IDs,
and catalogs available vs. unavailable/outage dates.

Saves:
- data/events_2026/mosdac_catalog_index_2026.json
"""

import json
import time
from pathlib import Path
from datetime import date, timedelta
import requests

OUT_INDEX = Path("data/events_2026/mosdac_catalog_index_2026.json")
SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"

def crawl_catalog():
    print("=" * 75)
    print("CRAWLING MOSDAC OPENSEARCH CATALOG FOR 2026 MONSOON GRANULES")
    print("=" * 75)

    start_d = date(2026, 5, 31)
    end_d = date(2026, 10, 10)
    total_days = (end_d - start_d).days + 1

    catalog = {
        "metadata": {
            "dataset_id": "3SIMG_L2B_HEM",
            "target_version": "V01R00",
            "start_date": str(start_d),
            "end_date": str(end_d),
            "total_days_probed": total_days,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "days": {},
        "records": {},  # filename -> {id, identifier, updated, date}
        "outage_days": []
    }

    curr = start_d
    total_found = 0

    while curr <= end_d:
        d_str = curr.strftime("%Y-%m-%d")
        params = {
            "datasetId": "3SIMG_L2B_HEM",
            "startTime": d_str,
            "endTime": d_str,
            "count": 100
        }

        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                entries = data.get("entries", [])
                v1_entries = [e for e in entries if "_V01R00.h5" in e.get("identifier", "")]
                
                catalog["days"][d_str] = {
                    "status": "AVAILABLE" if len(v1_entries) > 0 else "EMPTY",
                    "count": len(v1_entries),
                    "http_code": 200
                }

                for e in v1_entries:
                    fname = e.get("identifier")
                    catalog["records"][fname] = {
                        "id": str(e.get("id")),
                        "identifier": fname,
                        "updated": e.get("updated"),
                        "date": d_str
                    }
                total_found += len(v1_entries)
                print(f"[{d_str}] HTTP 200 | {len(v1_entries)} granules (Total: {total_found})")

            elif resp.status_code == 500:
                print(f"[{d_str}] HTTP 500 | SATELLITE OUTAGE / DATA UNAVAILABLE")
                catalog["days"][d_str] = {
                    "status": "OUTAGE_UNAVAILABLE",
                    "count": 0,
                    "http_code": 500,
                    "error": "HTTP 500 Data unavailable for given parameter"
                }
                catalog["outage_days"].append(d_str)

            else:
                print(f"[{d_str}] HTTP {resp.status_code}")
                catalog["days"][d_str] = {
                    "status": f"HTTP_{resp.status_code}",
                    "count": 0,
                    "http_code": resp.status_code
                }

        except Exception as ex:
            print(f"[{d_str}] ERROR: {ex}")
            catalog["days"][d_str] = {
                "status": "CONNECTION_ERROR",
                "count": 0,
                "error": str(ex)
            }

        curr += timedelta(days=1)
        time.sleep(0.15)  # respectful request rate

    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_INDEX, "w") as f:
        json.dump(catalog, f, indent=2)

    print("\n" + "=" * 75)
    print("CATALOG CRAWL COMPLETE")
    print(f"Total operational granules discovered: {len(catalog['records'])}")
    print(f"Confirmed outage days: {catalog['outage_days']}")
    print(f"Catalog index saved to: {OUT_INDEX}")
    print("=" * 75)

if __name__ == "__main__":
    crawl_catalog()
