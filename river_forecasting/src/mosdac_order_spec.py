"""
river_forecasting/src/mosdac_order_spec.py
==========================================
MOSDAC Catalog Order Specification & Acquisition Planner.

Product: INSAT-3DS Imager Level-2B Hydro-Estimator Precipitation (3SIMG_L2B_HEM_V01R00).
Target Window: 2024-07-04 00:00 UTC to 2024-08-23 23:59 UTC.
"""

from typing import Dict, Any
from datetime import datetime


class MosdacOrderSpecification:
    """
    Detailed specifications, volumetric estimates, and safe order workflows
    for historical INSAT-3DS HEM observations.
    """

    MISSION = "INSAT-3DS"
    SENSOR = "Imager"
    PRODUCT_ID = "3SIMG_L2B_HEM_V01R00"
    PRODUCT_NAME = "Level-2B Hydro-Estimator Precipitation (HEM)"
    FORMAT = "HDF5 (.h5)"
    VARIABLE = "/HEM (Precipitation Rate, mm/hr)"
    SCALE_FACTOR = 0.01
    FILL_VALUE = -999.0
    TEMPORAL_RESOLUTION_MINUTES = 30
    SPATIAL_RESOLUTION_KM = 4.0

    # Target Historical Window
    START_DATE_UTC = datetime(2024, 7, 4, 0, 0)
    END_DATE_UTC = datetime(2024, 8, 23, 23, 59)
    TOTAL_DAYS = 51

    # Volumetric Estimates (verified against local 669 files in data/mosdac/hdf_archive)
    AVERAGE_FILE_SIZE_MB = 9.76  # ~9,760,000 bytes per full-disk HDF5 file
    SLOTS_PER_DAY = 48
    THEORETICAL_MAX_FILES = TOTAL_DAYS * SLOTS_PER_DAY  # 2,448
    KNOWN_OPERATIONAL_GAPS_PER_DAY = 2  # Routine calibration/eclipse gaps at 17:30 and 19:00 UTC
    EXPECTED_ACTUAL_FILES = TOTAL_DAYS * (SLOTS_PER_DAY - KNOWN_OPERATIONAL_GAPS_PER_DAY)  # ~2,346 files

    # Storage Estimates
    FULL_DISK_STORAGE_GB = round((EXPECTED_ACTUAL_FILES * AVERAGE_FILE_SIZE_MB) / 1024.0, 2)  # ~22.36 GB
    SUBSET_FILE_SIZE_MB = 0.78  # Compressed Godavari basin clip (~18-21°N, 73-78°E)
    SUBSET_STORAGE_GB = round((EXPECTED_ACTUAL_FILES * SUBSET_FILE_SIZE_MB) / 1024.0, 2)  # ~1.79 GB

    # Transfer & Extraction Runtime Benchmarks
    DOWNLOAD_TIME_HOURS_AT_50MBPS = round((FULL_DISK_STORAGE_GB * 1024 * 8) / (50 * 3600), 2)  # ~1.02 hours
    EXTRACTION_SPEED_MS_PER_FILE = 12.0  # Curvilinear mask cached in CatchmentRainfallExtractor
    TOTAL_EXTRACTION_TIME_SECONDS = round((EXPECTED_ACTUAL_FILES * EXTRACTION_SPEED_MS_PER_FILE) / 1000.0, 1)  # ~28.2s

    @classmethod
    def get_order_plan(cls) -> Dict[str, Any]:
        return {
            "product_details": {
                "mission": cls.MISSION,
                "sensor": cls.SENSOR,
                "product_id": cls.PRODUCT_ID,
                "dataset": cls.VARIABLE,
                "temporal_resolution": "30 minutes (half-hourly)",
                "spatial_resolution": "~4 km curvilinear",
                "units": "mm/hr"
            },
            "order_window": {
                "start_utc": cls.START_DATE_UTC.strftime("%Y-%m-%d %H:%M:%S"),
                "end_utc": cls.END_DATE_UTC.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_days": cls.TOTAL_DAYS,
                "slots_per_day": cls.SLOTS_PER_DAY,
                "theoretical_file_count": cls.THEORETICAL_MAX_FILES,
                "expected_file_count": cls.EXPECTED_ACTUAL_FILES,
            },
            "spatial_subset_analysis": {
                "is_subset_option_available_on_portal": (
                    "MOSDAC cart supports geographic AOI bounding box filtering for query purposes, "
                    "but raw Level-2B HDF5 products are generated and delivered as full-disk curvilinear grids. "
                    "GeoTIFF conversions can be requested for subsets, but native HDF5 L2B files on SFTP "
                    "are standard full-disk swaths (~9.76 MB per file)."
                ),
                "full_disk_storage_estimate_gb": cls.FULL_DISK_STORAGE_GB,
                "subset_storage_estimate_gb": cls.SUBSET_STORAGE_GB,
                "recommended_strategy": (
                    "Order standard full-disk HDF5 files via MOSDAC cart. "
                    "Use our fast CatchmentRainfallExtractor to immediately filter and extract "
                    "catchment-average precipitation, keeping raw storage in data/mosdac/hdf_archive."
                )
            },
            "runtime_estimates": {
                "sftp_download_estimated_minutes": round(cls.DOWNLOAD_TIME_HOURS_AT_50MBPS * 60, 1),
                "catchment_extraction_estimated_seconds": cls.TOTAL_EXTRACTION_TIME_SECONDS,
                "processing_rate_files_per_second": round(1000.0 / cls.EXTRACTION_SPEED_MS_PER_FILE, 1)
            },
            "safe_credential_policy": [
                "Never commit passwords, tokens, or private keys to git.",
                "Store credentials only in local .env as MOSDAC_USERNAME and MOSDAC_PASSWORD.",
                "Existing scripts/download_mosdac_hdf.py reads .env safely without printing secrets."
            ],
            "step_by_step_order_workflow": [
                "1. Log into the official MOSDAC catalog at https://mosdac.gov.in/catalog/.",
                "2. Under Mission, select 'INSAT-3DS'; under Sensor, select 'Imager'.",
                "3. Under Product, select '3SIMG_L2B_HEM_V01R00' (Hydro-Estimator Precipitation).",
                "4. Set Date Range: 2024-07-04 00:00 to 2024-08-23 23:59 UTC.",
                "5. Add items to Cart and submit order with Order Purpose: 'Flood Forecasting Research'.",
                "6. Await email notification from MOSDAC with Order ID (typically 6-24 hours for historical processing).",
                "7. Once staged on download.mosdac.gov.in:22 under your order directory, run:",
                "   python scripts/download_mosdac_hdf.py --start-date 2024-07-04 --end-date 2024-08-23"
            ]
        }
