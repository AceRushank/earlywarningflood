"""
river_forecasting/tests/test_mosdac_retriever.py
================================================
Unit tests for MOSDAC retrieval, version filtering, and HDF5 validation.
"""

import unittest
from pathlib import Path
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from river_forecasting.src.mosdac_retriever import (
    search_valid_records,
    RELEASE_VERSION_PATTERN
)


class TestMosdacRetriever(unittest.TestCase):

    def test_version_regex_pattern(self):
        valid_name = "3SIMG_04JUL2024_2330_L2B_HEM_V01R00.h5"
        invalid_name = "3SIMG_22AUG2024_2330_L2B_HEM_V00R05.h5"
        other_name = "3SIMG_04JUL2024_2330_L2B_HEM_V01R01.h5"

        self.assertIsNotNone(RELEASE_VERSION_PATTERN.match(valid_name))
        self.assertIsNone(RELEASE_VERSION_PATTERN.match(invalid_name))
        self.assertIsNone(RELEASE_VERSION_PATTERN.match(other_name))

    def test_search_filters_obsolete_records(self):
        """
        Queries a date known to contain both V00R05 and V01R00 records,
        and verifies that only V01R00 records are returned.
        """
        records = search_valid_records(
            dataset_id="3SIMG_L2B_HEM",
            start_time="2024-08-22",
            end_time="2024-08-22",
            target_version="V01R00",
            max_records=4
        )
        self.assertGreater(len(records), 0)
        for r in records:
            self.assertTrue(r["identifier"].endswith("_V01R00.h5"))
            self.assertFalse("V00R05" in r["identifier"])

    def test_downloaded_hdf5_integrity(self):
        """
        Validates the integrity, dimensions, and variables of verified local HDF5 files.
        """
        sample_path = REPO_ROOT / "river_forecasting" / "data" / "mosdac_raw" / "3SIMG_L2B_HEM" / "2024" / "04JUL" / "3SIMG_04JUL2024_2330_L2B_HEM_V01R00.h5"
        self.assertTrue(sample_path.exists(), f"Sample HDF5 missing: {sample_path}")
        self.assertGreater(sample_path.stat().st_size, 10_000_000)

        with h5py.File(sample_path, "r") as hf:
            # Datasets
            expected_keys = {"HEM", "Latitude", "Longitude", "time", "GeoX", "GeoY"}
            self.assertTrue(expected_keys.issubset(set(hf.keys())))

            # HEM dataset properties
            hem = hf["HEM"]
            self.assertEqual(hem.shape, (1, 2816, 2805))
            self.assertEqual(hem.attrs.get("units"), b"mm/hr")
            self.assertEqual(hem.attrs.get("_FillValue")[0], -999.0)

            # Time dataset
            time_ds = hf["time"]
            self.assertEqual(time_ds.attrs.get("units"), b"minutes since 2000-01-01 00:00:00")
            self.assertEqual(time_ds[0], 12890850.0)


if __name__ == "__main__":
    unittest.main()
