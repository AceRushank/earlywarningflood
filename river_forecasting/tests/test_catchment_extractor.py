"""
river_forecasting/tests/test_catchment_extractor.py
===================================================
Unit tests for CatchmentRainfallExtractor with authentic HydroBASINS delineated
upstream Godavari polygon and real INSAT-3DS HEM HDF5 granules.
"""

import unittest
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from river_forecasting.src.catchment_extractor import CatchmentRainfallExtractor
from river_forecasting.src.catchment_boundary_spec import VishnupuriCatchmentSpecification


class TestCatchmentRainfallExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.polygon_path = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "vishnupuri_upstream_catchment.geojson"
        cls.hdf_dir = REPO_ROOT / "river_forecasting" / "data" / "mosdac_raw" / "3SIMG_L2B_HEM" / "2024" / "04JUL"
        cls.sample_hdf = cls.hdf_dir / "3SIMG_04JUL2024_2330_L2B_HEM_V01R00.h5"

    def test_polygon_exists_and_validates(self):
        self.assertTrue(self.polygon_path.exists(), "Catchment polygon GeoJSON missing")
        extractor = CatchmentRainfallExtractor(self.polygon_path)
        self.assertIsNotNone(extractor.catchment_geom)

        # Validate area against CWC benchmark tolerance [49000, 53000] sq km
        area_km2 = extractor.catchment_area_sq_km
        self.assertGreaterEqual(area_km2, VishnupuriCatchmentSpecification.TOLERANCE_AREA_MIN_SQ_KM)
        self.assertLessEqual(area_km2, VishnupuriCatchmentSpecification.TOLERANCE_AREA_MAX_SQ_KM)

        # Bounds check
        min_lon, min_lat, max_lon, max_lat = extractor.catchment_bbox
        self.assertGreater(min_lon, 73.0)
        self.assertLess(max_lon, 78.0)
        self.assertGreater(min_lat, 18.0)
        self.assertLess(max_lat, 21.0)

    def test_curvilinear_mask_generation(self):
        self.assertTrue(self.sample_hdf.exists(), "Sample HDF5 granule missing")
        extractor = CatchmentRainfallExtractor(self.polygon_path)
        n_pixels = extractor.build_catchment_pixel_mask(self.sample_hdf)

        # Upper Godavari basin spans ~2800 INSAT-3DS pixels (~4 km grid)
        self.assertGreater(n_pixels, 2500)
        self.assertLess(n_pixels, 3500)
        self.assertEqual(extractor.total_catchment_pixels, n_pixels)

    def test_single_timestep_extraction(self):
        extractor = CatchmentRainfallExtractor(self.polygon_path)
        res = extractor.extract_single_timestep(self.sample_hdf)

        self.assertTrue(res["success"])
        self.assertEqual(res["qc_flag"], "OK")
        self.assertEqual(res["coverage_ratio"], 1.0)
        self.assertGreaterEqual(res["mean_rainfall_mm_hr"], 0.0)
        self.assertLessEqual(res["mean_rainfall_mm_hr"], 150.0)
        self.assertEqual(res["timestamp_utc"], "2024-07-04 23:30:00")
        self.assertEqual(res["timestamp_ist"], "2024-07-05 05:00:00")

    def test_sequence_extraction(self):
        hdf_files = sorted(list(self.hdf_dir.glob("*.h5")))
        self.assertGreaterEqual(len(hdf_files), 2)

        extractor = CatchmentRainfallExtractor(self.polygon_path)
        df, summary = extractor.extract_sequence(hdf_files, expected_interval_minutes=30)

        self.assertEqual(summary["total_files_evaluated"], len(hdf_files))
        self.assertEqual(summary["successful_extractions"], len(hdf_files))
        self.assertEqual(summary["missing_timesteps_count"], 0)
        self.assertIn("mean_rainfall_mm_hr", df.columns)
        self.assertEqual(len(df), len(hdf_files))


if __name__ == "__main__":
    unittest.main()
