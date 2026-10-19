"""
river_forecasting/tests/test_historical_ingestion.py
===================================================
Unit tests for the historical rainfall ingestion, catchment parameters,
and dashboard data bundling.
"""

import unittest
from pathlib import Path
import json
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestHistoricalIngestion(unittest.TestCase):
    def test_imd_daily_dataset_exists(self):
        csv_path = REPO_ROOT / "river_forecasting" / "outputs" / "historical_catchment_rainfall_daily.csv"
        self.assertTrue(csv_path.exists(), "IMD daily catchment rainfall CSV must exist.")
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 300, "Should contain full calendar year daily entries.")
        self.assertIn("rainfall_1d_mm", df.columns)
        self.assertIn("rainfall_3d_accum_mm", df.columns)
        self.assertIn("rainfall_7d_accum_mm", df.columns)
        self.assertIn("active_cells_observed", df.columns)
        self.assertEqual(df["total_catchment_cells"].iloc[0], 71)

    def test_catchment_physiography_json(self):
        json_path = REPO_ROOT / "river_forecasting" / "outputs" / "catchment_physiographic_parameters.json"
        self.assertTrue(json_path.exists(), "Catchment physiographic parameters JSON must exist.")
        with open(json_path) as f:
            params = json.load(f)
        self.assertIn("station_information", params)
        self.assertIn("catchment_physiography", params)
        self.assertAlmostEqual(params["catchment_physiography"]["drainage_area_hydrobasins_sq_km"], 51768.25, places=2)
        self.assertAlmostEqual(params["catchment_physiography"]["main_channel_length_km"], 650.0, places=1)
        self.assertAlmostEqual(params["catchment_physiography"]["average_river_channel_slope_m_per_km"], 1.10, places=2)

    def test_dashboard_bundle_integrity(self):
        bundle_path = REPO_ROOT / "ui" / "dashboard_data.json"
        self.assertTrue(bundle_path.exists(), "Dashboard data bundle must exist.")
        with open(bundle_path) as f:
            data = json.load(f)
        self.assertIn("timeline", data)
        self.assertIn("catchment_geojson", data)
        self.assertIn("annotations", data)
        self.assertIn("evaluation_metrics", data)
        self.assertGreater(len(data["timeline"]), 1000)

        # Verify rising-limb predictions are mapped
        sample_with_pred = next((pt for pt in data["timeline"] if pt["predictions"] is not None), None)
        self.assertIsNotNone(sample_with_pred)
        self.assertIn("1h", sample_with_pred["predictions"])
        self.assertIn("persistence", sample_with_pred["predictions"]["1h"])
        self.assertIn("wl_ridge", sample_with_pred["predictions"]["1h"])


if __name__ == "__main__":
    unittest.main()
