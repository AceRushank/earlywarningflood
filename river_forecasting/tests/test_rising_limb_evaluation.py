"""
river_forecasting/tests/test_rising_limb_evaluation.py
======================================================
Unit tests for the scientifically valid rising-limb evaluation:
- Verifies strict chronological constraints and non-leakage
- Verifies training feasibility gate outcomes
- Verifies prediction counts and consistency
- Verifies that all 9 leakage audit checks pass
"""

import sys
import unittest
from pathlib import Path
import json
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.rising_limb_evaluation import (
    compute_metrics,
    build_experiment_timeseries,
    OUT_PREDICTIONS_CSV,
    OUT_RESULTS_JSON,
    OUT_REPORT_MD
)


class TestRisingLimbEvaluation(unittest.TestCase):
    def test_metrics_calculation(self):
        y_true = np.array([353.0, 354.0, 355.0])
        y_pred = np.array([353.1, 353.9, 355.2])
        m = compute_metrics(y_true, y_pred)
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["mae"], 0.1333, places=3)
        self.assertAlmostEqual(m["max_ae"], 0.2000, places=3)
        self.assertGreater(m["nse"], 0.95)

    def test_timeseries_causality(self):
        df = build_experiment_timeseries()
        # Stage features <= t: feat_stage_lag1 at index 10 equals stage_m at index 9
        idx = 15
        self.assertEqual(df.loc[idx, "feat_stage_lag1"], df.loc[idx - 1, "stage_m"])
        self.assertEqual(df.loc[idx, "feat_stage_t"], df.loc[idx, "stage_m"])

        # Rainfall features strictly prior to t (< t): feat_rain_lag1 at index 10 equals hourly_depth_mm at index 9
        # Check non-empty rainfall section
        rain_rows = df[df["hourly_depth_mm"].notna()].index
        if len(rain_rows) > 5:
            r_idx = rain_rows[3]
            self.assertEqual(df.loc[r_idx, "feat_rain_lag1"], df.loc[r_idx - 1, "hourly_depth_mm"])

    def test_output_artifacts_exist_and_consistent(self):
        self.assertTrue(OUT_PREDICTIONS_CSV.exists())
        self.assertTrue(OUT_RESULTS_JSON.exists())
        self.assertTrue(OUT_REPORT_MD.exists())

        # Check JSON
        with open(OUT_RESULTS_JSON, "r") as f:
            res = json.load(f)

        self.assertIn("training_feasibility_gate", res)
        self.assertIn("models", res)
        self.assertIn("leakage_audit", res)

        # Check all 9 leakage audit checks pass
        audit = res["leakage_audit"]
        self.assertEqual(len(audit), 9)
        for k, v in audit.items():
            self.assertEqual(v["status"], "PASS")

        # Check sample counts
        self.assertEqual(res["models"]["Persistence Baseline"]["1h"]["n"], 34)
        self.assertEqual(res["models"]["Persistence Baseline"]["2h"]["n"], 33)
        self.assertEqual(res["models"]["Persistence Baseline"]["3h"]["n"], 32)

        # Check Rainfall model is unavailable
        self.assertIsNone(res["models"]["Rainfall-Enhanced ML (Ridge)"]["1h"]["n"])
        self.assertEqual(
            res["models"]["Rainfall-Enhanced ML (Ridge)"]["status"],
            "UNAVAILABLE_INSUFFICIENT_TRAINING_DATA"
        )

        # Check Predictions CSV
        df_pred = pd.read_csv(OUT_PREDICTIONS_CSV)
        self.assertEqual(len(df_pred), 198)  # (34 + 33 + 32) * 2 models
        self.assertFalse(df_pred["predicted_stage_m"].isna().any())
        self.assertFalse(df_pred["actual_stage_m"].isna().any())
        self.assertFalse(df_pred["absolute_error_m"].isna().any())


if __name__ == "__main__":
    unittest.main()
