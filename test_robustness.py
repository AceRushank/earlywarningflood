"""
test_robustness.py

Comprehensive scientific and robustness tests for the Mumbai Flood Early-Warning System:
  1. ML Model & Metadata Validation
  2. IMD 24-Hour Rainfall Category Mapping
  3. IMD Hourly Intensity Spell Mapping
  4. BRIMSTOWAD Drainage Benchmark Ratios
  5. Zero-Rain Safety Guarantee (no warnings when dry)
  6. High-Susceptibility + Low Rain Edge Case
  7. Low-Susceptibility + Extreme Rain Edge Case
  8. Weather Failure State & Graceful Degradation
  9. Multi-Horizon Scaling (6h, 12h, 24h)
"""

import math
import pickle
import unittest
import numpy as np
import pandas as pd

from app.data_loader import load_ward_data
from app.scoring import (
    classify_imd_24h_rainfall,
    classify_imd_intensity,
    calculate_drainage_intensity_ratio,
    compute_risk_scores,
    generate_alert_channels,
    _MODEL_PATH,
)


class TestMLModelArtifact(unittest.TestCase):
    def test_model_bundle_integrity(self):
        self.assertTrue(_MODEL_PATH.exists(), f"Model artifact {_MODEL_PATH} must exist")
        with open(_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

        self.assertEqual(bundle.get("target"), "inundation_ratio")
        self.assertEqual(bundle.get("training_ward_count"), 23)
        self.assertIn("Ridge", bundle.get("model_type"))
        metrics = bundle.get("metrics", {})
        self.assertGreater(metrics.get("R2", -1.0), 0.0, "Ridge must achieve positive R2")
        self.assertAlmostEqual(metrics.get("R2"), 0.076144, places=4)
        self.assertAlmostEqual(metrics.get("RMSE"), 0.034567, places=4)


class TestIMDClassifications(unittest.TestCase):
    def test_imd_24h_rainfall_categories(self):
        cases = [
            (0.0,   "NO_RAIN"),
            (5.0,   "VERY_LIGHT_TO_LIGHT"),
            (20.0,  "MODERATE"),
            (70.0,  "HEAVY"),
            (120.0, "VERY_HEAVY"),
            (205.0, "EXTREMELY_HEAVY"),
        ]
        for rain, expected in cases:
            actual = classify_imd_24h_rainfall(rain)
            self.assertEqual(actual, expected, f"Failed for {rain}mm: expected {expected}, got {actual}")

    def test_imd_intensity_spell_categories(self):
        cases = [
            (5.0,   "LIGHT_SPELL"),
            (15.0,  "MODERATE_SPELL"),
            (25.0,  "INTENSE_SPELL"),
            (40.0,  "VERY_INTENSE_SPELL"),
            (75.0,  "EXTREMELY_INTENSE_SPELL"),
            (110.0, "EXCEPTIONALLY_HEAVY_SPELL"),
        ]
        for intensity, expected in cases:
            actual = classify_imd_intensity(intensity)
            self.assertEqual(actual, expected, f"Failed for {intensity}mm/h: expected {expected}, got {actual}")

    def test_drainage_intensity_ratio(self):
        # 50 mm/hr benchmark
        self.assertEqual(calculate_drainage_intensity_ratio(25.0), 0.5)
        self.assertEqual(calculate_drainage_intensity_ratio(50.0), 1.0)
        self.assertEqual(calculate_drainage_intensity_ratio(75.0), 1.5)
        self.assertIsNone(calculate_drainage_intensity_ratio(None))


class TestScoringBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_ward_data()

    def test_zero_rain_safety(self):
        """CRITICAL: Zero rainfall must yield hybrid_risk_index == 0.0 and early_warning == False for all wards."""
        df = self.df.copy()
        df["weather_available"] = True
        df["next_6h_rainfall_mm"] = 0.0
        df["next_12h_rainfall_mm"] = 0.0
        df["next_24h_rainfall_mm"] = 0.0
        df["peak_hourly_mm"] = 0.0

        scored = compute_risk_scores(df, horizon="next_24h")

        self.assertTrue((scored["hybrid_risk_index"] == 0.0).all(), "All risk indices must be 0.0 when dry")
        self.assertFalse(scored["early_warning"].any(), "Zero early warnings on dry days")
        self.assertTrue((scored["risk_level"] == "LOW").all(), "All risk levels must be LOW on dry days")

    def test_high_susceptibility_low_rain(self):
        """High susceptibility with 5mm light rain must NOT trigger an early warning."""
        df = self.df.copy()
        df["weather_available"] = True
        df["next_24h_rainfall_mm"] = 5.0
        df["peak_hourly_mm"] = 2.0

        scored = compute_risk_scores(df, horizon="next_24h")
        self.assertFalse(scored["early_warning"].any(), "5mm light rain must not trigger early warnings")

    def test_extreme_rain_triggers_warnings(self):
        """250mm extreme rain must trigger early warnings across all wards."""
        df = self.df.copy()
        df["weather_available"] = True
        df["next_24h_rainfall_mm"] = 250.0
        df["peak_hourly_mm"] = 60.0

        scored = compute_risk_scores(df, horizon="next_24h")
        self.assertTrue(scored["early_warning"].all(), "Extreme rain must trigger warnings citywide")
        self.assertTrue((scored["risk_level"].isin(["HIGH", "CRITICAL"])).all())

    def test_weather_failure_handling(self):
        """If weather is unavailable, response must be safe and not silently zero-filled."""
        df = self.df.copy()
        df["weather_available"] = False
        df["next_6h_rainfall_mm"] = None
        df["next_12h_rainfall_mm"] = None
        df["next_24h_rainfall_mm"] = None
        df["peak_hourly_mm"] = None

        scored = compute_risk_scores(df, horizon="next_24h")
        self.assertTrue(scored["hybrid_risk_index"].isna().all())
        self.assertTrue((scored["risk_level"] == "WEATHER_UNAVAILABLE").all())
        self.assertFalse(scored["early_warning"].any())

    def test_alert_channel_disclaimers(self):
        """Simulated alert messages must contain prototype research disclaimer."""
        df = self.df.copy()
        df["weather_available"] = True
        df["next_24h_rainfall_mm"] = 150.0
        df["peak_hourly_mm"] = 40.0

        scored = compute_risk_scores(df, horizon="next_24h")
        alerted_row = scored[scored["early_warning"]].iloc[0]
        alerts = generate_alert_channels(alerted_row)

        self.assertIsNotNone(alerts)
        self.assertIn("NOT an official IMD alert", alerts["sms_draft"])
        self.assertIn("NOT an official IMD alert", alerts["push_notification"]["body"])
        self.assertIn("SIMULATED_DISSEMINATION", alerts["dashboard_entry"])

    def test_multi_horizon_hazard_scaling(self):
        """Verify that 204.4 mm is ONLY used for next_24h accumulation denominator, not 6h/12h."""
        df = self.df.copy()
        df["weather_available"] = True
        df["next_6h_rainfall_mm"] = 50.0
        df["next_12h_rainfall_mm"] = 80.0
        df["next_24h_rainfall_mm"] = 120.0
        df["peak_hourly_mm"] = 25.0  # 25 mm/hr -> intensity ratio = 0.50

        # For next_24h: h_accum = 120 / 204.4 = 0.5871, h_intensity = 25 / 50 = 0.50 -> hazard = 0.5871
        scored_24h = compute_risk_scores(df, horizon="next_24h")
        self.assertAlmostEqual(scored_24h["rainfall_hazard"].iloc[0], round(120.0 / 204.4, 4), places=3)

        # For next_6h: must NOT divide 50 by 204.4 (which would have been 0.2446).
        # Instead, hazard relies on peak hourly intensity benchmark: 25 / 50 = 0.50
        scored_6h = compute_risk_scores(df, horizon="next_6h")
        self.assertEqual(scored_6h["rainfall_hazard"].iloc[0], 0.50)
        self.assertEqual(scored_6h["forecast_rainfall_mm"].iloc[0], 50.0)

        # For next_12h: must NOT divide 80 by 204.4 (which would have been 0.3914).
        # Instead, hazard relies on peak hourly intensity benchmark: 25 / 50 = 0.50
        scored_12h = compute_risk_scores(df, horizon="next_12h")
        self.assertEqual(scored_12h["rainfall_hazard"].iloc[0], 0.50)
        self.assertEqual(scored_12h["forecast_rainfall_mm"].iloc[0], 80.0)


if __name__ == "__main__":
    unittest.main()
