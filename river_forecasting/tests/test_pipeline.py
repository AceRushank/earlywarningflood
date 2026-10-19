"""
river_forecasting/tests/test_pipeline.py
========================================
Comprehensive unit tests for river_forecasting modules:
- Catchment boundary specification & area validator
- MOSDAC order calculations
- Feature pipeline (anti-leakage guarantee & missing-hour handling)
- Embargo buffer boundary protection
- Persistence and Water-Level-Only models (Ridge & Gradient Boosting)
- Missing hour handling in origin, lags, and future targets
- Evaluation metrics and rapid-rise regime separation
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.catchment_boundary_spec import VishnupuriCatchmentSpecification
from river_forecasting.src.mosdac_order_spec import MosdacOrderSpecification
from river_forecasting.src.feature_pipeline import (
    prepare_hourly_timeseries,
    build_forecasting_features,
    chronological_split
)
from river_forecasting.src.baseline_models import (
    PersistenceBaselineModel,
    WaterLevelOnlyForecaster,
    RainfallRunoffForecaster
)
from river_forecasting.src.evaluate import compute_hydrological_metrics, evaluate_forecasts


class TestCatchmentBoundarySpec(unittest.TestCase):
    def test_benchmark_area(self):
        self.assertAlmostEqual(VishnupuriCatchmentSpecification.VERIFIED_DRAINAGE_AREA_SQ_KM, 51390.78, places=2)
        self.assertAlmostEqual(VishnupuriCatchmentSpecification.POUR_POINT_LAT, 19.126944, places=5)
        self.assertAlmostEqual(VishnupuriCatchmentSpecification.POUR_POINT_LON, 77.280833, places=5)

    def test_area_validation(self):
        val_pass = VishnupuriCatchmentSpecification.validate_polygon_geometry(51400.0)
        self.assertTrue(val_pass["is_valid"])
        self.assertEqual(val_pass["status"], "VALIDATED")

        val_fail = VishnupuriCatchmentSpecification.validate_polygon_geometry(5000.0)
        self.assertFalse(val_fail["is_valid"])
        self.assertEqual(val_fail["status"], "AREA_OUT_OF_BOUNDS")


class TestMosdacOrderSpec(unittest.TestCase):
    def test_order_estimates(self):
        plan = MosdacOrderSpecification.get_order_plan()
        self.assertEqual(plan["product_details"]["product_id"], "3SIMG_L2B_HEM_V01R00")
        self.assertGreater(plan["order_window"]["expected_file_count"], 2000)
        self.assertGreater(plan["spatial_subset_analysis"]["full_disk_storage_estimate_gb"], 15.0)


class TestFeaturePipeline(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-07-04 00:00", periods=50, freq="1h")
        stages = 350.0 + np.sin(np.linspace(0, 3, 50)) * 2.0
        self.sample_df = pd.DataFrame({
            "timestamp_ist": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "stage_m": stages,
            "quality_flag": ["VALID"] * 50
        })

    def test_alignment_and_lags(self):
        aligned = prepare_hourly_timeseries(self.sample_df)
        self.assertEqual(len(aligned), 50)

        feat_df, feat_cols, target_cols = build_forecasting_features(
            aligned, horizons=[1, 2, 3], stage_lags=[1, 2, 3], rolling_windows=[3, 6]
        )
        self.assertIn("feat_stage_t", feat_cols)
        self.assertIn("feat_stage_lag_1h", feat_cols)
        self.assertIn("target_stage_1h", target_cols)

        # Verify no future lookahead in lag_1h: at index 5, lag_1h should equal stage at index 4
        self.assertAlmostEqual(feat_df.loc[5, "feat_stage_lag_1h"], feat_df.loc[4, "feat_stage_t"])

    def test_no_future_leakage_in_features(self):
        aligned = prepare_hourly_timeseries(self.sample_df)
        feat_df1, feat_cols, _ = build_forecasting_features(aligned, horizons=[1, 2])

        # Create modified copy with altered future values after index 20
        aligned_mod = aligned.copy()
        aligned_mod.loc[25:, "stage_m"] = aligned_mod.loc[25:, "stage_m"] + 10.0
        feat_df2, _, _ = build_forecasting_features(aligned_mod, horizons=[1, 2])

        # Features at or before index 20 must be exactly identical
        np.testing.assert_array_almost_equal(
            feat_df1.loc[:20, feat_cols].to_numpy(),
            feat_df2.loc[:20, feat_cols].to_numpy()
        )

    def test_missing_hours_in_origin(self):
        # Create gap at index 15: missing observation
        df_gap = self.sample_df.copy()
        df_gap.drop(index=15, inplace=True)
        aligned = prepare_hourly_timeseries(df_gap)

        self.assertEqual(len(aligned), 50)
        self.assertFalse(aligned.loc[15, "is_observed"])

        feat_df, feat_cols, _ = build_forecasting_features(aligned, horizons=[1, 2])
        # Origin at index 15 should NOT be a valid forecast origin
        self.assertFalse(feat_df.loc[15, "is_valid_forecast_origin"])

    def test_missing_hours_in_target(self):
        # Missing observation at index 20
        df_gap = self.sample_df.copy()
        df_gap.drop(index=20, inplace=True)
        aligned = prepare_hourly_timeseries(df_gap)

        feat_df, _, target_cols = build_forecasting_features(aligned, horizons=[1, 2])

        # For origin at index 19, 1h target is index 20 (which is missing)
        self.assertTrue(np.isnan(feat_df.loc[19, "target_stage_1h"]))
        # For origin at index 18, 2h target is index 20 (which is missing)
        self.assertTrue(np.isnan(feat_df.loc[18, "target_stage_2h"]))

    def test_embargo_buffer_split(self):
        aligned = prepare_hourly_timeseries(self.sample_df)
        feat_df, feat_cols, target_cols = build_forecasting_features(aligned, horizons=[1, 2, 3])
        train_df, val_df, test_df, split_info = chronological_split(
            feat_df, train_frac=0.6, val_frac=0.2, horizon_embargo_hours=3
        )

        train_max = train_df["dt"].max()
        val_min = val_df["dt"].min()
        val_max = val_df["dt"].max()
        test_min = test_df["dt"].min()

        # Enforce 3-hour gap between train max and val min
        self.assertGreaterEqual((val_min - train_max).total_seconds() / 3600.0, 3.0)
        # Enforce 3-hour gap between val max and test min
        self.assertGreaterEqual((test_min - val_max).total_seconds() / 3600.0, 3.0)


class TestModelsAndEvaluation(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-07-04 00:00", periods=70, freq="1h")
        stages = 350.0 + np.linspace(0, 3, 70)
        self.sample_df = pd.DataFrame({
            "timestamp_ist": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "stage_m": stages,
            "quality_flag": ["VALID"] * 70
        })
        aligned = prepare_hourly_timeseries(self.sample_df)
        self.feat_df, self.feat_cols, self.target_cols = build_forecasting_features(aligned, horizons=[1, 2, 3])

    def test_persistence_model_multi_horizons(self):
        model = PersistenceBaselineModel(horizons=[1, 2, 3])
        preds = model.predict(self.feat_df)
        for h in [1, 2, 3]:
            np.testing.assert_array_equal(preds[h], self.feat_df["feat_stage_t"].to_numpy())

    def test_water_level_only_ridge_and_gb(self):
        train_df = self.feat_df.iloc[:45].copy()
        test_df = self.feat_df.iloc[45:].copy()

        # Ridge
        model_ridge = WaterLevelOnlyForecaster(horizons=[1, 2, 3], model_type="ridge")
        model_ridge.fit(train_df, self.feat_cols)
        preds_ridge = model_ridge.predict(test_df)
        self.assertIn(1, preds_ridge)
        self.assertEqual(len(preds_ridge[1]), len(test_df))

        # Gradient Boosting
        model_gb = WaterLevelOnlyForecaster(horizons=[1, 2, 3], model_type="gradient_boosting")
        model_gb.fit(train_df, self.feat_cols)
        preds_gb = model_gb.predict(test_df)
        self.assertIn(1, preds_gb)
        self.assertEqual(len(preds_gb[1]), len(test_df))

    def test_rainfall_runoff_protection(self):
        # Strict rule: RainfallRunoffForecaster must not fit without real rainfall features
        model = RainfallRunoffForecaster(horizons=[1, 2])
        train_df = self.feat_df.iloc[:45].copy()
        with self.assertRaises(RuntimeError):
            model.fit(train_df, self.feat_cols)

    def test_hydrological_metrics(self):
        yt = np.array([350.0, 351.0, 352.0, 353.0])
        yp = np.array([350.1, 350.9, 352.0, 353.2])
        metrics = compute_hydrological_metrics(yt, yp)
        self.assertAlmostEqual(metrics["mae_m"], 0.1, places=3)
        self.assertGreater(metrics["nse"], 0.95)

    def test_rapid_rise_regime_distinction(self):
        # Create series with a surge
        dates = pd.date_range("2024-07-04 00:00", periods=20, freq="1h")
        stages = np.array([350.0] * 10 + [350.3, 350.6, 350.9] + [350.9] * 7)
        df_surge = pd.DataFrame({
            "timestamp_ist": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "stage_m": stages,
            "quality_flag": ["VALID"] * 20
        })
        aligned = prepare_hourly_timeseries(df_surge)
        feat_df, feat_cols, target_cols = build_forecasting_features(aligned, horizons=[1, 2])

        model = PersistenceBaselineModel(horizons=[1, 2])
        eval_res = evaluate_forecasts(model, feat_df, horizons=[1], rapid_rise_threshold_m_hr=0.15)

        h1 = eval_res["horizons"]["1h"]
        self.assertIn("antecedent_rapid_rise", h1)
        self.assertIn("antecedent_rapid_change", h1)
        self.assertIn("future_rapid_rise", h1)

        # Antecedent rise occurs when diff_1h >= 0.15
        self.assertGreater(h1["antecedent_rapid_rise"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
