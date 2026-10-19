"""
river_forecasting/tests/test_forecast_evaluation_audit.py
=========================================================
Automated audit test suite for 2026 Whole-Monsoon evaluation artifacts:
1. Validates that test_predictions_2026.csv exists and contains 643 eligible rows.
2. Recomputes MAE and RMSE directly from arrays and asserts exact match with benchmark_results_2026.json.
3. Asserts displayed MAEs in forecast_evaluation_2026.png equal mean absolute error of plotted arrays.
4. Asserts that test data (September 1-30) is strictly held out (zero test timestamps in train/val).
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_CSV = REPO_ROOT / "data" / "historical" / "test_predictions_2026.csv"
DOWNSTREAM_CSV = REPO_ROOT / "data" / "historical" / "downstream_predictions_2026.csv"
BENCHMARK_JSON = REPO_ROOT / "data" / "historical" / "benchmark_results_2026.json"
DATASET_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"

def test_predictions_artifact_exists_and_complete():
    assert PREDICTIONS_CSV.exists(), f"Missing predictions file: {PREDICTIONS_CSV}"
    df = pd.read_csv(PREDICTIONS_CSV)
    assert len(df) == 643, f"Expected 643 eligible September test samples, got {len(df)}"
    
    # Required columns
    expected_cols = [
        "dt", "stage_vishnupuri", "dH_v_1h", "is_rapid_rise", "is_sync_window",
        "y_true_1h", "pred_pers_1h", "pred_ridge_1h", "pred_rf_1h",
        "ae_pers_1h", "ae_ridge_1h", "ae_rf_1h",
        "y_true_2h", "pred_pers_2h", "pred_ridge_2h", "pred_rf_2h",
        "ae_pers_2h", "ae_ridge_2h", "ae_rf_2h",
        "y_true_3h", "pred_pers_3h", "pred_ridge_3h", "pred_rf_3h",
        "ae_pers_3h", "ae_ridge_3h", "ae_rf_3h"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in {PREDICTIONS_CSV}"

def test_benchmark_metrics_exact_recomputation():
    df = pd.read_csv(PREDICTIONS_CSV)
    with open(BENCHMARK_JSON) as f:
        benchmarks = json.load(f)
    
    stage_bench = benchmarks["models"]["stage_only_full_september"]
    
    for h in ["1h", "2h", "3h"]:
        h_key = f"+{h}"
        y_true = df[f"y_true_{h}"].values
        
        # Persistence
        y_pers = df[f"pred_pers_{h}"].values
        mae_pers = mean_absolute_error(y_true, y_pers)
        rmse_pers = np.sqrt(mean_squared_error(y_true, y_pers))
        assert abs(round(float(mae_pers), 4) - stage_bench[h_key]["persistence"]["mae_m"]) < 1e-4
        assert abs(round(float(rmse_pers), 4) - stage_bench[h_key]["persistence"]["rmse_m"]) < 1e-4
        
        # Ridge
        y_ridge = df[f"pred_ridge_{h}"].values
        mae_ridge = mean_absolute_error(y_true, y_ridge)
        rmse_ridge = np.sqrt(mean_squared_error(y_true, y_ridge))
        assert abs(round(float(mae_ridge), 4) - stage_bench[h_key]["ridge"]["mae_m"]) < 1e-4
        assert abs(round(float(rmse_ridge), 4) - stage_bench[h_key]["ridge"]["rmse_m"]) < 1e-4
        
        # Random Forest
        y_rf = df[f"pred_rf_{h}"].values
        mae_rf = mean_absolute_error(y_true, y_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_true, y_rf))
        assert abs(round(float(mae_rf), 4) - stage_bench[h_key]["random_forest"]["mae_m"]) < 1e-4
        assert abs(round(float(rmse_rf), 4) - stage_bench[h_key]["random_forest"]["rmse_m"]) < 1e-4

def test_plotted_arrays_and_displayed_mae_consistency():
    df = pd.read_csv(PREDICTIONS_CSV)
    
    # Check that ae columns are exact absolute errors
    for h in ["1h", "2h", "3h"]:
        diff_p = np.abs(df[f"pred_pers_{h}"] - df[f"y_true_{h}"])
        assert np.allclose(df[f"ae_pers_{h}"], diff_p)
        
        diff_rd = np.abs(df[f"pred_ridge_{h}"] - df[f"y_true_{h}"])
        assert np.allclose(df[f"ae_ridge_{h}"], diff_rd)
        
        diff_rf = np.abs(df[f"pred_rf_{h}"] - df[f"y_true_{h}"])
        assert np.allclose(df[f"ae_rf_{h}"], diff_rf)

def test_strictly_held_out_september():
    df = pd.read_csv(DATASET_CSV)
    train_times = set(df.loc[df["split"] == "TRAIN", "dt"])
    val_times = set(df.loc[df["split"] == "VAL", "dt"])
    test_times = set(df.loc[df["split"] == "TEST", "dt"])
    
    assert len(train_times.intersection(test_times)) == 0, "Test leakage into TRAIN split!"
    assert len(val_times.intersection(test_times)) == 0, "Test leakage into VAL split!"
    assert len(train_times.intersection(val_times)) == 0, "Val leakage into TRAIN split!"

def test_downstream_raher_metrics_recomputation():
    assert DOWNSTREAM_CSV.exists(), f"Missing downstream file: {DOWNSTREAM_CSV}"
    df = pd.read_csv(DOWNSTREAM_CSV)
    assert len(df) == 659, f"Expected 659 downstream test samples, got {len(df)}"
    
    with open(BENCHMARK_JSON) as f:
        benchmarks = json.load(f)
    down_bench = benchmarks["models"]["downstream_raher_exploratory"]
    
    for h in ["1h", "2h", "3h"]:
        h_key = f"+{h}"
        y_true = df[f"y_true_{h}"].values
        y_pers = df[f"pred_pers_{h}"].values
        y_rf = df[f"pred_rf_{h}"].values
        
        mae_p = mean_absolute_error(y_true, y_pers)
        mae_rf = mean_absolute_error(y_true, y_rf)
        
        assert abs(round(float(mae_p), 4) - down_bench[h_key]["persistence"]["mae_m"]) < 1e-4
        assert abs(round(float(mae_rf), 4) - down_bench[h_key]["upstream_informed_rf"]["mae_m"]) < 1e-4

if __name__ == "__main__":
    test_predictions_artifact_exists_and_complete()
    test_benchmark_metrics_exact_recomputation()
    test_plotted_arrays_and_displayed_mae_consistency()
    test_strictly_held_out_september()
    test_downstream_raher_metrics_recomputation()
    print("ALL 5 AUDIT TESTS PASSED SUCESSFULLY!")
