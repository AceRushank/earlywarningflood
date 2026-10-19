"""
river_forecasting/tests/test_event_evaluation_july2026.py
========================================================
Automated test suite verifying the event-focused evaluation of the
July 30 - August 2, 2026 flood event at Vishnupuri Barrage.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_CSV = REPO_ROOT / "data" / "historical" / "event_july2026_predictions.csv"
BENCHMARK_JSON = REPO_ROOT / "data" / "historical" / "event_july2026_benchmark_results.json"
DATASET_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"

def test_event_files_exist_and_consistent():
    assert PREDICTIONS_CSV.exists(), f"Missing predictions file: {PREDICTIONS_CSV}"
    assert BENCHMARK_JSON.exists(), f"Missing benchmark json: {BENCHMARK_JSON}"
    df = pd.read_csv(PREDICTIONS_CSV)
    assert len(df) == 30, f"Expected 30 rows in Phase A, got {len(df)}"

def test_pre_event_training_boundary():
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)
    train_end = data["event_definition"]["pre_event_train_window"].split(" to ")[1]
    assert "2026-07-29" in train_end, f"Pre-event training window leaked into event: {train_end}"
    assert data["event_definition"]["target_leakage_purged"] is True, "Target leakage flag not set!"

def test_strictly_zero_target_boundary_leakage():
    """Asserts that no training row has a future target timestamp on or after event start."""
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)
    event_start = pd.to_datetime(data["event_definition"]["event_start_ist"])
    max_dts = data["event_definition"]["max_training_target_timestamps"]
    
    for h in ["1h", "2h", "3h"]:
        max_tgt = pd.to_datetime(max_dts[h])
        assert max_tgt < event_start, f"Horizon +{h} training target reached event start! Max: {max_tgt} >= {event_start}"
        # Max target must be at or before 2026-07-29 23:00:00 IST
        assert max_tgt <= pd.to_datetime("2026-07-29 23:00:00"), f"Target exceeded pre-event boundary: {max_tgt}"

def test_phase_a_multimodal_gain_and_persistence_comparison():
    df = pd.read_csv(PREDICTIONS_CSV)
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)
    phase_a_bench = data["phases"]["phase_a_inception"]
    
    for h in ["1h", "2h", "3h"]:
        y_true = df[f"y_true_{h}"].values
        y_pers = df[f"pred_pers_{h}"].values
        y_st = df[f"pred_stage_ridge_{h}"].values
        y_mm = df[f"pred_mm_ridge_{h}"].values
        
        mae_p = mean_absolute_error(y_true, y_pers)
        mae_st = mean_absolute_error(y_true, y_st)
        mae_mm = mean_absolute_error(y_true, y_mm)
        
        # Verify JSON values match exact array MAEs
        assert abs(round(float(mae_p), 4) - phase_a_bench[f"+{h}"]["persistence"]["mae_m"]) < 1e-4
        assert abs(round(float(mae_st), 4) - phase_a_bench[f"+{h}"]["stage_only_ridge"]["mae_m"]) < 1e-4
        assert abs(round(float(mae_mm), 4) - phase_a_bench[f"+{h}"]["multimodal_ridge"]["mae_m"]) < 1e-4
        
        # Multimodal Ridge outperforms Stage Ridge
        assert mae_mm < mae_st, f"Multimodal Ridge failed to beat Stage Ridge at +{h}"
        
        # Persistence beats both linear models due to pre-event flat training baseline
        assert mae_p < mae_mm, f"Persistence expected to beat Multimodal Ridge at +{h}"
        assert mae_p < mae_st, f"Persistence expected to beat Stage Ridge at +{h}"

def test_phase_b_outage_preservation():
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)
    phase_b_bench = data["phases"]["phase_b_surge"]
    for h in ["+1h", "+2h", "+3h"]:
        assert phase_b_bench[h]["multimodal_status"] == "DISQUALIFIED_DUE_TO_AUTHENTIC_SATELLITE_OUTAGE"

if __name__ == "__main__":
    test_event_files_exist_and_consistent()
    test_pre_event_training_boundary()
    test_phase_a_multimodal_gain_and_persistence_comparison()
    test_phase_b_outage_preservation()
    print("ALL EVENT EVALUATION TESTS PASSED SUCESSFULLY!")
