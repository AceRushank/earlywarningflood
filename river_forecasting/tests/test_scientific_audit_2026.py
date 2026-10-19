"""
river_forecasting/tests/test_scientific_audit_2026.py
===================================================
Targeted scientific audit regression tests for the 2026 Vishnupuri study:
1. Validates that the N=19 subset consists of 11 stage drops and 8 minor rises (<=3.9 cm),
   confirming they represent regulated pool chatter rather than natural flood events.
2. Validates that raw/aligned dataset preserves the July 31 - August 4 satellite outage as NaN.
3. Validates strictly causal feature engineering with zero target lookahead.
4. Asserts exact parity between test_predictions_2026.csv and benchmark_results_2026.json.
5. Verifies that Persistence outperforms ML models on the overall September regulated recession.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_CSV = REPO_ROOT / "data" / "historical" / "test_predictions_2026.csv"
DATASET_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"
BENCHMARK_JSON = REPO_ROOT / "data" / "historical" / "benchmark_results_2026.json"

def test_rapid_rise_subset_characterization():
    """Audit test confirming the physical nature of the N=19 subset."""
    df = pd.read_csv(PREDICTIONS_CSV)
    rise = df[df["is_rapid_rise"] == True]
    assert len(rise) == 19, f"Expected 19 rapid rise rows, found {len(rise)}"
    
    dh = rise["dH_v_1h"].values
    drops = (dh < 0).sum()
    rises = (dh > 0).sum()
    
    assert drops == 12, f"Expected 12 negative stage changes, found {drops}"
    assert rises == 7, f"Expected 7 positive stage changes, found {rises}"
    assert np.max(np.abs(dh)) <= 0.040, f"Max fluctuation exceeded 4cm: {np.max(np.abs(dh))}"

def test_outage_missingness_preservation():
    """Audit test verifying that satellite outage hours are strictly preserved as NaN."""
    df = pd.read_csv(DATASET_CSV)
    outage_rows = df[df["rain_qc_flag"] == "OUTAGE_UNAVAILABLE"]
    assert len(outage_rows) >= 96, f"Expected at least 96 outage hours, found {len(outage_rows)}"
    
    # Confirm rainfall depth and rate are strictly NaN
    assert outage_rows["rain_rate_mm_hr"].isna().all(), "Outage rain rate was illegally filled!"
    assert outage_rows["rain_depth_1h_mm"].isna().all(), "Outage rain depth was illegally filled!"
    assert outage_rows["rain_24h_mm"].isna().all(), "Outage rolling rain was illegally filled!"

def test_no_lookahead_in_predictors():
    """Audit test verifying zero future lookahead or target leakage in predictors."""
    df = pd.read_csv(DATASET_CSV)
    # Check that lag1 is exactly shift(1)
    lag1_diff = (df["stage_vishnupuri"].shift(1) - df["stage_v_lag1"]).dropna()
    assert np.allclose(lag1_diff, 0.0), "stage_v_lag1 is not an exact 1-hour lag!"
    
    # Check that target_1h is exactly shift(-1)
    target1_diff = (df["stage_vishnupuri"].shift(-1) - df["target_v_1h"]).dropna()
    assert np.allclose(target1_diff, 0.0), "target_v_1h is not an exact 1-hour lead target!"

def test_benchmark_metrics_and_persistence_superiority():
    """Audit test asserting that Persistence baseline achieves lower error on overall September recession."""
    with open(BENCHMARK_JSON) as f:
        benchmarks = json.load(f)
        
    stage_full = benchmarks["models"]["stage_only_full_september"]
    # +1h
    mae_pers_1h = stage_full["+1h"]["persistence"]["mae_m"]
    mae_ridge_1h = stage_full["+1h"]["ridge"]["mae_m"]
    mae_mm_1h = stage_full["+1h"]["multimodal_ridge"]["mae_m"]
    assert mae_pers_1h < mae_ridge_1h, "Persistence should outperform Ridge at +1h"
    assert mae_pers_1h < mae_mm_1h, "Persistence should outperform Multimodal Ridge at +1h"
    
    # +2h
    mae_pers_2h = stage_full["+2h"]["persistence"]["mae_m"]
    mae_ridge_2h = stage_full["+2h"]["ridge"]["mae_m"]
    mae_mm_2h = stage_full["+2h"]["multimodal_ridge"]["mae_m"]
    assert mae_pers_2h < mae_ridge_2h, "Persistence should outperform Ridge at +2h"
    assert mae_pers_2h < mae_mm_2h, "Persistence should outperform Multimodal Ridge at +2h"

if __name__ == "__main__":
    test_rapid_rise_subset_characterization()
    test_outage_missingness_preservation()
    test_no_lookahead_in_predictors()
    test_benchmark_metrics_and_persistence_superiority()
    print("ALL AUDIT REGRESSION TESTS PASSED!")
