"""
river_forecasting/tests/test_multivariable_forecasting.py
=========================================================
Automated test suite verifying the multivariable forecasting research upgrade
for Vishnupuri Barrage (+1h, +2h, +3h).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.multivariable_pipeline import (
    get_feature_audit_matrix,
    prepare_multivariable_dataset,
    MultivariableModelComparator,
    FRL_ELEVATION_M,
    REGIME_HIGH_POOL_THRESHOLD_M
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_JSON = REPO_ROOT / "data" / "historical" / "multivariable_benchmark_results.json"
AUDIT_JSON = REPO_ROOT / "data" / "historical" / "multivariable_feature_audit.json"


def test_feature_audit_matrix_completeness():
    matrix = get_feature_audit_matrix()
    assert len(matrix) >= 14, f"Expected >= 14 audited features, got {len(matrix)}"

    # Check that unavailable features are explicitly marked
    unavail_names = ["ground_rain_imd", "nwp_rainfall_forecasts", "gate_openings_and_discharge"]
    for u_name in unavail_names:
        item = next((f for f in matrix if f["feature_name"] == u_name), None)
        assert item is not None, f"Missing audit item: {u_name}"
        assert item["available_at_t"] is False
        assert item["season_coverage_pct"] == 0.0


def test_multivariable_dataset_causality_and_regimes():
    df = prepare_multivariable_dataset()
    assert len(df) == 2592
    assert "dH_v_6h" in df.columns
    assert "accel_v_1h" in df.columns
    assert "stage_mudgal" in df.columns
    assert "freeboard_frl_m" in df.columns
    assert "api_7d_mm" in df.columns

    # Test FRL freeboard
    assert np.allclose(
        df["freeboard_frl_m"].dropna().values[:10],
        (FRL_ELEVATION_M - df["stage_vishnupuri"].dropna().values[:10])
    )

    # Test API non-negativity
    valid_api = df["api_7d_mm"].dropna()
    assert (valid_api >= 0.0).all()


def test_stage_change_model_beats_persistence_on_phase_a():
    """Verify that Model D (Stage Change Increment dH) beats Persistence across all horizons on Phase A."""
    assert BENCHMARK_JSON.exists(), f"Missing benchmark JSON: {BENCHMARK_JSON}"
    with open(BENCHMARK_JSON) as f:
        res = json.load(f)

    phase_a_res = res["phase_a_event_eval"]
    for h in ["+1h", "+2h", "+3h"]:
        h_data = phase_a_res[h]
        mae_pers = h_data["A_Persistence"]["mae_m"]
        mae_lvl_ridge = h_data["C_Multimodal_Ridge_Level"]["mae_m"]
        mae_dh_ridge = h_data["D_Multimodal_Ridge_dH"]["mae_m"]

        # 1. Model D strictly beats Persistence
        assert mae_dh_ridge < mae_pers, f"Model D failed to beat Persistence at {h} ({mae_dh_ridge} >= {mae_pers})"
        # 2. Model D drastically outperforms Level Ridge
        assert mae_dh_ridge < mae_lvl_ridge * 0.85, f"Model D failed to beat Level Ridge at {h}"


def test_ablation_shows_rainfall_value_over_stage_dynamics_only():
    """Verify that rainfall features provide incremental gain over stage dynamics alone."""
    with open(BENCHMARK_JSON) as f:
        res = json.load(f)

    ab_res = res["ablation_study"]
    for h in ["+1h", "+2h", "+3h"]:
        h_ab = ab_res[h]
        mae_st_only = h_ab["2_StageDynamicsOnly"]["mae_m"]
        mae_st_rain = h_ab["3_StageDyn_Plus_Rainfall"]["mae_m"]
        mae_pers = h_ab["1_Persistence"]["mae_m"]

        # Stage dynamics alone does NOT beat persistence
        assert mae_st_only >= mae_pers
        # Adding rainfall beats both stage dynamics only and persistence
        assert mae_st_rain < mae_st_only
        assert mae_st_rain < mae_pers


def test_september_stability_and_no_level_shrinkage():
    """Verify that Model D achieves < 9mm MAE on September holdout without shrinkage bias."""
    with open(BENCHMARK_JSON) as f:
        res = json.load(f)

    sep_res = res["september_sync_eval"]
    for h in ["+1h", "+2h", "+3h"]:
        h_data = sep_res[h]
        mae_dh_ridge = h_data["D_Multimodal_Ridge_dH"]["mae_m"]
        mae_lvl_ridge = h_data["C_Multimodal_Ridge_Level"]["mae_m"]

        # Level model has massive ~4-6cm shrinkage error
        assert mae_lvl_ridge > 0.040
        # Stage change model achieves < 9mm error
        assert mae_dh_ridge < 0.009
