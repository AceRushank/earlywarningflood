"""
river_forecasting/tests/test_monsoon_event_evaluation.py
=========================================================
Automated test suite verifying the full-monsoon 2026 rising-stage
episode evaluation at Vishnupuri Barrage.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.monsoon_event_evaluation import (
    DATASET_CSV,
    identify_rising_episodes,
    fit_pre_event_models,
    evaluate_episode,
    audit_stage_datum,
    STAGE_FEATURES,
    MULTIMODAL_FEATURES
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVENTS_SUMMARY_CSV = REPO_ROOT / "data" / "historical" / "monsoon_2026_events_summary.csv"
BENCHMARK_JSON = REPO_ROOT / "data" / "historical" / "monsoon_2026_event_benchmark_results.json"


def test_dataset_exists_and_covers_full_monsoon():
    assert DATASET_CSV.exists(), f"Missing dataset: {DATASET_CSV}"
    df = pd.read_csv(DATASET_CSV)
    df["dt"] = pd.to_datetime(df["dt"])
    assert df["dt"].min() <= pd.to_datetime("2026-06-15 00:00:00")
    assert df["dt"].max() >= pd.to_datetime("2026-09-30 23:00:00")
    assert len(df) == 2592


def test_stage_only_rising_episode_detection():
    df = pd.read_csv(DATASET_CSV)
    episodes = identify_rising_episodes(df, min_net_rise=0.05, min_duration=2)

    assert len(episodes) >= 8, f"Expected at least 8 rising episodes, got {len(episodes)}"
    for ep in episodes:
        assert ep["net_rise_m"] >= 0.05, f"Episode {ep['episode_id']} has rise < 0.05m"
        assert ep["duration_hrs"] >= 2, f"Episode {ep['episode_id']} has duration < 2h"
        assert ep["peak_stage_m"] > ep["start_stage_m"]


def test_complete_case_preserves_outages():
    """Verify that during satellite outages, complete-case evaluation is preserved as None."""
    assert EVENTS_SUMMARY_CSV.exists()
    df_sum = pd.read_csv(EVENTS_SUMMARY_CSV)

    # Surge limb (Episode 6B) must be OUTAGE_OR_INCOMPLETE with None for complete-case MAE
    surge_row = df_sum[df_sum["episode_id"] == "EP_06_B"].iloc[0]
    assert surge_row["mm_status"] == "OUTAGE_OR_INCOMPLETE"
    assert pd.isna(surge_row["mae_mm_1h"])
    assert pd.isna(surge_row["mae_mm_3h"])
    # But persistence and stage-only ridge must have valid predictions
    assert not pd.isna(surge_row["mae_pers_1h"])
    assert not pd.isna(surge_row["mae_stage_1h"])


def test_missingness_aware_sensitivity_reveals_zero_imputation_penalty():
    """Verify that naive zero imputation during surge limb produces severe negative bias."""
    assert EVENTS_SUMMARY_CSV.exists()
    df_sum = pd.read_csv(EVENTS_SUMMARY_CSV)

    surge_row = df_sum[df_sum["episode_id"] == "EP_06_B"].iloc[0]
    # Zero imputation bias at +3h must be strongly negative (underpredicting surge by > 0.5m)
    assert surge_row["zero_impute_bias_3h"] < -0.50
    # Zero imputation MAE must be substantially higher than persistence
    assert surge_row["zero_impute_mae_3h"] > surge_row["mae_pers_3h"] * 1.3


def test_persistence_beats_linear_models_pooled():
    """Verify that persistence achieves lower error than linear models across pooled episodes."""
    assert EVENTS_SUMMARY_CSV.exists()
    df_sum = pd.read_csv(EVENTS_SUMMARY_CSV)
    complete = df_sum[df_sum["mm_status"] == "COMPLETE"]

    for h in ["1h", "3h"]:
        p_w = np.average(complete[f"mae_pers_{h}"], weights=complete["duration_hrs"])
        st_w = np.average(complete[f"mae_stage_{h}"], weights=complete["duration_hrs"])
        mm_w = np.average(complete[f"mae_mm_{h}"], weights=complete["duration_hrs"])

        # Multimodal Ridge outperforms Stage Ridge on complete rising cases
        assert mm_w < st_w, f"Multimodal Ridge failed to beat Stage Ridge pooled at +{h}"
        # But Persistence outperforms both linear models
        assert p_w < mm_w, f"Persistence failed to beat Multimodal Ridge pooled at +{h}"
        assert p_w < st_w, f"Persistence failed to beat Stage Ridge pooled at +{h}"


def test_stage_datum_audit():
    audit = audit_stage_datum()
    assert audit["official_nwic_metadata"]["MeanSeaLevel_field_val"] == 366.5
    assert "structural" in audit["datum_resolution"]
    assert "Is_DischargeDataAvailable" in audit["unresolved_operational_uncertainty"]
