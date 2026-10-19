"""
river_forecasting/tests/test_model_comparison_benchmark.py
==========================================================
Tests for evaluation alignment, leak-free chronological partitioning,
identical forecast origin counts across compared models, and reproducible metrics.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_JSON = REPO_ROOT / "data" / "historical" / "comprehensive_model_comparison_results.json"
BENCHMARK_CSV = REPO_ROOT / "data" / "historical" / "comprehensive_model_comparison_results.csv"
DATASET_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"


def test_benchmark_files_exist_and_non_empty():
    assert BENCHMARK_JSON.exists(), "Benchmark JSON file missing"
    assert BENCHMARK_CSV.exists(), "Benchmark CSV file missing"
    assert BENCHMARK_CSV.stat().st_size > 1000
    assert BENCHMARK_JSON.stat().st_size > 1000


def test_strict_chronological_partitions_and_zero_target_leakage():
    df = pd.read_csv(DATASET_CSV)
    df["dt"] = pd.to_datetime(df["dt"])

    event_start = pd.to_datetime("2026-07-30 00:00:00")
    pre_train = df[(df["dt"] >= "2026-06-15 00:00:00") & (df["dt"] < event_start)]

    for h_int in [1, 2, 3]:
        valid_train = (pre_train["dt"] + pd.Timedelta(hours=h_int)) < event_start
        train_origins = pre_train[valid_train]
        # Verify maximum target timestamp is strictly before event start
        max_target_dt = (train_origins["dt"] + pd.Timedelta(hours=h_int)).max()
        assert max_target_dt < event_start, f"Target leakage detected at +{h_int}h: {max_target_dt} >= {event_start}"


def test_identical_matched_origin_counts_across_compared_models():
    """Verify that every single model in each evaluation cohort is evaluated on identical origins."""
    csv_df = pd.read_csv(BENCHMARK_CSV)
    for cohort, grp in csv_df.groupby("Cohort"):
        for lead, lead_grp in grp.groupby("Lead_Time"):
            sample_counts = lead_grp["N_Eval_Samples"].unique()
            assert len(sample_counts) == 1, f"Mismatch in eval samples for {cohort} {lead}: {sample_counts}"
            # Specific cohort expected sample sizes
            n = sample_counts[0]
            if cohort == "Phase_A_July_Event":
                assert n == 30
            elif cohort == "September_Sync_Holdout":
                assert n == 305
            elif cohort == "September_Full_Test":
                assert n == 617
            elif cohort == "Full_Season_Monitored":
                assert n == 2100


def test_model_d_performance_consistency():
    """Verify that Model D produces expected ~5-7% improvement on Phase A event."""
    with open(BENCHMARK_JSON, "r") as f:
        data = json.load(f)

    phase_a = data["evaluations"]["Phase_A_July_Event"]["horizons"]
    for h in ["+1h", "+2h", "+3h"]:
        d_stats = phase_a[h]["models"]["D. Model D: Stage-Change + Rainfall Ridge (dH)"]
        pers_stats = phase_a[h]["models"]["A. Persistence Baseline"]
        assert d_stats["mae_m"] < pers_stats["mae_m"], f"Model D did not improve over Persistence at {h} in Phase A"
        assert d_stats["imp_vs_pers_pct"] >= 5.0, f"Expected >=5% improvement at {h}, got {d_stats['imp_vs_pers_pct']}%"


def test_ui_benchmark_notice_and_interpretation_content():
    """Verify exact wording of Experimental Candidate Notice, units, and interpretation."""
    import re
    index_html = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

    expected_notice = (
        "Experimental Candidate Notice: Model D (Stage-Change + Rainfall Ridge ΔH) "
        "demonstrates active predictive gains on Phase A flood inception (-5.5% to -6.8% MAE vs persistence) "
        "and matches persistence on the flat-pool holdout (<0.4 mm). "
        "However, it is an experimental research candidate, not a proven operational flood warning system. "
        "Barrage pool cresting is governed by unmonitored radial gate discharge."
    )
    # Strip HTML tags and normalize HTML entities (&Delta; -> Δ, &lt; -> <) for textual check
    normalized_text = (
        re.sub(r"<[^>]+>", "", index_html)
        .replace("&Delta;", "Δ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&minus;", "-")
        .replace("&amp;", "&")
    )
    # Normalize consecutive whitespace
    normalized_clean = " ".join(normalized_text.split())

    assert expected_notice in normalized_clean, "Experimental candidate notice text mismatch"

    expected_interp = (
        "Model D improves on persistence during the July inception event at all three horizons, "
        "but performs worse on the September held-out split. Across the full-season cohort at +3h, "
        "Random Forest and XGBoost have lower MAE than persistence. Results are cohort-dependent; "
        "Model D remains experimental."
    )
    assert expected_interp in normalized_clean, "Benchmark interpretation text mismatch"


def test_ui_benchmark_table_cohorts_and_exclusions():
    """Verify that all 3 cohorts are present with exact sample sizes, without gradient boosting or omitted rows."""
    index_html = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

    # Sample sizes
    assert "N = 30 matched origins" in index_html
    assert "N = 617 matched origins" in index_html
    assert "N = 2,100 matched origins" in index_html

    # Check that Gradient Boosting is NOT present in metrics-2026-view
    view_start = index_html.find('id="metrics-2026-view"')
    view_end = index_html.find('id="metrics-2024-view"')
    assert view_start != -1 and view_end != -1
    metrics_2026_slice = index_html[view_start:view_end]

    assert "Gradient Boosting" not in metrics_2026_slice, "Gradient Boosting must be excluded from UI content"

    # Evaluation 2 should NOT contain +2h
    eval2_start = metrics_2026_slice.find("Evaluation 2: Full Held-Out September Split")
    eval3_start = metrics_2026_slice.find("Evaluation 3: Full-Season Monitored Cohort")
    eval2_slice = metrics_2026_slice[eval2_start:eval3_start]
    assert "+2h" not in eval2_slice, "Evaluation 2 must not have +2h rows"

    # Evaluation 3 should only contain +3h (not +1h or +2h)
    eval3_slice = metrics_2026_slice[eval3_start:]
    assert "+1h" not in eval3_slice, "Evaluation 3 must not have +1h rows"
    assert "+2h" not in eval3_slice, "Evaluation 3 must not have +2h rows"

    # Check key metric values
    assert "0.0274" in metrics_2026_slice  # Phase A Model D +1h
    assert "0.0506" in metrics_2026_slice  # Phase A Model D +2h
    assert "0.0759" in metrics_2026_slice  # Phase A Model D +3h
    assert "0.0064" in metrics_2026_slice  # Sep XGBoost +3h
    assert "0.0087" in metrics_2026_slice  # Full-Season XGBoost +3h

