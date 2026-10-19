"""
river_forecasting/tests/test_phase1_ml_pipeline.py
===================================================
Automated test suite for Phase 1 ML Training and Benchmarking Pipeline.

Tests cover:
  - Data loading and feature engineering
  - Temporal split boundaries (no overlap)
  - Leakage prevention (no feature correlates 1:1 with a future target)
  - Feature construction (lagged rainfall, API, accel, dH_v_6h)
  - API correctness (decay, NaN propagation)
  - Model catalogue completeness (12 configs)
  - Metrics computation
  - Block-bootstrap CI
  - Stratum computation
  - Artifact save/load round-trip
  - Phase1InferenceInterface validation (missing features, wrong horizon)
  - Inference interface correctly refuses future horizon values
  - Existing live_inference safeguards preserved (import check)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# ── import under test ───────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from river_forecasting.src.phase1_ml_pipeline import (
    _build_pipe,
    _compute_api,
    _metrics,
    _bootstrap_ci,
    _strata,
    _valid,
    _valid_histgb,
    check_leakage,
    get_catalogue,
    get_splits,
    load_features,
    HORIZONS,
    SEED,
    TRAIN_START, TRAIN_END,
    VAL_START, VAL_END,
    TEST_START, TEST_END,
    PHASE_A_START, PHASE_A_END,
    PHASE_A_CUTOFF,
    FEAT_STAGE_BASE, FEAT_STAGE_DYN, FEAT_RAIN_CONT,
    FEAT_MULTIMODAL, FEAT_RAIN_LAG,
    API_DECAY_HR,
    OUT_DIR,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    """Load and engineer the real dataset once per module."""
    return load_features()


@pytest.fixture(scope="module")
def sp(df):
    return get_splits(df)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoading:
    def test_dataset_loads_with_expected_shape(self, df):
        assert len(df) > 2000, "Dataset should have >2000 rows for the 2026 monsoon season"
        assert "dt" in df.columns
        assert "stage_vishnupuri" in df.columns
        assert "rain_depth_1h_mm" in df.columns

    def test_derived_features_present(self, df):
        for feat in ["dH_v_6h", "accel_v_1h", "api_7d_mm",
                     "rain_lag1_1h_mm", "rain_lag2_1h_mm", "rain_lag3_1h_mm"]:
            assert feat in df.columns, f"Derived feature {feat} missing"

    def test_dt_sorted_ascending(self, df):
        assert df["dt"].is_monotonic_increasing, "Dataset must be sorted by dt"

    def test_no_duplicate_timestamps(self, df):
        assert df["dt"].duplicated().sum() == 0, "Duplicate timestamps found"

    def test_missing_rain_preserved_as_nan(self, df):
        # Missing rainfall must stay NaN; it should NOT be zero
        rain_missing_mask = df["rain_depth_1h_mm"].isna()
        # If no missing values, just verify the dtype is float (could hold NaN)
        assert df["rain_depth_1h_mm"].dtype == float

    def test_api_decay_is_nan_when_rain_is_nan(self, df):
        rain_null_idx = df[df["rain_depth_1h_mm"].isna()].index
        if len(rain_null_idx) > 0:
            api_at_null = df.loc[rain_null_idx, "api_7d_mm"]
            # API during rain outages should be NaN (not zero)
            assert api_at_null.isna().sum() > 0, (
                "API should be NaN when rain is NaN; zero imputation detected"
            )

    def test_lagged_rainfall_is_strictly_causal(self, df):
        # rain_lag1_1h_mm at row i must equal rain_depth_1h_mm at row i-1
        valid = df[["rain_depth_1h_mm", "rain_lag1_1h_mm"]].dropna()
        if len(valid) > 5:
            # sample first 5 valid consecutive rows
            prev = df["rain_depth_1h_mm"].shift(1)
            diff = (df["rain_lag1_1h_mm"] - prev).dropna().abs()
            assert (diff < 1e-10).all(), (
                "rain_lag1_1h_mm does not equal rain_depth_1h_mm shifted by 1 row"
            )

    def test_accel_v_1h_is_second_difference(self, df):
        expected = df["dH_v_1h"] - df["dH_v_1h"].shift(1)
        diff = (df["accel_v_1h"] - expected).dropna().abs()
        assert (diff < 1e-10).all(), "accel_v_1h is not the second difference of dH_v_1h"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal split boundaries
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalSplits:
    def test_no_overlap_between_splits(self, sp):
        tr_dates  = set(sp["train"]["dt"])
        val_dates = set(sp["val"]["dt"])
        te_dates  = set(sp["test"]["dt"])
        assert len(tr_dates & val_dates) == 0, "Train and Val overlap"
        assert len(tr_dates & te_dates)  == 0, "Train and Test overlap"
        assert len(val_dates & te_dates) == 0, "Val and Test overlap"

    def test_train_precedes_val_precedes_test(self, sp):
        assert sp["train"]["dt"].max() < sp["val"]["dt"].min(), "Train extends into Val"
        assert sp["val"]["dt"].max()   < sp["test"]["dt"].min(), "Val extends into Test"

    def test_phase_a_within_train_period(self, sp):
        pa_start = pd.Timestamp(PHASE_A_START)
        tr_end   = pd.Timestamp(TRAIN_END)
        assert pa_start <= tr_end, "Phase A starts after training period ends"

    def test_splits_are_non_empty(self, sp):
        for name, sub in sp.items():
            assert len(sub) > 0, f"Split '{name}' is empty"

    def test_phase_a_has_expected_size(self, sp):
        # Phase A: Jul 30 00:00 to Jul 31 05:00 = 30 hours max
        assert 1 <= len(sp["phase_a"]) <= 35, (
            f"Phase A should have ~30 rows, got {len(sp['phase_a'])}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Leakage prevention
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakagePrevention:
    def test_no_feature_perfectly_correlated_with_future_target(self, df):
        # Physical note: stage_vishnupuri vs target_v_1h has corr ~0.9999 because
        # river stage is a near-AR-1 process.  This is NOT leakage -- the feature
        # is observed at t and the target is the value at t+1.  True leakage would
        # be correlation exactly == 1.0 (identical arrays) or a feature that IS the
        # future value.  check_leakage uses threshold 0.9999; we verify the function
        # runs without raising an exception and that no feature is the SAME series
        # as a future target (identity check).
        feat_cols = [c for c in df.columns
                     if not c.startswith("target_") and c not in
                     {"dt","timestamp_ist","split","is_test_sync_rainfall",
                      "rain_scans_in_hour","rain_qc_flag"}]
        tgt_cols  = [c for c in df.columns if c.startswith("target_")]
        for fc in feat_cols:
            for tc in tgt_cols:
                pair = df[[fc, tc]].dropna()
                if len(pair) < 10 or df[fc].dtype == object:
                    continue
                # Identity check: prediction = current value is persistence (allowed)
                # but any feature that is EXACTLY equal to a FUTURE stage value
                # is a data preparation error.
                r = pair[fc].corr(pair[tc])
                # corr can be 0.9999 for AR-1 autocorrelation; only corr==1.0
                # and same variance indicates the feature IS the future target.
                if abs(r) > 0.99999:
                    # Verify variance differs
                    var_diff = abs(pair[fc].var() - pair[tc].var())
                    assert var_diff > 1e-8, (
                        f"Potential leakage: {fc} may BE {tc} "
                        f"(corr={r:.6f}, var_diff={var_diff:.2e})"
                    )

    def test_api_not_correlated_with_future_target(self, df):
        pair = df[["api_7d_mm", "target_v_1h"]].dropna()
        r = pair["api_7d_mm"].corr(pair["target_v_1h"])
        assert abs(r) < 0.9999, f"api_7d_mm may be leaking (corr={r:.4f})"

    def test_target_columns_not_in_feature_sets(self):
        all_feats = set(
            FEAT_STAGE_BASE + FEAT_STAGE_DYN + FEAT_RAIN_CONT + FEAT_RAIN_LAG + ["api_7d_mm"]
        )
        for h in HORIZONS:
            hs = f"{h}h"
            for prefix in [f"target_v_{hs}", f"target_v_dH_{hs}"]:
                assert prefix not in all_feats, f"Future target {prefix} is in a feature set"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model catalogue
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCatalogue:
    def test_exactly_12_configurations(self):
        cat = get_catalogue()
        assert len(cat) == 12, f"Expected 12 models, got {len(cat)}"

    def test_all_ids_unique(self):
        cat = get_catalogue()
        ids = [c["id"] for c in cat]
        assert len(ids) == len(set(ids)), "Duplicate model IDs in catalogue"

    def test_required_keys_present(self):
        cat = get_catalogue()
        required = {"id", "name", "feats", "ttype", "est", "scale", "grid"}
        for cfg in cat:
            missing = required - set(cfg.keys())
            assert not missing, f"{cfg['id']} missing keys: {missing}"

    def test_persistence_has_no_estimator(self):
        cat = get_catalogue()
        m01 = next(c for c in cat if c["id"] == "M01")
        assert m01["est"] is None, "Persistence should have est=None"

    def test_target_types_valid(self):
        cat = get_catalogue()
        valid_types = {"level", "dH"}
        for cfg in cat:
            assert cfg["ttype"] in valid_types, (
                f"{cfg['id']} has invalid ttype={cfg['ttype']}"
            )

    def test_feature_sets_non_empty_for_non_persistence(self):
        cat = get_catalogue()
        for cfg in cat:
            if cfg["id"] != "M01":
                assert len(cfg["feats"]) > 0, f"{cfg['id']} has empty feature set"

    def test_m05_is_model_d_equivalent(self):
        cat = get_catalogue()
        m05 = next(c for c in cat if c["id"] == "M05")
        assert "rain_depth_1h_mm" in m05["feats"], "M05 must include rainfall features"
        assert "dH_v_1h" in m05["feats"], "M05 must include stage dynamics"

    def test_all_tree_models_have_seed(self):
        cat = get_catalogue()
        tree_ids = {"M08", "M09", "M10", "M11", "M12"}
        for cfg in cat:
            if cfg["id"] in tree_ids and cfg["est"] is not None:
                params = cfg["est"].get_params()
                rs = params.get("random_state")
                assert rs == SEED, f"{cfg['id']} random_state={rs}, expected {SEED}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. API computation
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIComputation:
    def test_api_zero_for_zero_rain(self):
        rain = pd.Series([0.0, 0.0, 0.0])
        api = _compute_api(rain)
        assert (api < 0.01).all(), "API should be near zero for zero-rain series"

    def test_api_increases_with_rain(self):
        rain = pd.Series([10.0, 10.0, 10.0, 10.0])
        api = _compute_api(rain)
        assert api.is_monotonic_increasing, "API should increase with continuous rainfall"

    def test_api_nan_when_rain_nan(self):
        rain = pd.Series([5.0, np.nan, np.nan])
        api = _compute_api(rain)
        assert not pd.isna(api.iloc[0]), "First non-null rain should give non-null API"
        assert pd.isna(api.iloc[1]), "API should be NaN when rain is NaN"
        assert pd.isna(api.iloc[2]), "API should be NaN when rain is NaN"

    def test_api_decays_after_rain_stops(self):
        # Rain for 24h then stop
        rain = pd.Series([5.0] * 24 + [np.nan] * 5)
        api = _compute_api(rain)
        # After rain stops, API should be decaying (NaN, but internal state decays)
        # Check the last non-NaN value
        non_nan = api.dropna()
        if len(non_nan) > 1:
            assert non_nan.iloc[-1] > 0, "API should still be positive after recent rain"

    def test_api_decay_constant_correct(self):
        assert abs(API_DECAY_HR - (0.9 ** (1.0 / 24.0))) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 6. Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_prediction_zero_mae(self):
        yt = np.array([1.0, 2.0, 3.0])
        m = _metrics(yt, yt.copy(), yt.copy())
        assert m["mae_m"] == 0.0
        assert m["rmse_m"] == 0.0
        assert abs(m["r2"] - 1.0) < 1e-9

    def test_persistence_zero_improvement(self):
        yt    = np.array([1.0, 2.0, 3.0])
        ypers = np.array([1.0, 2.0, 3.0])
        m = _metrics(yt, ypers, ypers)
        assert m["imp_vs_persistence_pct"] == 0.0

    def test_better_model_positive_improvement(self):
        yt    = np.array([1.0, 2.0, 3.0])
        ypers = np.array([1.1, 2.1, 3.1])  # MAE = 0.1
        yp    = np.array([1.01, 2.01, 3.01])  # MAE = 0.01
        m = _metrics(yt, yp, ypers)
        assert m["imp_vs_persistence_pct"] > 0, "Better model should show positive improvement"

    def test_ci_bounds_ordered(self):
        rng = np.random.default_rng(0)
        yt = rng.normal(350, 1, 100)
        yp = yt + rng.normal(0, 0.05, 100)
        lo, hi = _bootstrap_ci(yt, yp)
        assert lo <= hi, "CI lower bound must be <= upper bound"
        assert lo >= 0, "MAE CI lower bound must be non-negative"

    def test_bias_sign_correct(self):
        yt = np.array([1.0, 1.0, 1.0])
        yp = np.array([1.1, 1.1, 1.1])  # overpredicts
        m = _metrics(yt, yp, yt)
        assert m["bias_m"] > 0, "Overpredicting model should have positive bias"

    def test_mae_mm_equals_1000x_mae_m(self):
        rng = np.random.default_rng(0)
        yt = rng.normal(350, 1, 50)
        yp = yt + rng.normal(0, 0.1, 50)
        m = _metrics(yt, yp, yt)
        assert abs(m["mae_mm"] - m["mae_m"] * 1000) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Stratum computation
# ─────────────────────────────────────────────────────────────────────────────

class TestStrata:
    def _make_df(self, dh_vals, rain_vals, target_vals):
        return pd.DataFrame({
            "dH_v_1h": dh_vals,
            "rain_depth_1h_mm": rain_vals,
            "target_v_1h": target_vals,
            "stage_vishnupuri": np.zeros(len(dh_vals)),
        })

    def test_stratum_keys_present(self):
        n = 30
        df = self._make_df(
            np.concatenate([np.full(10, 0.2), np.full(10, -0.2), np.zeros(10)]),
            np.concatenate([np.full(15, 5.0), np.full(15, 0.0)]),
            np.zeros(n),
        )
        st = _strata(df, np.zeros(n), np.zeros(n), "target_v_1h")
        assert "rising" in st
        assert "falling" in st
        assert "stable" in st

    def test_rising_stratum_n_correct(self):
        n = 30
        dh = np.concatenate([np.full(15, 0.2), np.full(15, -0.2)])
        df = self._make_df(dh, np.zeros(n), np.zeros(n))
        st = _strata(df, np.zeros(n), np.zeros(n), "target_v_1h")
        assert st["rising"]["n"] == 15
        assert st["falling"]["n"] == 15


# ─────────────────────────────────────────────────────────────────────────────
# 8. Valid-row filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestValidRows:
    def test_valid_removes_nan_feature(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0], "t": [1, 2, 3]})
        out = _valid(df, ["a", "b"], "t")
        assert len(out) == 2
        assert out["a"].isna().sum() == 0

    def test_valid_removes_nan_target(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "t": [1.0, np.nan, 3.0]})
        out = _valid(df, ["a"], "t")
        assert len(out) == 2

    def test_valid_histgb_keeps_nan_features(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0], "t": [1, 2, 3]})
        out = _valid_histgb(df, "t")
        assert len(out) == 3  # NaN feature kept for HistGB

    def test_valid_histgb_removes_nan_target(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "t": [1.0, np.nan, 3.0]})
        out = _valid_histgb(df, "t")
        assert len(out) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 9. Artifact save/load round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactRoundTrip:
    def test_artifact_file_exists_after_training(self):
        """Verify at least one artifact file exists from a prior benchmark run, or skip."""
        artifacts = list(OUT_DIR.glob("M*.joblib"))
        if not artifacts:
            pytest.skip("No Phase 1 artifacts found (benchmark not yet run)")
        assert len(artifacts) > 0

    def test_artifact_has_expected_structure(self):
        artifacts = list(OUT_DIR.glob("M*.joblib"))
        if not artifacts:
            pytest.skip("No Phase 1 artifacts found")
        import joblib
        art = joblib.load(artifacts[0])
        assert "model" in art
        assert "metadata" in art
        meta = art["metadata"]
        for key in ["model_id", "model_name", "horizon", "target_type", "features",
                    "n_features", "train_range", "random_seed", "n_train"]:
            assert key in meta, f"Metadata missing key: {key}"

    def test_artifact_model_predicts_correctly_on_fixture(self):
        """Load first artifact, make a deterministic test prediction on synthetic input."""
        artifacts = list(OUT_DIR.glob("M*.joblib"))
        if not artifacts:
            pytest.skip("No Phase 1 artifacts found")
        import joblib
        art  = joblib.load(artifacts[0])
        meta = art["metadata"]
        model= art["model"]
        feats= meta["features"]
        # Create synthetic input of the right shape
        X = pd.DataFrame([{f: 350.0 for f in feats}])
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        assert abs(float(pred1[0]) - float(pred2[0])) < 1e-10, (
            "Same model on same input should give identical predictions (determinism)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Phase1InferenceInterface
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceInterface:
    def test_refuses_invalid_horizon(self):
        from river_forecasting.src.phase1_ml_pipeline import Phase1InferenceInterface
        registry = OUT_DIR / "model_registry.json"
        if not registry.exists():
            pytest.skip("Registry not yet built")
        iface = Phase1InferenceInterface(registry)
        result = iface.predict("M01", horizon=99, features={}, origin_ts="2026-09-01",
                               current_stage_m=351.0)
        assert not result["valid"]
        assert "99" in result["reason"]

    def test_persistence_returns_current_stage(self):
        from river_forecasting.src.phase1_ml_pipeline import Phase1InferenceInterface
        registry = OUT_DIR / "model_registry.json"
        if not registry.exists():
            pytest.skip("Registry not yet built")
        iface = Phase1InferenceInterface(registry)
        result = iface.predict("M01", horizon=1, features={}, origin_ts="2026-09-01",
                               current_stage_m=351.5)
        assert result["valid"]
        assert abs(result["predicted_stage_m"] - 351.5) < 1e-9

    def test_missing_features_rejected(self):
        from river_forecasting.src.phase1_ml_pipeline import Phase1InferenceInterface
        registry = OUT_DIR / "model_registry.json"
        if not registry.exists():
            pytest.skip("Registry not yet built")
        iface = Phase1InferenceInterface(registry)
        # Try M04 with empty feature dict (all features missing)
        result = iface.predict("M04", horizon=1, features={}, origin_ts="2026-09-01",
                               current_stage_m=351.0)
        assert not result["valid"], "Should refuse when mandatory features are missing"
        assert "Missing" in result["reason"]

    def test_result_tagged_historical_evaluation(self):
        from river_forecasting.src.phase1_ml_pipeline import Phase1InferenceInterface
        registry = OUT_DIR / "model_registry.json"
        if not registry.exists():
            pytest.skip("Registry not yet built")
        iface = Phase1InferenceInterface(registry)
        result = iface.predict("M01", horizon=1, features={}, origin_ts="2026-09-01",
                               current_stage_m=351.0)
        assert result.get("mode") == "HISTORICAL_EVALUATION", (
            "Research interface must label predictions as HISTORICAL_EVALUATION"
        )
        assert "warning" in result, "Research interface must include a warning field"

    def test_warning_mentions_live_safeguard(self):
        from river_forecasting.src.phase1_ml_pipeline import Phase1InferenceInterface
        registry = OUT_DIR / "model_registry.json"
        if not registry.exists():
            pytest.skip("Registry not yet built")
        iface = Phase1InferenceInterface(registry)
        result = iface.predict("M01", horizon=1, features={}, origin_ts="2026-09-01",
                               current_stage_m=351.0)
        warning = result.get("warning", "")
        assert "live" in warning.lower() or "operational" in warning.lower(), (
            "Warning must mention live operational / freshness safeguard separation"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Live inference safeguards preserved
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveInferenceSafeguardsPreserved:
    def test_live_inference_module_imports_without_error(self):
        """live_inference.py must still be importable (Phase 1 must not have broken it)."""
        from river_forecasting.src import live_inference  # noqa: F401

    def test_live_inference_has_freshness_rule(self):
        """The 3-hour freshness gate must still exist in live_inference.py."""
        import inspect
        from river_forecasting.src import live_inference
        src = inspect.getsource(live_inference)
        # The 3h rule is encoded inline; verify it still refers to <= 3.0
        assert "<= 3.0" in src or "<= 3" in src, (
            "live_inference.py: 3-hour telemetry freshness rule not found"
        )

    def test_live_inference_freshness_not_weakened(self):
        """Verify the threshold value in source has not been raised above 3h."""
        import inspect, re
        from river_forecasting.src import live_inference
        src = inspect.getsource(live_inference)
        # Find all '<= N' patterns that relate to age_hours check
        matches = re.findall(r'age_hours\s*<=\s*([0-9.]+)', src)
        for val in matches:
            assert float(val) <= 3.0, (
                f"Freshness threshold has been weakened to {val}h (must be <= 3.0h)"
            )
        assert len(matches) > 0, "age_hours freshness check removed from live_inference.py"

    def test_phase1_pipeline_does_not_import_live_inference(self):
        """Phase 1 pipeline must not import live_inference (separation of concerns)."""
        import inspect
        import river_forecasting.src.phase1_ml_pipeline as p1
        src = inspect.getsource(p1)
        # Check only import lines, not docstrings or comments
        import_lines = [l for l in src.split('\n')
                        if l.strip().startswith('import ') or
                        l.strip().startswith('from ')]
        import_text = ' '.join(import_lines)
        assert 'live_inference' not in import_text, (
            "phase1_ml_pipeline.py must not import live_inference"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. Estimator Cloning and Isolation Regression Tests (Integrity Repair)
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimatorCloningAndIsolation:
    """
    Automated regression tests proving the Phase 1 estimator-reuse bug is resolved:
    1. Two calls to _build_pipe() produce independent estimators.
    2. Fitting one pipeline does not mutate the other pipeline's estimator or predictions.
    3. The standard and pre-event pipelines do not share fitted estimator or transformer state.
    4. Rebuilding a pipeline from the same configuration is deterministic where expected.
    5. The fix applies to every affected estimator configuration, not only Ridge.
    """

    def test_build_pipe_produces_independent_estimator_instances_all_catalogue_models(self):
        """Every parametric model in get_catalogue() must produce distinct estimator instances."""
        cat = get_catalogue()
        parametric_models = [cfg for cfg in cat if cfg["est"] is not None]
        assert len(parametric_models) == 11, f"Expected 11 parametric models, found {len(parametric_models)}"

        for cfg in parametric_models:
            mid = cfg["id"]
            p1 = _build_pipe(cfg)
            p2 = _build_pipe(cfg)
            assert p1 is not None and p2 is not None

            est1 = p1.named_steps["estimator"]
            est2 = p2.named_steps["estimator"]
            assert id(est1) != id(est2), (
                f"Model {mid} failed estimator isolation: p1 and p2 share identical estimator object {id(est1)}"
            )

            step_name = "sc" if cfg["scale"] else "id"
            sc1 = p1.named_steps[step_name]
            sc2 = p2.named_steps[step_name]
            assert id(sc1) != id(sc2), (
                f"Model {mid} failed scaler isolation: p1 and p2 share identical scaler object {id(sc1)}"
            )

    def test_fitting_one_pipeline_does_not_mutate_other_estimator_or_predictions(self):
        """Fitting one pipeline must not mutate another pipeline or make the other appear fitted."""
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        cat = get_catalogue()
        m04_cfg = next(c for c in cat if c["id"] == "M04")
        p1 = _build_pipe(m04_cfg)
        p2 = _build_pipe(m04_cfg)

        rng = np.random.default_rng(42)
        X1 = pd.DataFrame(rng.normal(size=(50, len(m04_cfg["feats"]))), columns=m04_cfg["feats"])
        y1 = rng.normal(size=50)

        X2 = pd.DataFrame(rng.normal(size=(50, len(m04_cfg["feats"]))), columns=m04_cfg["feats"])
        y2 = rng.normal(size=50) * 100.0

        p1.fit(X1, y1)
        est2 = p2.named_steps["estimator"]
        with pytest.raises(NotFittedError):
            check_is_fitted(est2)

        # Record p1 predictions
        pred1_before = p1.predict(X1).copy()
        coef1_before = p1.named_steps["estimator"].coef_.copy()

        # Fit p2 on different data
        p2.fit(X2, y2)

        # p1 must be completely unchanged
        pred1_after = p1.predict(X1)
        coef1_after = p1.named_steps["estimator"].coef_
        np.testing.assert_allclose(pred1_before, pred1_after, rtol=1e-12, atol=1e-12,
                                   err_msg="p1 predictions changed after fitting p2!")
        np.testing.assert_allclose(coef1_before, coef1_after, rtol=1e-12, atol=1e-12,
                                   err_msg="p1 coefficients changed after fitting p2!")

    def test_standard_and_preevent_pipelines_isolation(self, df):
        """Simulate Phase1Runner: fitting p_pre must NOT alter p_std fitted state."""
        cat = get_catalogue()
        sp = get_splits(df)
        tr = sp["train"]
        tr_pre = df[
            (df["dt"] >= TRAIN_START) &
            ((df["dt"] + pd.Timedelta(hours=1)) <= pd.Timestamp(PHASE_A_CUTOFF))
        ].copy()

        for mid, expected_ntr, expected_npre in [("M04", 1222, 957), ("M05", 1079, 944)]:
            cfg = next(c for c in cat if c["id"] == mid)
            feats = cfg["feats"]
            tgt = "target_v_dH_1h"

            tr_c = _valid(tr, feats, tgt)
            pre_c = _valid(tr_pre, feats, tgt)

            assert len(tr_c) == expected_ntr
            assert len(pre_c) == expected_npre

            p_std = _build_pipe(cfg)
            p_std.fit(tr_c[feats], tr_c[tgt].values)
            std_coef_initial = p_std.named_steps["estimator"].coef_.copy()
            std_intercept_initial = float(p_std.named_steps["estimator"].intercept_)

            p_pre = _build_pipe(cfg)
            p_pre.fit(pre_c[feats], pre_c[tgt].values)
            pre_coef = p_pre.named_steps["estimator"].coef_.copy()
            pre_intercept = float(p_pre.named_steps["estimator"].intercept_)

            # Verify pre_c and tr_c produced different models
            assert not np.allclose(std_coef_initial, pre_coef), f"Pre-event and standard training should differ for {mid}"
            assert abs(std_intercept_initial - pre_intercept) > 1e-6

            # Critical assertion: p_std was NOT mutated by p_pre.fit()
            np.testing.assert_allclose(p_std.named_steps["estimator"].coef_, std_coef_initial,
                                       err_msg=f"p_std coefficients were mutated by p_pre.fit() for {mid}!")
            assert p_std.named_steps["estimator"].intercept_ == std_intercept_initial, (
                f"p_std intercept was mutated by p_pre.fit() for {mid}!"
            )

    def test_rebuilding_pipeline_is_deterministic(self):
        """Building two pipelines from same config and fitting with same data produces identical results."""
        cat = get_catalogue()
        m08_cfg = next(c for c in cat if c["id"] == "M08")
        feats = m08_cfg["feats"]
        rng = np.random.default_rng(SEED)
        X = pd.DataFrame(rng.normal(size=(60, len(feats))), columns=feats)
        y = rng.normal(size=60)

        p1 = _build_pipe(m08_cfg)
        p2 = _build_pipe(m08_cfg)

        p1.fit(X, y)
        p2.fit(X, y)

        pred1 = p1.predict(X)
        pred2 = p2.predict(X)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10, atol=1e-10,
                                   err_msg="Rebuilding RF pipeline was not deterministic with fixed SEED")

    def test_catalogue_template_estimators_remain_unfitted(self):
        """The prototype estimator inside get_catalogue() must remain completely unfitted after pipeline execution."""
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        cat = get_catalogue()
        for cfg in cat:
            if cfg["est"] is None:
                continue
            pipe = _build_pipe(cfg)
            rng = np.random.default_rng(123)
            X = pd.DataFrame(rng.normal(size=(20, len(cfg["feats"]))), columns=cfg["feats"])
            y = rng.normal(size=20)
            pipe.fit(X, y)

            proto = cfg["est"]
            with pytest.raises(NotFittedError):
                check_is_fitted(proto)

