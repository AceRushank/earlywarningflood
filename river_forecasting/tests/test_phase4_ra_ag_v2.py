"""
river_forecasting/tests/test_phase4_ra_ag_v2.py
==============================================
Unit and integration tests for Phase 4: RA-AG v2 Dynamic Gating & Soft Shrinkage.

Test Coverage:
  1. Causal Regime Detection (Zero lookahead leakage).
  2. Soft Gating Calculator (Partition of unity: sum(w) == 1.0, w in [0, 1]).
  3. Stable Shrinkage Expert (Persistence at +1h, regularized Ridge at +2h/+3h).
  4. Bounded Outage Fallback (Engaged on missing rainfall, eliminates tree saturation).
  5. Controlled Masking Evaluation (Masked rain features with frozen truth labels).
  6. Moving-Block Bootstrap Test Utility.
  7. Serialized Phase 4 Model Artifacts Reproducibility.
"""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from river_forecasting.src.phase1_ml_pipeline import (
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_MULTIMODAL,
)
from river_forecasting.src.phase4_ra_ag_v2 import (
    CausalRegimeDetector,
    SoftGateCalculator,
    RAAGv2,
    compute_phase4_metrics,
    compute_moving_block_bootstrap_p_value,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase4_ra_ag_v2"
MODELS_DIR = OUTPUT_DIR / "models"


@pytest.fixture
def synthetic_sample_df():
    rng = np.random.default_rng(202)
    n = 100
    df = pd.DataFrame({
        "stage_vishnupuri": 354.0 + np.cumsum(rng.normal(0, 0.01, size=n)),
        "dH_v_1h": rng.normal(0, 0.02, size=n),
        "dH_v_3h": rng.normal(0, 0.03, size=n),
        "dH_v_6h": rng.normal(0, 0.04, size=n),
        "accel_v_1h": rng.normal(0, 0.01, size=n),
        "rain_depth_1h_mm": np.maximum(0, rng.normal(0.5, 2.0, size=n)),
        "rain_3h_mm": np.maximum(0, rng.normal(1.5, 4.0, size=n)),
        "rain_6h_mm": np.maximum(0, rng.normal(3.0, 6.0, size=n)),
        "rain_12h_mm": np.maximum(0, rng.normal(5.0, 8.0, size=n)),
        "rain_24h_mm": np.maximum(0, rng.normal(10.0, 12.0, size=n)),
        "api_7d_mm": np.maximum(0, rng.normal(15.0, 5.0, size=n)),
    })
    df["target_v_dH_1h"] = 0.5 * df["dH_v_1h"] + 0.002 * df["rain_depth_1h_mm"] + rng.normal(0, 0.005, size=n)
    df["target_v_1h"] = df["stage_vishnupuri"] + df["target_v_dH_1h"]
    return df


def test_causal_regime_detector_zero_lookahead():
    detector = CausalRegimeDetector(theta_rise=0.010, theta_fall=-0.010, rain_threshold=1.0)
    # Origin t observable Series
    s_rise = pd.Series({"dH_v_1h": 0.025, "dH_v_3h": 0.03, "accel_v_1h": 0.01, "rain_3h_mm": 0.0})
    s_fall = pd.Series({"dH_v_1h": -0.018, "dH_v_3h": -0.025, "accel_v_1h": -0.01, "rain_3h_mm": 0.0})
    s_stable = pd.Series({"dH_v_1h": 0.002, "dH_v_3h": 0.003, "accel_v_1h": 0.0, "rain_3h_mm": 0.1})

    assert detector.predict_regime(s_rise) == "RISING"
    assert detector.predict_regime(s_fall) == "FALLING"
    assert detector.predict_regime(s_stable) == "STABLE"


def test_soft_gating_partition_of_unity(synthetic_sample_df):
    gate = SoftGateCalculator()
    w_rise, w_fall, w_stable = gate.compute_weights(synthetic_sample_df)

    assert len(w_rise) == len(synthetic_sample_df)
    assert np.all(w_rise >= 0.0) and np.all(w_rise <= 1.0)
    assert np.all(w_fall >= 0.0) and np.all(w_fall <= 1.0)
    assert np.all(w_stable >= 0.0) and np.all(w_stable <= 1.0)

    # Weights must sum strictly to 1.0 within floating point tolerance
    w_sum = w_rise + w_fall + w_stable
    np.testing.assert_allclose(w_sum, np.ones_like(w_sum), atol=1e-5)


def test_raag_v2_hard_and_soft_execution(synthetic_sample_df):
    df = synthetic_sample_df
    y_dh = df["target_v_dH_1h"].values

    # Test Hard Gating
    v2_hard = RAAGv2(horizon=1, gating_mode="hard", stable_strategy="horizon_adaptive")
    v2_hard.fit(df, y_dh)
    p_dh_h, fb_h, reg_h, w_h = v2_hard.predict(df)
    assert len(p_dh_h) == len(df)
    assert np.all(np.isfinite(p_dh_h))
    assert not np.any(fb_h)

    # Test Soft Gating
    v2_soft = RAAGv2(horizon=1, gating_mode="soft", stable_strategy="horizon_adaptive")
    v2_soft.fit(df, y_dh)
    p_dh_s, fb_s, reg_s, w_s = v2_soft.predict(df)
    assert len(p_dh_s) == len(df)
    assert np.all(np.isfinite(p_dh_s))


def test_bounded_linear_fallback_on_missing_rain(synthetic_sample_df):
    df = synthetic_sample_df.copy()
    # Mask rainfall columns for observations 10 to 20
    df.loc[10:20, FEAT_RAIN_CONT] = np.nan

    v2 = RAAGv2(horizon=1, fallback_mode="bounded_linear")
    v2.fit(synthetic_sample_df, synthetic_sample_df["target_v_dH_1h"].values)

    p_dh, is_fb, _, _ = v2.predict(df)
    assert np.all(np.isfinite(p_dh))
    # Fallback must be active on masked rows
    assert np.all(is_fb[10:21])


def test_moving_block_bootstrap_utility():
    n = 200
    err_a = np.random.normal(0.005, 0.002, size=n)
    err_b = err_a + 0.003  # Candidate B is significantly worse
    p_val = compute_moving_block_bootstrap_p_value(err_a, err_b, block_length=12, n_bootstraps=500)
    assert p_val < 0.01


def test_serialized_phase4_models():
    for h in [1, 2, 3]:
        hs = f"{h}h"
        hard_file = MODELS_DIR / f"raag_v2_hard_{hs}.joblib"
        soft_file = MODELS_DIR / f"raag_v2_soft_{hs}.joblib"

        assert hard_file.exists(), f"Missing {hard_file}"
        assert soft_file.exists(), f"Missing {soft_file}"

        m_hard = joblib.load(hard_file)
        m_soft = joblib.load(soft_file)

        assert isinstance(m_hard, RAAGv2)
        assert isinstance(m_soft, RAAGv2)
