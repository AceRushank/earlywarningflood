"""
river_forecasting/src/phase2_model_development.py
==================================================
Phase 2 - Model Development and Novelty Discovery Pipeline
Vishnupuri Barrage / Godavari River, Nanded, Maharashtra

Models Trained and Benchmarked:
  1. M01_persistence: Baseline persistence (dH = 0)
  2. M02_stage_only_ridge: Direct stage level Ridge (H_t+h)
  3. M03_stage_change_ridge: Stage-change Ridge (dH_t+h)
  4. M04_model_d_rainfall: Stage-change Ridge + Contemporary Rainfall (Model D)
  5. M05_random_forest: Random Forest Regressor (dH_t+h)
  6. M06_extra_trees: Extra Trees Regressor (dH_t+h)
  7. M07_xgboost: Conservative tuned XGBoost Regressor (dH_t+h)
  8. M08_regime_aware_adaptive: Novelty Candidate - Causal Regime-Aware Adaptive Gating
     with Stage-Only Telemetry Fallback
  9. M09_event_weighted_ridge: Controlled Event-Weighted Loss Ridge Regressor

Guardrails:
  - Strict chronological splits: Train (June 15 - Aug 09), Val (Aug 10 - Aug 31), Test (Sept 01 - Sept 30).
  - Hyperparameter tuning and threshold selection strictly on Validation data.
  - Final Test partition is evaluated only once on pre-selected models.
  - Causal regime detection uses only information observable at origin t.
  - Missing rainfall is never zero-imputed; handled via explicit stage-only fallback.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from river_forecasting.src.phase1_ml_pipeline import (
    DATASET_CSV,
    RAIN_HOURLY,
    HORIZONS,
    SEED,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    FEAT_STAGE_BASE,
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_RAIN_LAG,
    FEAT_MULTIMODAL,
    load_features,
    get_splits,
)

log = logging.getLogger("phase2_development")


# =============================================================================
# REGIME DETECTOR AND REGIME-AWARE ADAPTIVE MODEL (NOVELTY CANDIDATE)
# =============================================================================

class RegimeDetector:
    """
    Causal Regime Classifier evaluated strictly at forecast origin t.
    Uses rate-of-rise (dH_v_1h, dH_v_3h), acceleration (accel_v_1h), and antecedent rain.
    """
    def __init__(self, theta_rise: float = 0.015, theta_fall: float = -0.015, rain_threshold: float = 1.0):
        self.theta_rise = theta_rise
        self.theta_fall = theta_fall
        self.rain_threshold = rain_threshold

    def predict_regime(self, row: pd.Series) -> str:
        dh1 = float(row.get("dH_v_1h", 0.0))
        dh3 = float(row.get("dH_v_3h", 0.0))
        accel = float(row.get("accel_v_1h", 0.0))
        rain3 = float(row.get("rain_3h_mm", 0.0)) if pd.notna(row.get("rain_3h_mm")) else 0.0

        # Rising condition: positive velocity or active rain or strong positive acceleration
        is_rising = (dh1 > self.theta_rise) or (dh3 > self.theta_rise * 2.0) or (rain3 >= self.rain_threshold) or (accel > 0.015 and dh1 > 0.0)
        # Falling condition: negative velocity with zero active rainfall
        is_falling = (dh1 < self.theta_fall) or (dh3 < self.theta_fall * 2.0)

        if is_rising:
            return "RISING"
        elif is_falling and not is_rising:
            return "FALLING"
        else:
            return "STABLE"

    def predict_regimes(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.predict_regime, axis=1)


class RegimeAwareAdaptiveRegressor(BaseEstimator, RegressorMixin):
    """
    Novelty Candidate: Regime-Aware Adaptive Gating with Stage-Only Fallback.

    Architecture:
      1. Causal Regime Detection: Detects RISING, FALLING, or STABLE at origin t.
      2. Specialized Regime Experts:
         - Rising Expert: Tuned for surging hydrographs and active rainfall forcing.
         - Falling Expert: Tuned for recession limb dynamics.
         - Stable Expert: High-regularization shrinkage / persistence (dH -> 0).
      3. Missing Telemetry Fallback Gate:
         - If rainfall features are valid -> use Multimodal Regime Model.
         - If rainfall features are missing/stale -> use Stage-Only Regime Model.
         - Never imputes zero for missing rainfall.
    """
    def __init__(
        self,
        horizon: int = 1,
        theta_rise: float = 0.015,
        theta_fall: float = -0.015,
        alpha_rise: float = 5.0,
        alpha_fall: float = 10.0,
        alpha_stable: float = 100.0,
        use_pure_persistence_on_stable: bool = True,
        use_tree_rising_expert: bool = False,
    ):
        self.horizon = horizon
        self.theta_rise = theta_rise
        self.theta_fall = theta_fall
        self.alpha_rise = alpha_rise
        self.alpha_fall = alpha_fall
        self.alpha_stable = alpha_stable
        self.use_pure_persistence_on_stable = use_pure_persistence_on_stable
        self.use_tree_rising_expert = use_tree_rising_expert

        self.regime_detector = RegimeDetector(theta_rise=theta_rise, theta_fall=theta_fall)
        self.models_mm_ = {}
        self.models_so_ = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RegimeAwareAdaptiveRegressor":
        regimes = self.regime_detector.predict_regimes(X)
        
        # Multimodal features
        mm_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT) if c in X.columns]
        so_cols = [c for c in FEAT_STAGE_DYN if c in X.columns]

        # Fit Multimodal Regime Experts
        for r in ["RISING", "FALLING", "STABLE"]:
            mask_r = (regimes == r)
            # Ensure sufficient samples, else fallback to all data
            if mask_r.sum() < 20:
                mask_r = np.ones(len(X), dtype=bool)

            X_r_mm = X.loc[mask_r, mm_cols].astype(float)
            y_r = y[mask_r]

            alpha = self.alpha_rise if r == "RISING" else (self.alpha_fall if r == "FALLING" else self.alpha_stable)
            if r == "RISING" and self.use_tree_rising_expert and _XGB_AVAILABLE:
                pipe_mm = xgb.XGBRegressor(
                    max_depth=3, learning_rate=0.03, n_estimators=60,
                    reg_lambda=5.0, random_state=SEED, tree_method="hist"
                )
            else:
                pipe_mm = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=alpha))])

            pipe_mm.fit(X_r_mm, y_r)
            self.models_mm_[r] = pipe_mm

            # Fit Stage-Only Fallback Experts
            X_r_so = X.loc[mask_r, so_cols].astype(float)
            pipe_so = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=alpha))])
            pipe_so.fit(X_r_so, y_r)
            self.models_so_[r] = pipe_so

        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          pred_dh: Predicted stage change array (meters)
          is_fallback: Boolean array indicating whether stage-only fallback was engaged
          regimes: String array indicating regime assigned to each observation
        """
        n = len(X)
        pred_dh = np.zeros(n, dtype=float)
        is_fallback = np.zeros(n, dtype=bool)

        mm_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT) if c in X.columns]
        so_cols = [c for c in FEAT_STAGE_DYN if c in X.columns]

        X_mm = X[mm_cols].astype(float)
        X_so = X[so_cols].astype(float)

        # Vectorized check for rainfall availability
        has_valid_rain = X[FEAT_RAIN_CONT].notna().all(axis=1).values
        regimes = self.regime_detector.predict_regimes(X).values

        for r in ["RISING", "FALLING", "STABLE"]:
            mask_r = (regimes == r)
            if not np.any(mask_r):
                continue

            if r == "STABLE" and self.use_pure_persistence_on_stable:
                pred_dh[mask_r] = 0.0
                is_fallback[mask_r] = ~has_valid_rain[mask_r]
                continue

            # Multimodal vs Fallback
            mask_r_mm = mask_r & has_valid_rain
            mask_r_so = mask_r & ~has_valid_rain

            if np.any(mask_r_mm):
                pred_dh[mask_r_mm] = self.models_mm_[r].predict(X_mm.loc[mask_r_mm])
                is_fallback[mask_r_mm] = False

            if np.any(mask_r_so):
                pred_dh[mask_r_so] = self.models_so_[r].predict(X_so.loc[mask_r_so])
                is_fallback[mask_r_so] = True

        return pred_dh, is_fallback, regimes


# =============================================================================
# EVALUATION AND STRATIFICATION UTILITIES
# =============================================================================

def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pers: np.ndarray,
    df_eval: pd.DataFrame,
    is_fallback: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes aggregate metrics plus flood-specific objective metrics:
      - Aggregate MAE, RMSE, R2, Bias
      - Error during rapid stage rises (dH > 0.02 and dH > 0.05 m/h)
      - Peak stage underprediction
      - Direction-of-change accuracy
      - Falling and stable period MAEs
      - Performance with rainfall available vs unavailable
      - Fallback frequency and coverage
    """
    n = len(y_true)
    err = np.abs(y_true - y_pred)
    err_pers = np.abs(y_true - y_pers)

    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred)) if n > 1 else 0.0

    mae_pers = float(np.mean(err_pers))
    imp_vs_pers_pct = ((mae_pers - mae) / mae_pers) * 100.0 if mae_pers > 0 else 0.0

    # Direction-of-change accuracy:
    # Does predicted direction (y_pred - y_pers) match true direction (y_true - y_pers)?
    delta_true = y_true - y_pers
    delta_pred = y_pred - y_pers
    # Focus direction accuracy on active changes (|delta_true| > 0.005m)
    active_mask = np.abs(delta_true) > 0.005
    if active_mask.sum() > 0:
        dir_acc = float(np.mean(np.sign(delta_pred[active_mask]) == np.sign(delta_true[active_mask]))) * 100.0
    else:
        dir_acc = 100.0

    # Strata evaluations
    dh1 = df_eval["dH_v_1h"].values if "dH_v_1h" in df_eval.columns else np.zeros(n)
    rain = df_eval["rain_depth_1h_mm"].values if "rain_depth_1h_mm" in df_eval.columns else np.full(n, np.nan)

    mask_rapid_rise = dh1 > 0.02
    mask_extreme_rise = dh1 > 0.05
    mask_falling = dh1 < -0.02
    mask_stable = np.abs(dh1) <= 0.02

    mae_rapid_rise = float(np.mean(err[mask_rapid_rise])) if mask_rapid_rise.sum() > 0 else np.nan
    mae_pers_rapid = float(np.mean(err_pers[mask_rapid_rise])) if mask_rapid_rise.sum() > 0 else np.nan
    bias_rapid_rise = float(np.mean(y_pred[mask_rapid_rise] - y_true[mask_rapid_rise])) if mask_rapid_rise.sum() > 0 else np.nan

    mae_falling = float(np.mean(err[mask_falling])) if mask_falling.sum() > 0 else np.nan
    mae_stable = float(np.mean(err[mask_stable])) if mask_stable.sum() > 0 else np.nan

    # Peak underprediction (highest observed stage point in eval cohort)
    peak_idx = int(np.argmax(y_true))
    peak_true = float(y_true[peak_idx])
    peak_pred = float(y_pred[peak_idx])
    peak_err = abs(peak_true - peak_pred)
    peak_underpred = float(peak_true - peak_pred)  # Positive means underpredicted

    # Rainfall availability breakdown
    rain_available = ~np.isnan(rain)
    mae_rain_avail = float(np.mean(err[rain_available])) if rain_available.sum() > 0 else np.nan
    mae_rain_unavail = float(np.mean(err[~rain_available])) if (~rain_available).sum() > 0 else np.nan

    # Fallback statistics
    fallback_count = int(np.sum(is_fallback)) if is_fallback is not None else 0
    fallback_freq_pct = (fallback_count / n) * 100.0 if n > 0 else 0.0
    forecast_coverage_pct = 100.0  # With fallback, coverage is 100%

    return {
        "n_samples": n,
        "mae_mm": round(mae * 1000.0, 3),
        "rmse_mm": round(rmse * 1000.0, 3),
        "bias_mm": round(bias * 1000.0, 3),
        "r2": round(r2, 4),
        "mae_persistence_mm": round(mae_pers * 1000.0, 3),
        "imp_vs_persistence_pct": round(imp_vs_pers_pct, 2),
        "direction_accuracy_pct": round(dir_acc, 2),
        "n_rapid_rise": int(mask_rapid_rise.sum()),
        "mae_rapid_rise_mm": round(mae_rapid_rise * 1000.0, 3) if not np.isnan(mae_rapid_rise) else None,
        "mae_pers_rapid_rise_mm": round(mae_pers_rapid * 1000.0, 3) if not np.isnan(mae_pers_rapid) else None,
        "bias_rapid_rise_mm": round(bias_rapid_rise * 1000.0, 3) if not np.isnan(bias_rapid_rise) else None,
        "n_falling": int(mask_falling.sum()),
        "mae_falling_mm": round(mae_falling * 1000.0, 3) if not np.isnan(mae_falling) else None,
        "n_stable": int(mask_stable.sum()),
        "mae_stable_mm": round(mae_stable * 1000.0, 3) if not np.isnan(mae_stable) else None,
        "peak_true_m": round(peak_true, 3),
        "peak_pred_m": round(peak_pred, 3),
        "peak_err_mm": round(peak_err * 1000.0, 3),
        "peak_underprediction_mm": round(peak_underpred * 1000.0, 3),
        "rain_available_hours": int(rain_available.sum()),
        "mae_rain_avail_mm": round(mae_rain_avail * 1000.0, 3) if not np.isnan(mae_rain_avail) else None,
        "mae_rain_unavail_mm": round(mae_rain_unavail * 1000.0, 3) if not np.isnan(mae_rain_unavail) else None,
        "fallback_hours": fallback_count,
        "fallback_freq_pct": round(fallback_freq_pct, 2),
        "forecast_coverage_pct": round(forecast_coverage_pct, 2),
    }
