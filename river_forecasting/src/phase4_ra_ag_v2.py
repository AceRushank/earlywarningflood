"""
river_forecasting/src/phase4_ra_ag_v2.py
=======================================
Phase 4: RA-AG v2 Architecture - Dynamic Lead-Time Gating, Soft Shrinkage,
and Bounded Telemetry Outage Fallback.
Vishnupuri Barrage / Godavari River, Nanded, Maharashtra

Architectural Innovations:
  1. Horizon-Dependent Stable Gating:
     - At +1h: Pure persistence (dH = 0) gated by origin-t stability threshold.
     - At +2h & +3h: Adaptive shrinkage Ridge expert or soft-blended persistence to
       capture multi-hour reservoir storage drainage without tree noise.
  2. Tuned XGBoost Rising Expert:
     - Dedicated multimodal XGBoost regressor for rising hydrograph limbs.
  3. Bounded Linear Falling Expert:
     - Regularized Ridge recession expert for drawdown limbs.
  4. Robust Telemetry Fallback Gate:
     - Bounded linear Ridge fallback during rainfall telemetry outages (avoids tree leaf saturation).
  5. Soft Gating Candidate:
     - Continuous probabilistic/sigmoid weighting partition of unity:
       Delta_H = w_stable * Delta_H_stable + w_rise * Delta_H_rise + w_fall * Delta_H_fall.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
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
    SEED,
    HORIZONS,
    FEAT_STAGE_BASE,
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_RAIN_LAG,
    FEAT_MULTIMODAL,
    load_features,
    get_splits,
)

log = logging.getLogger("phase4_ra_ag_v2")


# =============================================================================
# 1. CAUSAL REGIME DETECTORS & SOFT GATING
# =============================================================================

class CausalRegimeDetector:
    """
    Causal Regime Classifier evaluated strictly at forecast origin t.
    Uses rate-of-rise (dH_v_1h, dH_v_3h), acceleration (accel_v_1h), and antecedent rain (rain_3h_mm).
    Zero future leakage.
    """
    def __init__(self, theta_rise: float = 0.010, theta_fall: float = -0.010, rain_threshold: float = 1.0):
        self.theta_rise = theta_rise
        self.theta_fall = theta_fall
        self.rain_threshold = rain_threshold

    def predict_regime(self, row: pd.Series) -> str:
        dh1 = float(row.get("dH_v_1h", 0.0))
        dh3 = float(row.get("dH_v_3h", 0.0))
        accel = float(row.get("accel_v_1h", 0.0))
        
        # Explicitly distinguish missing rainfall from observed zero rainfall (Stage 0 audit fix)
        val_rain3 = row.get("rain_3h_mm", np.nan)
        has_valid_rain = pd.notna(val_rain3)
        rain3 = float(val_rain3) if has_valid_rain else None

        is_rising_stage = (
            (dh1 > self.theta_rise) or
            (dh3 > self.theta_rise * 2.0) or
            (accel > 0.015 and dh1 > 0.0)
        )
        is_rising_rain = (rain3 is not None) and (rain3 >= self.rain_threshold)
        is_rising = is_rising_stage or is_rising_rain
        is_falling = (dh1 < self.theta_fall) or (dh3 < self.theta_fall * 2.0)

        if is_rising:
            return "RISING"
        elif is_falling and not is_rising:
            return "FALLING"
        else:
            return "STABLE"

    def predict_regimes(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.predict_regime, axis=1)


class SoftGateCalculator:
    """
    Continuous Soft Gating Weight Calculator evaluated strictly at origin t.
    Computes smooth weights (w_rise, w_fall, w_stable) in [0, 1] summing strictly to 1.0.
    """
    def __init__(
        self,
        th_rise_low: float = 0.005,
        th_rise_high: float = 0.020,
        th_fall_low: float = 0.005,
        th_fall_high: float = 0.020,
        rain_scale_mm: float = 3.0,
    ):
        self.th_rise_low = th_rise_low
        self.th_rise_high = th_rise_high
        self.th_fall_low = th_fall_low
        self.th_fall_high = th_fall_high
        self.rain_scale_mm = rain_scale_mm

    def compute_weights(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(df)
        dh1 = df["dH_v_1h"].values if "dH_v_1h" in df.columns else np.zeros(n)
        
        # Audit Fix: Eliminate fillna(0.0) - identify missing rainfall explicitly
        if "rain_3h_mm" in df.columns:
            rain_raw = df["rain_3h_mm"]
            has_rain = rain_raw.notna().values
            rain3_clean = np.where(has_rain, rain_raw.values, 0.0)
        else:
            has_rain = np.zeros(n, dtype=bool)
            rain3_clean = np.zeros(n)

        # Rising weight based on velocity and rainfall trigger
        v_rise_raw = np.clip((dh1 - self.th_rise_low) / (self.th_rise_high - self.th_rise_low + 1e-6), 0.0, 1.0)
        # Rainfall trigger is only evaluated when genuine rainfall telemetry is valid
        r_rise_raw = np.where(has_rain, np.clip(rain3_clean / self.rain_scale_mm, 0.0, 1.0), 0.0)
        w_rise_unnorm = np.maximum(v_rise_raw, r_rise_raw)

        # Falling weight based on recession velocity
        v_fall_raw = np.clip((-dh1 - self.th_fall_low) / (self.th_fall_high - self.th_fall_low + 1e-6), 0.0, 1.0)
        # Suppress falling weight ONLY when rainfall is confirmed present (>= 0.5 mm)
        # If rainfall is missing, do not suppress recession dynamics based on missing telemetry
        w_fall_unnorm = np.where(has_rain & (rain3_clean >= 0.5), 0.0, v_fall_raw)

        # Ensure rise and fall do not conflict
        w_active_sum = w_rise_unnorm + w_fall_unnorm
        scale = np.where(w_active_sum > 1.0, w_active_sum, 1.0)
        w_rise = w_rise_unnorm / scale
        w_fall = w_fall_unnorm / scale
        w_stable = np.maximum(0.0, 1.0 - (w_rise + w_fall))

        return w_rise, w_fall, w_stable


# =============================================================================
# 2. RA-AG v2 ARCHITECTURE (HARD & SOFT VARIANTS)
# =============================================================================

class RAAGv2(BaseEstimator, RegressorMixin):
    """
    RA-AG v2: Regime-Aware Adaptive Gating with Horizon-Aware Stable Shrinkage,
    Tuned XGBoost Rising Expert, Linear Recession Expert, and Bounded Outage Fallback.

    Parameters:
      horizon: Forecast lead time in hours (1, 2, or 3).
      gating_mode: 'hard' (discrete regime routing) or 'soft' (continuous blending).
      stable_strategy: 'persistence' (dH=0), 'shrinkage_ridge' (regularized Ridge),
                       or 'horizon_adaptive' (persistence for +1h, shrinkage for +2h/+3h).
      ridge_alpha_stable: Regularization strength for stable Ridge expert.
      ridge_alpha_fall: Regularization strength for recession expert.
      fallback_mode: 'bounded_linear' (Ridge slope continuation) or 'tree' (XGBoost fallback).
      theta_rise: Threshold for rising regime.
      theta_fall: Threshold for falling regime.
      xgb_params: Hyperparameters for XGBoost rising expert.
    """
    def __init__(
        self,
        horizon: int = 1,
        gating_mode: str = "hard",
        stable_strategy: str = "horizon_adaptive",
        ridge_alpha_stable: float = 100.0,
        ridge_alpha_fall: float = 10.0,
        fallback_mode: str = "bounded_linear",
        theta_rise: float = 0.010,
        theta_fall: float = -0.010,
        xgb_params: Optional[Dict[str, Any]] = None,
    ):
        self.horizon = horizon
        self.gating_mode = gating_mode
        self.stable_strategy = stable_strategy
        self.ridge_alpha_stable = ridge_alpha_stable
        self.ridge_alpha_fall = ridge_alpha_fall
        self.fallback_mode = fallback_mode
        self.theta_rise = theta_rise
        self.theta_fall = theta_fall
        self.xgb_params = xgb_params or {"max_depth": 2, "learning_rate": 0.02, "reg_lambda": 5.0}

        self.regime_detector = CausalRegimeDetector(theta_rise=theta_rise, theta_fall=theta_fall)
        self.soft_gate = SoftGateCalculator()

        # Component models
        self.rising_expert_mm_ = None
        self.rising_expert_so_ = None
        self.falling_expert_mm_ = None
        self.falling_expert_so_ = None
        self.stable_expert_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RAAGv2":
        regimes = self.regime_detector.predict_regimes(X)

        mm_cols = [c for c in FEAT_MULTIMODAL if c in X.columns]
        so_cols = [c for c in FEAT_STAGE_DYN if c in X.columns]
        fall_mm_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT) if c in X.columns]

        # 1. Rising Expert: Tuned XGBoost on rising observations
        mask_rise = (regimes == "RISING")
        if mask_rise.sum() < 20:
            mask_rise = np.ones(len(X), dtype=bool)

        X_rise_mm = X.loc[mask_rise, mm_cols].astype(float)
        X_rise_so = X.loc[mask_rise, so_cols].astype(float)
        y_rise = y[mask_rise]

        self.rising_expert_mm_ = xgb.XGBRegressor(
            max_depth=self.xgb_params.get("max_depth", 2),
            learning_rate=self.xgb_params.get("learning_rate", 0.02),
            n_estimators=self.xgb_params.get("n_estimators", 60),
            reg_lambda=self.xgb_params.get("reg_lambda", 5.0),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            tree_method="hist",
        )
        self.rising_expert_mm_.fit(X_rise_mm, y_rise)

        # Fallback rising model: either bounded linear Ridge or Stage-Only XGBoost
        if self.fallback_mode == "bounded_linear":
            self.rising_expert_so_ = Pipeline([
                ("sc", StandardScaler()),
                ("est", Ridge(alpha=5.0)),
            ])
            self.rising_expert_so_.fit(X_rise_so, y_rise)
        else:
            self.rising_expert_so_ = xgb.XGBRegressor(
                max_depth=self.xgb_params.get("max_depth", 2),
                learning_rate=self.xgb_params.get("learning_rate", 0.02),
                n_estimators=self.xgb_params.get("n_estimators", 60),
                reg_lambda=self.xgb_params.get("reg_lambda", 5.0),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                tree_method="hist",
            )
            self.rising_expert_so_.fit(X_rise_so, y_rise)

        # 2. Falling Expert: Regularized Ridge on recession limbs
        mask_fall = (regimes == "FALLING")
        if mask_fall.sum() < 20:
            mask_fall = np.ones(len(X), dtype=bool)

        X_fall_mm = X.loc[mask_fall, fall_mm_cols].astype(float)
        X_fall_so = X.loc[mask_fall, so_cols].astype(float)
        y_fall = y[mask_fall]

        self.falling_expert_mm_ = Pipeline([
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.ridge_alpha_fall)),
        ])
        self.falling_expert_mm_.fit(X_fall_mm, y_fall)

        self.falling_expert_so_ = Pipeline([
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.ridge_alpha_fall)),
        ])
        self.falling_expert_so_.fit(X_fall_so, y_fall)

        # 3. Stable Expert: Fit Ridge with shrinkage on stable subset
        mask_stable = (regimes == "STABLE")
        if mask_stable.sum() < 20:
            mask_stable = np.ones(len(X), dtype=bool)

        X_st = X.loc[mask_stable, so_cols].astype(float)
        y_st = y[mask_stable]

        self.stable_expert_ = Pipeline([
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.ridge_alpha_stable)),
        ])
        self.stable_expert_.fit(X_st, y_st)

        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          pred_dh: Predicted stage change array (meters)
          is_fallback: Boolean array indicating whether stage-only fallback was active
          regimes: String array indicating regime assigned to each observation
          weights: Array of shape (n, 3) containing [w_rise, w_fall, w_stable]
        """
        n = len(X)
        pred_dh = np.zeros(n, dtype=float)
        is_fallback = np.zeros(n, dtype=bool)

        mm_cols = [c for c in FEAT_MULTIMODAL if c in X.columns]
        so_cols = [c for c in FEAT_STAGE_DYN if c in X.columns]
        fall_mm_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT) if c in X.columns]

        X_mm = X[mm_cols].astype(float)
        X_so = X[so_cols].astype(float)
        X_fall_mm = X[fall_mm_cols].astype(float)

        avail_rain_cols = [c for c in FEAT_RAIN_CONT if c in X.columns]
        if len(avail_rain_cols) == len(FEAT_RAIN_CONT) and len(FEAT_RAIN_CONT) > 0:
            has_valid_rain = X[FEAT_RAIN_CONT].notna().all(axis=1).values
        else:
            has_valid_rain = np.zeros(n, dtype=bool)
        regimes = self.regime_detector.predict_regimes(X).values

        # Determine stable prediction
        is_pure_pers = (
            self.stable_strategy == "persistence" or
            (self.stable_strategy == "horizon_adaptive" and self.horizon == 1)
        )
        if is_pure_pers:
            dh_stable = np.zeros(n, dtype=float)
        else:
            dh_stable = self.stable_expert_.predict(X_so)

        # Determine rising predictions
        dh_rise = np.zeros(n, dtype=float)
        mask_rain_avail = has_valid_rain
        mask_rain_unavail = ~has_valid_rain

        if np.any(mask_rain_avail):
            dh_rise[mask_rain_avail] = self.rising_expert_mm_.predict(X_mm.loc[mask_rain_avail])
        if np.any(mask_rain_unavail):
            dh_rise[mask_rain_unavail] = self.rising_expert_so_.predict(X_so.loc[mask_rain_unavail])

        # Determine falling predictions
        dh_fall = np.zeros(n, dtype=float)
        if np.any(mask_rain_avail):
            dh_fall[mask_rain_avail] = self.falling_expert_mm_.predict(X_fall_mm.loc[mask_rain_avail])
        if np.any(mask_rain_unavail):
            dh_fall[mask_rain_unavail] = self.falling_expert_so_.predict(X_so.loc[mask_rain_unavail])

        # Gating Execution
        if self.gating_mode == "soft":
            w_rise, w_fall, w_stable = self.soft_gate.compute_weights(X)
            pred_dh = w_rise * dh_rise + w_fall * dh_fall + w_stable * dh_stable
            is_fallback = ~has_valid_rain
            weights = np.column_stack([w_rise, w_fall, w_stable])
        else:
            weights = np.zeros((n, 3), dtype=float)
            for r in ["RISING", "FALLING", "STABLE"]:
                mask_r = (regimes == r)
                if not np.any(mask_r):
                    continue

                if r == "STABLE":
                    pred_dh[mask_r] = dh_stable[mask_r]
                    is_fallback[mask_r] = ~has_valid_rain[mask_r]
                    weights[mask_r, 2] = 1.0
                elif r == "RISING":
                    pred_dh[mask_r] = dh_rise[mask_r]
                    is_fallback[mask_r] = ~has_valid_rain[mask_r]
                    weights[mask_r, 0] = 1.0
                elif r == "FALLING":
                    pred_dh[mask_r] = dh_fall[mask_r]
                    is_fallback[mask_r] = ~has_valid_rain[mask_r]
                    weights[mask_r, 1] = 1.0

        return pred_dh, is_fallback, regimes, weights


# =============================================================================
# 3. STATISTICAL EVALUATION & CONTROLLED MASKING UTILITIES
# =============================================================================

def compute_phase4_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pers: np.ndarray,
    df_eval: pd.DataFrame,
    is_fallback: Optional[np.ndarray] = None,
    deadband_tau_m: float = 0.005,
) -> Dict[str, Any]:
    """
    Computes audited metrics for Phase 4:
      - MAE, RMSE, Bias, R2 in mm
      - Ternary Direction Accuracy (with 5 mm deadband)
      - Active-Movement Concordance, Reversals, and Holding rates
      - Peak underprediction on test period peak
      - Stratified MAEs: Rapid-Rise, Falling, Stable, Rainfall Available vs Outage
      - Forecast availability & fallback frequency
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

    delta_true = y_true - y_pers
    delta_pred = y_pred - y_pers

    def to_class(d: np.ndarray, tau: float) -> np.ndarray:
        c = np.zeros_like(d, dtype=int)
        c[d > tau] = 1
        c[d < -tau] = -1
        return c

    class_true = to_class(delta_true, deadband_tau_m)
    class_pred = to_class(delta_pred, deadband_tau_m)
    dir_acc_ternary = float(np.mean(class_true == class_pred)) * 100.0

    active_mask = np.abs(delta_true) > deadband_tau_m
    if active_mask.sum() > 0:
        dir_acc_active = float(np.mean(class_pred[active_mask] == class_true[active_mask])) * 100.0
        reversal_rate = float(np.mean(class_pred[active_mask] == -class_true[active_mask])) * 100.0
        holding_rate = float(np.mean(class_pred[active_mask] == 0)) * 100.0
    else:
        dir_acc_active, reversal_rate, holding_rate = 100.0, 0.0, 0.0

    dh1 = df_eval["dH_v_1h"].values if "dH_v_1h" in df_eval.columns else np.zeros(n)
    rain = df_eval["rain_depth_1h_mm"].values if "rain_depth_1h_mm" in df_eval.columns else np.full(n, np.nan)

    mask_rapid_rise = dh1 > 0.020
    mask_falling = dh1 < -0.010
    mask_stable = np.abs(dh1) <= 0.010

    mae_rapid_rise = float(np.mean(err[mask_rapid_rise])) if mask_rapid_rise.sum() > 0 else np.nan
    mae_falling = float(np.mean(err[mask_falling])) if mask_falling.sum() > 0 else np.nan
    mae_stable = float(np.mean(err[mask_stable])) if mask_stable.sum() > 0 else np.nan

    peak_idx = int(np.argmax(y_true))
    peak_true = float(y_true[peak_idx])
    peak_pred = float(y_pred[peak_idx])
    peak_underpred_mm = float(peak_true - peak_pred) * 1000.0

    rain_avail = ~np.isnan(rain)
    mae_rain_avail = float(np.mean(err[rain_avail])) if rain_avail.sum() > 0 else np.nan
    mae_rain_unavail = float(np.mean(err[~rain_avail])) if (~rain_avail).sum() > 0 else np.nan

    fb_count = int(np.sum(is_fallback)) if is_fallback is not None else 0
    fb_freq_pct = (fb_count / n) * 100.0 if n > 0 else 0.0

    return {
        "n_samples": n,
        "mae_mm": round(mae * 1000.0, 3),
        "rmse_mm": round(rmse * 1000.0, 3),
        "bias_mm": round(bias * 1000.0, 3),
        "r2": round(r2, 4),
        "mae_persistence_mm": round(mae_pers * 1000.0, 3),
        "imp_vs_persistence_pct": round(imp_vs_pers_pct, 2),
        "dir_acc_ternary_pct": round(dir_acc_ternary, 2),
        "dir_acc_active_pct": round(dir_acc_active, 2),
        "reversal_rate_pct": round(reversal_rate, 2),
        "holding_rate_pct": round(holding_rate, 2),
        "n_rapid_rise": int(mask_rapid_rise.sum()),
        "mae_rapid_rise_mm": round(mae_rapid_rise * 1000.0, 3) if not np.isnan(mae_rapid_rise) else None,
        "n_falling": int(mask_falling.sum()),
        "mae_falling_mm": round(mae_falling * 1000.0, 3) if not np.isnan(mae_falling) else None,
        "n_stable": int(mask_stable.sum()),
        "mae_stable_mm": round(mae_stable * 1000.0, 3) if not np.isnan(mae_stable) else None,
        "peak_underpred_mm": round(peak_underpred_mm, 3),
        "mae_rain_avail_mm": round(mae_rain_avail * 1000.0, 3) if not np.isnan(mae_rain_avail) else None,
        "mae_rain_unavail_mm": round(mae_rain_unavail * 1000.0, 3) if not np.isnan(mae_rain_unavail) else None,
        "fallback_hours": fb_count,
        "fallback_freq_pct": round(fb_freq_pct, 2),
        "forecast_coverage_pct": 100.0,
    }


def compute_moving_block_bootstrap_p_value(
    err_a: np.ndarray,
    err_b: np.ndarray,
    block_length: int = 12,
    n_bootstraps: int = 2000,
    seed: int = SEED,
) -> float:
    """
    Computes two-sided moving-block bootstrap p-value for the difference in MAE.
    Accounts for temporal serial dependence in hourly forecasts.
    """
    rng = np.random.default_rng(seed)
    d = (err_a - err_b) * 1000.0  # in mm
    n = len(d)
    obs_diff = float(np.mean(d))

    if n < block_length * 2:
        return 1.0

    n_blocks = int(math.ceil(n / block_length))
    boot_diffs = np.zeros(n_bootstraps)

    # Valid starting indices for blocks
    max_start = n - block_length + 1
    starts = rng.integers(0, max_start, size=(n_bootstraps, n_blocks))

    for b in range(n_bootstraps):
        sample_indices = []
        for s in starts[b]:
            sample_indices.extend(range(s, s + block_length))
        sample_indices = sample_indices[:n]
        boot_diffs[b] = float(np.mean(d[sample_indices]))

    # Centered bootstrap distribution under H0: mean(d) = 0
    boot_centered = boot_diffs - obs_diff
    p_val = float(np.mean(np.abs(boot_centered) >= np.abs(obs_diff)))
    return round(p_val, 5)
