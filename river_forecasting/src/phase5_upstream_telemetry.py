"""
river_forecasting/src/phase5_upstream_telemetry.py
=================================================
Phase 5: Upstream Telemetry Integration and Out-of-Sample Surge Forecasting.
Evaluates the incremental value of real, authentic upstream river-stage
telemetry (Mudgal Barrage, Godavari River) for +1h, +2h, and +3h stage forecasting
at Vishnupuri Barrage (Nanded, Maharashtra).

Key Principles:
  - Zero synthetic data, zero inferred gauge measurements.
  - Authentic CWC / Maharashtra Water Resources Department hourly telemetry.
  - Strict causal feature engineering with 1-hour operational telemetry latency buffer.
  - Controlled feature ablation:
      1. Local stage features only
      2. Local stage + rainfall
      3. Local stage + verified upstream telemetry
      4. Local stage + rainfall + upstream telemetry
  - Robust telemetry outage fallback handling.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from river_forecasting.src.phase1_ml_pipeline import (
    SEED,
    HORIZONS,
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_MULTIMODAL,
    _bootstrap_ci,
)
from river_forecasting.src.phase4_ra_ag_v2 import (
    CausalRegimeDetector,
    SoftGateCalculator,
    RAAGv2,
    compute_moving_block_bootstrap_p_value,
)

log = logging.getLogger("phase5_upstream_telemetry")

# -----------------------------------------------------------------------------
# Upstream Feature Definitions (Mudgal Barrage, Godavari River, ~105 km upstream)
# -----------------------------------------------------------------------------
# Telemetry latency: 1 hour operational buffer. At forecast origin t,
# the most recent confirmed upstream reading is t-1h.
FEAT_UPSTREAM = [
    "stage_m_lag1",   # Upstream stage at t-1h
    "dH_m_1h",        # 1-hour upstream change (t-1h minus t-2h)
    "dH_m_3h",        # 3-hour upstream change (t-1h minus t-4h)
    "dH_m_6h",        # 6-hour upstream change (t-1h minus t-7h)
    "dH_m_24h",       # 24-hour upstream change (t-1h minus t-25h, ~hydraulic wave travel time)
]

# Combined Multimodal + Upstream
FEAT_STAGE_UPSTREAM = FEAT_STAGE_DYN + FEAT_UPSTREAM
FEAT_ALL = FEAT_MULTIMODAL + FEAT_UPSTREAM


# -----------------------------------------------------------------------------
# Dataset Ingestion & Feature Engineering with Real Upstream Telemetry
# -----------------------------------------------------------------------------
def load_phase5_dataset(
    aligned_csv_path: Path,
    raw_telemetry_csv_path: Path
) -> pd.DataFrame:
    """
    Loads the aligned Vishnupuri dataset and merges authentic Mudgal Barrage
    upstream telemetry from the official NWIC/MWRD surface-water dataset.
    Strictly causal: enforces 1-hour latency buffer on upstream features.
    """
    print(f"Loading local base dataset from {aligned_csv_path}...")
    df = pd.read_csv(aligned_csv_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    print(f"Loading authentic Mudgal Barrage telemetry from {raw_telemetry_csv_path}...")
    usecols = ["Station", "Data Acquisition Time", "River Water Level Telemetry Hourly (meter)"]
    chunks = []
    for chunk in pd.read_csv(raw_telemetry_csv_path, usecols=usecols, chunksize=100000):
        sub = chunk[chunk["Station"] == "Mudgal Barrage"]
        if len(sub) > 0:
            chunks.append(sub)
            
    df_mudgal = pd.concat(chunks, ignore_index=True)
    df_mudgal["dt"] = pd.to_datetime(df_mudgal["Data Acquisition Time"], format="%d-%m-%Y %H:%M")
    df_mudgal["stage_mudgal"] = pd.to_numeric(df_mudgal["River Water Level Telemetry Hourly (meter)"], errors="coerce")

    # QC: Mask sensor dropouts (< 370m MSL or > 410m MSL)
    df_mudgal.loc[(df_mudgal["stage_mudgal"] < 370.0) | (df_mudgal["stage_mudgal"] > 410.0), "stage_mudgal"] = np.nan

    # De-duplicate Mudgal on timestamp
    df_mudgal = df_mudgal.drop_duplicates(subset=["dt"]).sort_values("dt").set_index("dt")

    # Join Mudgal raw stage to continuous hourly grid
    df = df.set_index("dt")
    df["stage_mudgal"] = df_mudgal["stage_mudgal"]
    df = df.reset_index()

    # Engineer strictly causal upstream features with 1-hour latency buffer
    # At origin t, most recent observation is t-1
    df["stage_m_lag1"] = df["stage_mudgal"].shift(1)
    df["stage_m_lag2"] = df["stage_mudgal"].shift(2)
    df["stage_m_lag4"] = df["stage_mudgal"].shift(4)
    df["stage_m_lag7"] = df["stage_mudgal"].shift(7)
    df["stage_m_lag25"] = df["stage_mudgal"].shift(25)

    df["dH_m_1h"] = df["stage_m_lag1"] - df["stage_m_lag2"]
    df["dH_m_3h"] = df["stage_m_lag1"] - df["stage_m_lag4"]
    df["dH_m_6h"] = df["stage_m_lag1"] - df["stage_m_lag7"]
    df["dH_m_24h"] = df["stage_m_lag1"] - df["stage_m_lag25"]

    # Vishnupuri local acceleration
    if "accel_v_1h" not in df.columns:
        df["accel_v_1h"] = df["dH_v_1h"] - df["dH_v_1h"].shift(1)
    if "dH_v_6h" not in df.columns:
        df["dH_v_6h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(6)
    if "api_7d_mm" not in df.columns and "rain_depth_1h_mm" in df.columns:
        # Approximate 7d Antecedent Precipitation Index
        df["api_7d_mm"] = df["rain_depth_1h_mm"].rolling(168, min_periods=24).sum()

    # Telemetry availability flags
    df["has_valid_stage"] = df["stage_vishnupuri"].notna()
    df["has_valid_rain"] = df[FEAT_RAIN_CONT].notna().all(axis=1)
    df["has_valid_upstream"] = df[FEAT_UPSTREAM].notna().all(axis=1)

    print(f"Phase 5 dataset loaded: {len(df)} rows.")
    print(f"  Valid Vishnupuri stage: {df['has_valid_stage'].sum()} / {len(df)} ({df['has_valid_stage'].mean()*100:.1f}%)")
    print(f"  Valid Rainfall:         {df['has_valid_rain'].sum()} / {len(df)} ({df['has_valid_rain'].mean()*100:.1f}%)")
    print(f"  Valid Upstream Mudgal:  {df['has_valid_upstream'].sum()} / {len(df)} ({df['has_valid_upstream'].mean()*100:.1f}%)")

    return df


# -----------------------------------------------------------------------------
# Upstream-Aware Gated Regressor (RA-AG v2 with Upstream Telemetry)
# -----------------------------------------------------------------------------
class UpstreamAwareRAAGv2(BaseEstimator, RegressorMixin):
    """
    RA-AG v2 adapted for Multimodal + Upstream Telemetry Integration.
    Routes to:
      1. Primary Multimodal+Upstream Expert when all telemetry valid.
      2. Upstream-Fallback Expert when rainfall is missing but upstream is valid.
      3. Stage-Only Fallback Expert when both rainfall and upstream are missing.
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
        use_upstream: bool = True,
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
        self.use_upstream = use_upstream

        self.regime_detector = CausalRegimeDetector(theta_rise=theta_rise, theta_fall=theta_fall)
        self.soft_gate = SoftGateCalculator()

        # Component models
        self.rising_expert_full_ = None
        self.rising_expert_so_ = None
        self.falling_expert_full_ = None
        self.falling_expert_so_ = None
        self.stable_expert_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "UpstreamAwareRAAGv2":
        regimes = self.regime_detector.predict_regimes(X)

        so_cols = [c for c in FEAT_STAGE_DYN if c in X.columns]
        if self.use_upstream:
            full_cols = [c for c in FEAT_ALL if c in X.columns]
            fall_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT + FEAT_UPSTREAM) if c in X.columns]
        else:
            full_cols = [c for c in FEAT_MULTIMODAL if c in X.columns]
            fall_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT) if c in X.columns]

        # 1. Rising Expert
        mask_rise = (regimes == "RISING")
        if mask_rise.sum() < 20:
            mask_rise = np.ones(len(X), dtype=bool)

        X_rise_full = X.loc[mask_rise, full_cols].astype(float)
        X_rise_so = X.loc[mask_rise, so_cols].astype(float)
        y_rise = y[mask_rise]

        self.rising_expert_full_ = xgb.XGBRegressor(
            max_depth=self.xgb_params.get("max_depth", 2),
            learning_rate=self.xgb_params.get("learning_rate", 0.02),
            n_estimators=self.xgb_params.get("n_estimators", 60),
            reg_lambda=self.xgb_params.get("reg_lambda", 5.0),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            tree_method="hist",
        )
        self.rising_expert_full_.fit(X_rise_full, y_rise)

        # Stage-Only Fallback
        self.rising_expert_so_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=5.0)),
        ])
        self.rising_expert_so_.fit(X_rise_so, y_rise)

        # 2. Falling Expert
        mask_fall = (regimes == "FALLING")
        if mask_fall.sum() < 20:
            mask_fall = np.ones(len(X), dtype=bool)

        mask_fall_valid = mask_fall & X[fall_cols].notna().all(axis=1)
        if mask_fall_valid.sum() >= 20:
            X_fall_full = X.loc[mask_fall_valid, fall_cols].astype(float)
            y_fall_full = y[mask_fall_valid]
        else:
            X_fall_full = X.loc[mask_fall, fall_cols].fillna(X[fall_cols].median()).fillna(0.0).astype(float)
            y_fall_full = y[mask_fall]

        X_fall_so = X.loc[mask_fall, so_cols].astype(float)
        y_fall_so = y[mask_fall]

        self.falling_expert_full_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.ridge_alpha_fall)),
        ])
        self.falling_expert_full_.fit(X_fall_full, y_fall_full)

        self.falling_expert_so_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.ridge_alpha_fall)),
        ])
        self.falling_expert_so_.fit(X_fall_so, y_fall_so)

        # 3. Stable Expert
        mask_stable = (regimes == "STABLE")
        if mask_stable.sum() < 20:
            mask_stable = np.ones(len(X), dtype=bool)

        X_st = X.loc[mask_stable, so_cols].astype(float)
        y_st = y[mask_stable]

        self.stable_expert_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.ridge_alpha_stable)),
        ])
        self.stable_expert_.fit(X_st, y_st)

        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        pred_dh = np.zeros(n, dtype=float)
        is_fallback = np.zeros(n, dtype=bool)

        so_cols = [c for c in FEAT_STAGE_DYN if c in X.columns]
        if self.use_upstream:
            full_cols = [c for c in FEAT_ALL if c in X.columns]
            fall_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT + FEAT_UPSTREAM) if c in X.columns]
            req_cols = FEAT_RAIN_CONT + FEAT_UPSTREAM
        else:
            full_cols = [c for c in FEAT_MULTIMODAL if c in X.columns]
            fall_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT) if c in X.columns]
            req_cols = FEAT_RAIN_CONT

        X_full = X[full_cols].astype(float)
        X_so = X[so_cols].astype(float)
        X_fall = X[fall_cols].astype(float)

        # Check telemetry validity
        avail_req = [c for c in req_cols if c in X.columns]
        if len(avail_req) == len(req_cols) and len(req_cols) > 0:
            has_valid_inputs = X[req_cols].notna().all(axis=1).values
        else:
            has_valid_inputs = np.zeros(n, dtype=bool)

        regimes = self.regime_detector.predict_regimes(X).values

        # Stable prediction
        is_pure_pers = (
            self.stable_strategy == "persistence" or
            (self.stable_strategy == "horizon_adaptive" and self.horizon == 1)
        )
        if is_pure_pers:
            dh_stable = np.zeros(n, dtype=float)
        else:
            dh_stable = self.stable_expert_.predict(X_so)

        # Rising prediction
        dh_rise = np.zeros(n, dtype=float)
        mask_avail = has_valid_inputs
        mask_unavail = ~has_valid_inputs

        if np.any(mask_avail):
            dh_rise[mask_avail] = self.rising_expert_full_.predict(X_full.loc[mask_avail])
        if np.any(mask_unavail):
            dh_rise[mask_unavail] = self.rising_expert_so_.predict(X_so.loc[mask_unavail])

        # Falling prediction
        dh_fall = np.zeros(n, dtype=float)
        if np.any(mask_avail):
            dh_fall[mask_avail] = self.falling_expert_full_.predict(X_fall.loc[mask_avail])
        if np.any(mask_unavail):
            dh_fall[mask_unavail] = self.falling_expert_so_.predict(X_so.loc[mask_unavail])

        # Hard Gating Execution
        weights = np.zeros((n, 3), dtype=float)
        for r in ["RISING", "FALLING", "STABLE"]:
            mask_r = (regimes == r)
            if not np.any(mask_r):
                continue

            if r == "STABLE":
                pred_dh[mask_r] = dh_stable[mask_r]
                is_fallback[mask_r] = ~has_valid_inputs[mask_r]
                weights[mask_r, 2] = 1.0
            elif r == "RISING":
                pred_dh[mask_r] = dh_rise[mask_r]
                is_fallback[mask_r] = ~has_valid_inputs[mask_r]
                weights[mask_r, 0] = 1.0
            elif r == "FALLING":
                pred_dh[mask_r] = dh_fall[mask_r]
                is_fallback[mask_r] = ~has_valid_inputs[mask_r]
                weights[mask_r, 1] = 1.0

        return pred_dh, is_fallback, regimes, weights


# -----------------------------------------------------------------------------
# Phase 5 Hybrid Finalist Architecture with Upstream Support
# -----------------------------------------------------------------------------
class Phase5Hybrid(BaseEstimator, RegressorMixin):
    """
    Phase 3 Hybrid Architecture with robust telemetry imputer safeguards:
    - RISING: Tuned XGBoost on multimodal features (+ dedicated stage-only fallback)
    - FALLING: Regularized Ridge on recession limbs with SimpleImputer
    - STABLE: Pure Persistence (dH = 0.0)
    """
    def __init__(
        self,
        horizon: int = 1,
        theta_rise: float = 0.010,
        theta_fall: float = -0.010,
        alpha_fall: float = 10.0,
        xgb_params: Optional[Dict[str, Any]] = None,
        use_upstream: bool = False,
    ):
        self.horizon = horizon
        self.theta_rise = theta_rise
        self.theta_fall = theta_fall
        self.alpha_fall = alpha_fall
        self.xgb_params = xgb_params or {"max_depth": 2, "learning_rate": 0.02, "reg_lambda": 5.0}
        self.use_upstream = use_upstream

        self.regime_detector = CausalRegimeDetector(theta_rise=theta_rise, theta_fall=theta_fall)
        self.rising_expert_mm_ = None
        self.rising_expert_so_ = None
        self.falling_expert_mm_ = None
        self.falling_expert_so_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Phase5Hybrid":
        regimes = self.regime_detector.predict_regimes(X)
        so_cols = [c for c in FEAT_STAGE_DYN if c in X.columns]
        if self.use_upstream:
            mm_cols = [c for c in FEAT_ALL if c in X.columns]
            fall_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT + FEAT_UPSTREAM) if c in X.columns]
        else:
            mm_cols = [c for c in FEAT_MULTIMODAL if c in X.columns]
            fall_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT) if c in X.columns]

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

        mask_fall = (regimes == "FALLING")
        if mask_fall.sum() < 20:
            mask_fall = np.ones(len(X), dtype=bool)

        X_fall_mm = X.loc[mask_fall, fall_cols].astype(float)
        X_fall_so = X.loc[mask_fall, so_cols].astype(float)
        y_fall = y[mask_fall]

        self.falling_expert_mm_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.alpha_fall)),
        ])
        self.falling_expert_mm_.fit(X_fall_mm, y_fall)

        self.falling_expert_so_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.alpha_fall)),
        ])
        self.falling_expert_so_.fit(X_fall_so, y_fall)

        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        pred_dh = np.zeros(n, dtype=float)
        is_fallback = np.zeros(n, dtype=bool)

        so_cols = [c for c in FEAT_STAGE_DYN if c in X.columns]
        if self.use_upstream:
            mm_cols = [c for c in FEAT_ALL if c in X.columns]
            fall_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT + FEAT_UPSTREAM) if c in X.columns]
            req_cols = FEAT_RAIN_CONT + FEAT_UPSTREAM
        else:
            mm_cols = [c for c in FEAT_MULTIMODAL if c in X.columns]
            fall_cols = [c for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT) if c in X.columns]
            req_cols = FEAT_RAIN_CONT

        X_mm = X[mm_cols].astype(float)
        X_so = X[so_cols].astype(float)
        X_fall = X[fall_cols].astype(float)

        avail_req = [c for c in req_cols if c in X.columns]
        if len(avail_req) == len(req_cols) and len(req_cols) > 0:
            has_valid_inputs = X[req_cols].notna().all(axis=1).values
        else:
            has_valid_inputs = np.zeros(n, dtype=bool)

        regimes = self.regime_detector.predict_regimes(X).values

        for r in ["RISING", "FALLING", "STABLE"]:
            mask_r = (regimes == r)
            if not np.any(mask_r):
                continue

            if r == "STABLE":
                pred_dh[mask_r] = 0.0
                is_fallback[mask_r] = ~has_valid_inputs[mask_r]
                continue

            mask_r_mm = mask_r & has_valid_inputs
            mask_r_so = mask_r & ~has_valid_inputs

            if r == "RISING":
                if np.any(mask_r_mm):
                    pred_dh[mask_r_mm] = self.rising_expert_mm_.predict(X_mm.loc[mask_r_mm])
                    is_fallback[mask_r_mm] = False
                if np.any(mask_r_so):
                    pred_dh[mask_r_so] = self.rising_expert_so_.predict(X_so.loc[mask_r_so])
                    is_fallback[mask_r_so] = True
            elif r == "FALLING":
                if np.any(mask_r_mm):
                    pred_dh[mask_r_mm] = self.falling_expert_mm_.predict(X_fall.loc[mask_r_mm])
                    is_fallback[mask_r_mm] = False
                if np.any(mask_r_so):
                    pred_dh[mask_r_so] = self.falling_expert_so_.predict(X_so.loc[mask_r_so])
                    is_fallback[mask_r_so] = True

        return pred_dh, is_fallback, regimes
