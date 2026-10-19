"""
river_forecasting/src/phase6_medium_range.py
============================================
Phase 6: Medium-Range Upstream Signal and Flood-Wave Routing Experiment.
Evaluates whether authentic Mudgal Barrage telemetry provides predictive value
for medium-range river-stage forecasts (+18h, +24h, +36h) at Vishnupuri Barrage.

Architectural Standards:
  - Genuine targets directly from observed Vishnupuri stage measurements.
  - Zero lookahead: targets never enter feature sets; training origins separated by >= H hours.
  - Latency-aware: upstream features strictly lagged by >= 1 hour.
  - Robust telemetry fallbacks: no silent zero-filling of missing rainfall or upstream telemetry.
"""

from __future__ import annotations

import json
import logging
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

from river_forecasting.src.phase1_ml_pipeline import SEED, _bootstrap_ci
from river_forecasting.src.phase4_ra_ag_v2 import compute_moving_block_bootstrap_p_value

log = logging.getLogger("phase6_medium_range")

# -----------------------------------------------------------------------------
# Feature Set Definitions for Medium-Range Horizons (+18h, +24h, +36h)
# -----------------------------------------------------------------------------
# Local Vishnupuri Stage History (available at origin t)
FEAT_LOCAL_STAGE = [
    "stage_vishnupuri",
    "dH_v_1h",
    "dH_v_3h",
    "dH_v_6h",
    "dH_v_12h",
    "dH_v_24h",
    "accel_v_1h",
]

# Catchment Rainfall Features (available at origin t)
FEAT_RAINFALL = [
    "rain_depth_1h_mm",
    "rain_3h_mm",
    "rain_6h_mm",
    "rain_12h_mm",
    "rain_24h_mm",
    "rain_48h_mm",
    "api_7d_mm",
]

# Upstream Telemetry Features (Mudgal Barrage, strictly lagged >= 1 hour for latency)
FEAT_UPSTREAM_MUDGAL = [
    "stage_m_lag1",
    "dH_m_1h",
    "dH_m_3h",
    "dH_m_6h",
    "dH_m_12h",
    "dH_m_18h",
    "dH_m_24h",
]

FEAT_STAGE_RAIN = FEAT_LOCAL_STAGE + FEAT_RAINFALL
FEAT_STAGE_UPSTREAM = FEAT_LOCAL_STAGE + FEAT_UPSTREAM_MUDGAL
FEAT_ALL = FEAT_LOCAL_STAGE + FEAT_RAINFALL + FEAT_UPSTREAM_MUDGAL

HORIZONS_MEDIUM = [18, 24, 36]


# -----------------------------------------------------------------------------
# Simple Lagged-Upstream-Change Baseline (Config F)
# -----------------------------------------------------------------------------
class SimpleLaggedUpstreamBaseline(BaseEstimator, RegressorMixin):
    """
    Kinematic Flood-Wave Routing Proxy Baseline (Config F).
    Estimates Vishnupuri stage change directly from upstream Mudgal rate of change
    at the hydrologically identified flood-wave lag:
        Delta_H_v(t + h) = alpha + beta * Delta_H_m(t - tau_wave)
    If upstream data is missing, falls back cleanly to persistence (dH = 0.0).
    """
    def __init__(self, horizon: int = 18, wave_lag_hr: int = 14):
        self.horizon = horizon
        self.wave_lag_hr = wave_lag_hr
        self.alpha_ = 0.0
        self.beta_ = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SimpleLaggedUpstreamBaseline":
        # Feature column for upstream change
        col_up = "dH_m_12h" if "dH_m_12h" in X.columns else "dH_m_6h"
        x_up = X[col_up].values
        
        valid = ~np.isnan(x_up) & ~np.isnan(y)
        if valid.sum() > 20:
            p = np.polyfit(x_up[valid], y[valid], 1)
            self.beta_ = float(p[0])
            self.alpha_ = float(p[1])
        else:
            self.alpha_ = 0.0
            self.beta_ = 0.0
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        col_up = "dH_m_12h" if "dH_m_12h" in X.columns else "dH_m_6h"
        x_up = X[col_up].values if col_up in X.columns else np.full(len(X), np.nan)
        
        is_missing = np.isnan(x_up)
        pred = np.where(is_missing, 0.0, self.alpha_ + self.beta_ * np.nan_to_num(x_up, nan=0.0))
        return pred, is_missing


# -----------------------------------------------------------------------------
# Regularized Ridge Model with Telemetry Outage Fallback (Config G)
# -----------------------------------------------------------------------------
class MediumRangeRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Regularized Ridge Regressor with SimpleImputer and Stage-Only Outage Fallback.
    When rainfall or upstream telemetry drops out, automatically routes to
    a dedicated stage-only regularized model rather than imputing zero.
    """
    def __init__(self, alpha: float = 50.0, use_upstream: bool = True, use_rainfall: bool = True):
        self.alpha = alpha
        self.use_upstream = use_upstream
        self.use_rainfall = use_rainfall
        self.model_full_ = None
        self.model_so_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MediumRangeRidgeRegressor":
        so_cols = [c for c in FEAT_LOCAL_STAGE if c in X.columns]
        req_cols = []
        if self.use_rainfall:
            req_cols += [c for c in FEAT_RAINFALL if c in X.columns]
        if self.use_upstream:
            req_cols += [c for c in FEAT_UPSTREAM_MUDGAL if c in X.columns]
        full_cols = so_cols + req_cols

        # 1. Fit Stage-Only Model (Fallback)
        self.model_so_ = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=self.alpha)),
        ])
        self.model_so_.fit(X[so_cols].astype(float), y)

        # 2. Fit Full Multimodal Model
        if req_cols:
            self.model_full_ = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("est", Ridge(alpha=self.alpha)),
            ])
            self.model_full_.fit(X[full_cols].astype(float), y)
        else:
            self.model_full_ = self.model_so_

        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        so_cols = [c for c in FEAT_LOCAL_STAGE if c in X.columns]
        req_cols = []
        if self.use_rainfall:
            req_cols += [c for c in FEAT_RAINFALL if c in X.columns]
        if self.use_upstream:
            req_cols += [c for c in FEAT_UPSTREAM_MUDGAL if c in X.columns]
        full_cols = so_cols + req_cols

        if req_cols:
            has_valid_inputs = X[req_cols].notna().all(axis=1).values
        else:
            has_valid_inputs = np.ones(n, dtype=bool)

        pred_dh = np.zeros(n, dtype=float)
        is_fallback = ~has_valid_inputs

        if np.any(has_valid_inputs):
            pred_dh[has_valid_inputs] = self.model_full_.predict(X.loc[has_valid_inputs, full_cols].astype(float))
        if np.any(is_fallback):
            pred_dh[is_fallback] = self.model_so_.predict(X.loc[is_fallback, so_cols].astype(float))

        return pred_dh, is_fallback


# -----------------------------------------------------------------------------
# Carefully Regularized Nonlinear XGBoost Model (Config H)
# -----------------------------------------------------------------------------
class MediumRangeXGBRegressor(BaseEstimator, RegressorMixin):
    """
    Carefully Regularized Tree Ensemble for Medium-Range Forecasting.
    Uses conservative max_depth=2, n_estimators=50, reg_lambda=10.0 to prevent
    overfitting to high-frequency noise. Includes dedicated stage-only fallback.
    """
    def __init__(
        self,
        max_depth: int = 2,
        learning_rate: float = 0.02,
        n_estimators: int = 50,
        reg_lambda: float = 10.0,
        use_upstream: bool = True,
        use_rainfall: bool = True,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.use_upstream = use_upstream
        self.use_rainfall = use_rainfall
        self.xgb_full_ = None
        self.xgb_so_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MediumRangeXGBRegressor":
        so_cols = [c for c in FEAT_LOCAL_STAGE if c in X.columns]
        req_cols = []
        if self.use_rainfall:
            req_cols += [c for c in FEAT_RAINFALL if c in X.columns]
        if self.use_upstream:
            req_cols += [c for c in FEAT_UPSTREAM_MUDGAL if c in X.columns]
        full_cols = so_cols + req_cols

        # 1. Fit Stage-Only XGBoost Fallback
        self.xgb_so_ = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            reg_lambda=self.reg_lambda,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            tree_method="hist",
        )
        self.xgb_so_.fit(X[so_cols].astype(float), y)

        # 2. Fit Full Model
        if req_cols:
            self.xgb_full_ = xgb.XGBRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                reg_lambda=self.reg_lambda,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                tree_method="hist",
            )
            self.xgb_full_.fit(X[full_cols].astype(float), y)
        else:
            self.xgb_full_ = self.xgb_so_

        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        so_cols = [c for c in FEAT_LOCAL_STAGE if c in X.columns]
        req_cols = []
        if self.use_rainfall:
            req_cols += [c for c in FEAT_RAINFALL if c in X.columns]
        if self.use_upstream:
            req_cols += [c for c in FEAT_UPSTREAM_MUDGAL if c in X.columns]
        full_cols = so_cols + req_cols

        if req_cols:
            has_valid_inputs = X[req_cols].notna().all(axis=1).values
        else:
            has_valid_inputs = np.ones(n, dtype=bool)

        pred_dh = np.zeros(n, dtype=float)
        is_fallback = ~has_valid_inputs

        if np.any(has_valid_inputs):
            pred_dh[has_valid_inputs] = self.xgb_full_.predict(X.loc[has_valid_inputs, full_cols].astype(float))
        if np.any(is_fallback):
            pred_dh[is_fallback] = self.xgb_so_.predict(X.loc[is_fallback, so_cols].astype(float))

        return pred_dh, is_fallback
