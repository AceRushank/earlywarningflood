"""
river_forecasting/src/baseline_models.py
========================================
Forecasting Models for Godavari River Stage at Vishnupuri Barrage.

Includes:
1. Persistence Baseline: H_hat(t+h) = H(t).
2. Water-Level-Only Autoregressive Model: Ridge & Gradient Boosting predicting stage deltas.
3. Water-Level + Catchment-Rainfall Model: Modular framework awaiting authentic MOSDAC rainfall.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class PersistenceBaselineModel:
    """
    Standard hydrological persistence benchmark:
    Predicts that future river stage at horizon h equals current observed stage at time t.
    """

    def __init__(self, horizons: List[int] = [1, 2, 3]):
        self.horizons = horizons
        self.name = "Persistence Baseline"

    def fit(self, X: pd.DataFrame, y: Dict[int, pd.Series]) -> "PersistenceBaselineModel":
        # Persistence has no parameters to fit
        return self

    def predict(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Returns {h: array_of_predictions} where each pred equals df['feat_stage_t'].
        """
        predictions = {}
        stage_t = df["feat_stage_t"].to_numpy()
        for h in self.horizons:
            predictions[h] = stage_t.copy()
        return predictions


class WaterLevelOnlyForecaster:
    """
    Autoregressive stage forecasting model utilizing past stage lags, differences,
    and rolling dynamics available up to issue time t.
    Predicts the stage delta: dH(t -> t+h), then adds H(t) to ensure physical continuity.
    """

    def __init__(self, horizons: List[int] = [1, 2, 3], model_type: str = "ridge"):
        self.horizons = horizons
        self.model_type = model_type.lower()
        self.models: Dict[int, Any] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.feature_cols: List[str] = []
        self.name = f"Water-Level-Only ({self.model_type.capitalize()})"

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        val_df: Optional[pd.DataFrame] = None
    ) -> "WaterLevelOnlyForecaster":
        self.feature_cols = feature_cols

        for h in self.horizons:
            delta_target_col = f"target_delta_{h}h"
            stage_target_col = f"target_stage_{h}h"

            # Filter valid rows where target and all features are observed
            valid_train = train_df[
                train_df[delta_target_col].notna() &
                train_df[stage_target_col].notna() &
                train_df[self.feature_cols].notna().all(axis=1)
            ]
            X_train = valid_train[self.feature_cols].to_numpy()
            y_train = valid_train[delta_target_col].to_numpy()

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            if self.model_type == "ridge":
                regressor = Ridge(alpha=10.0, random_state=42)
            elif self.model_type == "gradient_boosting":
                regressor = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
                )
            else:
                regressor = Ridge(alpha=1.0, random_state=42)

            regressor.fit(X_train_scaled, y_train)

            self.models[h] = regressor
            self.scalers[h] = scaler

        return self

    def predict(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Predicts future stage: H_hat(t+h) = H(t) + delta_hat(t -> t+h).
        """
        predictions = {}
        stage_t = df["feat_stage_t"].to_numpy()

        X_df = df[self.feature_cols].copy()
        if X_df.isna().any().any():
            X_df = X_df.bfill().ffill().fillna(0.0)

        for h in self.horizons:
            scaler = self.scalers[h]
            model = self.models[h]
            X = X_df.to_numpy()
            X_scaled = scaler.transform(X)
            delta_preds = model.predict(X_scaled)
            predictions[h] = stage_t + delta_preds

        return predictions


class RainfallRunoffForecaster:
    """
    Joint stage-rainfall forecasting framework.
    Combines autoregressive stage features with upstream catchment rainfall accumulations.
    
    Status: Prepared in code; awaiting authentic MOSDAC HEM rainfall extraction.
    Strictly prohibits synthetic data generation or fake metric reporting.
    """

    def __init__(self, horizons: List[int] = [1, 2, 3]):
        self.horizons = horizons
        self.name = "Water-Level + Catchment-Rainfall Model"
        self.status = "AWAITING_REAL_RAINFALL"
        self.models: Dict[int, Any] = {}

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> "RainfallRunoffForecaster":
        # Check if real rainfall features are present
        rain_features = [c for c in feature_cols if "rain" in c.lower()]
        if not rain_features:
            raise RuntimeError(
                "Cannot fit RainfallRunoffForecaster: No authentic rainfall features found in dataset. "
                "Synthetic rainfall training is strictly prohibited by project protocol."
            )
        self.status = "TRAINED"
        return self

    def predict(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        if self.status != "TRAINED":
            raise RuntimeError("RainfallRunoffForecaster has not been trained (awaiting real rainfall data).")
        return {}
