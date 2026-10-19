"""
river_forecasting/src/feature_pipeline.py
=========================================
Feature Engineering, Timestamp Alignment, and Leakage-Free Dataset Preparation.

Designed for real hourly river stage observations and optional catchment rainfall.
Guarantees:
1. Strict chronological ordering with zero future lookahead (forecasts at time t use only data <= t).
2. Preserves authentic observations and tracks anomaly flags.
3. No synthetic rainfall or stage generation.
4. Flexible multi-horizon target generation (t+1, t+2, t+3 hours).
"""

from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd


def prepare_hourly_timeseries(stage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns stage observations to a regular 1-hour grid, preserving raw observations
    and flagging gaps and anomalies.
    """
    df = stage_df.copy()
    if "timestamp_ist" not in df.columns:
        raise ValueError("Missing 'timestamp_ist' column in stage DataFrame.")

    df["dt"] = pd.to_datetime(df["timestamp_ist"])
    df.sort_values(by="dt", inplace=True)
    df.drop_duplicates(subset=["dt"], keep="first", inplace=True)

    min_dt = df["dt"].min()
    max_dt = df["dt"].max()

    # Regular 1-hour grid
    hourly_index = pd.date_range(start=min_dt, end=max_dt, freq="1h", name="dt")
    full_df = pd.DataFrame(index=hourly_index).reset_index()

    # Merge observations
    merged = pd.merge(full_df, df, on="dt", how="left")

    # Tracking flags
    merged["is_observed"] = merged["stage_m"].notna()
    merged["is_anomaly_flagged"] = merged["quality_flag"].fillna("MISSING").ne("VALID")

    return merged


def build_forecasting_features(
    aligned_df: pd.DataFrame,
    rainfall_df: Optional[pd.DataFrame] = None,
    horizons: List[int] = [1, 2, 3],
    stage_lags: List[int] = [1, 2, 3, 4, 6],
    rolling_windows: List[int] = [3, 6, 12],
    rainfall_windows: List[int] = [1, 3, 6, 12, 24, 48, 72]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Constructs leakage-free autoregressive stage features and optional catchment rainfall features.
    
    Returns:
        feature_df: DataFrame with timestamps, features, and target columns.
        feature_cols: List of generated predictor feature column names.
        target_cols: List of target column names for horizons.
    """
    df = aligned_df.copy().sort_values(by="dt").reset_index(drop=True)

    # Autoregressive past fill for feature computation ONLY (max limit 3 hours, never modifies raw stage_m)
    # This prevents short 1-hour network outages from discarding subsequent rows while strictly using past data <= t
    past_stage = df["stage_m"].copy()
    filled_past_stage = past_stage.ffill(limit=3)

    feature_cols = []

    # 1. Base stage at issue time t
    df["feat_stage_t"] = filled_past_stage
    feature_cols.append("feat_stage_t")

    # 2. Stage lags strictly prior to t: t-1, t-2, t-3, ...
    for lag in stage_lags:
        col_name = f"feat_stage_lag_{lag}h"
        df[col_name] = filled_past_stage.shift(lag)
        feature_cols.append(col_name)

    # 3. Differences / rates of change (dH/dt)
    df["feat_stage_diff_1h"] = filled_past_stage - filled_past_stage.shift(1)
    df["feat_stage_diff_2h"] = filled_past_stage.shift(1) - filled_past_stage.shift(2)
    df["feat_stage_diff_3h"] = filled_past_stage - filled_past_stage.shift(3)
    df["feat_stage_diff_6h"] = filled_past_stage - filled_past_stage.shift(6)
    feature_cols.extend([
        "feat_stage_diff_1h", "feat_stage_diff_2h",
        "feat_stage_diff_3h", "feat_stage_diff_6h"
    ])

    # 4. Rolling statistics over past window up to time t
    for w in rolling_windows:
        mean_col = f"feat_stage_rolling_mean_{w}h"
        std_col = f"feat_stage_rolling_std_{w}h"
        min_col = f"feat_stage_rolling_min_{w}h"
        max_col = f"feat_stage_rolling_max_{w}h"

        roll = filled_past_stage.rolling(window=w, min_periods=max(2, w // 2))
        df[mean_col] = roll.mean()
        df[std_col] = roll.std()
        df[min_col] = roll.min()
        df[max_col] = roll.max()

        feature_cols.extend([mean_col, std_col, min_col, max_col])

    # 5. Anomaly context flag at issue time t
    df["feat_is_anomaly"] = df["is_anomaly_flagged"].astype(float)
    feature_cols.append("feat_is_anomaly")

    # 6. Catchment Rainfall Features (IF real rainfall is available)
    if rainfall_df is not None and not rainfall_df.empty:
        rain = rainfall_df.copy()
        if "dt" not in rain.columns and "timestamp_ist" in rain.columns:
            rain["dt"] = pd.to_datetime(rain["timestamp_ist"])
        elif "dt" not in rain.columns and "timestamp_utc" in rain.columns:
            rain["dt"] = pd.to_datetime(rain["timestamp_utc"]) + pd.Timedelta(hours=5, minutes=30)

        # Aggregate rainfall hourly
        hourly_rain = rain.groupby(pd.Grouper(key="dt", freq="1h"))["mean_rainfall_mm_hr"].sum().reset_index()
        df = pd.merge(df, hourly_rain.rename(columns={"mean_rainfall_mm_hr": "rain_1h"}), on="dt", how="left")
        df["rain_1h"] = df["rain_1h"].fillna(0.0)

        for rw in rainfall_windows:
            rcol = f"feat_rain_accum_{rw}h"
            df[rcol] = df["rain_1h"].rolling(window=rw, min_periods=1).sum()
            feature_cols.append(rcol)

    # 7. Targets for horizons h (e.g. 1h, 2h, 3h ahead)
    # IMPORTANT: Target MUST be authentic un-imputed ground observation stage_m
    target_cols = []
    for h in horizons:
        target_stage_col = f"target_stage_{h}h"
        target_delta_col = f"target_delta_{h}h"

        df[target_stage_col] = df["stage_m"].shift(-h)
        df[target_delta_col] = df[target_stage_col] - df["stage_m"]

        target_cols.append(target_stage_col)
        target_cols.append(target_delta_col)

    # Valid forecast origin flag:
    # Must have authentic ground observation at time t and no missing features
    df["is_valid_forecast_origin"] = (
        df["is_observed"] &
        df[feature_cols].notna().all(axis=1)
    )

    return df, feature_cols, target_cols


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    horizon_embargo_hours: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Splits dataset chronologically into train, validation, and test subsets.
    No random shuffling is applied to prevent temporal leakage.
    
    If horizon_embargo_hours > 0, removes origins at the tail of train and validation
    so that multi-hour target horizons never cross into subsequent evaluation sets.
    """
    valid_df = df[df["is_valid_forecast_origin"]].copy().sort_values(by="dt").reset_index(drop=True)
    n = len(valid_df)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_raw = valid_df.iloc[:train_end].copy()
    val_raw = valid_df.iloc[train_end:val_end].copy()
    test_df = valid_df.iloc[val_end:].copy()

    if horizon_embargo_hours > 0:
        val_start_dt = val_raw["dt"].min()
        test_start_dt = test_df["dt"].min()

        # Enforce embargo buffer: train origins must end <= val_start - embargo
        train_df = train_raw[train_raw["dt"] <= (val_start_dt - pd.Timedelta(hours=horizon_embargo_hours))].copy()
        # Validation origins must end <= test_start - embargo
        val_df = val_raw[val_raw["dt"] <= (test_start_dt - pd.Timedelta(hours=horizon_embargo_hours))].copy()
    else:
        train_df = train_raw
        val_df = val_raw

    split_info = {
        "total_valid_origins": n,
        "horizon_embargo_hours": horizon_embargo_hours,
        "train": {
            "count": len(train_df),
            "start": train_df["dt"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": train_df["dt"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "min_stage": float(train_df["stage_m"].min()),
            "max_stage": float(train_df["stage_m"].max()),
        },
        "val": {
            "count": len(val_df),
            "start": val_df["dt"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": val_df["dt"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "min_stage": float(val_df["stage_m"].min()),
            "max_stage": float(val_df["stage_m"].max()),
        },
        "test": {
            "count": len(test_df),
            "start": test_df["dt"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": test_df["dt"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "min_stage": float(test_df["stage_m"].min()),
            "max_stage": float(test_df["stage_m"].max()),
        }
    }

    return train_df, val_df, test_df, split_info
