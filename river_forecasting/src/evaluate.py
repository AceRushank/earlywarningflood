"""
river_forecasting/src/evaluate.py
=================================
Model Evaluation Module for River Stage Forecasting.

Computes:
1. Standard hydrological metrics: MAE, RMSE, MaxAE, Nash-Sutcliffe Efficiency (NSE).
2. Segmented evaluation: Overall, Peak-stage (H >= 354.80 m), and Rapid-rise (|dH/dt| >= 0.15 m/hr).
3. Prediction exports for visual inspection and auditability.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def compute_hydrological_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Optional[float]]:
    """
    Computes MAE, RMSE, MaxAE, and Nash-Sutcliffe Efficiency (NSE).
    Filters out any NaN pairs.
    """
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        return {
            "count": 0,
            "mae_m": None,
            "rmse_m": None,
            "max_ae_m": None,
            "nse": None
        }

    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    n = len(yt)

    errors = yp - yt
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_ae = float(np.max(np.abs(errors)))

    # Nash-Sutcliffe Efficiency (NSE)
    var_obs = np.sum((yt - np.mean(yt)) ** 2)
    if var_obs > 1e-9:
        nse = float(1.0 - (np.sum(errors ** 2) / var_obs))
    else:
        nse = None

    return {
        "count": int(n),
        "mae_m": round(mae, 4),
        "rmse_m": round(rmse, 4),
        "max_ae_m": round(max_ae, 4),
        "nse": round(nse, 4) if nse is not None else None
    }


def evaluate_forecasts(
    model: Any,
    test_df: pd.DataFrame,
    horizons: List[int] = [1, 2, 3],
    peak_threshold_m: float = 354.80,
    rapid_rise_threshold_m_hr: float = 0.15
) -> Dict[str, Any]:
    """
    Evaluates model across horizons and segments (overall, peak-stage, rapid-rise).
    """
    preds_by_horizon = model.predict(test_df)

    results_by_horizon = {}
    for h in horizons:
        target_col = f"target_stage_{h}h"
        y_true = test_df[target_col].to_numpy()
        y_pred = preds_by_horizon[h]

        # 1. Overall
        overall_metrics = compute_hydrological_metrics(y_true, y_pred)

        # 2. Peak stage regime (where true target stage or issue stage >= peak_threshold_m)
        peak_mask = (test_df["stage_m"] >= peak_threshold_m) | (test_df[target_col] >= peak_threshold_m)
        peak_metrics = compute_hydrological_metrics(
            y_true[peak_mask.to_numpy()],
            y_pred[peak_mask.to_numpy()]
        )

        # 3. Antecedent Rapid Rise regime (dH_t = H_t - H_{t-1} >= +threshold; strictly positive, available at issue time t)
        diff_1h = test_df["feat_stage_diff_1h"]
        ant_rise_mask = diff_1h >= rapid_rise_threshold_m_hr
        ant_rise_metrics = compute_hydrological_metrics(
            y_true[ant_rise_mask.to_numpy()],
            y_pred[ant_rise_mask.to_numpy()]
        )
        ant_rise_metrics["timestamps"] = test_df.loc[ant_rise_mask, "dt"].dt.strftime("%Y-%m-%d %H:%M").tolist()

        # 4. Antecedent Rapid Change / Volatility regime (|dH_t| >= threshold; includes both steep rises and steep recessions)
        ant_change_mask = diff_1h.abs() >= rapid_rise_threshold_m_hr
        ant_change_metrics = compute_hydrological_metrics(
            y_true[ant_change_mask.to_numpy()],
            y_pred[ant_change_mask.to_numpy()]
        )
        ant_change_metrics["timestamps"] = test_df.loc[ant_change_mask, "dt"].dt.strftime("%Y-%m-%d %H:%M").tolist()

        # 5. Future Rapid Rise regime (H_{t+h} - H_t >= threshold; post-hoc evaluation of surge events over the forecast horizon)
        target_delta_col = f"target_delta_{h}h"
        fut_rise_mask = test_df[target_delta_col] >= rapid_rise_threshold_m_hr
        fut_rise_metrics = compute_hydrological_metrics(
            y_true[fut_rise_mask.to_numpy()],
            y_pred[fut_rise_mask.to_numpy()]
        )
        fut_rise_metrics["timestamps"] = test_df.loc[fut_rise_mask, "dt"].dt.strftime("%Y-%m-%d %H:%M").tolist()

        results_by_horizon[f"{h}h"] = {
            "overall": overall_metrics,
            "peak_stage": peak_metrics,
            "antecedent_rapid_rise": ant_rise_metrics,
            "antecedent_rapid_change": ant_change_metrics,
            "future_rapid_rise": fut_rise_metrics,
            # Backward-compatible alias pointing to true antecedent rapid rise
            "rapid_rise": ant_rise_metrics
        }

    return {
        "model_name": getattr(model, "name", "Model"),
        "horizons": results_by_horizon
    }
