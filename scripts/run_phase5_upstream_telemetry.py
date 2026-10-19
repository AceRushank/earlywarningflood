"""
scripts/run_phase5_upstream_telemetry.py
========================================
Full execution pipeline for Phase 5: Upstream Telemetry Integration and Out-of-Sample Surge Forecasting.
Trains and evaluates Persistence, XGBoost, Hybrid, and RA-AG v2 Hard across:
  1. Local stage features only
  2. Local stage plus rainfall
  3. Local stage plus verified upstream telemetry
  4. Local stage plus rainfall plus upstream telemetry

Saves all deliverables in river_forecasting/outputs/phase5_upstream_telemetry/
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.phase1_ml_pipeline import (
    SEED,
    HORIZONS,
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_MULTIMODAL,
    _bootstrap_ci,
)
from river_forecasting.src.phase3_finalist_comparison import RAAGXGBHybrid
from river_forecasting.src.phase4_ra_ag_v2 import (
    RAAGv2,
    compute_moving_block_bootstrap_p_value,
)
from river_forecasting.src.phase5_upstream_telemetry import (
    FEAT_UPSTREAM,
    FEAT_STAGE_UPSTREAM,
    FEAT_ALL,
    load_phase5_dataset,
    UpstreamAwareRAAGv2,
    Phase5Hybrid,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ALIGNED_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"
RAW_TEL_CSV = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"
OUTPUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase5_upstream_telemetry"
MODELS_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.02,
    "n_estimators": 60,
    "reg_lambda": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "tree_method": "hist",
}

FEATURE_SETS = {
    "stage_only": FEAT_STAGE_DYN,
    "stage_rain": FEAT_MULTIMODAL,
    "stage_upstream": FEAT_STAGE_UPSTREAM,
    "stage_rain_upstream": FEAT_ALL,
}

MODELS = ["Persistence", "XGBoost", "Hybrid", "RAAG_v2_Hard"]


def train_and_evaluate():
    print("=" * 80)
    print("STARTING PHASE 5: CONTROLLED UPSTREAM TELEMETRY EXPERIMENT")
    print("=" * 80)

    # 1. Ingest Data
    df = load_phase5_dataset(ALIGNED_CSV, RAW_TEL_CSV)

    # Chronological Split masks
    train_mask = (df["dt"] >= "2026-06-15 00:00:00") & (df["dt"] <= "2026-08-10 23:00:00")
    val_mask   = (df["dt"] >= "2026-08-11 00:00:00") & (df["dt"] <= "2026-08-31 23:00:00")
    test_mask  = (df["dt"] >= "2026-09-01 00:00:00") & (df["dt"] <= "2026-09-30 23:00:00")
    test_sync_mask = (df["dt"] >= "2026-09-08 06:00:00") & (df["dt"] <= "2026-09-22 19:00:00")

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()
    df_test_sync = df[test_sync_mask].copy()

    print(f"Data Splits: Train={len(df_train)}h, Val={len(df_val)}h, Test(Sep)={len(df_test)}h, TestSync={len(df_test_sync)}h")

    # Tracking containers
    feature_ablation_records = []
    horizon_records = []
    event_records = []
    outage_records = []
    paired_records = []
    predictions_df = df_test[["dt", "timestamp_ist", "stage_vishnupuri", "stage_mudgal", "is_test_sync_rainfall"]].copy()

    # Model artifact cache
    model_manifest = {
        "phase": 5,
        "title": "Phase 5 Upstream Telemetry Integration and Out-of-Sample Surge Forecasting",
        "upstream_station": "Mudgal Barrage (Godavari River, Parbhani)",
        "upstream_distance_km": 105.0,
        "telemetry_latency_buffer_hr": 1,
        "horizons": HORIZONS,
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "xgb_params": XGB_PARAMS,
        "models_trained": {},
    }

    # Evaluate for each horizon
    for h in HORIZONS:
        tgt_dh = f"target_v_dH_{h}h"
        tgt_stage = f"target_v_{h}h"
        print(f"\n=======================================================")
        print(f"EVALUATING LEAD TIME: +{h}h (Target: {tgt_dh})")
        print(f"=======================================================")

        # Valid rows in train and test
        valid_train = df_train[df_train[tgt_dh].notna() & df_train["has_valid_stage"]].copy()
        valid_test = df_test[df_test[tgt_dh].notna() & df_test["has_valid_stage"]].copy()
        valid_test_sync = df_test_sync[df_test_sync[tgt_dh].notna() & df_test_sync["has_valid_stage"]].copy()

        y_train = valid_train[tgt_dh].values
        y_test = valid_test[tgt_dh].values
        y_test_sync = valid_test_sync[tgt_dh].values

        stage_test = valid_test["stage_vishnupuri"].values
        true_stage_test = valid_test[tgt_stage].values
        stage_test_sync = valid_test_sync["stage_vishnupuri"].values
        true_stage_test_sync = valid_test_sync[tgt_stage].values

        # Prepare training dataframe with features imputed for linear expert fitting
        valid_train_fit = valid_train.copy()
        for c in (FEAT_STAGE_DYN + FEAT_RAIN_CONT + FEAT_UPSTREAM):
            if c in valid_train_fit.columns:
                valid_train_fit[c] = valid_train_fit[c].fillna(valid_train_fit[c].median()).fillna(0.0)

        # Store predictions per model & feature set for this horizon
        preds_dh_test = {}
        preds_dh_sync = {}

        # ---------------------------------------------------------------------
        # Train & Predict Models Across 4 Feature Configurations
        # ---------------------------------------------------------------------
        for fset_name, fcols in FEATURE_SETS.items():
            print(f"\n--- Feature Configuration: {fset_name} (Features: {len(fcols)}) ---")

            # Subset available columns
            X_tr = valid_train_fit[fcols].copy()
            X_te = valid_test[fcols].copy()
            X_sy = valid_test_sync[fcols].copy()

            for m_name in MODELS:
                m_key = f"{m_name}__{fset_name}__{h}h"

                if m_name == "Persistence":
                    pred_dh_te = np.zeros(len(valid_test))
                    pred_dh_sy = np.zeros(len(valid_test_sync))
                    fitted_model = None

                elif m_name == "XGBoost":
                    xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
                    xgb_m.fit(X_tr, y_train)
                    pred_dh_te = xgb_m.predict(X_te)
                    pred_dh_sy = xgb_m.predict(X_sy)
                    fitted_model = xgb_m

                elif m_name == "Hybrid":
                    use_up = ("upstream" in fset_name)
                    hyb = Phase5Hybrid(horizon=h, xgb_params=XGB_PARAMS, use_upstream=use_up)
                    hyb.fit(valid_train_fit, y_train)
                    pred_dh_te, _, _ = hyb.predict(valid_test)
                    pred_dh_sy, _, _ = hyb.predict(valid_test_sync)
                    fitted_model = hyb

                elif m_name == "RAAG_v2_Hard":
                    use_up = ("upstream" in fset_name)
                    raag = UpstreamAwareRAAGv2(
                        horizon=h,
                        gating_mode="hard",
                        stable_strategy="horizon_adaptive",
                        use_upstream=use_up,
                        xgb_params=XGB_PARAMS
                    )
                    raag.fit(valid_train_fit, y_train)
                    pred_dh_te, _, _, _ = raag.predict(valid_test)
                    pred_dh_sy, _, _, _ = raag.predict(valid_test_sync)
                    fitted_model = raag

                # Save fitted model artifact
                if fitted_model is not None:
                    art_file = MODELS_DIR / f"{m_key}.joblib"
                    joblib.dump(fitted_model, art_file)
                    model_manifest["models_trained"][m_key] = str(art_file.name)

                # Store predictions
                preds_dh_test[m_key] = pred_dh_te
                preds_dh_sync[m_key] = pred_dh_sy

                # Add to row-level predictions DataFrame
                col_pred_dh = f"pred_dH_{h}h__{m_name}__{fset_name}"
                col_pred_st = f"pred_stage_{h}h__{m_name}__{fset_name}"
                predictions_df.loc[valid_test.index, col_pred_dh] = pred_dh_te
                predictions_df.loc[valid_test.index, col_pred_st] = stage_test + pred_dh_te

                # Compute Metrics on Synchronized Test Subwindow (Verified Rainfall)
                err_sy = np.abs(pred_dh_sy - y_test_sync)
                mae_sy = float(mean_absolute_error(y_test_sync, pred_dh_sy))
                rmse_sy = float(np.sqrt(mean_squared_error(y_test_sync, pred_dh_sy)))
                bias_sy = float(np.mean(pred_dh_sy - y_test_sync))
                r2_sy = float(r2_score(y_test_sync, pred_dh_sy))

                # Persistence baseline for comparison
                base_err_sy = np.abs(y_test_sync)
                base_mae_sy = float(np.mean(base_err_sy))
                imp_vs_pers_pct = ((base_mae_sy - mae_sy) / base_mae_sy) * 100.0 if base_mae_sy > 0 else 0.0

                ci_lo, ci_hi = _bootstrap_ci(y_test_sync, pred_dh_sy)

                feature_ablation_records.append({
                    "horizon": f"+{h}h",
                    "model": m_name,
                    "feature_configuration": fset_name,
                    "n_features": len(fcols),
                    "test_cohort": "Synchronized_Sept8_22",
                    "n_samples": len(valid_test_sync),
                    "mae_m": round(mae_sy, 5),
                    "mae_mm": round(mae_sy * 1000.0, 2),
                    "rmse_m": round(rmse_sy, 5),
                    "rmse_mm": round(rmse_sy * 1000.0, 2),
                    "bias_m": round(bias_sy, 5),
                    "r2": round(r2_sy, 4),
                    "mae_ci_low_mm": round(ci_lo * 1000.0, 2),
                    "mae_ci_high_mm": round(ci_hi * 1000.0, 2),
                    "imp_vs_persistence_pct": round(imp_vs_pers_pct, 2),
                })

                # Also evaluate on Whole September Test
                mae_te = float(mean_absolute_error(y_test, pred_dh_te))
                rmse_te = float(np.sqrt(mean_squared_error(y_test, pred_dh_te)))
                bias_te = float(np.mean(pred_dh_te - y_test))
                r2_te = float(r2_score(y_test, pred_dh_te))
                base_mae_te = float(np.mean(np.abs(y_test)))
                imp_te = ((base_mae_te - mae_te) / base_mae_te) * 100.0

                feature_ablation_records.append({
                    "horizon": f"+{h}h",
                    "model": m_name,
                    "feature_configuration": fset_name,
                    "n_features": len(fcols),
                    "test_cohort": "Whole_September",
                    "n_samples": len(valid_test),
                    "mae_m": round(mae_te, 5),
                    "mae_mm": round(mae_te * 1000.0, 2),
                    "rmse_m": round(rmse_te, 5),
                    "rmse_mm": round(rmse_te * 1000.0, 2),
                    "bias_m": round(bias_te, 5),
                    "r2": round(r2_te, 4),
                    "mae_ci_low_mm": round(ci_lo * 1000.0, 2),
                    "mae_ci_high_mm": round(ci_hi * 1000.0, 2),
                    "imp_vs_persistence_pct": round(imp_te, 2),
                })

                print(f"  {m_name:15s} | Sync MAE: {mae_sy*1000:6.2f} mm | RMSE: {rmse_sy*1000:6.2f} mm | Imp: {imp_vs_pers_pct:5.2f}%")

        # ---------------------------------------------------------------------
        # Statistical Paired Hypothesis Tests (Bootstrap Block Comparison)
        # ---------------------------------------------------------------------
        # Test 1: Stage+Rain vs Stage+Rain+Upstream (Incremental Value of Upstream)
        for m_name in ["XGBoost", "Hybrid", "RAAG_v2_Hard"]:
            key_base = f"{m_name}__stage_rain__{h}h"
            key_up = f"{m_name}__stage_rain_upstream__{h}h"
            p_base = preds_dh_sync[key_base]
            p_up = preds_dh_sync[key_up]

            err_base = np.abs(p_base - y_test_sync)
            err_up = np.abs(p_up - y_test_sync)
            delta_mae = (np.mean(err_base) - np.mean(err_up)) * 1000.0  # Positive means upstream is better

            p_val = compute_moving_block_bootstrap_p_value(err_base, err_up, block_length=12, n_bootstraps=1000)

            paired_records.append({
                "horizon": f"+{h}h",
                "model": m_name,
                "comparison": "Stage+Rain vs Stage+Rain+Upstream",
                "cohort": "Synchronized_Sept8_22",
                "n_samples": len(y_test_sync),
                "mae_stage_rain_mm": round(np.mean(err_base) * 1000.0, 2),
                "mae_stage_rain_upstream_mm": round(np.mean(err_up) * 1000.0, 2),
                "delta_mae_mm": round(delta_mae, 2),
                "p_value": round(p_val, 4),
                "statistically_significant_05": p_val < 0.05,
                "conclusion": "Upstream improves model" if (p_val < 0.05 and delta_mae > 0) else "No statistically significant improvement",
            })

            # Test 2: Stage-Only vs Stage+Upstream
            key_so = f"{m_name}__stage_only__{h}h"
            key_so_up = f"{m_name}__stage_upstream__{h}h"
            p_so = preds_dh_sync[key_so]
            p_so_up = preds_dh_sync[key_so_up]
            err_so = np.abs(p_so - y_test_sync)
            err_so_up = np.abs(p_so_up - y_test_sync)
            delta_so = (np.mean(err_so) - np.mean(err_so_up)) * 1000.0
            p_val_so = compute_moving_block_bootstrap_p_value(err_so, err_so_up, block_length=12, n_bootstraps=1000)

            paired_records.append({
                "horizon": f"+{h}h",
                "model": m_name,
                "comparison": "Stage-Only vs Stage+Upstream",
                "cohort": "Synchronized_Sept8_22",
                "n_samples": len(y_test_sync),
                "mae_stage_rain_mm": round(np.mean(err_so) * 1000.0, 2),
                "mae_stage_rain_upstream_mm": round(np.mean(err_so_up) * 1000.0, 2),
                "delta_mae_mm": round(delta_so, 2),
                "p_value": round(p_val_so, 4),
                "statistically_significant_05": p_val_so < 0.05,
                "conclusion": "Upstream improves stage-only" if (p_val_so < 0.05 and delta_so > 0) else "No statistically significant improvement",
            })

        # ---------------------------------------------------------------------
        # Event & Subgroup Metrics (Surge, Rising Limb, Falling Limb, Peak Stage)
        # ---------------------------------------------------------------------
        dh_v = valid_test_sync["dH_v_1h"].values
        subgroups = {
            "Rising_Limb": dh_v > 0.010,
            "Falling_Limb": dh_v < -0.010,
            "Stable_Limb": np.abs(dh_v) <= 0.010,
            "Surge_Approach_Sept11_13": (valid_test_sync["dt"] >= "2026-09-11 00:00:00") & (valid_test_sync["dt"] <= "2026-09-13 23:00:00"),
        }

        for sg_name, sg_mask in subgroups.items():
            n_sg = sg_mask.sum()
            if n_sg < 5:
                continue

            y_sg = y_test_sync[sg_mask]
            st_sg = true_stage_test_sync[sg_mask]

            for m_name in MODELS:
                for fset_name in ["stage_only", "stage_rain", "stage_upstream", "stage_rain_upstream"]:
                    m_key = f"{m_name}__{fset_name}__{h}h"
                    p_sg = preds_dh_sync[m_key][sg_mask]
                    pred_st_sg = stage_test_sync[sg_mask] + p_sg

                    mae_sg = float(mean_absolute_error(y_sg, p_sg))
                    rmse_sg = float(np.sqrt(mean_squared_error(y_sg, p_sg)))
                    bias_sg = float(np.mean(p_sg - y_sg))

                    # Peak Stage Underprediction: mean error when stage is high
                    peak_underpred_mm = float(np.mean(np.maximum(0, st_sg - pred_st_sg))) * 1000.0

                    event_records.append({
                        "horizon": f"+{h}h",
                        "subgroup": sg_name,
                        "n_samples": int(n_sg),
                        "model": m_name,
                        "feature_configuration": fset_name,
                        "mae_mm": round(mae_sg * 1000.0, 2),
                        "rmse_mm": round(rmse_sg * 1000.0, 2),
                        "bias_mm": round(bias_sg * 1000.0, 2),
                        "peak_underpred_mm": round(peak_underpred_mm, 2),
                    })

        # ---------------------------------------------------------------------
        # Telemetry Outage Resilience Evaluation
        # ---------------------------------------------------------------------
        # Simulate:
        # 1. Normal (All Valid)
        # 2. Rainfall Outage (Rainfall masked to NaN)
        # 3. Upstream Outage (Mudgal telemetry masked to NaN)
        # 4. Dual Outage (Both Rainfall and Upstream masked to NaN)
        outage_scenarios = {
            "Normal_Telemetry": (False, False),
            "Rainfall_Outage": (True, False),
            "Upstream_Outage": (False, True),
            "Dual_Outage": (True, True),
        }

        for sc_name, (mask_rain, mask_up) in outage_scenarios.items():
            X_out = valid_test_sync[FEAT_ALL].copy()
            if mask_rain:
                X_out[FEAT_RAIN_CONT] = np.nan
            if mask_up:
                X_out[FEAT_UPSTREAM] = np.nan

            # Evaluate RA-AG v2 Hard and XGBoost
            # RA-AG v2 Hard
            raag_m = joblib.load(MODELS_DIR / f"RAAG_v2_Hard__stage_rain_upstream__{h}h.joblib")
            p_raag_out, is_fb, _, _ = raag_m.predict(X_out)
            mae_raag = float(mean_absolute_error(y_test_sync, p_raag_out))

            # XGBoost
            xgb_m = joblib.load(MODELS_DIR / f"XGBoost__stage_rain_upstream__{h}h.joblib")
            p_xgb_out = xgb_m.predict(X_out)
            mae_xgb = float(mean_absolute_error(y_test_sync, p_xgb_out))

            outage_records.append({
                "horizon": f"+{h}h",
                "scenario": sc_name,
                "rainfall_masked": mask_rain,
                "upstream_masked": mask_up,
                "model": "RAAG_v2_Hard",
                "mae_mm": round(mae_raag * 1000.0, 2),
                "rmse_mm": round(float(np.sqrt(mean_squared_error(y_test_sync, p_raag_out))) * 1000.0, 2),
                "fallback_active_pct": round(float(np.mean(is_fb)) * 100.0, 1),
            })
            outage_records.append({
                "horizon": f"+{h}h",
                "scenario": sc_name,
                "rainfall_masked": mask_rain,
                "upstream_masked": mask_up,
                "model": "XGBoost",
                "mae_mm": round(mae_xgb * 1000.0, 2),
                "rmse_mm": round(float(np.sqrt(mean_squared_error(y_test_sync, p_xgb_out))) * 1000.0, 2),
                "fallback_active_pct": 0.0,
            })

    # -------------------------------------------------------------------------
    # Save All Deliverables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAVING DELIVERABLES")
    print("=" * 80)

    # 1. feature_ablation_results.csv
    df_ablation = pd.DataFrame(feature_ablation_records)
    ablation_path = OUTPUT_DIR / "feature_ablation_results.csv"
    df_ablation.to_csv(ablation_path, index=False)
    print(f"Saved -> {ablation_path}")

    # 2. horizon_metrics.csv
    df_horizon = df_ablation[df_ablation["test_cohort"] == "Synchronized_Sept8_22"].copy()
    horizon_path = OUTPUT_DIR / "horizon_metrics.csv"
    df_horizon.to_csv(horizon_path, index=False)
    print(f"Saved -> {horizon_path}")

    # 3. event_metrics.csv
    df_event = pd.DataFrame(event_records)
    event_path = OUTPUT_DIR / "event_metrics.csv"
    df_event.to_csv(event_path, index=False)
    print(f"Saved -> {event_path}")

    # 4. outage_metrics.csv
    df_outage = pd.DataFrame(outage_records)
    outage_path = OUTPUT_DIR / "outage_metrics.csv"
    df_outage.to_csv(outage_path, index=False)
    print(f"Saved -> {outage_path}")

    # 5. paired_comparisons.csv
    df_paired = pd.DataFrame(paired_records)
    paired_path = OUTPUT_DIR / "paired_comparisons.csv"
    df_paired.to_csv(paired_path, index=False)
    print(f"Saved -> {paired_path}")

    # 6. experiment_predictions.csv
    pred_path = OUTPUT_DIR / "experiment_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved -> {pred_path}")

    # 7. model_manifest.json
    manifest_path = OUTPUT_DIR / "model_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(model_manifest, f, indent=2)
    print(f"Saved -> {manifest_path}")

    print("\nPhase 5 pipeline completed successfully!")


if __name__ == "__main__":
    train_and_evaluate()
