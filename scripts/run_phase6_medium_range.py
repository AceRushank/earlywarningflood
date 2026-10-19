"""
scripts/run_phase6_medium_range.py
==================================
Execution orchestrator for Phase 6: Medium-Range Upstream Signal and Flood-Wave Routing Experiment.
Trains, evaluates, and tests 8 model configurations (A through H) across
+18h, +24h, and +36h horizons for Vishnupuri Barrage.

Produces all deliverables in river_forecasting/outputs/phase6_medium_range/
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.phase1_ml_pipeline import SEED, _bootstrap_ci
from river_forecasting.src.phase4_ra_ag_v2 import compute_moving_block_bootstrap_p_value
from river_forecasting.src.phase6_medium_range import (
    FEAT_LOCAL_STAGE,
    FEAT_RAINFALL,
    FEAT_UPSTREAM_MUDGAL,
    FEAT_STAGE_RAIN,
    FEAT_STAGE_UPSTREAM,
    FEAT_ALL,
    HORIZONS_MEDIUM,
    SimpleLaggedUpstreamBaseline,
    MediumRangeRidgeRegressor,
    MediumRangeXGBRegressor,
)

OUTPUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase6_medium_range"
MODELS_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_RAW = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"
CSV_RAIN = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_hourly_2026.csv"


def prepare_dataset() -> pd.DataFrame:
    print("Loading raw telemetry and rainfall datasets...")
    usecols = ["Station", "Data Acquisition Time", "River Water Level Telemetry Hourly (meter)"]
    chunks = []
    for chunk in pd.read_csv(CSV_RAW, usecols=usecols, chunksize=100000):
        sub = chunk[chunk["Station"].isin(["Vishnupuri", "Mudgal Barrage"])]
        if len(sub) > 0:
            chunks.append(sub)
    df_raw = pd.concat(chunks, ignore_index=True)
    df_raw["dt"] = pd.to_datetime(df_raw["Data Acquisition Time"], format="%d-%m-%Y %H:%M")
    df_raw["stage"] = pd.to_numeric(df_raw["River Water Level Telemetry Hourly (meter)"], errors="coerce")

    df_raw.loc[(df_raw["Station"] == "Vishnupuri") & (df_raw["stage"] < 345.0), "stage"] = np.nan
    df_raw.loc[(df_raw["Station"] == "Mudgal Barrage") & (df_raw["stage"] < 370.0), "stage"] = np.nan

    start_dt = pd.to_datetime("2026-06-15 00:00:00")
    end_dt = pd.to_datetime("2026-09-30 23:00:00")
    grid = pd.date_range(start_dt, end_dt, freq="1h")

    piv = df_raw.pivot_table(index="dt", columns="Station", values="stage", aggfunc="first").reindex(grid)
    piv.rename(columns={"Vishnupuri": "stage_vishnupuri", "Mudgal Barrage": "stage_mudgal"}, inplace=True)

    # Ingest Rainfall
    df_rain = pd.read_csv(CSV_RAIN)
    df_rain["dt"] = pd.to_datetime(df_rain["timestamp_ist"])
    df_rain.set_index("dt", inplace=True)
    piv["rain_depth_1h_mm"] = df_rain["hourly_rainfall_depth_mm"]
    piv["rain_qc_flag"] = df_rain["completeness"]

    piv["rain_3h_mm"] = piv["rain_depth_1h_mm"].rolling(3, min_periods=2).mean() * 3
    piv["rain_6h_mm"] = piv["rain_depth_1h_mm"].rolling(6, min_periods=4).mean() * 6
    piv["rain_12h_mm"] = piv["rain_depth_1h_mm"].rolling(12, min_periods=8).mean() * 12
    piv["rain_24h_mm"] = piv["rain_depth_1h_mm"].rolling(24, min_periods=16).mean() * 24
    piv["rain_48h_mm"] = piv["rain_depth_1h_mm"].rolling(48, min_periods=32).mean() * 48
    piv["api_7d_mm"] = piv["rain_depth_1h_mm"].rolling(168, min_periods=24).sum()

    rain_outage = (piv["rain_qc_flag"] == "OUTAGE_UNAVAILABLE")
    piv.loc[rain_outage, FEAT_RAINFALL] = np.nan

    # Local stage dynamics
    piv["dH_v_1h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(1)
    piv["dH_v_3h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(3)
    piv["dH_v_6h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(6)
    piv["dH_v_12h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(12)
    piv["dH_v_24h"] = piv["stage_vishnupuri"] - piv["stage_vishnupuri"].shift(24)
    piv["accel_v_1h"] = piv["dH_v_1h"] - piv["dH_v_1h"].shift(1)

    # Upstream features (strictly lagged >= 1 hour)
    piv["stage_m_lag1"] = piv["stage_mudgal"].shift(1)
    piv["stage_m_lag2"] = piv["stage_mudgal"].shift(2)
    piv["stage_m_lag4"] = piv["stage_mudgal"].shift(4)
    piv["stage_m_lag7"] = piv["stage_mudgal"].shift(7)
    piv["stage_m_lag13"] = piv["stage_mudgal"].shift(13)
    piv["stage_m_lag19"] = piv["stage_mudgal"].shift(19)
    piv["stage_m_lag25"] = piv["stage_mudgal"].shift(25)

    piv["dH_m_1h"] = piv["stage_m_lag1"] - piv["stage_m_lag2"]
    piv["dH_m_3h"] = piv["stage_m_lag1"] - piv["stage_m_lag4"]
    piv["dH_m_6h"] = piv["stage_m_lag1"] - piv["stage_m_lag7"]
    piv["dH_m_12h"] = piv["stage_m_lag1"] - piv["stage_m_lag13"]
    piv["dH_m_18h"] = piv["stage_m_lag1"] - piv["stage_m_lag19"]
    piv["dH_m_24h"] = piv["stage_m_lag1"] - piv["stage_m_lag25"]

    # Genuine Targets (+18h, +24h, +36h)
    for h in HORIZONS_MEDIUM:
        piv[f"target_v_{h}h"] = piv["stage_vishnupuri"].shift(-h)
        piv[f"target_v_dH_{h}h"] = piv[f"target_v_{h}h"] - piv["stage_vishnupuri"]

    piv["timestamp_ist"] = piv.index.strftime("%Y-%m-%d %H:%M:%S")
    piv["is_test_sync_rainfall"] = (piv.index >= "2026-09-08 06:00:00") & (piv.index <= "2026-09-22 19:00:00")

    return piv


def run_experiment():
    print("=" * 80)
    print("STARTING PHASE 6: MEDIUM-RANGE UPSTREAM EXPERIMENT (+18h, +24h, +36h)")
    print("=" * 80)

    piv = prepare_dataset()

    # Save config.json
    config = {
        "phase": 6,
        "title": "Phase 6 Medium-Range Upstream Signal and Flood-Wave Routing Experiment",
        "horizons_hours": HORIZONS_MEDIUM,
        "upstream_station": "Mudgal Barrage (Godavari River, Parbhani)",
        "upstream_distance_km": 105.0,
        "operational_latency_buffer_hr": 1,
        "feature_sets": {
            "Config_A_Persistence": [],
            "Config_B_Local_Stage": FEAT_LOCAL_STAGE,
            "Config_C_Local_Stage_Rain": FEAT_STAGE_RAIN,
            "Config_D_Local_Stage_Upstream": FEAT_STAGE_UPSTREAM,
            "Config_E_Local_Stage_Rain_Upstream": FEAT_ALL,
            "Config_F_Simple_Lagged_Upstream_Proxy": ["dH_m_12h"],
            "Config_G_Ridge_Local_Upstream": FEAT_STAGE_UPSTREAM,
            "Config_H_XGBoost_Local_Upstream": FEAT_STAGE_UPSTREAM,
        },
        "model_parameters": {
            "Ridge_alpha_candidates": [1.0, 10.0, 50.0, 100.0, 500.0],
            "XGBoost_params": {
                "max_depth": 2,
                "learning_rate": 0.02,
                "n_estimators": 50,
                "reg_lambda": 10.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": SEED,
            }
        },
        "chronological_partitions": {
            "nominal_train": "2026-06-15 00:00 to 2026-08-10 23:00",
            "nominal_val": "2026-08-11 00:00 to 2026-08-31 23:00",
            "nominal_test": "2026-09-01 00:00 to 2026-09-30 23:00",
            "synchronized_test": "2026-09-08 06:00 to 2026-09-22 19:00",
            "boundary_separation_applied": True,
        }
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration -> {OUTPUT_DIR / 'config.json'}")

    feature_ablation_records = []
    horizon_records = []
    event_records = []
    outage_records = []
    paired_records = []

    # Row level predictions dataframe
    predictions_df = piv[["timestamp_ist", "stage_vishnupuri", "stage_mudgal", "is_test_sync_rainfall"]].copy()

    model_manifest = {
        "phase": 6,
        "models_trained": {},
        "optimal_alphas": {},
    }

    # Evaluate across each horizon
    for h in HORIZONS_MEDIUM:
        tgt_dh = f"target_v_dH_{h}h"
        tgt_stage = f"target_v_{h}h"

        print(f"\n" + "=" * 60)
        print(f"EVALUATING HORIZON: +{h}h (Target: {tgt_dh})")
        print(f"=" * 60)

        # Enforce boundary separation:
        train_end_h = pd.to_datetime("2026-08-10 23:00:00") - pd.Timedelta(hours=h)
        val_end_h   = pd.to_datetime("2026-08-31 23:00:00") - pd.Timedelta(hours=h)
        test_end_h  = pd.to_datetime("2026-09-30 23:00:00") - pd.Timedelta(hours=h)

        df_train = piv.loc["2026-06-15 00:00:00":train_end_h].copy()
        df_val   = piv.loc["2026-08-11 00:00:00":val_end_h].copy()
        df_test  = piv.loc["2026-09-01 00:00:00":test_end_h].copy()
        df_sync  = piv.loc["2026-09-08 06:00:00":min(pd.to_datetime("2026-09-22 19:00:00"), test_end_h)].copy()

        # Valid target masks
        v_train = df_train[df_train[tgt_dh].notna() & df_train["stage_vishnupuri"].notna()].copy()
        v_val   = df_val[df_val[tgt_dh].notna() & df_val["stage_vishnupuri"].notna()].copy()
        v_test  = df_test[df_test[tgt_dh].notna() & df_test["stage_vishnupuri"].notna()].copy()
        v_sync  = df_sync[df_sync[tgt_dh].notna() & df_sync["stage_vishnupuri"].notna()].copy()

        y_train = v_train[tgt_dh].values
        y_val   = v_val[tgt_dh].values
        y_test  = v_test[tgt_dh].values
        y_sync  = v_sync[tgt_dh].values

        stage_sync = v_sync["stage_vishnupuri"].values
        true_stage_sync = v_sync[tgt_stage].values

        stage_test = v_test["stage_vishnupuri"].values
        true_stage_test = v_test[tgt_stage].values

        print(f"Valid origins: Train={len(v_train)}, Val={len(v_val)}, Test(Whole)={len(v_test)}, Test(Sync)={len(v_sync)}")

        # 1. Tune Ridge alpha on Validation Split
        best_alpha = 50.0
        best_val_mae = float("inf")
        for alpha_cand in [1.0, 10.0, 50.0, 100.0, 500.0]:
            r_temp = MediumRangeRidgeRegressor(alpha=alpha_cand, use_upstream=True, use_rainfall=True)
            r_temp.fit(v_train, y_train)
            p_val, _ = r_temp.predict(v_val)
            val_mae = mean_absolute_error(y_val, p_val)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_alpha = alpha_cand
        model_manifest["optimal_alphas"][f"+{h}h"] = best_alpha
        print(f"Optimal Ridge alpha tuned on Val: {best_alpha} (Val MAE: {best_val_mae*1000:.2f} mm)")

        # 2. Instantiate and Fit All 8 Configurations
        models_dict = {
            "Config_A_Persistence": None,
            "Config_B_Local_Stage": MediumRangeRidgeRegressor(alpha=best_alpha, use_upstream=False, use_rainfall=False),
            "Config_C_Local_Stage_Rain": MediumRangeRidgeRegressor(alpha=best_alpha, use_upstream=False, use_rainfall=True),
            "Config_D_Local_Stage_Upstream": MediumRangeRidgeRegressor(alpha=best_alpha, use_upstream=True, use_rainfall=False),
            "Config_E_Local_Stage_Rain_Upstream": MediumRangeRidgeRegressor(alpha=best_alpha, use_upstream=True, use_rainfall=True),
            "Config_F_Simple_Lagged_Upstream_Proxy": SimpleLaggedUpstreamBaseline(horizon=h, wave_lag_hr=14),
            "Config_G_Ridge_Local_Upstream": MediumRangeRidgeRegressor(alpha=best_alpha, use_upstream=True, use_rainfall=False),
            "Config_H_XGBoost_Local_Upstream": MediumRangeXGBRegressor(
                max_depth=2, learning_rate=0.02, n_estimators=50, reg_lambda=10.0, use_upstream=True, use_rainfall=False
            ),
        }

        preds_sync = {}
        preds_test = {}

        for cfg_name, model in models_dict.items():
            m_key = f"{cfg_name}__{h}h"
            if model is None:
                # Persistence: dH = 0.0
                p_sy = np.zeros(len(v_sync))
                p_te = np.zeros(len(v_test))
            else:
                model.fit(v_train, y_train)
                res_sy = model.predict(v_sync)
                res_te = model.predict(v_test)
                p_sy = res_sy[0] if isinstance(res_sy, tuple) else res_sy
                p_te = res_te[0] if isinstance(res_te, tuple) else res_te

                # Serialize artifact
                art_p = MODELS_DIR / f"{m_key}.joblib"
                joblib.dump(model, art_p)
                model_manifest["models_trained"][m_key] = str(art_p.name)

            preds_sync[cfg_name] = p_sy
            preds_test[cfg_name] = p_te

            # Store in row-level predictions
            predictions_df.loc[v_test.index, f"pred_dH_{h}h__{cfg_name}"] = p_te
            predictions_df.loc[v_test.index, f"pred_stage_{h}h__{cfg_name}"] = stage_test + p_te

            # Metrics on Synchronized Test Period (Verified on-disk Rainfall)
            mae_sy = float(mean_absolute_error(y_sync, p_sy))
            rmse_sy = float(np.sqrt(mean_squared_error(y_sync, p_sy)))
            bias_sy = float(np.mean(p_sy - y_sync))
            r2_sy = float(r2_score(y_sync, p_sy))
            ci_lo, ci_hi = _bootstrap_ci(y_sync, p_sy)

            # Persistence benchmark
            base_mae = float(np.mean(np.abs(y_sync)))
            imp_pct = ((base_mae - mae_sy) / base_mae) * 100.0 if base_mae > 0 else 0.0

            feature_ablation_records.append({
                "horizon": f"+{h}h",
                "model_configuration": cfg_name,
                "test_cohort": "Synchronized_Sept8_22",
                "n_samples": len(v_sync),
                "mae_mm": round(mae_sy * 1000.0, 2),
                "rmse_mm": round(rmse_sy * 1000.0, 2),
                "bias_mm": round(bias_sy * 1000.0, 2),
                "r2": round(r2_sy, 4),
                "mae_ci_low_mm": round(ci_lo * 1000.0, 2),
                "mae_ci_high_mm": round(ci_hi * 1000.0, 2),
                "imp_vs_persistence_pct": round(imp_pct, 2),
            })

            # Metrics on Whole September Test Period
            mae_te = float(mean_absolute_error(y_test, p_te))
            rmse_te = float(np.sqrt(mean_squared_error(y_test, p_te)))
            bias_te = float(np.mean(p_te - y_test))
            r2_te = float(r2_score(y_test, p_te))
            base_mae_te = float(np.mean(np.abs(y_test)))
            imp_te = ((base_mae_te - mae_te) / base_mae_te) * 100.0

            feature_ablation_records.append({
                "horizon": f"+{h}h",
                "model_configuration": cfg_name,
                "test_cohort": "Whole_September",
                "n_samples": len(v_test),
                "mae_mm": round(mae_te * 1000.0, 2),
                "rmse_mm": round(rmse_te * 1000.0, 2),
                "bias_mm": round(bias_te * 1000.0, 2),
                "r2": round(r2_te, 4),
                "mae_ci_low_mm": round(ci_lo * 1000.0, 2),
                "mae_ci_high_mm": round(ci_hi * 1000.0, 2),
                "imp_vs_persistence_pct": round(imp_te, 2),
            })

            print(f"  {cfg_name:38s} | Sync MAE: {mae_sy*1000:6.2f} mm | RMSE: {rmse_sy*1000:6.2f} mm | Imp: {imp_pct:6.2f}%")

        # 3. Statistical Paired Comparisons (Moving Block Bootstrap, B=1000, block=24h)
        comparisons = [
            ("Config_C_Local_Stage_Rain", "Config_E_Local_Stage_Rain_Upstream", "Incremental Value of Upstream (Stage+Rain vs Full)"),
            ("Config_B_Local_Stage", "Config_D_Local_Stage_Upstream", "Upstream Addition to Stage-Only"),
            ("Config_A_Persistence", "Config_D_Local_Stage_Upstream", "Stage+Upstream vs Persistence Baseline"),
            ("Config_A_Persistence", "Config_G_Ridge_Local_Upstream", "Ridge Local+Upstream vs Persistence"),
            ("Config_A_Persistence", "Config_H_XGBoost_Local_Upstream", "XGBoost Local+Upstream vs Persistence"),
            ("Config_B_Local_Stage", "Config_F_Simple_Lagged_Upstream_Proxy", "Lagged Upstream Proxy vs Local Stage"),
        ]

        for m_base_name, m_test_name, comp_desc in comparisons:
            p_base = preds_sync[m_base_name]
            p_test = preds_sync[m_test_name]
            err_base = np.abs(p_base - y_sync)
            err_test = np.abs(p_test - y_sync)

            delta_mae = (np.mean(err_base) - np.mean(err_test)) * 1000.0
            p_val = compute_moving_block_bootstrap_p_value(err_base, err_test, block_length=24, n_bootstraps=1000)

            paired_records.append({
                "horizon": f"+{h}h",
                "comparison_description": comp_desc,
                "baseline_model": m_base_name,
                "candidate_model": m_test_name,
                "test_cohort": "Synchronized_Sept8_22",
                "n_samples": len(y_sync),
                "baseline_mae_mm": round(np.mean(err_base) * 1000.0, 2),
                "candidate_mae_mm": round(np.mean(err_test) * 1000.0, 2),
                "delta_mae_mm": round(delta_mae, 2),
                "p_value": round(p_val, 4),
                "statistically_significant_05": p_val < 0.05,
                "conclusion": "Candidate significantly outperforms baseline" if (p_val < 0.05 and delta_mae > 0) else "No statistically significant improvement",
            })

        # 4. Event Subgroups (Stable, Rising, Falling, Peak Underprediction)
        dh_v = v_sync["dH_v_1h"].values
        subgroups = {
            "Stable_Stage": np.abs(dh_v) <= 0.010,
            "Rising_Limb": dh_v > 0.010,
            "Falling_Limb": dh_v < -0.010,
            "Major_Surge_Sept11_13": (v_sync.index >= "2026-09-11 00:00:00") & (v_sync.index <= "2026-09-13 23:00:00"),
        }

        for sg_name, sg_mask in subgroups.items():
            n_sg = sg_mask.sum()
            if n_sg < 5:
                continue
            y_sg = y_sync[sg_mask]
            st_sg = true_stage_sync[sg_mask]
            origin_st_sg = stage_sync[sg_mask]

            for cfg_name in models_dict.keys():
                p_sg = preds_sync[cfg_name][sg_mask]
                pred_st_sg = origin_st_sg + p_sg

                mae_sg = float(mean_absolute_error(y_sg, p_sg))
                rmse_sg = float(np.sqrt(mean_squared_error(y_sg, p_sg)))
                bias_sg = float(np.mean(p_sg - y_sg))
                peak_underpred_mm = float(np.mean(np.maximum(0, st_sg - pred_st_sg))) * 1000.0

                event_records.append({
                    "horizon": f"+{h}h",
                    "subgroup": sg_name,
                    "n_samples": int(n_sg),
                    "model_configuration": cfg_name,
                    "mae_mm": round(mae_sg * 1000.0, 2),
                    "rmse_mm": round(rmse_sg * 1000.0, 2),
                    "bias_mm": round(bias_sg * 1000.0, 2),
                    "peak_underpred_mm": round(peak_underpred_mm, 2),
                })

        # 5. Outage Scenarios
        outage_cases = {
            "Normal_Telemetry": (False, False),
            "Rainfall_Outage": (True, False),
            "Upstream_Outage": (False, True),
            "Dual_Outage": (True, True),
        }

        for sc_name, (mask_rain, mask_up) in outage_cases.items():
            v_out = v_sync.copy()
            if mask_rain:
                v_out[FEAT_RAINFALL] = np.nan
            if mask_up:
                v_out[FEAT_UPSTREAM_MUDGAL] = np.nan

            # Evaluate Config E (Ridge Full) and Config H (XGBoost)
            ridge_m = joblib.load(MODELS_DIR / f"Config_E_Local_Stage_Rain_Upstream__{h}h.joblib")
            xgb_m   = joblib.load(MODELS_DIR / f"Config_H_XGBoost_Local_Upstream__{h}h.joblib")

            p_ridge_out, is_fb_ridge = ridge_m.predict(v_out)
            p_xgb_out, is_fb_xgb     = xgb_m.predict(v_out)

            outage_records.append({
                "horizon": f"+{h}h",
                "scenario": sc_name,
                "model_configuration": "Config_E_Ridge_Full",
                "mae_mm": round(float(mean_absolute_error(y_sync, p_ridge_out)) * 1000.0, 2),
                "rmse_mm": round(float(np.sqrt(mean_squared_error(y_sync, p_ridge_out))) * 1000.0, 2),
                "fallback_active_pct": round(float(np.mean(is_fb_ridge)) * 100.0, 1),
            })
            outage_records.append({
                "horizon": f"+{h}h",
                "scenario": sc_name,
                "model_configuration": "Config_H_XGBoost_Local_Upstream",
                "mae_mm": round(float(mean_absolute_error(y_sync, p_xgb_out)) * 1000.0, 2),
                "rmse_mm": round(float(np.sqrt(mean_squared_error(y_sync, p_xgb_out))) * 1000.0, 2),
                "fallback_active_pct": round(float(np.mean(is_fb_xgb)) * 100.0, 1),
            })

    # Save Deliverables
    print("\n" + "=" * 80)
    print("SAVING DELIVERABLES TO", OUTPUT_DIR)
    print("=" * 80)

    # 1. feature_ablation_results.csv
    df_ablation = pd.DataFrame(feature_ablation_records)
    df_ablation.to_csv(OUTPUT_DIR / "feature_ablation_results.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'feature_ablation_results.csv'}")

    # 2. horizon_metrics.csv
    df_horiz = df_ablation[df_ablation["test_cohort"] == "Synchronized_Sept8_22"].copy()
    df_horiz.to_csv(OUTPUT_DIR / "horizon_metrics.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'horizon_metrics.csv'}")

    # 3. event_metrics.csv
    df_ev = pd.DataFrame(event_records)
    df_ev.to_csv(OUTPUT_DIR / "event_metrics.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'event_metrics.csv'}")

    # 4. outage_metrics.csv
    df_out = pd.DataFrame(outage_records)
    df_out.to_csv(OUTPUT_DIR / "outage_metrics.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'outage_metrics.csv'}")

    # 5. paired_comparisons.csv
    df_paired = pd.DataFrame(paired_records)
    df_paired.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'paired_comparisons.csv'}")

    # 6. row_level_predictions.csv
    predictions_df.to_csv(OUTPUT_DIR / "row_level_predictions.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR / 'row_level_predictions.csv'}")

    # 7. model_manifest.json
    with open(OUTPUT_DIR / "model_manifest.json", "w") as f:
        json.dump(model_manifest, f, indent=2)
    print(f"Saved -> {OUTPUT_DIR / 'model_manifest.json'}")

    print("\nPhase 6 experiment orchestration completed successfully!")


if __name__ == "__main__":
    run_experiment()
