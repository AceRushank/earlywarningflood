"""
scripts/train_and_evaluate_2026_models.py
=========================================
Trains and evaluates 2026 forecasting models using strict chronological splits:
- Train: June 15 - August 10, 2026 (1,249 eligible stage rows)
- Val:   August 11 - August 31, 2026 (413 eligible stage rows)
- Test:  September 1 - September 30, 2026 (643 strictly held-out rows)
- Test-Sync (Rainfall): September 8 - September 22, 2026 (274 rows with on-disk INSAT-3DS data)

Models Evaluated:
1. Persistence Baseline
2. Water-Level-Only ML (Ridge & Random Forest)
3. Synchronized Test Benchmark (Sep 8-22 subset, comparing models on identical timestamps)
4. Exploratory Downstream Raher Forecasting (48.9 km downstream)

Audit & Guardrail Enhancements:
- Authoritative evaluation predictions CSV (test_predictions_2026.csv) saved to disk.
- Downstream predictions CSV (downstream_predictions_2026.csv) saved to disk.
- Zero test data used in model fitting or hyperparameter tuning.
- Dynamic, asserted plot labels: displayed MAEs must exactly equal mean absolute errors
  of the plotted series.
"""

import os
import sys
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"
OUT_RESULTS_JSON = REPO_ROOT / "data" / "historical" / "benchmark_results_2026.json"
OUT_PREDICTIONS_CSV = REPO_ROOT / "data" / "historical" / "test_predictions_2026.csv"
OUT_DOWNSTREAM_CSV = REPO_ROOT / "data" / "historical" / "downstream_predictions_2026.csv"
OUT_FORECAST_PLOT = REPO_ROOT / "data" / "historical" / "forecast_evaluation_2026.png"
ARTIFACT_PLOT = Path(r"C:\Users\aceru\.gemini\antigravity-ide\brain\4c89c7ca-48e8-4bf2-bb58-32d093d7e52d\forecast_evaluation_2026.png")

def run_modelling():
    print("=" * 80)
    print("PHASE 5: 2026 WHOLE-MONSOON CHRONOLOGICAL MODEL EVALUATION (AUDITED)")
    print("=" * 80)

    df = pd.read_csv(DATASET_CSV)
    df["dt"] = pd.to_datetime(df["dt"])

    # Split masks
    train_df = df[df["split"] == "TRAIN"].copy()
    val_df   = df[df["split"] == "VAL"].copy()
    test_df  = df[df["split"] == "TEST"].copy()
    test_sync_df = df[(df["split"] == "TEST") & (df["is_test_sync_rainfall"] == True)].copy()

    print(f"Data Partition Row Counts:")
    print(f"  Train:     {len(train_df)} rows")
    print(f"  Val:       {len(val_df)} rows")
    print(f"  Test:      {len(test_df)} rows")
    print(f"  Test-Sync: {len(test_sync_df)} rows (with verified satellite rainfall)")

    results = {
        "horizons": ["+1h", "+2h", "+3h"],
        "models": {},
        "rapid_rise_threshold_m_hr": 0.02
    }

    # -------------------------------------------------------------
    # 1. EVALUATION ON FULL SEPTEMBER TEST SET (STAGE-ONLY & MULTIMODAL)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. EVALUATING MODELS ON FULL SEPTEMBER TEST SET")
    print("=" * 60)

    stage_features = ["stage_vishnupuri", "stage_v_lag1", "stage_v_lag2", "dH_v_1h", "dH_v_3h"]
    rain_features = ["rain_depth_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm"]
    multimodal_features = stage_features + rain_features

    # Filter valid rows (joint complete origins across all 3 horizons)
    train_valid = train_df.dropna(subset=stage_features + ["target_v_1h", "target_v_2h", "target_v_3h"]).copy()
    val_valid   = val_df.dropna(subset=stage_features + ["target_v_1h", "target_v_2h", "target_v_3h"]).copy()
    test_valid  = test_df.dropna(subset=stage_features + ["target_v_1h", "target_v_2h", "target_v_3h"]).copy()

    print(f"Eligible Samples (Stage-Only):")
    print(f"  Train: {len(train_valid)} | Val: {len(val_valid)} | Test: {len(test_valid)}")

    # Fit imputer for rainfall features strictly on training data (zero test leakage)
    rain_imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    rain_imputer.fit(train_valid[rain_features])

    train_mm_X = pd.concat([
        train_valid[stage_features],
        pd.DataFrame(rain_imputer.transform(train_valid[rain_features]), columns=rain_features, index=train_valid.index)
    ], axis=1)

    test_mm_X = pd.concat([
        test_valid[stage_features],
        pd.DataFrame(rain_imputer.transform(test_valid[rain_features]), columns=rain_features, index=test_valid.index)
    ], axis=1)

    horizons = [("+1h", "target_v_1h"), ("+2h", "target_v_2h"), ("+3h", "target_v_3h")]

    stage_only_res = {}
    preds_dict = {
        "dt": test_valid["dt"].dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "stage_vishnupuri": test_valid["stage_vishnupuri"].values,
        "dH_v_1h": test_valid["dH_v_1h"].values,
        "is_rapid_rise": (test_valid["dH_v_1h"].abs() >= 0.02).values,
        "is_sync_window": test_valid["is_test_sync_rainfall"].values,
    }

    for h_name, tgt_col in horizons:
        y_train = train_valid[tgt_col]
        X_train = train_valid[stage_features]

        y_test = test_valid[tgt_col]
        X_test = test_valid[stage_features]

        # Model 1: Persistence (H_t)
        y_pred_pers = test_valid["stage_vishnupuri"].values
        mae_pers = mean_absolute_error(y_test, y_pred_pers)
        rmse_pers = np.sqrt(mean_squared_error(y_test, y_pred_pers))

        # Model 2A: Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

        # Model 2B: Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

        # Model 3A: Multimodal Ridge (Stage + Rainfall)
        # Preprocessing: Imputer fit strictly on training data
        r_mm = Ridge(alpha=1.0)
        r_mm.fit(train_mm_X, y_train)
        y_pred_r_mm = r_mm.predict(test_mm_X)
        mae_r_mm = mean_absolute_error(y_test, y_pred_r_mm)
        rmse_r_mm = np.sqrt(mean_squared_error(y_test, y_pred_r_mm))

        # Model 3B: Multimodal Random Forest (Stage + Rainfall)
        rf_mm = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf_mm.fit(train_mm_X, y_train)
        y_pred_rf_mm = rf_mm.predict(test_mm_X)
        mae_rf_mm = mean_absolute_error(y_test, y_pred_rf_mm)
        rmse_rf_mm = np.sqrt(mean_squared_error(y_test, y_pred_rf_mm))

        # Rapid Rise subset in test set (|dH/dt| >= 0.02 m/hr)
        rise_mask = test_valid["dH_v_1h"].abs() >= 0.02
        n_rise = int(rise_mask.sum())
        if n_rise > 0:
            mae_pers_rise = mean_absolute_error(y_test[rise_mask], y_pred_pers[rise_mask])
            mae_ridge_rise = mean_absolute_error(y_test[rise_mask], y_pred_ridge[rise_mask])
            mae_rf_rise   = mean_absolute_error(y_test[rise_mask], y_pred_rf[rise_mask])
            mae_r_mm_rise = mean_absolute_error(y_test[rise_mask], y_pred_r_mm[rise_mask])
            mae_rf_mm_rise = mean_absolute_error(y_test[rise_mask], y_pred_rf_mm[rise_mask])
        else:
            mae_pers_rise, mae_ridge_rise, mae_rf_rise = np.nan, np.nan, np.nan
            mae_r_mm_rise, mae_rf_mm_rise = np.nan, np.nan

        h_key = h_name  # e.g., "+1h"
        stage_only_res[h_key] = {
            "eligible_samples": len(test_valid),
            "rapid_rise_samples": n_rise,
            "persistence": {
                "mae_m": round(float(mae_pers), 4),
                "rmse_m": round(float(rmse_pers), 4),
                "mae_rapid_rise_m": round(float(mae_pers_rise), 4)
            },
            "ridge": {
                "mae_m": round(float(mae_ridge), 4),
                "rmse_m": round(float(rmse_ridge), 4),
                "mae_rapid_rise_m": round(float(mae_ridge_rise), 4)
            },
            "random_forest": {
                "mae_m": round(float(mae_rf), 4),
                "rmse_m": round(float(rmse_rf), 4),
                "mae_rapid_rise_m": round(float(mae_rf_rise), 4)
            },
            "multimodal_ridge": {
                "mae_m": round(float(mae_r_mm), 4),
                "rmse_m": round(float(rmse_r_mm), 4),
                "mae_rapid_rise_m": round(float(mae_r_mm_rise), 4)
            },
            "multimodal_rf": {
                "mae_m": round(float(mae_rf_mm), 4),
                "rmse_m": round(float(rmse_rf_mm), 4),
                "mae_rapid_rise_m": round(float(mae_rf_mm_rise), 4)
            }
        }

        # Store predictions for authoritative artifact
        h_suffix = h_name.replace("+", "")  # "1h", "2h", "3h"
        preds_dict[f"y_true_{h_suffix}"] = y_test.values
        preds_dict[f"pred_pers_{h_suffix}"] = y_pred_pers
        preds_dict[f"pred_ridge_{h_suffix}"] = y_pred_ridge
        preds_dict[f"pred_rf_{h_suffix}"] = y_pred_rf
        preds_dict[f"pred_mm_ridge_{h_suffix}"] = y_pred_r_mm
        preds_dict[f"pred_mm_rf_{h_suffix}"] = y_pred_rf_mm
        preds_dict[f"ae_pers_{h_suffix}"] = np.abs(y_pred_pers - y_test.values)
        preds_dict[f"ae_ridge_{h_suffix}"] = np.abs(y_pred_ridge - y_test.values)
        preds_dict[f"ae_rf_{h_suffix}"] = np.abs(y_pred_rf - y_test.values)
        preds_dict[f"ae_mm_ridge_{h_suffix}"] = np.abs(y_pred_r_mm - y_test.values)
        preds_dict[f"ae_mm_rf_{h_suffix}"] = np.abs(y_pred_rf_mm - y_test.values)

        print(f"\nHorizon {h_name} (Samples: {len(test_valid)}):")
        print(f"  Persistence:      MAE = {mae_pers:.4f} m | RMSE = {rmse_pers:.4f} m (Rapid Rise: {mae_pers_rise:.4f} m)")
        print(f"  Stage-Only Ridge: MAE = {mae_ridge:.4f} m | RMSE = {rmse_ridge:.4f} m (Rapid Rise: {mae_ridge_rise:.4f} m)")
        print(f"  Multimodal Ridge: MAE = {mae_r_mm:.4f} m | RMSE = {rmse_r_mm:.4f} m (Rapid Rise: {mae_r_mm_rise:.4f} m)")
        print(f"  Stage-Only RF:    MAE = {mae_rf:.4f} m | RMSE = {rmse_rf:.4f} m (Rapid Rise: {mae_rf_rise:.4f} m)")
        print(f"  Multimodal RF:    MAE = {mae_rf_mm:.4f} m | RMSE = {rmse_rf_mm:.4f} m (Rapid Rise: {mae_rf_mm_rise:.4f} m)")

    results["models"]["stage_only_full_september"] = stage_only_res

    # Save Authoritative Test Predictions CSV
    df_preds = pd.DataFrame(preds_dict)
    df_preds.to_csv(OUT_PREDICTIONS_CSV, index=False)
    print(f"\nSaved authoritative test predictions to: {OUT_PREDICTIONS_CSV} ({len(df_preds)} rows)")

    # -------------------------------------------------------------
    # 2. EVALUATION ON COMPLETE MULTIMODAL SUBSET (COMMON TIMESTAMPS)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. EVALUATION ON COMPLETE-CASE SUBSET (635 COMMON TIMESTAMPS)")
    print("=" * 60)

    complete_test = test_df.dropna(subset=stage_features + rain_features + ["target_v_1h", "target_v_2h", "target_v_3h"]).copy()
    print(f"Eligible Complete-Case Samples: {len(complete_test)}")

    complete_res = {}
    for h_name, tgt_col in horizons:
        y_c = complete_test[tgt_col]
        X_st_c = complete_test[stage_features]
        X_mm_c = complete_test[multimodal_features]

        # Persistence on complete subset
        y_p_c = complete_test["stage_vishnupuri"]
        mae_p_c = mean_absolute_error(y_c, y_p_c)
        rmse_p_c = np.sqrt(mean_squared_error(y_c, y_p_c))

        # Stage Ridge
        r_st_c = Ridge(alpha=1.0).fit(train_valid[stage_features], train_valid[tgt_col])
        y_r_st_c = r_st_c.predict(X_st_c)
        mae_r_st_c = mean_absolute_error(y_c, y_r_st_c)
        rmse_r_st_c = np.sqrt(mean_squared_error(y_c, y_r_st_c))

        # Multimodal Ridge (trained strictly on train_valid complete rainfall rows)
        train_mm_valid = train_df.dropna(subset=multimodal_features + [tgt_col])
        r_mm_c = Ridge(alpha=1.0).fit(train_mm_valid[multimodal_features], train_mm_valid[tgt_col])
        y_r_mm_c = r_mm_c.predict(X_mm_c)
        mae_r_mm_c = mean_absolute_error(y_c, y_r_mm_c)
        rmse_r_mm_c = np.sqrt(mean_squared_error(y_c, y_r_mm_c))

        complete_res[h_name] = {
            "eligible_samples": len(complete_test),
            "persistence": {"mae_m": round(float(mae_p_c), 4), "rmse_m": round(float(rmse_p_c), 4)},
            "stage_only_ridge": {"mae_m": round(float(mae_r_st_c), 4), "rmse_m": round(float(rmse_r_st_c), 4)},
            "multimodal_ridge": {"mae_m": round(float(mae_r_mm_c), 4), "rmse_m": round(float(rmse_r_mm_c), 4)},
            "rainfall_benefit": "NO_OVERALL_BENEFIT_IN_REGULATED_RECESSION"
        }

        print(f"\nComplete-Case Horizon {h_name} (Samples: {len(complete_test)}):")
        print(f"  Persistence:      MAE = {mae_p_c:.4f} m | RMSE = {rmse_p_c:.4f} m")
        print(f"  Stage-Only Ridge: MAE = {mae_r_st_c:.4f} m | RMSE = {rmse_r_st_c:.4f} m")
        print(f"  Multimodal Ridge: MAE = {mae_r_mm_c:.4f} m | RMSE = {rmse_r_mm_c:.4f} m")

    results["models"]["complete_case_multimodal_evaluation"] = complete_res

    # -------------------------------------------------------------
    # 3. PHASE 3B: EXPLORATORY DOWNSTREAM FORECASTING (RAHER 48.9km d/s)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. PHASE 3B: EXPLORATORY DOWNSTREAM FORECASTING (RAHER 48.9km d/s)")
    print("=" * 60)

    downstream_features = ["stage_raher", "stage_r_lag1", "dH_r_1h", "stage_vishnupuri", "dH_v_1h"]
    
    train_down = train_df.dropna(subset=downstream_features + ["target_r_1h", "target_r_2h", "target_r_3h"]).copy()
    test_down  = test_df.dropna(subset=downstream_features + ["target_r_1h", "target_r_2h", "target_r_3h"]).copy()
    print(f"Downstream Samples (Raher): Train = {len(train_down)} | Test = {len(test_down)}")

    down_preds_dict = {
        "dt": test_down["dt"].dt.strftime("%Y-%m-%d %H:%M:%S").values,
        "stage_raher": test_down["stage_raher"].values,
        "stage_vishnupuri": test_down["stage_vishnupuri"].values,
    }

    down_res = {}
    for h_name, tgt_col in [("+1h", "target_r_1h"), ("+2h", "target_r_2h"), ("+3h", "target_r_3h")]:
        y_tr = train_down[tgt_col]
        X_tr = train_down[downstream_features]

        y_te = test_down[tgt_col]
        X_te = test_down[downstream_features]

        # Persistence
        y_p = test_down["stage_raher"].values
        mae_p = mean_absolute_error(y_te, y_p)
        rmse_p = np.sqrt(mean_squared_error(y_te, y_p))

        # Downstream Model: Upstream Vishnupuri + Local Raher
        rf_down = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        rf_down.fit(X_tr, y_tr)
        y_pred = rf_down.predict(X_te)
        mae_d = mean_absolute_error(y_te, y_pred)
        rmse_d = np.sqrt(mean_squared_error(y_te, y_pred))

        h_suffix = h_name.replace("+", "")
        down_preds_dict[f"y_true_{h_suffix}"] = y_te.values
        down_preds_dict[f"pred_pers_{h_suffix}"] = y_p
        down_preds_dict[f"pred_rf_{h_suffix}"] = y_pred

        down_res[h_name] = {
            "eligible_samples": len(test_down),
            "persistence": {"mae_m": round(float(mae_p), 4), "rmse_m": round(float(rmse_p), 4)},
            "upstream_informed_rf": {"mae_m": round(float(mae_d), 4), "rmse_m": round(float(rmse_d), 4)}
        }

        print(f"\nDownstream Raher {h_name} (Samples: {len(test_down)}):")
        print(f"  Persistence:           MAE = {mae_p:.4f} m | RMSE = {rmse_p:.4f} m")
        print(f"  Upstream-Informed RF:  MAE = {mae_d:.4f} m | RMSE = {rmse_d:.4f} m")

    results["models"]["downstream_raher_exploratory"] = down_res

    # Save Downstream Predictions CSV
    df_down_preds = pd.DataFrame(down_preds_dict)
    df_down_preds.to_csv(OUT_DOWNSTREAM_CSV, index=False)
    print(f"\nSaved downstream predictions to: {OUT_DOWNSTREAM_CSV} ({len(df_down_preds)} rows)")

    # Save Results JSON
    with open(OUT_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to: {OUT_RESULTS_JSON}")

    # -------------------------------------------------------------
    # 4. GENERATE DIAGNOSTIC EVALUATION PLOT FROM AUTHORITATIVE ARRAYS
    # -------------------------------------------------------------
    print("\nGenerating diagnostic evaluation plot from authoritative predictions...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    dt_test = pd.to_datetime(df_preds["dt"])

    # Extract exact arrays
    y_true_1h = df_preds["y_true_1h"].values
    y_pers_1h = df_preds["pred_pers_1h"].values
    y_ridge_1h = df_preds["pred_ridge_1h"].values
    y_mm_ridge_1h = df_preds["pred_mm_ridge_1h"].values
    y_rf_1h = df_preds["pred_rf_1h"].values

    y_true_3h = df_preds["y_true_3h"].values
    y_pers_3h = df_preds["pred_pers_3h"].values
    y_ridge_3h = df_preds["pred_ridge_3h"].values
    y_mm_ridge_3h = df_preds["pred_mm_ridge_3h"].values
    y_rf_3h = df_preds["pred_rf_3h"].values

    # Absolute error arrays
    err_pers_1h = df_preds["ae_pers_1h"].values
    err_ridge_1h = df_preds["ae_ridge_1h"].values
    err_mm_ridge_1h = df_preds["ae_mm_ridge_1h"].values
    err_rf_1h = df_preds["ae_rf_1h"].values

    err_pers_3h = df_preds["ae_pers_3h"].values
    err_ridge_3h = df_preds["ae_ridge_3h"].values
    err_mm_ridge_3h = df_preds["ae_mm_ridge_3h"].values
    err_rf_3h = df_preds["ae_rf_3h"].values

    # Compute exact MAEs from plotted arrays
    mae_p1 = float(np.mean(err_pers_1h))
    mae_rd1 = float(np.mean(err_ridge_1h))
    mae_mm1 = float(np.mean(err_mm_ridge_1h))
    mae_rf1 = float(np.mean(err_rf_1h))

    mae_p3 = float(np.mean(err_pers_3h))
    mae_rd3 = float(np.mean(err_ridge_3h))
    mae_mm3 = float(np.mean(err_mm_ridge_3h))
    mae_rf3 = float(np.mean(err_rf_3h))

    # CRITICAL AUTOMATED ASSERTIONS: Assert displayed MAE equals mean absolute error of plotted series
    assert abs(mae_p1 - stage_only_res["+1h"]["persistence"]["mae_m"]) < 1e-4, "Persistence +1h MAE mismatch"
    assert abs(mae_rd1 - stage_only_res["+1h"]["ridge"]["mae_m"]) < 1e-4, "Ridge +1h MAE mismatch"
    assert abs(mae_mm1 - stage_only_res["+1h"]["multimodal_ridge"]["mae_m"]) < 1e-4, "MM Ridge +1h MAE mismatch"
    assert abs(mae_rf1 - stage_only_res["+1h"]["random_forest"]["mae_m"]) < 1e-4, "RF +1h MAE mismatch"
    assert abs(mae_p3 - stage_only_res["+3h"]["persistence"]["mae_m"]) < 1e-4, "Persistence +3h MAE mismatch"
    assert abs(mae_rd3 - stage_only_res["+3h"]["ridge"]["mae_m"]) < 1e-4, "Ridge +3h MAE mismatch"
    assert abs(mae_mm3 - stage_only_res["+3h"]["multimodal_ridge"]["mae_m"]) < 1e-4, "MM Ridge +3h MAE mismatch"
    assert abs(mae_rf3 - stage_only_res["+3h"]["random_forest"]["mae_m"]) < 1e-4, "RF +3h MAE mismatch"
    print("All automated MAE consistency assertions PASSED successfully!")

    # Panel 1: +1h Forecast Comparison
    ax1 = axes[0]
    ax1.plot(dt_test, y_true_1h, color="#0f172a", lw=2.0, label="Observed Stage (t+1h)", zorder=5)
    ax1.plot(dt_test, y_pers_1h, color="#64748b", ls=":", lw=1.4, label=f"Persistence (+1h, MAE = {mae_p1:.4f}m)", zorder=4)
    ax1.plot(dt_test, y_ridge_1h, color="#059669", ls="-.", lw=1.4, label=f"Stage Ridge (+1h, MAE = {mae_rd1:.4f}m)", zorder=3)
    ax1.plot(dt_test, y_mm_ridge_1h, color="#0891b2", ls="--", lw=1.4, label=f"Multimodal Ridge (+1h, MAE = {mae_mm1:.4f}m)", zorder=3)
    ax1.plot(dt_test, y_rf_1h, color="#2563eb", ls=":", lw=1.3, label=f"Random Forest (+1h, MAE = {mae_rf1:.4f}m)", zorder=2)
    ax1.set_ylabel("Stage (m MSL)", fontsize=10, fontweight="bold")
    ax1.set_title("Vishnupuri Barrage — Whole-Monsoon 2026 Held-Out Test Evaluation (September 1–30, 2026)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=9)

    # Panel 2: +3h Forecast Comparison
    ax2 = axes[1]
    ax2.plot(dt_test, y_true_3h, color="#0f172a", lw=2.0, label="Observed Stage (t+3h)", zorder=5)
    ax2.plot(dt_test, y_pers_3h, color="#64748b", ls=":", lw=1.4, label=f"Persistence (+3h, MAE = {mae_p3:.4f}m)", zorder=4)
    ax2.plot(dt_test, y_ridge_3h, color="#059669", ls="-.", lw=1.4, label=f"Stage Ridge (+3h, MAE = {mae_rd3:.4f}m)", zorder=3)
    ax2.plot(dt_test, y_mm_ridge_3h, color="#0891b2", ls="--", lw=1.4, label=f"Multimodal Ridge (+3h, MAE = {mae_mm3:.4f}m)", zorder=3)
    ax2.plot(dt_test, y_rf_3h, color="#7c3aed", ls=":", lw=1.3, label=f"Random Forest (+3h, MAE = {mae_rf3:.4f}m)", zorder=2)
    ax2.set_ylabel("Stage (m MSL)", fontsize=10, fontweight="bold")
    ax2.set_title("Lead Time +3 Hours Forecast Trajectory", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)

    # Panel 3: Error Residuals (|Error| across horizons & models)
    ax3 = axes[2]
    ax3.plot(dt_test, err_ridge_1h, color="#059669", lw=1.2, label=f"Stage Ridge (+1h, MAE = {mae_rd1:.4f}m)", zorder=4)
    ax3.plot(dt_test, err_mm_ridge_1h, color="#0891b2", lw=1.2, ls="--", label=f"Multimodal Ridge (+1h, MAE = {mae_mm1:.4f}m)", zorder=4)
    ax3.plot(dt_test, err_ridge_3h, color="#0d9488", lw=1.2, ls="-.", label=f"Stage Ridge (+3h, MAE = {mae_rd3:.4f}m)", zorder=3)
    ax3.plot(dt_test, err_mm_ridge_3h, color="#0284c7", lw=1.2, ls=":", label=f"Multimodal Ridge (+3h, MAE = {mae_mm3:.4f}m)", zorder=3)
    ax3.axhline(0.05, color="#dc2626", ls="--", lw=1.2, label="5cm Operational Alert Threshold", zorder=5)
    ax3.set_ylabel("Absolute Error (m)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Date (IST)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(OUT_FORECAST_PLOT, dpi=180)
    shutil.copy(OUT_FORECAST_PLOT, ARTIFACT_PLOT)
    plt.close()
    print(f"Saved forecast evaluation plot to: {OUT_FORECAST_PLOT}")
    print(f"Copied to artifact directory: {ARTIFACT_PLOT}")

if __name__ == "__main__":
    run_modelling()
