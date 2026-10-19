"""
scripts/run_phase2_model_development.py
=======================================
Phase 2 Execution Script: Model Development, Novelty Discovery, and Benchmark.

Executes:
  1. Validation-based hyperparameter tuning across horizons +1h, +2h, +3h.
  2. Model training on chronological Train partition (June 15 - Aug 09).
  3. Evaluation of Candidate Leaderboard on untouched September Test partition.
  4. Real-world flood objective evaluation: rapid-rise performance, peak underprediction,
     direction accuracy, falling/stable stratification, and fallback behavior.
  5. Authentic event evaluations: July 30-31 Inception, July 31 Surge Limb (Outage),
     and September high-water fluctuations.
  6. Feature ablation study.
  7. Model serialization (.joblib artifacts).

Outputs saved to:
  - artifacts/phase2_model_development/
  - river_forecasting/outputs/phase2/
"""

import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from river_forecasting.src.phase1_ml_pipeline import (
    HORIZONS,
    SEED,
    FEAT_STAGE_BASE,
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_RAIN_LAG,
    FEAT_MULTIMODAL,
    load_features,
    get_splits,
)
from river_forecasting.src.phase2_model_development import (
    RegimeAwareAdaptiveRegressor,
    compute_comprehensive_metrics,
)

# Output directories
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase2_model_development"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

PHASE2_OUT_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase2"
PHASE2_MODELS_DIR = PHASE2_OUT_DIR / "models"
PHASE2_MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase2_runner")


def main():
    log.info("=" * 80)
    log.info("STARTING PHASE 2 -- MODEL DEVELOPMENT AND NOVELTY DISCOVERY")
    log.info("=" * 80)

    # 1. Load data and splits
    df = load_features()
    sp = get_splits(df)
    tr, val, te = sp["train"], sp["val"], sp["test"]
    sync_feats = FEAT_STAGE_DYN + FEAT_RAIN_LAG + ["api_7d_mm"]

    log.info("Train set: %d rows (%s to %s)", len(tr), tr["dt"].min(), tr["dt"].max())
    log.info("Val set:   %d rows (%s to %s)", len(val), val["dt"].min(), val["dt"].max())
    log.info("Test set:  %d rows (%s to %s)", len(te), te["dt"].min(), te["dt"].max())

    # -------------------------------------------------------------------------
    # PART 1: VALIDATION HYPERPARAMETER TUNING & MODEL SELECTION
    # -------------------------------------------------------------------------
    log.info("\n--- PART 1: HYPERPARAMETER TUNING ON VALIDATION SPLIT ---")
    val_tuning_records = []
    best_configs = {}

    for h in HORIZONS:
        hs = f"{h}h"
        tl = f"target_v_{hs}"
        tdh = f"target_v_dH_{hs}"

        tr_c = tr.dropna(subset=FEAT_MULTIMODAL + [tl, tdh, "stage_vishnupuri"]).copy()
        val_c = val.dropna(subset=FEAT_MULTIMODAL + [tl, tdh, "stage_vishnupuri"]).copy().reset_index(drop=True)

        yc_tr = tr_c["stage_vishnupuri"].values
        yc_val = val_c["stage_vishnupuri"].values
        y_val_true = val_c[tl].values

        # 1. Persistence Baseline
        mae_pers = float(mean_absolute_error(y_val_true, yc_val))
        val_tuning_records.append({
            "horizon": f"+{hs}", "model_id": "M01_persistence", "params": "None",
            "val_mae_mm": round(mae_pers * 1000.0, 3), "val_imp_vs_pers_pct": 0.0
        })

        # 2. Stage-only Ridge (level)
        p_lvl = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_lvl.fit(tr_c[FEAT_STAGE_BASE], tr_c[tl])
        mae_lvl = float(mean_absolute_error(y_val_true, p_lvl.predict(val_c[FEAT_STAGE_BASE])))
        val_tuning_records.append({
            "horizon": f"+{hs}", "model_id": "M02_stage_only_ridge_level", "params": "alpha=10.0",
            "val_mae_mm": round(mae_lvl * 1000.0, 3), "val_imp_vs_pers_pct": round((mae_pers - mae_lvl)/mae_pers*100, 2)
        })

        # 3. Stage-change Ridge (stage-only dH)
        p_m04 = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_m04.fit(tr_c[FEAT_STAGE_DYN], tr_c[tdh])
        mae_m04 = float(mean_absolute_error(y_val_true, yc_val + p_m04.predict(val_c[FEAT_STAGE_DYN])))
        val_tuning_records.append({
            "horizon": f"+{hs}", "model_id": "M03_stage_change_ridge", "params": "alpha=10.0",
            "val_mae_mm": round(mae_m04 * 1000.0, 3), "val_imp_vs_pers_pct": round((mae_pers - mae_m04)/mae_pers*100, 2)
        })

        # 4. Model D (Stage-change + Contemporary Rain) - Tune alpha
        best_alpha_d, best_mae_d = 10.0, 999.0
        for a in [0.5, 1.0, 5.0, 10.0, 20.0, 50.0]:
            p = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=a))])
            p.fit(tr_c[FEAT_STAGE_DYN + FEAT_RAIN_CONT], tr_c[tdh])
            mae = float(mean_absolute_error(y_val_true, yc_val + p.predict(val_c[FEAT_STAGE_DYN + FEAT_RAIN_CONT])))
            if mae < best_mae_d:
                best_mae_d, best_alpha_d = mae, a
        val_tuning_records.append({
            "horizon": f"+{hs}", "model_id": "M04_model_d_rainfall", "params": f"alpha={best_alpha_d}",
            "val_mae_mm": round(best_mae_d * 1000.0, 3), "val_imp_vs_pers_pct": round((mae_pers - best_mae_d)/mae_pers*100, 2)
        })

        # 5. Random Forest - Tune max_depth
        best_rf_md, best_mae_rf = 4, 999.0
        for md in [3, 4, 5, 6]:
            rf = RandomForestRegressor(n_estimators=100, max_depth=md, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
            rf.fit(tr_c[FEAT_MULTIMODAL], tr_c[tdh])
            mae = float(mean_absolute_error(y_val_true, yc_val + rf.predict(val_c[FEAT_MULTIMODAL])))
            if mae < best_mae_rf:
                best_mae_rf, best_rf_md = mae, md
        val_tuning_records.append({
            "horizon": f"+{hs}", "model_id": "M05_random_forest", "params": f"max_depth={best_rf_md}, n_est=100",
            "val_mae_mm": round(best_mae_rf * 1000.0, 3), "val_imp_vs_pers_pct": round((mae_pers - best_mae_rf)/mae_pers*100, 2)
        })

        # 6. Extra Trees - Tune max_depth
        best_et_md, best_mae_et = 4, 999.0
        for md in [3, 4, 5, 6]:
            et = ExtraTreesRegressor(n_estimators=100, max_depth=md, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
            et.fit(tr_c[FEAT_MULTIMODAL], tr_c[tdh])
            mae = float(mean_absolute_error(y_val_true, yc_val + et.predict(val_c[FEAT_MULTIMODAL])))
            if mae < best_mae_et:
                best_mae_et, best_et_md = mae, md
        val_tuning_records.append({
            "horizon": f"+{hs}", "model_id": "M06_extra_trees", "params": f"max_depth={best_et_md}, n_est=100",
            "val_mae_mm": round(best_mae_et * 1000.0, 3), "val_imp_vs_pers_pct": round((mae_pers - best_mae_et)/mae_pers*100, 2)
        })

        # 7. XGBoost - Tune max_depth & reg_lambda
        best_xgb_p, best_mae_xgb = {}, 999.0
        if _XGB_AVAILABLE:
            for md in [2, 3]:
                for lr in [0.02, 0.05]:
                    for l2 in [2.0, 5.0, 10.0]:
                        m = xgb.XGBRegressor(
                            max_depth=md, learning_rate=lr, n_estimators=60,
                            reg_lambda=l2, subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, tree_method="hist"
                        )
                        m.fit(tr_c[FEAT_MULTIMODAL], tr_c[tdh])
                        mae = float(mean_absolute_error(y_val_true, yc_val + m.predict(val_c[FEAT_MULTIMODAL])))
                        if mae < best_mae_xgb:
                            best_mae_xgb = mae
                            best_xgb_p = {"max_depth": md, "lr": lr, "reg_lambda": l2}
            val_tuning_records.append({
                "horizon": f"+{hs}", "model_id": "M07_xgboost", "params": str(best_xgb_p),
                "val_mae_mm": round(best_mae_xgb * 1000.0, 3), "val_imp_vs_pers_pct": round((mae_pers - best_mae_xgb)/mae_pers*100, 2)
            })

        # 8. Novelty: Regime-Aware Adaptive Gating (RA-AG)
        best_rag_cfg, best_mae_rag = {}, 999.0
        for th_rise in [0.010, 0.015, 0.020]:
            for tree_expert in [False, True]:
                rag = RegimeAwareAdaptiveRegressor(
                    horizon=h, theta_rise=th_rise, theta_fall=-th_rise,
                    alpha_rise=5.0, alpha_fall=10.0,
                    use_pure_persistence_on_stable=True,
                    use_tree_rising_expert=tree_expert
                )
                rag.fit(tr_c, tr_c[tdh].values)
                p_dh, is_fb, _ = rag.predict(val_c)
                mae_rag = float(mean_absolute_error(y_val_true, yc_val + p_dh))
                if mae_rag < best_mae_rag:
                    best_mae_rag = mae_rag
                    best_rag_cfg = {"theta_rise": th_rise, "tree_expert": tree_expert}
        val_tuning_records.append({
            "horizon": f"+{hs}", "model_id": "M08_regime_aware_adaptive", "params": str(best_rag_cfg),
            "val_mae_mm": round(best_mae_rag * 1000.0, 3), "val_imp_vs_pers_pct": round((mae_pers - best_mae_rag)/mae_pers*100, 2)
        })

        # 9. Event-Weighted Ridge (Controlled asymmetric penalty on rises)
        w_tr = 1.0 + 2.0 * np.maximum(0, tr_c[tdh].values / (np.std(tr_c[tdh].values) + 1e-6))
        p_ew = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_ew.fit(tr_c[FEAT_STAGE_DYN + FEAT_RAIN_CONT], tr_c[tdh], est__sample_weight=w_tr)
        mae_ew = float(mean_absolute_error(y_val_true, yc_val + p_ew.predict(val_c[FEAT_STAGE_DYN + FEAT_RAIN_CONT])))
        val_tuning_records.append({
            "horizon": f"+{hs}", "model_id": "M09_event_weighted_ridge", "params": "alpha=10.0, rise_weight=2.0",
            "val_mae_mm": round(mae_ew * 1000.0, 3), "val_imp_vs_pers_pct": round((mae_pers - mae_ew)/mae_pers*100, 2)
        })

        best_configs[h] = {
            "alpha_d": best_alpha_d,
            "rf_md": best_rf_md,
            "et_md": best_et_md,
            "xgb_p": best_xgb_p,
            "rag_cfg": best_rag_cfg
        }
        log.info("+%s Val Tuning: Best Ridge alpha=%.1f | Best XGB=%s | Best RA-AG=%s",
                 hs, best_alpha_d, best_xgb_p, best_rag_cfg)

    df_val_results = pd.DataFrame(val_tuning_records)
    df_val_results.to_csv(ARTIFACT_DIR / "phase2_validation_results.csv", index=False)
    log.info("Saved phase2_validation_results.csv successfully.")

    # -------------------------------------------------------------------------
    # PART 2: FINAL TEST BENCHMARK & CANDIDATE LEADERBOARD
    # -------------------------------------------------------------------------
    log.info("\n--- PART 2: COMPREHENSIVE BENCHMARK ON UNTOUCHED TEST SET ---")
    leaderboard_records = []
    test_prediction_rows = []
    trained_artifacts = {}

    for h in HORIZONS:
        hs = f"{h}h"
        tl = f"target_v_{hs}"
        tdh = f"target_v_dH_{hs}"

        # Matched training and evaluation subsets
        tr_c = tr.dropna(subset=FEAT_MULTIMODAL + [tl, tdh, "stage_vishnupuri"]).copy()
        # Synchronized test cohort (complete cases across all features)
        te_sync = te.dropna(subset=sync_feats + [tl, tdh, "stage_vishnupuri"]).copy().reset_index(drop=True)
        # Full test cohort (with real-world missing rainfall to evaluate fallback)
        te_full = te.dropna(subset=FEAT_STAGE_DYN + [tl, tdh, "stage_vishnupuri"]).copy().reset_index(drop=True)

        yc_sync = te_sync["stage_vishnupuri"].values
        yt_sync = te_sync[tl].values

        cfg_h = best_configs[h]

        # 1. M01: Persistence
        pred_pers = yc_sync
        m_pers = compute_comprehensive_metrics(yt_sync, pred_pers, yc_sync, te_sync)
        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M01_persistence", "model_name": "Persistence Baseline (dH = 0)",
            "test_mae_mm": m_pers["mae_mm"], "test_rmse_mm": m_pers["rmse_mm"], "test_bias_mm": m_pers["bias_mm"],
            "test_r2": m_pers["r2"], "imp_vs_pers_pct": 0.0, "direction_accuracy_pct": m_pers["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_pers["mae_rapid_rise_mm"], "peak_underprediction_mm": m_pers["peak_underprediction_mm"],
            "mae_falling_mm": m_pers["mae_falling_mm"], "mae_stable_mm": m_pers["mae_stable_mm"],
            "fallback_freq_pct": 0.0, "forecast_coverage_pct": 100.0
        })

        # 2. M02: Stage-Only Ridge (Direct Level)
        p_lvl = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_lvl.fit(tr_c[FEAT_STAGE_BASE], tr_c[tl])
        pred_lvl = p_lvl.predict(te_sync[FEAT_STAGE_BASE])
        m_lvl = compute_comprehensive_metrics(yt_sync, pred_lvl, yc_sync, te_sync)
        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M02_stage_only_ridge_level", "model_name": "Ridge (Stage History Direct Level)",
            "test_mae_mm": m_lvl["mae_mm"], "test_rmse_mm": m_lvl["rmse_mm"], "test_bias_mm": m_lvl["bias_mm"],
            "test_r2": m_lvl["r2"], "imp_vs_pers_pct": m_lvl["imp_vs_persistence_pct"], "direction_accuracy_pct": m_lvl["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_lvl["mae_rapid_rise_mm"], "peak_underprediction_mm": m_lvl["peak_underprediction_mm"],
            "mae_falling_mm": m_lvl["mae_falling_mm"], "mae_stable_mm": m_lvl["mae_stable_mm"],
            "fallback_freq_pct": 0.0, "forecast_coverage_pct": 100.0
        })

        # 3. M03: Stage-Change Ridge (Stage-Only dH)
        p_m04 = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_m04.fit(tr_c[FEAT_STAGE_DYN], tr_c[tdh])
        pred_m04 = yc_sync + p_m04.predict(te_sync[FEAT_STAGE_DYN])
        m_m04 = compute_comprehensive_metrics(yt_sync, pred_m04, yc_sync, te_sync)
        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M03_stage_change_ridge", "model_name": "Ridge (Stage-Change dH, Stage-Only)",
            "test_mae_mm": m_m04["mae_mm"], "test_rmse_mm": m_m04["rmse_mm"], "test_bias_mm": m_m04["bias_mm"],
            "test_r2": m_m04["r2"], "imp_vs_pers_pct": m_m04["imp_vs_persistence_pct"], "direction_accuracy_pct": m_m04["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_m04["mae_rapid_rise_mm"], "peak_underprediction_mm": m_m04["peak_underprediction_mm"],
            "mae_falling_mm": m_m04["mae_falling_mm"], "mae_stable_mm": m_m04["mae_stable_mm"],
            "fallback_freq_pct": 0.0, "forecast_coverage_pct": 100.0
        })

        # 4. M04: Model D (Stage-Change + Contemporary Rain)
        p_m05 = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=cfg_h["alpha_d"]))])
        p_m05.fit(tr_c[FEAT_STAGE_DYN + FEAT_RAIN_CONT], tr_c[tdh])
        pred_m05 = yc_sync + p_m05.predict(te_sync[FEAT_STAGE_DYN + FEAT_RAIN_CONT])
        m_m05 = compute_comprehensive_metrics(yt_sync, pred_m05, yc_sync, te_sync)
        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M04_model_d_rainfall", "model_name": f"Ridge (Stage-Change + Rain [Model D, alpha={cfg_h['alpha_d']}])",
            "test_mae_mm": m_m05["mae_mm"], "test_rmse_mm": m_m05["rmse_mm"], "test_bias_mm": m_m05["bias_mm"],
            "test_r2": m_m05["r2"], "imp_vs_pers_pct": m_m05["imp_vs_persistence_pct"], "direction_accuracy_pct": m_m05["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_m05["mae_rapid_rise_mm"], "peak_underprediction_mm": m_m05["peak_underprediction_mm"],
            "mae_falling_mm": m_m05["mae_falling_mm"], "mae_stable_mm": m_m05["mae_stable_mm"],
            "fallback_freq_pct": 0.0, "forecast_coverage_pct": 100.0
        })
        trained_artifacts[f"M04_model_d_{hs}"] = p_m05

        # 5. M05: Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=cfg_h["rf_md"], min_samples_leaf=5, random_state=SEED, n_jobs=-1)
        rf.fit(tr_c[FEAT_MULTIMODAL], tr_c[tdh])
        pred_rf = yc_sync + rf.predict(te_sync[FEAT_MULTIMODAL])
        m_rf = compute_comprehensive_metrics(yt_sync, pred_rf, yc_sync, te_sync)
        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M05_random_forest", "model_name": f"Random Forest (Stage+Rain+API, depth={cfg_h['rf_md']})",
            "test_mae_mm": m_rf["mae_mm"], "test_rmse_mm": m_rf["rmse_mm"], "test_bias_mm": m_rf["bias_mm"],
            "test_r2": m_rf["r2"], "imp_vs_pers_pct": m_rf["imp_vs_persistence_pct"], "direction_accuracy_pct": m_rf["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_rf["mae_rapid_rise_mm"], "peak_underprediction_mm": m_rf["peak_underprediction_mm"],
            "mae_falling_mm": m_rf["mae_falling_mm"], "mae_stable_mm": m_rf["mae_stable_mm"],
            "fallback_freq_pct": 0.0, "forecast_coverage_pct": 100.0
        })

        # 6. M06: Extra Trees
        et = ExtraTreesRegressor(n_estimators=100, max_depth=cfg_h["et_md"], min_samples_leaf=5, random_state=SEED, n_jobs=-1)
        et.fit(tr_c[FEAT_MULTIMODAL], tr_c[tdh])
        pred_et = yc_sync + et.predict(te_sync[FEAT_MULTIMODAL])
        m_et = compute_comprehensive_metrics(yt_sync, pred_et, yc_sync, te_sync)
        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M06_extra_trees", "model_name": f"Extra Trees (Stage+Rain+API, depth={cfg_h['et_md']})",
            "test_mae_mm": m_et["mae_mm"], "test_rmse_mm": m_et["rmse_mm"], "test_bias_mm": m_et["bias_mm"],
            "test_r2": m_et["r2"], "imp_vs_pers_pct": m_et["imp_vs_persistence_pct"], "direction_accuracy_pct": m_et["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_et["mae_rapid_rise_mm"], "peak_underprediction_mm": m_et["peak_underprediction_mm"],
            "mae_falling_mm": m_et["mae_falling_mm"], "mae_stable_mm": m_et["mae_stable_mm"],
            "fallback_freq_pct": 0.0, "forecast_coverage_pct": 100.0
        })

        # 7. M07: XGBoost
        xgb_p = cfg_h["xgb_p"]
        xg = xgb.XGBRegressor(
            max_depth=xgb_p.get("max_depth", 2),
            learning_rate=xgb_p.get("lr", 0.02),
            n_estimators=60,
            reg_lambda=xgb_p.get("reg_lambda", 5.0),
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, tree_method="hist"
        )
        xg.fit(tr_c[FEAT_MULTIMODAL], tr_c[tdh])
        pred_xg = yc_sync + xg.predict(te_sync[FEAT_MULTIMODAL])
        m_xg = compute_comprehensive_metrics(yt_sync, pred_xg, yc_sync, te_sync)
        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M07_xgboost", "model_name": f"XGBoost (depth={xgb_p.get('max_depth',2)}, lr={xgb_p.get('lr',0.02)}, l2={xgb_p.get('reg_lambda',5.0)})",
            "test_mae_mm": m_xg["mae_mm"], "test_rmse_mm": m_xg["rmse_mm"], "test_bias_mm": m_xg["bias_mm"],
            "test_r2": m_xg["r2"], "imp_vs_pers_pct": m_xg["imp_vs_persistence_pct"], "direction_accuracy_pct": m_xg["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_xg["mae_rapid_rise_mm"], "peak_underprediction_mm": m_xg["peak_underprediction_mm"],
            "mae_falling_mm": m_xg["mae_falling_mm"], "mae_stable_mm": m_xg["mae_stable_mm"],
            "fallback_freq_pct": 0.0, "forecast_coverage_pct": 100.0
        })
        trained_artifacts[f"M07_xgboost_{hs}"] = xg

        # 8. M08: Novelty - Regime-Aware Adaptive Gating (RA-AG)
        rag_p = cfg_h["rag_cfg"]
        rag = RegimeAwareAdaptiveRegressor(
            horizon=h,
            theta_rise=rag_p.get("theta_rise", 0.015),
            theta_fall=-rag_p.get("theta_rise", 0.015),
            alpha_rise=5.0, alpha_fall=10.0,
            use_pure_persistence_on_stable=True,
            use_tree_rising_expert=rag_p.get("tree_expert", False)
        )
        rag.fit(tr_c, tr_c[tdh].values)
        
        # Evaluate on synchronized test cohort
        pred_dh_rag_sync, fb_sync, reg_sync = rag.predict(te_sync)
        pred_rag_sync = yc_sync + pred_dh_rag_sync
        m_rag_sync = compute_comprehensive_metrics(yt_sync, pred_rag_sync, yc_sync, te_sync, is_fallback=fb_sync)
        
        # Also evaluate on full test cohort (testing authentic missing-rainfall fallback)
        yc_full = te_full["stage_vishnupuri"].values
        yt_full = te_full[tl].values
        pred_dh_rag_full, fb_full, reg_full = rag.predict(te_full)
        pred_rag_full = yc_full + pred_dh_rag_full
        m_rag_full = compute_comprehensive_metrics(yt_full, pred_rag_full, yc_full, te_full, is_fallback=fb_full)

        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M08_regime_aware_adaptive", "model_name": f"Regime-Aware Adaptive Gating (RA-AG, theta={rag_p.get('theta_rise', 0.015)})",
            "test_mae_mm": m_rag_sync["mae_mm"], "test_rmse_mm": m_rag_sync["rmse_mm"], "test_bias_mm": m_rag_sync["bias_mm"],
            "test_r2": m_rag_sync["r2"], "imp_vs_pers_pct": m_rag_sync["imp_vs_persistence_pct"], "direction_accuracy_pct": m_rag_sync["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_rag_sync["mae_rapid_rise_mm"], "peak_underprediction_mm": m_rag_sync["peak_underprediction_mm"],
            "mae_falling_mm": m_rag_sync["mae_falling_mm"], "mae_stable_mm": m_rag_sync["mae_stable_mm"],
            "fallback_freq_pct": m_rag_full["fallback_freq_pct"], "forecast_coverage_pct": 100.0
        })
        trained_artifacts[f"M08_regime_adaptive_{hs}"] = rag

        # 9. M09: Event-Weighted Ridge Regressor
        w_tr = 1.0 + 2.0 * np.maximum(0, tr_c[tdh].values / (np.std(tr_c[tdh].values) + 1e-6))
        p_ew = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_ew.fit(tr_c[FEAT_STAGE_DYN + FEAT_RAIN_CONT], tr_c[tdh], est__sample_weight=w_tr)
        pred_ew = yc_sync + p_ew.predict(te_sync[FEAT_STAGE_DYN + FEAT_RAIN_CONT])
        m_ew = compute_comprehensive_metrics(yt_sync, pred_ew, yc_sync, te_sync)
        leaderboard_records.append({
            "horizon": f"+{hs}", "model_id": "M09_event_weighted_ridge", "model_name": "Event-Weighted Ridge (Asymmetric Rise Penalty)",
            "test_mae_mm": m_ew["mae_mm"], "test_rmse_mm": m_ew["rmse_mm"], "test_bias_mm": m_ew["bias_mm"],
            "test_r2": m_ew["r2"], "imp_vs_pers_pct": m_ew["imp_vs_persistence_pct"], "direction_accuracy_pct": m_ew["direction_accuracy_pct"],
            "mae_rapid_rise_mm": m_ew["mae_rapid_rise_mm"], "peak_underprediction_mm": m_ew["peak_underprediction_mm"],
            "mae_falling_mm": m_ew["mae_falling_mm"], "mae_stable_mm": m_ew["mae_stable_mm"],
            "fallback_freq_pct": 0.0, "forecast_coverage_pct": 100.0
        })

        # Save row-level predictions for synchronized test cohort
        for i in range(len(te_sync)):
            test_prediction_rows.append({
                "horizon": f"+{hs}",
                "dt": str(te_sync.loc[i, "dt"]),
                "stage_current_m": yc_sync[i],
                "stage_target_m": yt_sync[i],
                "pred_m01_persistence_m": pred_pers[i],
                "pred_m03_stage_ridge_m": pred_m04[i],
                "pred_m04_model_d_m": pred_m05[i],
                "pred_m06_extra_trees_m": pred_et[i],
                "pred_m07_xgboost_m": pred_xg[i],
                "pred_m08_regime_adaptive_m": pred_rag_sync[i],
                "assigned_regime": reg_sync[i],
                "fallback_engaged": fb_sync[i],
                "error_m01_persistence_mm": abs(yt_sync[i] - pred_pers[i]) * 1000.0,
                "error_m04_model_d_mm": abs(yt_sync[i] - pred_m05[i]) * 1000.0,
                "error_m07_xgboost_mm": abs(yt_sync[i] - pred_xg[i]) * 1000.0,
                "error_m08_regime_adaptive_mm": abs(yt_sync[i] - pred_rag_sync[i]) * 1000.0,
            })

    # Save leaderboard and row predictions
    df_leaderboard = pd.DataFrame(leaderboard_records)
    df_leaderboard.to_csv(ARTIFACT_DIR / "phase2_candidate_leaderboard.csv", index=False)
    log.info("Saved phase2_candidate_leaderboard.csv successfully.")

    df_preds = pd.DataFrame(test_prediction_rows)
    df_preds.to_csv(PHASE2_OUT_DIR / "phase2_test_predictions.csv", index=False)
    log.info("Saved phase2_test_predictions.csv successfully (N=%d).", len(df_preds))

    # Serialize top model artifacts
    for name, art in trained_artifacts.items():
        art_path = PHASE2_MODELS_DIR / f"{name}.joblib"
        joblib.dump(art, art_path)
    log.info("Serialized %d Phase 2 model artifacts to %s.", len(trained_artifacts), PHASE2_MODELS_DIR)

    # -------------------------------------------------------------------------
    # PART 3: AUTHENTIC EVENT & REGIME EVALUATION (FLOOD OBJECTIVE)
    # -------------------------------------------------------------------------
    log.info("\n--- PART 3: EVALUATION ON AUTHENTIC FLOOD EVENTS & REGIMES ---")
    event_records = []

    # Pre-event models for chronological evaluation of July 30-31 flood
    pre_train = df[(df["dt"] >= "2026-06-15 00:00:00") & (df["dt"] <= "2026-07-29 23:00:00")].copy()
    
    event_definitions = [
        ("July 30–31 Flood Inception (Phase A)", "2026-07-30 00:00:00", "2026-07-31 05:00:00", True),
        ("July 31–Aug 01 Rapid Surge Limb (Phase B)", "2026-07-31 06:00:00", "2026-08-01 07:00:00", True),
        ("September 18 Test Inflow Pulse", "2026-09-18 01:00:00", "2026-09-18 08:00:00", False),
        ("September 21 Test Inflow Pulse", "2026-09-21 16:00:00", "2026-09-21 20:00:00", False),
    ]

    for ev_name, t_start, t_end, use_pre in event_definitions:
        sub_ev = df[(df["dt"] >= t_start) & (df["dt"] <= t_end)].copy().reset_index(drop=True)
        if sub_ev.empty:
            continue

        for h in HORIZONS:
            hs = f"{h}h"
            tl = f"target_v_{hs}"
            tdh = f"target_v_dH_{hs}"

            valid_ev = sub_ev.dropna(subset=[tl, tdh, "stage_vishnupuri"] + FEAT_STAGE_DYN).copy().reset_index(drop=True)
            if valid_ev.empty:
                continue

            yc_ev = valid_ev["stage_vishnupuri"].values
            yt_ev = valid_ev[tl].values

            # Fit pre-models if required
            if use_pre:
                valid_pre = pre_train.dropna(subset=FEAT_MULTIMODAL + [tl, tdh, "stage_vishnupuri"]).copy()
                m_m05_ev = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))]).fit(valid_pre[FEAT_STAGE_DYN + FEAT_RAIN_CONT], valid_pre[tdh])
                m_xg_ev = xgb.XGBRegressor(max_depth=3, learning_rate=0.03, n_estimators=60, reg_lambda=5.0, random_state=SEED, tree_method="hist").fit(valid_pre[FEAT_MULTIMODAL], valid_pre[tdh])
                m_rag_ev = RegimeAwareAdaptiveRegressor(horizon=h, theta_rise=0.015, theta_fall=-0.015, alpha_rise=5.0).fit(valid_pre, valid_pre[tdh].values)
            else:
                m_m05_ev = trained_artifacts[f"M04_model_d_{hs}"]
                m_xg_ev = trained_artifacts[f"M07_xgboost_{hs}"]
                m_rag_ev = trained_artifacts[f"M08_regime_adaptive_{hs}"]

            # 1. Persistence
            mae_p = float(mean_absolute_error(yt_ev, yc_ev))
            rmse_p = float(np.sqrt(np.mean((yt_ev - yc_ev)**2)))
            bias_p = float(np.mean(yc_ev - yt_ev))

            # 2. Model D
            has_rain = valid_ev[FEAT_RAIN_CONT].notna().all(axis=1)
            if has_rain.all():
                p_m05 = yc_ev + m_m05_ev.predict(valid_ev[FEAT_STAGE_DYN + FEAT_RAIN_CONT])
                mae_m05 = float(mean_absolute_error(yt_ev, p_m05))
                rmse_m05 = float(np.sqrt(np.mean((yt_ev - p_m05)**2)))
                bias_m05 = float(np.mean(p_m05 - yt_ev))
                status_m05 = "COMPLETE"
            else:
                mae_m05, rmse_m05, bias_m05 = None, None, None
                status_m05 = "OUTAGE (0% Available)"

            # 3. XGBoost
            has_mm = valid_ev[FEAT_MULTIMODAL].notna().all(axis=1)
            if has_mm.all():
                p_xg = yc_ev + m_xg_ev.predict(valid_ev[FEAT_MULTIMODAL])
                mae_xg = float(mean_absolute_error(yt_ev, p_xg))
                rmse_xg = float(np.sqrt(np.mean((yt_ev - p_xg)**2)))
                bias_xg = float(np.mean(p_xg - yt_ev))
                status_xg = "COMPLETE"
            else:
                mae_xg, rmse_xg, bias_xg = None, None, None
                status_xg = "OUTAGE (0% Available)"

            # 4. Regime-Aware Adaptive Gating (With Stage-Only Fallback)
            pred_dh_rag, is_fb, _ = m_rag_ev.predict(valid_ev)
            p_rag = yc_ev + pred_dh_rag
            mae_rag = float(mean_absolute_error(yt_ev, p_rag))
            rmse_rag = float(np.sqrt(np.mean((yt_ev - p_rag)**2)))
            bias_rag = float(np.mean(p_rag - yt_ev))
            status_rag = f"FALLBACK_ACTIVE ({is_fb.sum()}/{len(valid_ev)} hours)" if is_fb.any() else "COMPLETE"

            event_records.append({
                "episode": ev_name,
                "horizon": f"+{hs}",
                "n_samples": len(valid_ev),
                "persistence_mae_mm": round(mae_p * 1000.0, 2),
                "persistence_bias_mm": round(bias_p * 1000.0, 2),
                "status_model_d": status_m05,
                "model_d_mae_mm": round(mae_m05 * 1000.0, 2) if mae_m05 is not None else None,
                "model_d_bias_mm": round(bias_m05 * 1000.0, 2) if bias_m05 is not None else None,
                "status_xgboost": status_xg,
                "xgboost_mae_mm": round(mae_xg * 1000.0, 2) if mae_xg is not None else None,
                "xgboost_bias_mm": round(bias_xg * 1000.0, 2) if bias_xg is not None else None,
                "status_regime_adaptive": status_rag,
                "regime_adaptive_mae_mm": round(mae_rag * 1000.0, 2),
                "regime_adaptive_bias_mm": round(bias_rag * 1000.0, 2),
                "regime_adaptive_vs_pers_pct": round((mae_p - mae_rag) / mae_p * 100.0, 2)
            })

    df_events = pd.DataFrame(event_records)
    df_events.to_csv(ARTIFACT_DIR / "phase2_event_regime_evaluation.csv", index=False)
    log.info("Saved phase2_event_regime_evaluation.csv successfully.")

    # -------------------------------------------------------------------------
    # PART 4: FEATURE ABLATION STUDY
    # -------------------------------------------------------------------------
    log.info("\n--- PART 4: FEATURE ABLATION STUDY ---")
    ablation_records = []

    feature_suites = [
        ("F01_Stage_Base", FEAT_STAGE_BASE, "level"),
        ("F02_Stage_Dynamics", FEAT_STAGE_DYN, "dH"),
        ("F03_Stage_Dyn_Contemp_Rain", FEAT_STAGE_DYN + FEAT_RAIN_CONT, "dH"),
        ("F04_Stage_Dyn_Lagged_Rain", FEAT_STAGE_DYN + FEAT_RAIN_LAG, "dH"),
        ("F05_Full_Multimodal", FEAT_MULTIMODAL, "dH"),
    ]

    for h in HORIZONS:
        hs = f"{h}h"
        tl = f"target_v_{hs}"
        tdh = f"target_v_dH_{hs}"

        tr_c = tr.dropna(subset=sync_feats + [tl, tdh, "stage_vishnupuri"]).copy()
        val_c = val.dropna(subset=sync_feats + [tl, tdh, "stage_vishnupuri"]).copy().reset_index(drop=True)
        te_sync = te.dropna(subset=sync_feats + [tl, tdh, "stage_vishnupuri"]).copy().reset_index(drop=True)

        yc_val = val_c["stage_vishnupuri"].values
        yt_val = val_c[tl].values
        yc_te = te_sync["stage_vishnupuri"].values
        yt_te = te_sync[tl].values

        for f_name, f_cols, tgt_type in feature_suites:
            p = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
            if tgt_type == "level":
                p.fit(tr_c[f_cols], tr_c[tl])
                val_pred = p.predict(val_c[f_cols])
                test_pred = p.predict(te_sync[f_cols])
            else:
                p.fit(tr_c[f_cols], tr_c[tdh])
                val_pred = yc_val + p.predict(val_c[f_cols])
                test_pred = yc_te + p.predict(te_sync[f_cols])

            v_mae = float(mean_absolute_error(yt_val, val_pred))
            t_mae = float(mean_absolute_error(yt_te, test_pred))

            ablation_records.append({
                "horizon": f"+{hs}",
                "feature_suite": f_name,
                "n_features": len(f_cols),
                "target_type": tgt_type,
                "val_mae_mm": round(v_mae * 1000.0, 3),
                "test_mae_mm": round(t_mae * 1000.0, 3),
            })

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(ARTIFACT_DIR / "phase2_feature_ablation.csv", index=False)
    log.info("Saved phase2_feature_ablation.csv successfully.")

    log.info("\nPhase 2 execution script completed successfully.")


if __name__ == "__main__":
    main()
