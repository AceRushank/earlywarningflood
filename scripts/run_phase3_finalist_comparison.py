"""
scripts/run_phase3_finalist_comparison.py
=========================================
Phase 3 Execution Script: Finalist Model Comparison & Novelty Validation
Vishnupuri Barrage / Godavari River, Nanded, Maharashtra

Finalists Evaluated:
  - Finalist A: Tuned XGBoost Baseline (Standalone M07)
  - Finalist B: Existing RA-AG (Phase 2 Exact Reproduction M08)
  - Finalist C: RA-AG + XGBoost Rising Expert Hybrid (Novelty Candidate)
  - Baseline Ref: Persistence Baseline (M01)

Executes:
  1. Training on chronological Train partition (June 15 - Aug 09).
  2. Validation threshold & hyperparameter confirmation (Aug 10 - Aug 31).
  3. Comprehensive evaluation on untouched September Test partition (Sept 01 - Sept 30).
  4. Identical timestamp cohort alignment across all horizons (+1h, +2h, +3h).
  5. Regime stratification (Stable, Rising, Falling).
  6. Authentic event evaluation (July 30-31 Inception, July 31 MOSDAC Outage Surge, Sept 18 Peak, Sept 21 Pulse).
  7. Paired Newey-West HAC & Diebold-Mariano HLN hypothesis tests.
  8. Serialization of finalist models and manifest.
  9. Generation of all Phase 3 required deliverables.
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
from sklearn.metrics import mean_absolute_error

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from river_forecasting.src.phase3_finalist_comparison import (
    RegimeDetector,
    ExistingRAAG,
    RAAGXGBHybrid,
    TunedXGBoostPipeline,
    compute_audited_metrics,
    compute_paired_hac_dm,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase3_finalist_comparison"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = REPO_ROOT / "river_forecasting" / "outputs" / "phase3" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase3_runner")


def main():
    log.info("=" * 80)
    log.info("STARTING PHASE 3 -- FINALIST MODEL COMPARISON & NOVELTY VALIDATION")
    log.info("=" * 80)

    # 1. Load data and chronological partitions
    df = load_features()
    sp = get_splits(df)
    tr, val, te = sp["train"], sp["val"], sp["test"]
    sync_feats = FEAT_STAGE_DYN + FEAT_RAIN_LAG + ["api_7d_mm"]

    log.info("Data loaded successfully:")
    log.info("  Train set: %d rows (%s to %s)", len(tr), tr["dt"].min(), tr["dt"].max())
    log.info("  Val set:   %d rows (%s to %s)", len(val), val["dt"].min(), val["dt"].max())
    log.info("  Test set:  %d rows (%s to %s)", len(te), te["dt"].min(), te["dt"].max())

    # Pre-defined tuned hyperparameters from Phase 2 validation
    xgb_params_by_h = {
        1: {"max_depth": 2, "learning_rate": 0.02, "reg_lambda": 5.0, "n_estimators": 60},
        2: {"max_depth": 2, "learning_rate": 0.02, "reg_lambda": 10.0, "n_estimators": 60},
        3: {"max_depth": 2, "learning_rate": 0.02, "reg_lambda": 10.0, "n_estimators": 60},
    }

    # Data structures for outputs
    leaderboard_records = []
    regime_records = []
    event_records = []
    paired_comparison_records = []
    all_predictions_rows = []
    model_manifest = {}

    # Event definitions for peak & flood surge audit
    # 1. July 30-31 Inception: 2026-07-30 00:00 to 2026-07-31 05:00
    # 2. July 31 MOSDAC Outage Surge: 2026-07-31 06:00 to 2026-07-31 23:00 (Rainfall missing, rapid stage rise)
    # 3. September 18 Peak: 2026-09-17 12:00 to 2026-09-19 12:00 (Highest water in test period, ~354.12 m)
    # 4. September 21 Pulse: 2026-09-20 18:00 to 2026-09-22 18:00 (Secondary pulse, ~353.98 m)
    event_specs = {
        "event_sep18_peak": {
            "name": "September 18 Test Peak Pulse",
            "start": "2026-09-17 12:00:00",
            "end": "2026-09-19 12:00:00",
            "partition": "test",
        },
        "event_sep21_pulse": {
            "name": "September 21 Test Secondary Pulse",
            "start": "2026-09-20 18:00:00",
            "end": "2026-09-22 18:00:00",
            "partition": "test",
        },
        "event_july30_inception": {
            "name": "July 30-31 Phase A Inception Limb",
            "start": "2026-07-30 00:00:00",
            "end": "2026-07-31 05:00:00",
            "partition": "train_val",
        },
        "event_july31_surge_outage": {
            "name": "July 31 Phase B Surge Limb (MOSDAC Telemetry Outage)",
            "start": "2026-07-31 06:00:00",
            "end": "2026-07-31 23:00:00",
            "partition": "train_val",
        },
    }

    # =========================================================================
    # HORIZON EVALUATION LOOP (+1h, +2h, +3h)
    # =========================================================================
    for h in HORIZONS:
        hs = f"{h}h"
        tl = f"target_v_{hs}"
        tdh = f"target_v_dH_{hs}"
        log.info("\n" + "=" * 60)
        log.info("PROCESSING HORIZON +%s", hs)
        log.info("=" * 60)

        # ---------------------------------------------------------------------
        # 1. Dataset Preparation
        # ---------------------------------------------------------------------
        # Training set: complete multimodal cases
        tr_c = tr.dropna(subset=FEAT_MULTIMODAL + [tl, tdh, "stage_vishnupuri"]).copy()
        
        # Test sets:
        # te_sync: complete cases across all features (identical cohort for fair numerical comparison)
        te_sync = te.dropna(subset=sync_feats + [tl, tdh, "stage_vishnupuri"]).copy().reset_index(drop=True)
        # te_full: all available stage observations (evaluating fallback & coverage during missing rainfall)
        te_full = te.dropna(subset=FEAT_STAGE_DYN + [tl, tdh, "stage_vishnupuri"]).copy().reset_index(drop=True)

        yc_sync = te_sync["stage_vishnupuri"].values
        yt_sync = te_sync[tl].values
        yc_full = te_full["stage_vishnupuri"].values
        yt_full = te_full[tl].values

        n_sync = len(te_sync)
        n_full = len(te_full)
        log.info("+%s Test Cohorts: Synchronized = %d rows | Full Operational = %d rows", hs, n_sync, n_full)

        # ---------------------------------------------------------------------
        # 2. Fit Finalists on Training Partition
        # ---------------------------------------------------------------------
        # Baseline: Persistence (dH = 0)
        # Finalist A: Tuned XGBoost Baseline
        xgb_p = xgb_params_by_h[h]
        finalist_a = TunedXGBoostPipeline(horizon=h, xgb_params=xgb_p)
        finalist_a.fit(tr_c, tr_c[tdh].values)

        # Finalist B: Existing RA-AG (Phase 2 Exact Reproduction)
        finalist_b = ExistingRAAG(
            horizon=h,
            theta_rise=0.010,
            theta_fall=-0.010,
            alpha_rise=5.0,
            alpha_fall=10.0,
            use_pure_persistence_on_stable=True,
            use_tree_rising_expert=True,
        )
        finalist_b.fit(tr_c, tr_c[tdh].values)

        # Finalist C: RA-AG + XGBoost Rising Expert Hybrid
        finalist_c = RAAGXGBHybrid(
            horizon=h,
            theta_rise=0.010,
            theta_fall=-0.010,
            xgb_params=xgb_p,
            alpha_fall=10.0,
        )
        finalist_c.fit(tr_c, tr_c[tdh].values)

        # Serialize trained models
        fa_path = MODELS_DIR / f"finalist_a_xgboost_{hs}.joblib"
        fb_path = MODELS_DIR / f"finalist_b_raag_{hs}.joblib"
        fc_path = MODELS_DIR / f"finalist_c_hybrid_{hs}.joblib"

        joblib.dump(finalist_a, fa_path)
        joblib.dump(finalist_b, fb_path)
        joblib.dump(finalist_c, fc_path)

        def file_sha256(p: Path) -> str:
            hsh = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(8192):
                    hsh.update(chunk)
            return hsh.hexdigest()

        model_manifest[f"finalist_a_xgboost_{hs}"] = {
            "horizon": f"+{hs}",
            "model_type": "Tuned Standalone XGBoost with Stage-Only Fallback",
            "file": fa_path.name,
            "sha256": file_sha256(fa_path),
            "features_multimodal": FEAT_MULTIMODAL,
            "features_fallback": FEAT_STAGE_DYN,
            "hyperparameters": xgb_p,
        }
        model_manifest[f"finalist_b_raag_{hs}"] = {
            "horizon": f"+{hs}",
            "model_type": "Existing RA-AG (Phase 2 Exact Reproduction)",
            "file": fb_path.name,
            "sha256": file_sha256(fb_path),
            "regime_thresholds": {"theta_rise": 0.010, "theta_fall": -0.010},
            "stable_strategy": "Pure Persistence (dH = 0)",
            "falling_expert": "Ridge (alpha=10.0)",
            "rising_expert": "XGBoost (max_depth=3, lr=0.03, n_est=60, l2=5.0)",
            "fallback_expert": "Stage-Only Ridge",
        }
        model_manifest[f"finalist_c_hybrid_{hs}"] = {
            "horizon": f"+{hs}",
            "model_type": "RA-AG + Tuned XGBoost Rising Expert Hybrid",
            "file": fc_path.name,
            "sha256": file_sha256(fc_path),
            "regime_thresholds": {"theta_rise": 0.010, "theta_fall": -0.010},
            "stable_strategy": "Pure Persistence (dH = 0)",
            "falling_expert": "Ridge (alpha=10.0)",
            "rising_expert": f"Tuned XGBoost (depth={xgb_p['max_depth']}, lr={xgb_p['learning_rate']}, l2={xgb_p['reg_lambda']})",
            "fallback_rising_expert": "Stage-Only Tuned XGBoost",
            "fallback_falling_expert": "Stage-Only Ridge",
        }

        # ---------------------------------------------------------------------
        # 3. Synchronized Cohort Evaluation (Identical Timestamps, Rainfall Available)
        # ---------------------------------------------------------------------
        # Baseline Persistence
        pred_dh_pers = np.zeros(n_sync)
        pred_pers = yc_sync + pred_dh_pers

        # Finalist A: XGBoost
        pred_dh_fa_sync, fb_fa_sync, avail_fa_sync = finalist_a.predict(te_sync, use_fallback=False)
        pred_fa_sync = yc_sync + pred_dh_fa_sync

        # Finalist B: Existing RA-AG
        pred_dh_fb_sync, fb_fb_sync, reg_fb_sync = finalist_b.predict(te_sync)
        pred_fb_sync = yc_sync + pred_dh_fb_sync

        # Finalist C: Hybrid
        pred_dh_fc_sync, fb_fc_sync, reg_fc_sync = finalist_c.predict(te_sync)
        pred_fc_sync = yc_sync + pred_dh_fc_sync

        # Compute metrics on synchronized cohort
        m_pers_sync = compute_audited_metrics(yt_sync, pred_pers, yc_sync, te_sync)
        m_fa_sync = compute_audited_metrics(yt_sync, pred_fa_sync, yc_sync, te_sync, is_fallback=fb_fa_sync)
        m_fb_sync = compute_audited_metrics(yt_sync, pred_fb_sync, yc_sync, te_sync, is_fallback=fb_fb_sync)
        m_fc_sync = compute_audited_metrics(yt_sync, pred_fc_sync, yc_sync, te_sync, is_fallback=fb_fc_sync)

        # ---------------------------------------------------------------------
        # 4. Full Operational Cohort Evaluation (Including Missing-Rainfall Fallback)
        # ---------------------------------------------------------------------
        pred_dh_pers_full = np.zeros(n_full)
        pred_pers_full = yc_full + pred_dh_pers_full

        # Finalist A: Standalone XGBoost without fallback vs with stage-only fallback
        pred_dh_fa_nofb, fb_fa_nofb, avail_fa_nofb = finalist_a.predict(te_full, use_fallback=False)
        pred_dh_fa_full, fb_fa_full, avail_fa_full = finalist_a.predict(te_full, use_fallback=True)
        pred_fa_full = yc_full + pred_dh_fa_full

        # Finalist B: Existing RA-AG full
        pred_dh_fb_full, fb_fb_full, reg_fb_full = finalist_b.predict(te_full)
        pred_fb_full = yc_full + pred_dh_fb_full

        # Finalist C: Hybrid full
        pred_dh_fc_full, fb_fc_full, reg_fc_full = finalist_c.predict(te_full)
        pred_fc_full = yc_full + pred_dh_fc_full

        m_pers_full = compute_audited_metrics(yt_full, pred_pers_full, yc_full, te_full)
        m_fa_full = compute_audited_metrics(yt_full, pred_fa_full, yc_full, te_full, is_fallback=fb_fa_full)
        m_fb_full = compute_audited_metrics(yt_full, pred_fb_full, yc_full, te_full, is_fallback=fb_fb_full)
        m_fc_full = compute_audited_metrics(yt_full, pred_fc_full, yc_full, te_full, is_fallback=fb_fc_full)

        coverage_fa_nofb_pct = round((np.sum(avail_fa_nofb) / n_full) * 100.0, 2)

        # Append to finalist leaderboard
        # Entry for Persistence
        leaderboard_records.append({
            "horizon": f"+{hs}", "evaluation_cohort": "Synchronized (Complete Cases)",
            "model_id": "M01_persistence", "model_name": "Persistence Baseline (dH = 0)",
            "mae_mm": m_pers_sync["mae_mm"], "rmse_mm": m_pers_sync["rmse_mm"], "bias_mm": m_pers_sync["bias_mm"], "r2": m_pers_sync["r2"],
            "imp_vs_pers_pct": 0.0,
            "dir_acc_p2_pct": m_pers_sync["dir_acc_phase2_pct"],
            "dir_acc_ternary_pct": m_pers_sync["dir_acc_ternary_pct"],
            "dir_acc_active_pct": m_pers_sync["dir_acc_active_pct"],
            "peak_underpred_mm": m_pers_sync["peak_underpred_mm"],
            "mae_rapid_rise_mm": m_pers_sync["mae_rapid_rise_mm"],
            "mae_falling_mm": m_pers_sync["mae_falling_mm"],
            "mae_stable_mm": m_pers_sync["mae_stable_mm"],
            "forecast_coverage_pct": 100.0, "fallback_freq_pct": 0.0,
        })
        # Entry for Finalist A: XGBoost
        leaderboard_records.append({
            "horizon": f"+{hs}", "evaluation_cohort": "Synchronized (Complete Cases)",
            "model_id": "Finalist_A_XGBoost", "model_name": f"Tuned XGBoost Baseline (depth={xgb_p['max_depth']}, lr={xgb_p['learning_rate']}, l2={xgb_p['reg_lambda']})",
            "mae_mm": m_fa_sync["mae_mm"], "rmse_mm": m_fa_sync["rmse_mm"], "bias_mm": m_fa_sync["bias_mm"], "r2": m_fa_sync["r2"],
            "imp_vs_pers_pct": m_fa_sync["imp_vs_persistence_pct"],
            "dir_acc_p2_pct": m_fa_sync["dir_acc_phase2_pct"],
            "dir_acc_ternary_pct": m_fa_sync["dir_acc_ternary_pct"],
            "dir_acc_active_pct": m_fa_sync["dir_acc_active_pct"],
            "peak_underpred_mm": m_fa_sync["peak_underpred_mm"],
            "mae_rapid_rise_mm": m_fa_sync["mae_rapid_rise_mm"],
            "mae_falling_mm": m_fa_sync["mae_falling_mm"],
            "mae_stable_mm": m_fa_sync["mae_stable_mm"],
            "forecast_coverage_pct": 100.0, "fallback_freq_pct": 0.0,
        })
        # Entry for Finalist B: RA-AG
        leaderboard_records.append({
            "horizon": f"+{hs}", "evaluation_cohort": "Synchronized (Complete Cases)",
            "model_id": "Finalist_B_RAAG", "model_name": "Existing RA-AG (Phase 2 Exact Reproduction)",
            "mae_mm": m_fb_sync["mae_mm"], "rmse_mm": m_fb_sync["rmse_mm"], "bias_mm": m_fb_sync["bias_mm"], "r2": m_fb_sync["r2"],
            "imp_vs_pers_pct": m_fb_sync["imp_vs_persistence_pct"],
            "dir_acc_p2_pct": m_fb_sync["dir_acc_phase2_pct"],
            "dir_acc_ternary_pct": m_fb_sync["dir_acc_ternary_pct"],
            "dir_acc_active_pct": m_fb_sync["dir_acc_active_pct"],
            "peak_underpred_mm": m_fb_sync["peak_underpred_mm"],
            "mae_rapid_rise_mm": m_fb_sync["mae_rapid_rise_mm"],
            "mae_falling_mm": m_fb_sync["mae_falling_mm"],
            "mae_stable_mm": m_fb_sync["mae_stable_mm"],
            "forecast_coverage_pct": 100.0, "fallback_freq_pct": 0.0,
        })
        # Entry for Finalist C: Hybrid
        leaderboard_records.append({
            "horizon": f"+{hs}", "evaluation_cohort": "Synchronized (Complete Cases)",
            "model_id": "Finalist_C_Hybrid", "model_name": "RA-AG + Tuned XGBoost Rising Expert Hybrid",
            "mae_mm": m_fc_sync["mae_mm"], "rmse_mm": m_fc_sync["rmse_mm"], "bias_mm": m_fc_sync["bias_mm"], "r2": m_fc_sync["r2"],
            "imp_vs_pers_pct": m_fc_sync["imp_vs_persistence_pct"],
            "dir_acc_p2_pct": m_fc_sync["dir_acc_phase2_pct"],
            "dir_acc_ternary_pct": m_fc_sync["dir_acc_ternary_pct"],
            "dir_acc_active_pct": m_fc_sync["dir_acc_active_pct"],
            "peak_underpred_mm": m_fc_sync["peak_underpred_mm"],
            "mae_rapid_rise_mm": m_fc_sync["mae_rapid_rise_mm"],
            "mae_falling_mm": m_fc_sync["mae_falling_mm"],
            "mae_stable_mm": m_fc_sync["mae_stable_mm"],
            "forecast_coverage_pct": 100.0, "fallback_freq_pct": 0.0,
        })

        # Also add Full Operational Cohort records
        leaderboard_records.append({
            "horizon": f"+{hs}", "evaluation_cohort": "Full Operational (Real Missing Rain)",
            "model_id": "Finalist_A_XGBoost_NoFallback", "model_name": "Tuned XGBoost (No Fallback Gate)",
            "mae_mm": np.nan, "rmse_mm": np.nan, "bias_mm": np.nan, "r2": np.nan,
            "imp_vs_pers_pct": np.nan, "dir_acc_p2_pct": np.nan, "dir_acc_ternary_pct": np.nan, "dir_acc_active_pct": np.nan,
            "peak_underpred_mm": np.nan, "mae_rapid_rise_mm": np.nan, "mae_falling_mm": np.nan, "mae_stable_mm": np.nan,
            "forecast_coverage_pct": coverage_fa_nofb_pct, "fallback_freq_pct": 0.0,
        })
        leaderboard_records.append({
            "horizon": f"+{hs}", "evaluation_cohort": "Full Operational (Real Missing Rain)",
            "model_id": "Finalist_A_XGBoost_StageFallback", "model_name": "Tuned XGBoost (with Stage-Only Fallback)",
            "mae_mm": m_fa_full["mae_mm"], "rmse_mm": m_fa_full["rmse_mm"], "bias_mm": m_fa_full["bias_mm"], "r2": m_fa_full["r2"],
            "imp_vs_pers_pct": m_fa_full["imp_vs_persistence_pct"],
            "dir_acc_p2_pct": m_fa_full["dir_acc_phase2_pct"], "dir_acc_ternary_pct": m_fa_full["dir_acc_ternary_pct"], "dir_acc_active_pct": m_fa_full["dir_acc_active_pct"],
            "peak_underpred_mm": m_fa_full["peak_underpred_mm"], "mae_rapid_rise_mm": m_fa_full["mae_rapid_rise_mm"], "mae_falling_mm": m_fa_full["mae_falling_mm"], "mae_stable_mm": m_fa_full["mae_stable_mm"],
            "forecast_coverage_pct": 100.0, "fallback_freq_pct": m_fa_full["fallback_freq_pct"],
        })
        leaderboard_records.append({
            "horizon": f"+{hs}", "evaluation_cohort": "Full Operational (Real Missing Rain)",
            "model_id": "Finalist_B_RAAG", "model_name": "Existing RA-AG (Phase 2 Exact Reproduction)",
            "mae_mm": m_fb_full["mae_mm"], "rmse_mm": m_fb_full["rmse_mm"], "bias_mm": m_fb_full["bias_mm"], "r2": m_fb_full["r2"],
            "imp_vs_pers_pct": m_fb_full["imp_vs_persistence_pct"],
            "dir_acc_p2_pct": m_fb_full["dir_acc_phase2_pct"], "dir_acc_ternary_pct": m_fb_full["dir_acc_ternary_pct"], "dir_acc_active_pct": m_fb_full["dir_acc_active_pct"],
            "peak_underpred_mm": m_fb_full["peak_underpred_mm"], "mae_rapid_rise_mm": m_fb_full["mae_rapid_rise_mm"], "mae_falling_mm": m_fb_full["mae_falling_mm"], "mae_stable_mm": m_fb_full["mae_stable_mm"],
            "forecast_coverage_pct": 100.0, "fallback_freq_pct": m_fb_full["fallback_freq_pct"],
        })
        leaderboard_records.append({
            "horizon": f"+{hs}", "evaluation_cohort": "Full Operational (Real Missing Rain)",
            "model_id": "Finalist_C_Hybrid", "model_name": "RA-AG + Tuned XGBoost Rising Expert Hybrid",
            "mae_mm": m_fc_full["mae_mm"], "rmse_mm": m_fc_full["rmse_mm"], "bias_mm": m_fc_full["bias_mm"], "r2": m_fc_full["r2"],
            "imp_vs_pers_pct": m_fc_full["imp_vs_persistence_pct"],
            "dir_acc_p2_pct": m_fc_full["dir_acc_phase2_pct"], "dir_acc_ternary_pct": m_fc_full["dir_acc_ternary_pct"], "dir_acc_active_pct": m_fc_full["dir_acc_active_pct"],
            "peak_underpred_mm": m_fc_full["peak_underpred_mm"], "mae_rapid_rise_mm": m_fc_full["mae_rapid_rise_mm"], "mae_falling_mm": m_fc_full["mae_falling_mm"], "mae_stable_mm": m_fc_full["mae_stable_mm"],
            "forecast_coverage_pct": 100.0, "fallback_freq_pct": m_fc_full["fallback_freq_pct"],
        })

        # ---------------------------------------------------------------------
        # 5. Regime Stratification Evaluation
        # ---------------------------------------------------------------------
        # Strata:
        # 1. Stable periods: |dH_1h| <= 0.010 m/h
        # 2. Rising periods: dH_1h > 0.010 m/h
        # 3. Falling periods: dH_1h < -0.010 m/h
        # 4. Rapid-rise episodes: dH_1h > 0.020 m/h
        # 5. Rainfall Available vs Rainfall Missing
        dh1_sync = te_sync["dH_v_1h"].values
        reg_masks = {
            "STABLE (|dH_1h| <= 0.01 m/h)": np.abs(dh1_sync) <= 0.010,
            "RISING (dH_1h > 0.01 m/h)": dh1_sync > 0.010,
            "FALLING (dH_1h < -0.01 m/h)": dh1_sync < -0.010,
            "RAPID_RISE (dH_1h > 0.02 m/h)": dh1_sync > 0.020,
        }

        err_pers_sync = np.abs(yt_sync - pred_pers)
        err_fa_sync = np.abs(yt_sync - pred_fa_sync)
        err_fb_sync = np.abs(yt_sync - pred_fb_sync)
        err_fc_sync = np.abs(yt_sync - pred_fc_sync)

        for reg_name, r_mask in reg_masks.items():
            k_obs = int(r_mask.sum())
            if k_obs == 0:
                continue

            mae_p = float(np.mean(err_pers_sync[r_mask])) * 1000.0
            mae_fa = float(np.mean(err_fa_sync[r_mask])) * 1000.0
            mae_fb = float(np.mean(err_fb_sync[r_mask])) * 1000.0
            mae_fc = float(np.mean(err_fc_sync[r_mask])) * 1000.0

            regime_records.append({
                "horizon": f"+{hs}",
                "regime": reg_name,
                "n_observations": k_obs,
                "pct_cohort": round((k_obs / n_sync) * 100.0, 1),
                "persistence_mae_mm": round(mae_p, 3),
                "finalist_a_xgboost_mae_mm": round(mae_fa, 3),
                "finalist_b_raag_mae_mm": round(mae_fb, 3),
                "finalist_c_hybrid_mae_mm": round(mae_fc, 3),
                "hybrid_vs_pers_pct": round(((mae_p - mae_fc) / mae_p) * 100.0, 2) if mae_p > 0 else 0.0,
                "hybrid_vs_xgboost_pct": round(((mae_fa - mae_fc) / mae_fa) * 100.0, 2) if mae_fa > 0 else 0.0,
                "hybrid_vs_raag_pct": round(((mae_fb - mae_fc) / mae_fb) * 100.0, 2) if mae_fb > 0 else 0.0,
            })

        # Telemetry outage breakdown on full operational test cohort
        has_rain_full = te_full[FEAT_RAIN_CONT].notna().all(axis=1).values
        err_fa_full = np.abs(yt_full - pred_fa_full)
        err_fb_full = np.abs(yt_full - pred_fb_full)
        err_fc_full = np.abs(yt_full - pred_fc_full)
        err_pers_full = np.abs(yt_full - pred_pers_full)

        for tele_name, tele_mask in [("Rainfall Available", has_rain_full), ("Rainfall Outage (Fallback Active)", ~has_rain_full)]:
            k_obs = int(tele_mask.sum())
            if k_obs == 0:
                continue
            mae_p = float(np.mean(err_pers_full[tele_mask])) * 1000.0
            mae_fa = float(np.mean(err_fa_full[tele_mask])) * 1000.0
            mae_fb = float(np.mean(err_fb_full[tele_mask])) * 1000.0
            mae_fc = float(np.mean(err_fc_full[tele_mask])) * 1000.0

            regime_records.append({
                "horizon": f"+{hs}",
                "regime": f"TELEMETRY: {tele_name}",
                "n_observations": k_obs,
                "pct_cohort": round((k_obs / n_full) * 100.0, 1),
                "persistence_mae_mm": round(mae_p, 3),
                "finalist_a_xgboost_mae_mm": round(mae_fa, 3),
                "finalist_b_raag_mae_mm": round(mae_fb, 3),
                "finalist_c_hybrid_mae_mm": round(mae_fc, 3),
                "hybrid_vs_pers_pct": round(((mae_p - mae_fc) / mae_p) * 100.0, 2) if mae_p > 0 else 0.0,
                "hybrid_vs_xgboost_pct": round(((mae_fa - mae_fc) / mae_fa) * 100.0, 2) if mae_fa > 0 else 0.0,
                "hybrid_vs_raag_pct": round(((mae_fb - mae_fc) / mae_fb) * 100.0, 2) if mae_fb > 0 else 0.0,
            })

        # ---------------------------------------------------------------------
        # 6. Authentic Event Evaluation
        # ---------------------------------------------------------------------
        for ev_key, ev_info in event_specs.items():
            ev_start = pd.to_datetime(ev_info["start"])
            ev_end = pd.to_datetime(ev_info["end"])

            # Filter event data from full master df
            df_ev = df[(df["dt"] >= ev_start) & (df["dt"] <= ev_end)].copy()
            df_ev = df_ev.dropna(subset=FEAT_STAGE_DYN + [tl, tdh, "stage_vishnupuri"]).reset_index(drop=True)

            if len(df_ev) < 3:
                continue

            yc_ev = df_ev["stage_vishnupuri"].values
            yt_ev = df_ev[tl].values

            # Predictions on event
            pred_dh_fa_ev, fb_fa_ev, avail_fa_ev = finalist_a.predict(df_ev, use_fallback=True)
            pred_fa_ev = yc_ev + pred_dh_fa_ev

            pred_dh_fb_ev, fb_fb_ev, _ = finalist_b.predict(df_ev)
            pred_fb_ev = yc_ev + pred_dh_fb_ev

            pred_dh_fc_ev, fb_fc_ev, _ = finalist_c.predict(df_ev)
            pred_fc_ev = yc_ev + pred_dh_fc_ev

            pred_pers_ev = yc_ev

            # Peak underprediction on this authentic event
            peak_idx = int(np.argmax(yt_ev))
            peak_stage_m = float(yt_ev[peak_idx])

            peak_und_pers = (peak_stage_m - pred_pers_ev[peak_idx]) * 1000.0
            peak_und_fa = (peak_stage_m - pred_fa_ev[peak_idx]) * 1000.0
            peak_und_fb = (peak_stage_m - pred_fb_ev[peak_idx]) * 1000.0
            peak_und_fc = (peak_stage_m - pred_fc_ev[peak_idx]) * 1000.0

            mae_p = float(mean_absolute_error(yt_ev, pred_pers_ev)) * 1000.0
            mae_fa = float(mean_absolute_error(yt_ev, pred_fa_ev)) * 1000.0
            mae_fb = float(mean_absolute_error(yt_ev, pred_fb_ev)) * 1000.0
            mae_fc = float(mean_absolute_error(yt_ev, pred_fc_ev)) * 1000.0

            event_records.append({
                "horizon": f"+{hs}",
                "event_id": ev_key,
                "event_name": ev_info["name"],
                "n_hours": len(df_ev),
                "peak_observed_stage_m": round(peak_stage_m, 3),
                "persistence_mae_mm": round(mae_p, 3),
                "xgboost_mae_mm": round(mae_fa, 3),
                "raag_mae_mm": round(mae_fb, 3),
                "hybrid_mae_mm": round(mae_fc, 3),
                "peak_underpred_pers_mm": round(peak_und_pers, 3),
                "peak_underpred_xgboost_mm": round(peak_und_fa, 3),
                "peak_underpred_raag_mm": round(peak_und_fb, 3),
                "peak_underpred_hybrid_mm": round(peak_und_fc, 3),
                "rainfall_available_hours": int(df_ev[FEAT_RAIN_CONT].notna().all(axis=1).sum()),
                "hybrid_fallback_hours": int(np.sum(fb_fc_ev)),
            })

        # ---------------------------------------------------------------------
        # 7. Paired Statistical Hypothesis Testing (HAC & DM-HLN)
        # ---------------------------------------------------------------------
        # Pair 1: Hybrid vs Persistence
        test_h_vs_p = compute_paired_hac_dm(err_pers_sync, err_fc_sync, h=h)
        # Pair 2: Hybrid vs XGBoost Baseline
        test_h_vs_xgb = compute_paired_hac_dm(err_fa_sync, err_fc_sync, h=h)
        # Pair 3: Hybrid vs RA-AG Existing
        test_h_vs_raag = compute_paired_hac_dm(err_fb_sync, err_fc_sync, h=h)
        # Pair 4: XGBoost Baseline vs Persistence
        test_xgb_vs_p = compute_paired_hac_dm(err_pers_sync, err_fa_sync, h=h)
        # Pair 5: RA-AG Existing vs Persistence
        test_raag_vs_p = compute_paired_hac_dm(err_pers_sync, err_fb_sync, h=h)
        # Pair 6: RA-AG Existing vs XGBoost Baseline
        test_raag_vs_xgb = compute_paired_hac_dm(err_fa_sync, err_fb_sync, h=h)

        pairs_to_log = [
            ("Finalist_C_Hybrid", "M01_Persistence", test_h_vs_p),
            ("Finalist_C_Hybrid", "Finalist_A_XGBoost", test_h_vs_xgb),
            ("Finalist_C_Hybrid", "Finalist_B_RAAG", test_h_vs_raag),
            ("Finalist_A_XGBoost", "M01_Persistence", test_xgb_vs_p),
            ("Finalist_B_RAAG", "M01_Persistence", test_raag_vs_p),
            ("Finalist_B_RAAG", "Finalist_A_XGBoost", test_raag_vs_xgb),
        ]

        for mod_cand, mod_ref, stat_res in pairs_to_log:
            paired_comparison_records.append({
                "horizon": f"+{hs}",
                "model_candidate": mod_cand,
                "model_reference": mod_ref,
                "n_pairs": stat_res["n"],
                "mean_diff_mm": stat_res["mean_diff_mm"],  # Positive means candidate is better (lower error)
                "hac_se_mm": stat_res["hac_se_mm"],
                "hac_t_stat": stat_res["hac_t_stat"],
                "hac_p_value": stat_res["hac_p_value"],
                "dm_stat": stat_res["dm_stat"],
                "dm_hln_p_value": stat_res["dm_hln_p_value"],
                "significant_at_05": bool(stat_res["dm_hln_p_value"] < 0.05),
            })

        # ---------------------------------------------------------------------
        # 8. Store Row-Level Predictions
        # ---------------------------------------------------------------------
        for i in range(n_full):
            dt_i = str(te_full.loc[i, "dt"])
            has_rain_i = bool(te_full.loc[i, FEAT_RAIN_CONT].notna().all())
            all_predictions_rows.append({
                "horizon": f"+{hs}",
                "dt": dt_i,
                "observed_stage_current_m": round(yc_full[i], 3),
                "observed_stage_target_m": round(yt_full[i], 3),
                "observed_change_target_m": round(yt_full[i] - yc_full[i], 3),
                "pred_persistence_m": round(pred_pers_full[i], 3),
                "pred_finalist_a_xgboost_m": round(pred_fa_full[i], 3),
                "pred_finalist_b_raag_m": round(pred_fb_full[i], 3),
                "pred_finalist_c_hybrid_m": round(pred_fc_full[i], 3),
                "regime_finalist_b": reg_fb_full[i],
                "regime_finalist_c": reg_fc_full[i],
                "rainfall_available": has_rain_i,
                "fallback_finalist_a": bool(fb_fa_full[i]),
                "fallback_finalist_b": bool(fb_fb_full[i]),
                "fallback_finalist_c": bool(fb_fc_full[i]),
            })

    # =========================================================================
    # WRITE ALL ARTIFACTS AND CSV DELIVERABLES
    # =========================================================================
    # 1. Finalist Leaderboard CSV
    df_leaderboard = pd.DataFrame(leaderboard_records)
    df_leaderboard.to_csv(ARTIFACT_DIR / "finalist_leaderboard.csv", index=False)
    log.info("Saved %s", ARTIFACT_DIR / "finalist_leaderboard.csv")

    # 2. Regime Metrics CSV
    df_regime = pd.DataFrame(regime_records)
    df_regime.to_csv(ARTIFACT_DIR / "regime_metrics.csv", index=False)
    log.info("Saved %s", ARTIFACT_DIR / "regime_metrics.csv")

    # 3. Event Metrics CSV
    df_events = pd.DataFrame(event_records)
    df_events.to_csv(ARTIFACT_DIR / "event_metrics.csv", index=False)
    log.info("Saved %s", ARTIFACT_DIR / "event_metrics.csv")

    # 4. Paired Comparisons CSV
    df_paired = pd.DataFrame(paired_comparison_records)
    df_paired.to_csv(ARTIFACT_DIR / "paired_comparisons.csv", index=False)
    log.info("Saved %s", ARTIFACT_DIR / "paired_comparisons.csv")

    # 5. Finalist Predictions CSV
    df_preds = pd.DataFrame(all_predictions_rows)
    df_preds.to_csv(ARTIFACT_DIR / "finalist_predictions.csv", index=False)
    log.info("Saved %s (Total prediction rows: %d)", ARTIFACT_DIR / "finalist_predictions.csv", len(df_preds))

    # 6. Model Manifest JSON
    with open(ARTIFACT_DIR / "model_manifest.json", "w") as f:
        json.dump(model_manifest, f, indent=2)
    log.info("Saved %s", ARTIFACT_DIR / "model_manifest.json")

    log.info("\n" + "=" * 80)
    log.info("PHASE 3 EXECUTION COMPLETE -- ALL DATA STRUCTURES GENERATED")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
