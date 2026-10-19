"""
scripts/run_audit_v4_analysis.py
================================
Controlled rainfall ablation, statistical family reconciliation,
and final paired comparison analysis for Phase 1 Audit v4.

Outputs generated in: artifacts/phase1_integrity_audit_v4/
"""
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.phase1_ml_pipeline import (
    load_features,
    get_splits,
    get_catalogue,
    HORIZONS,
    SEED,
    FEAT_STAGE_DYN,
    FEAT_RAIN_CONT,
    FEAT_RAIN_LAG,
    FEAT_MULTIMODAL,
)

REPAIR_DIR = REPO_ROOT / "artifacts" / "phase1_integrity_repair_v2"
AUDIT_V3_DIR = REPO_ROOT / "artifacts" / "phase1_integrity_audit_v3"
AUDIT_V4_DIR = REPO_ROOT / "artifacts" / "phase1_integrity_audit_v4"
AUDIT_V4_DIR.mkdir(parents=True, exist_ok=True)


def diebold_mariano_hln(d: np.ndarray, h: int) -> Tuple[float, float, float, float]:
    n = len(d)
    mean_d = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=0))
    bw = max(1, h)
    gamma_sum = 0.0
    for lag in range(1, bw):
        w = 1.0 - (lag / bw)
        cov = float(np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d)))
        gamma_sum += 2.0 * w * cov
    var_d = (gamma0 + gamma_sum) / n
    se_hac = math.sqrt(max(1e-12, var_d))
    dm_stat = mean_d / se_hac
    hln_factor = math.sqrt(max(1e-6, (n + 1 - 2 * h + (h * (h - 1)) / n) / n))
    dm_hln = dm_stat * hln_factor
    p_two_sided = float(2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    return mean_d, se_hac, dm_hln, p_two_sided


def moving_block_bootstrap(
    d: np.ndarray, B: int = 2000, block_len: int = 6, seed: int = SEED
) -> Tuple[float, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    mean_d = float(np.mean(d))
    k = int(np.ceil(n / block_len))
    max_start = n - block_len + 1
    blocks = [d[i : i + block_len] for i in range(max_start)]

    boot_means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, max_start, size=k)
        sampled = np.concatenate([blocks[i] for i in idx])[:n]
        boot_means[b] = np.mean(sampled)

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    boot_se = float(np.std(boot_means, ddof=1))

    # Empirical plus-one p-value
    boot_centered = boot_means - mean_d
    exceedances = int(np.sum(np.abs(boot_centered) >= np.abs(mean_d)))
    p_plus_one = (exceedances + 1) / (B + 1)

    # Studentized asymptotic p-value
    z_score = mean_d / max(1e-12, boot_se)
    p_asymp = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))

    return ci_lo, ci_hi, boot_se, p_plus_one, z_score, p_asymp


def main():
    print("=" * 70)
    print("PHASE 1 AUDIT V4 -- CONTROLLED ABLATION & STATISTICAL RECONCILIATION")
    print("=" * 70)

    df = load_features()
    sp = get_splits(df)
    tr, te = sp["train"], sp["test"]
    sync_feats = FEAT_STAGE_DYN + FEAT_RAIN_LAG + ["api_7d_mm"]

    # -------------------------------------------------------------------------
    # PART A: CONTROLLED RAINFALL ABLATION
    # -------------------------------------------------------------------------
    ablation_rows = []

    for h in HORIZONS:
        hs = f"{h}h"
        tl = f"target_v_{hs}"
        tdh = f"target_v_dH_{hs}"

        # 1. Eligible training sets
        tr_all = tr.dropna(subset=FEAT_STAGE_DYN + [tdh, "stage_vishnupuri"]).copy()
        tr_match = tr.dropna(subset=FEAT_STAGE_DYN + FEAT_RAIN_CONT + [tdh, "stage_vishnupuri"]).copy()

        # Evaluation test cohorts
        # Both complete-case on rainfall (te_c) and synchronized common cohort (te_sync)
        te_sync = te.dropna(subset=sync_feats + [tl, tdh, "stage_vishnupuri"]).copy().reset_index(drop=True)
        N_eval = len(te_sync)

        yt = te_sync[tl].values
        yc = te_sync["stage_vishnupuri"].values

        # 1. M04-all: Stage-only on all eligible rows
        p_m04_all = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_m04_all.fit(tr_all[FEAT_STAGE_DYN], tr_all[tdh])
        pred_m04_all = yc + p_m04_all.predict(te_sync[FEAT_STAGE_DYN])
        e_m04_all = np.abs(yt - pred_m04_all)
        mae_m04_all = float(np.mean(e_m04_all))

        # 2. M04-matched: Stage-only on matched rows
        p_m04_match = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_m04_match.fit(tr_match[FEAT_STAGE_DYN], tr_match[tdh])
        pred_m04_match = yc + p_m04_match.predict(te_sync[FEAT_STAGE_DYN])
        e_m04_match = np.abs(yt - pred_m04_match)
        mae_m04_match = float(np.mean(e_m04_match))

        # 3. M05-all / M05-matched: Stage+Rain on matched rows
        p_m05_match = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=10.0))])
        p_m05_match.fit(tr_match[FEAT_STAGE_DYN + FEAT_RAIN_CONT], tr_match[tdh])
        pred_m05_match = yc + p_m05_match.predict(te_sync[FEAT_STAGE_DYN + FEAT_RAIN_CONT])
        e_m05_match = np.abs(yt - pred_m05_match)
        mae_m05_match = float(np.mean(e_m05_match))

        # Persistence on same test cohort
        e_pers = np.abs(yt - yc)
        mae_pers = float(np.mean(e_pers))

        # Decompositions
        # Total apparent difference: M04-all vs M05-match
        total_diff_m = mae_m04_all - mae_m05_match
        total_imp_pct = (mae_m04_all - mae_m05_match) / mae_m04_all * 100.0

        # Cohort effect: M04-all vs M04-matched
        cohort_diff_m = mae_m04_all - mae_m04_match
        cohort_pct = (mae_m04_all - mae_m04_match) / mae_m04_all * 100.0

        # Controlled rainfall effect: M04-matched vs M05-matched
        rain_diff_m = mae_m04_match - mae_m05_match
        rain_imp_pct = (mae_m04_match - mae_m05_match) / mae_m04_match * 100.0

        # Statistical tests on controlled rainfall effect (M04-matched vs M05-matched)
        d_rain = e_m04_match - e_m05_match
        ci_lo, ci_hi, se_boot, p_boot, z_boot, p_asymp = moving_block_bootstrap(d_rain, B=2000, block_len=6)
        dm_mean, dm_se, dm_t, dm_p = diebold_mariano_hln(d_rain, h=h)

        ablation_rows.append({
            "horizon": f"+{hs}",
            "n_train_m04_all": len(tr_all),
            "n_train_matched": len(tr_match),
            "n_eval_synchronized": N_eval,
            "mae_persistence_mm": mae_pers * 1000.0,
            "mae_m04_all_mm": mae_m04_all * 1000.0,
            "mae_m04_matched_mm": mae_m04_match * 1000.0,
            "mae_m05_matched_mm": mae_m05_match * 1000.0,
            "total_apparent_reduction_mm": total_diff_m * 1000.0,
            "total_apparent_improvement_pct": total_imp_pct,
            "training_cohort_effect_mm": cohort_diff_m * 1000.0,
            "training_cohort_effect_pct_of_total": (cohort_diff_m / total_diff_m) * 100.0,
            "controlled_rainfall_reduction_mm": rain_diff_m * 1000.0,
            "controlled_rainfall_improvement_pct": rain_imp_pct,
            "controlled_rainfall_pct_of_total": (rain_diff_m / total_diff_m) * 100.0,
            "rainfall_boot_ci_95_low_mm": ci_lo * 1000.0,
            "rainfall_boot_ci_95_high_mm": ci_hi * 1000.0,
            "rainfall_boot_p_plus_one": p_boot,
            "rainfall_dm_hln_stat": dm_t,
            "rainfall_dm_p_value": dm_p,
            "statistically_significant_05": dm_p < 0.05,
        })

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(AUDIT_V4_DIR / "controlled_ablation.csv", index=False)
    print("Saved controlled_ablation.csv successfully.")

    # -------------------------------------------------------------------------
    # PART B & C: RECONCILE STATISTICAL TEST FAMILY & METHODS
    # -------------------------------------------------------------------------
    # The confirmatory family consists of 12 distinct scientific hypotheses:
    # 4 comparison models (Persistence, M04, Extra Trees, XGBoost) x 3 horizons.
    # Sensitivity analyses (L=3, 12, 24) test robustness to block length choice.
    
    # Load predictions from v3
    df_paired_preds = pd.read_csv(AUDIT_V3_DIR / "paired_predictions.csv")

    confirmatory_rows = []
    
    comparisons = [
        ("Model D vs Persistence", "error_M01_persistence_m", "error_M05_model_d_m", "M05", "M01"),
        ("Model D vs Stage-Change Ridge (M04)", "error_M04_stage_ridge_m", "error_M05_model_d_m", "M05", "M04"),
        ("Model D vs Extra Trees (M09)", "error_M09_extra_trees_m", "error_M05_model_d_m", "M05", "M09"),
        ("Model D vs XGBoost (M12)", "error_M12_xgboost_m", "error_M05_model_d_m", "M05", "M12"),
    ]

    for h in HORIZONS:
        hs = f"{h}h"
        sub = df_paired_preds[df_paired_preds["horizon"] == f"+{h}h"].copy().reset_index(drop=True)
        N_c = len(sub)

        for desc, col_base, col_cand, m_cand, m_base in comparisons:
            e_base = sub[col_base].values
            e_cand = sub[col_cand].values
            
            # Loss differential: d_t = e_base - e_cand (positive => candidate is better)
            d = e_base - e_cand
            mean_d = float(np.mean(d))
            diff_mm = mean_d * 1000.0
            base_mae = float(np.mean(e_base))
            cand_mae = float(np.mean(e_cand))
            imp_pct = (base_mae - cand_mae) / base_mae * 100.0

            # 1. Primary Confirmatory Test: Moving Block Bootstrap with L=6h
            ci_lo, ci_hi, se_boot, p_boot, z_boot, p_asymp = moving_block_bootstrap(d, B=2000, block_len=6)

            # 2. Primary Confirmatory Test: Diebold-Mariano test with HLN modification
            dm_mean, dm_se, dm_t, dm_p = diebold_mariano_hln(d, h=h)

            confirmatory_rows.append({
                "hypothesis_id": f"H_{h}_{m_base}",
                "horizon": f"+{hs}",
                "comparison": desc,
                "model_candidate": m_cand,
                "model_baseline": m_base,
                "sample_count_n": N_c,
                "candidate_mae_mm": cand_mae * 1000.0,
                "baseline_mae_mm": base_mae * 1000.0,
                "mean_loss_diff_mm": diff_mm,
                "improvement_pct": imp_pct,
                "primary_block_length_h": 6,
                "boot_ci_95_low_mm": ci_lo * 1000.0,
                "boot_ci_95_high_mm": ci_hi * 1000.0,
                "boot_standard_error_mm": se_boot * 1000.0,
                "boot_empirical_p_plus_one": p_boot,
                "boot_studentized_z": z_boot,
                "boot_asymptotic_p": p_asymp,
                "dm_hln_t_stat": dm_t,
                "dm_degrees_of_freedom": N_c - 1,
                "dm_p_value": dm_p,
            })

    df_conf = pd.DataFrame(confirmatory_rows)
    m = len(df_conf)

    # Multiplicity adjustments on confirmatory family of 12 tests
    # Bonferroni
    df_conf["bonferroni_p_dm"] = np.minimum(1.0, df_conf["dm_p_value"] * m)
    df_conf["bonferroni_p_boot"] = np.minimum(1.0, df_conf["boot_empirical_p_plus_one"] * m)

    # Benjamini-Hochberg FDR
    for p_col, out_col in [("dm_p_value", "bh_fdr_p_dm"), ("boot_empirical_p_plus_one", "bh_fdr_p_boot")]:
        p_vals = df_conf[p_col].values
        sorted_idx = np.argsort(p_vals)
        bh_p = np.empty(m)
        for rank, idx in enumerate(sorted_idx, 1):
            bh_p[idx] = min(1.0, p_vals[idx] * m / rank)
        for i in range(m - 2, -1, -1):
            curr_idx = sorted_idx[i]
            next_idx = sorted_idx[i + 1]
            bh_p[curr_idx] = min(bh_p[curr_idx], bh_p[next_idx])
        df_conf[out_col] = np.round(bh_p, 5)

    df_conf["confirmatory_verdict"] = np.where(
        df_conf["bh_fdr_p_dm"] < 0.05,
        np.where(df_conf["mean_loss_diff_mm"] > 0, "SIGNIFICANT_CANDIDATE_SUPERIOR", "SIGNIFICANT_BASELINE_SUPERIOR"),
        "NOT_STATISTICALLY_SIGNIFICANT"
    )

    df_conf.to_csv(AUDIT_V4_DIR / "final_paired_comparisons.csv", index=False)
    print("Saved final_paired_comparisons.csv successfully.")

    # Statistical family reconciliation table
    recon_rows = []
    # Include 12 primary confirmatory tests + 36 sensitivity checks
    df_boot_all = pd.read_csv(AUDIT_V3_DIR / "bootstrap_reanalysis.csv")
    
    for _, r in df_boot_all.iterrows():
        is_confirmatory = (r["block_length_h"] == 6)
        recon_rows.append({
            "horizon": r["horizon"],
            "comparison": r["comparison"],
            "role": "PRIMARY_CONFIRMATORY_HYPOTHESIS" if is_confirmatory else f"SENSITIVITY_CHECK_L_{r['block_length_h']}H",
            "block_length_h": r["block_length_h"],
            "n_paired": r["n_paired"],
            "diff_mean_mm": r["diff_mean_mm"],
            "boot_ci_95_low_mm": r["boot_ci_95_low_mm"],
            "boot_ci_95_high_mm": r["boot_ci_95_high_mm"],
            "raw_empirical_p_plus_one": r["empirical_p_value_plus_one"],
            "raw_studentized_p": r["asymptotic_normal_p_value"],
            "multiplicity_family": "CONFIRMATORY_FAMILY_M12" if is_confirmatory else "NON_INDEPENDENT_SENSITIVITY",
        })

    df_recon = pd.DataFrame(recon_rows)
    df_recon.to_csv(AUDIT_V4_DIR / "statistical_family_reconciliation.csv", index=False)
    print("Saved statistical_family_reconciliation.csv successfully.")

    print("\nAudit v4 computational analysis complete.")


if __name__ == "__main__":
    main()
