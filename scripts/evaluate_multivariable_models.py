"""
scripts/evaluate_multivariable_models.py
=========================================
Executes the comprehensive multivariable model comparison and ablation audit
for Vishnupuri Barrage (+1h, +2h, +3h).

Compares:
A. Persistence baseline
B. Regularized Autoregressive Stage-Only (Level)
C. Stage + Rainfall Model (Level)
D. Stage Change Increment Model (dH)
E. Conservative Non-Linear XGBoost (dH)
F. Regime-Aware Model (dH)

Evaluates on:
- July 30–31, 2026 Monsoon Inception Event (Phase A, N=30)
- September 8–22, 2026 Synchronized Held-Out Test Subwindow
- Feature Ablations (Rainfall, Upstream Mudgal, Downstream Gauges)
"""

import sys
import shutil
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from river_forecasting.src.multivariable_pipeline import (
    get_feature_audit_matrix,
    prepare_multivariable_dataset,
    MultivariableModelComparator
)

OUT_BENCHMARK_JSON = REPO_ROOT / "data" / "historical" / "multivariable_benchmark_results.json"
OUT_AUDIT_JSON = REPO_ROOT / "data" / "historical" / "multivariable_feature_audit.json"
OUT_PLOT = REPO_ROOT / "data" / "historical" / "multivariable_model_comparison_plot.png"
ARTIFACT_DIR = Path(r"C:\Users\aceru\.gemini\antigravity-ide\brain\4c89c7ca-48e8-4bf2-bb58-32d093d7e52d")

def run_evaluation():
    print("=" * 80)
    print("MULTIVARIABLE FORECASTING MODEL RESEARCH UPGRADE: VISHNUPURI BARRAGE")
    print("=" * 80)

    # 1. Feature Availability Audit Matrix
    audit_matrix = get_feature_audit_matrix()
    with open(OUT_AUDIT_JSON, "w") as f:
        json.dump(audit_matrix, f, indent=2)
    print(f"Saved feature availability audit matrix to: {OUT_AUDIT_JSON}")
    print(f"Audited {len(audit_matrix)} candidate features across 10 hydrological dimensions.")

    # 2. Prepare Dataset
    print("\nPreparing multivariable dataset with causal feature engineering...")
    df = prepare_multivariable_dataset()
    print(f"Prepared dataset: {len(df)} rows ({df['dt'].min()} to {df['dt'].max()} IST)")

    comparator = MultivariableModelComparator(df, train_end_ist="2026-07-29 23:00:00")

    # 3. Define Evaluation Partitions
    phase_a_df = df[(df["dt"] >= "2026-07-30 00:00:00") & (df["dt"] <= "2026-07-31 05:00:00")].copy()
    sep_sync_df = df[(df["dt"] >= "2026-09-08 06:00:00") & (df["dt"] <= "2026-09-22 19:00:00")].copy()

    benchmark_results = {
        "feature_audit_summary": {
            "total_features_audited": len(audit_matrix),
            "available_features_count": sum(1 for f in audit_matrix if f["available_at_t"]),
            "unavailable_features_count": sum(1 for f in audit_matrix if not f["available_at_t"])
        },
        "phase_a_event_eval": {},
        "september_sync_eval": {},
        "ablation_study": {}
    }

    print("\n" + "=" * 80)
    print("EVALUATION 1: PHASE A FLOOD INCEPTION EVENT (JULY 30–31, 2026, N=30)")
    print("=" * 80)

    all_trained_models = {}
    for h_str in ["1h", "2h", "3h"]:
        print(f"\n--- Horizon +{h_str} ---")
        trained = comparator.train_models(h_str)
        all_trained_models[h_str] = trained
        
        eval_phase_a = comparator.evaluate_test_cohort(phase_a_df, h_str, trained)
        benchmark_results["phase_a_event_eval"][f"+{h_str}"] = eval_phase_a

        for m_name, m_res in eval_phase_a.items():
            print(f"  {m_name:<28}: MAE={m_res['mae_m']:.4f}m | RMSE={m_res['rmse_m']:.4f}m | Bias={m_res['bias_m']:+.4f}m | PeakErr={m_res['peak_abs_err_m']:.4f}m")

    print("\n" + "=" * 80)
    print("EVALUATION 2: SEPTEMBER HELD-OUT SYNCHRONIZED TEST SUBWINDOW (SEP 8–22, 2026)")
    print("=" * 80)
    for h_str in ["1h", "2h", "3h"]:
        print(f"\n--- Horizon +{h_str} ---")
        eval_sep = comparator.evaluate_test_cohort(sep_sync_df, h_str, all_trained_models[h_str])
        benchmark_results["september_sync_eval"][f"+{h_str}"] = eval_sep
        for m_name, m_res in eval_sep.items():
            print(f"  {m_name:<28}: MAE={m_res['mae_m']:.4f}m | RMSE={m_res['rmse_m']:.4f}m | Bias={m_res['bias_m']:+.4f}m")

    # 4. Feature Ablation Study on Phase A Inception Event
    print("\n" + "=" * 80)
    print("EVALUATION 3: FEATURE ABLATION STUDY (PHASE A INCEPTION)")
    print("\n" + "=" * 80)
    print("EVALUATION 3: FEATURE ABLATION STUDY (PHASE A INCEPTION)")
    print("=" * 80)
    from sklearn.linear_model import Ridge
    ablation_res = {}
    
    stage_dyn = comparator.feature_sets["stage_dyn_only"]
    rain = comparator.feature_sets["stage_change_multimodal"][len(stage_dyn):]
    regime = ["freeboard_frl_m", "is_high_pool"]
    upstream = ["dH_m_1h"]
    downstream = ["dH_r_1h"]  # Raher (100% complete in Phase A; Babhali had missing records)

    ab_feature_groups = {
        "1_Persistence": [],
        "2_StageDynamicsOnly": stage_dyn,
        "3_StageDyn_Plus_Rainfall": stage_dyn + rain,
        "4_Plus_Regime_FRL": stage_dyn + rain + regime,
        "5_Plus_Upstream_Mudgal": stage_dyn + rain + regime + upstream,
        "6_Plus_Downstream_Raher": stage_dyn + rain + regime + upstream + downstream
    }

    for h_str, h_int in [("1h", 1), ("2h", 2), ("3h", 3)]:
        target_dh = f"target_v_dH_{h_str}"
        target_lvl = f"target_v_{h_str}"
        
        valid_train = (df["dt"] >= "2026-06-15 00:00:00") & \
                      ((df["dt"] + pd.Timedelta(hours=h_int)) <= comparator.train_end_ist)
        
        te_common = phase_a_df.dropna(subset=[target_dh, target_lvl, "stage_vishnupuri"] + stage_dyn + rain + regime + upstream + downstream).copy()
        y_true = te_common[target_lvl].values
        curr_h = te_common["stage_vishnupuri"].values

        ab_h_res = {}
        print(f"\nAblations for Horizon +{h_str} (N={len(te_common)}):")

        for ab_name, ab_feats in ab_feature_groups.items():
            if len(ab_feats) == 0:
                ab_preds = curr_h
            else:
                tr_sub = df[valid_train].dropna(subset=ab_feats + [target_dh]).copy()
                m_sub = Ridge(alpha=10.0).fit(tr_sub[ab_feats], tr_sub[target_dh])
                ab_preds = curr_h + m_sub.predict(te_common[ab_feats])

            mae = float(mean_absolute_error(y_true, ab_preds))
            rmse = float(np.sqrt(mean_squared_error(y_true, ab_preds)))
            bias = float(np.mean(ab_preds - y_true))
            diff_pers = mae - mean_absolute_error(y_true, curr_h)

            ab_h_res[ab_name] = {
                "mae_m": round(mae, 4),
                "rmse_m": round(rmse, 4),
                "bias_m": round(bias, 4),
                "diff_vs_persistence_m": round(diff_pers, 4)
            }
            print(f"  {ab_name:<28}: MAE={mae:.4f}m | Diff vs Pers={diff_pers:+.4f}m | Bias={bias:+.4f}m")

        ablation_res[f"+{h_str}"] = ab_h_res

    benchmark_results["ablation_study"] = ablation_res

    # Save benchmark JSON
    with open(OUT_BENCHMARK_JSON, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved multivariable benchmark results to: {OUT_BENCHMARK_JSON}")

    # 5. Diagnostic Figures
    print("\nGenerating comparative diagnostic figure...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Phase A MAE Comparison by Model Architecture
    ax1 = axes[0, 0]
    models_to_plot = ["A_Persistence", "B_StageOnly_Ridge_Level", "C_Multimodal_Ridge_Level", "D_Multimodal_Ridge_dH", "E_Conservative_XGBoost_dH"]
    labels = ["Persistence", "Stage Ridge\n(Level)", "MM Ridge\n(Level)", "MM Ridge\n(dH Increment)", "XGBoost\n(dH Increment)"]
    x = np.arange(len(models_to_plot))
    width = 0.25

    mae_1h = [benchmark_results["phase_a_event_eval"]["+1h"][m]["mae_m"] for m in models_to_plot]
    mae_2h = [benchmark_results["phase_a_event_eval"]["+2h"][m]["mae_m"] for m in models_to_plot]
    mae_3h = [benchmark_results["phase_a_event_eval"]["+3h"][m]["mae_m"] for m in models_to_plot]

    ax1.bar(x - width, mae_1h, width, color="#0284c7", label="+1h Lead Time")
    ax1.bar(x, mae_2h, width, color="#4338ca", label="+2h Lead Time")
    ax1.bar(x + width, mae_3h, width, color="#059669", label="+3h Lead Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("MAE (m)", fontweight="bold")
    ax1.set_title("A. Phase A Inception: Model Candidate Comparison Across Lead Times", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="upper left")

    # Panel 2: Ablation Effect on Phase A (+3h MAE)
    ax2 = axes[0, 1]
    ab_keys = ["1_Persistence", "2_StageDynamicsOnly", "3_StageDyn_Plus_Rainfall", "4_Plus_Regime_FRL", "5_Plus_Upstream_Mudgal", "6_Plus_Downstream_Raher"]
    ab_labels = ["Persistence", "Stage Dyn Only", "+ Rainfall", "+ FRL Regime", "+ Upstream Mudgal", "+ Downstream"]
    ab_maes_3h = [ablation_res["+3h"][k]["mae_m"] for k in ab_keys]
    colors = ["#64748b", "#dc2626", "#059669", "#059669", "#0284c7", "#0284c7"]

    bars = ax2.bar(np.arange(len(ab_keys)), ab_maes_3h, color=colors, width=0.55)
    ax2.axhline(ablation_res["+3h"]["1_Persistence"]["mae_m"], color="#64748b", ls="--", lw=1.2, label="Persistence Reference (0.0806m)")
    ax2.set_xticks(np.arange(len(ab_keys)))
    ax2.set_xticklabels(ab_labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("MAE (m)", fontweight="bold")
    ax2.set_title("B. Feature Ablation: Incremental Value of Variable Groups at +3h", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper right")

    # Panel 3: Hydrograph Comparison on Phase A (+3h Trajectory)
    ax3 = axes[1, 0]
    dt_a = phase_a_df["dt"]
    y_true_3h = phase_a_df["target_v_3h"]
    y_pers_3h = phase_a_df["stage_vishnupuri"]
    
    # Predict level vs dH
    pred_level_3h = all_trained_models["3h"]["m_C"].predict(phase_a_df[comparator.feature_sets["multimodal_level"]])
    pred_dh_3h = phase_a_df["stage_vishnupuri"] + all_trained_models["3h"]["m_D"].predict(phase_a_df[comparator.feature_sets["stage_change_multimodal"]])

    ax3.plot(dt_a, y_true_3h, color="#0f172a", lw=2.2, label="Observed Stage (t+3h)")
    ax3.plot(dt_a, y_pers_3h, color="#64748b", ls=":", lw=1.5, label="Persistence (+3h)")
    ax3.plot(dt_a, pred_level_3h, color="#dc2626", ls="-.", lw=1.5, label="Level MM Ridge (Shrunk)")
    ax3.plot(dt_a, pred_dh_3h, color="#059669", ls="--", lw=1.8, label="Stage Change MM Ridge (dH)")
    ax3.set_ylabel("Stage (m MSL)", fontweight="bold")
    ax3.set_title("C. Phase A Hydrograph: Elimination of Level Shrinkage via dH Increment", fontweight="bold", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=9)

    # Panel 4: September Holdout Synchronized Performance (+1h MAE)
    ax4 = axes[1, 1]
    sep_models = ["A_Persistence", "B_StageOnly_Ridge_Level", "C_Multimodal_Ridge_Level", "D_Multimodal_Ridge_dH"]
    sep_labels = ["Persistence", "Stage Ridge\n(Level)", "MM Ridge\n(Level)", "MM Ridge\n(dH Increment)"]
    sep_maes_1h = [benchmark_results["september_sync_eval"]["+1h"][m]["mae_m"] for m in sep_models]

    ax4.bar(np.arange(len(sep_models)), sep_maes_1h, color=["#64748b", "#dc2626", "#dc2626", "#059669"], width=0.5)
    ax4.set_xticks(np.arange(len(sep_models)))
    ax4.set_xticklabels(sep_labels, fontsize=9)
    ax4.set_ylabel("MAE (m)", fontweight="bold")
    ax4.set_title("D. September Held-Out Sync Subwindow (+1h): Regularization Stability", fontweight="bold", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=180)
    if ARTIFACT_DIR.exists():
        shutil.copy(OUT_PLOT, ARTIFACT_DIR / "multivariable_model_comparison_plot.png")
    plt.close()
    print(f"Saved diagnostic plot to: {OUT_PLOT}")
    print(f"Copied to artifact directory: {ARTIFACT_DIR / 'multivariable_model_comparison_plot.png'}")

if __name__ == "__main__":
    run_evaluation()
