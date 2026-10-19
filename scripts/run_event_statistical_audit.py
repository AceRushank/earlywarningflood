"""
scripts/run_event_statistical_audit.py
=======================================
Executes the independent statistical audit of the July 2026 event predictions
and outputs verified metrics for the report and regression tests.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from river_forecasting.src.event_statistical_audit import (
    compute_loss_differential,
    compute_diebold_mariano,
    compute_moving_block_bootstrap,
    compute_effective_sample_size,
    compute_multiple_testing_adjustments
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_CSV = REPO_ROOT / "data" / "historical" / "event_july2026_predictions.csv"
OUT_AUDIT_JSON = REPO_ROOT / "data" / "historical" / "event_july2026_statistical_audit.json"

def run_audit():
    print("=" * 80)
    print("INDEPENDENT STATISTICAL AUDIT: JULY 2026 EVENT (PHASE A, N=30)")
    print("=" * 80)

    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Loaded {len(df)} predictions from {PREDICTIONS_CSV}")

    audit_results = {
        "dataset": {
            "path": str(PREDICTIONS_CSV),
            "samples": len(df),
            "event_window": "2026-07-30 00:00 to 2026-07-31 05:00 IST"
        },
        "horizons": {},
        "multiple_testing": {}
    }

    p_vals_rect = {}
    p_vals_bart = {}

    for h_str, h in [("1h", 1), ("2h", 2), ("3h", 3)]:
        y_true = df[f"y_true_{h_str}"].values
        y_st = df[f"pred_stage_ridge_{h_str}"].values
        y_mm = df[f"pred_mm_ridge_{h_str}"].values
        y_p = df[f"pred_pers_{h_str}"].values

        # 1. Loss differentials
        diff_mm_vs_st = compute_loss_differential(y_true, y_st, y_mm)
        diff_pers_vs_mm = compute_loss_differential(y_true, y_mm, y_p)

        d = diff_mm_vs_st["d_series"]
        delta = diff_mm_vs_st["delta_series"]

        # 2. Autocorrelation & Effective Sample Size
        ess = compute_effective_sample_size(d)

        # 3. Diebold-Mariano Tests
        # (a) Classic DM rectangular (lag = h - 1)
        dm_rect = compute_diebold_mariano(d, h=h, kernel="rectangular", use_hln=True)
        # (b) Bartlett / Newey-West (lag = 3)
        dm_bart = compute_diebold_mariano(d, h=h, kernel="bartlett", bandwidth=3, use_hln=True)

        p_vals_rect[f"+{h_str}"] = dm_rect["p_two_sided"]
        p_vals_bart[f"+{h_str}"] = dm_bart["p_two_sided"]

        # 4. Moving-Block Bootstrap across multiple block lengths
        boot_res = compute_moving_block_bootstrap(
            delta, block_lengths=[1, 2, 3, 4, 5, 6], n_boot=10000, seed=42
        )

        h_data = {
            "sample_size": len(df),
            "mae_persistence": round(diff_pers_vs_mm["mae_candidate"], 4),
            "mae_stage_ridge": round(diff_mm_vs_st["mae_baseline"], 4),
            "mae_multimodal_ridge": round(diff_mm_vs_st["mae_candidate"], 4),
            "delta_mae_m": round(diff_mm_vs_st["delta_mae"], 5),
            "relative_improvement_pct": round(diff_mm_vs_st["pct_gain"], 2),
            "multimodal_wins": diff_mm_vs_st["candidate_wins"],
            "multimodal_win_pct": round(diff_mm_vs_st["candidate_win_pct"], 1),
            "stage_ridge_wins": diff_mm_vs_st["baseline_wins"],
            "stage_ridge_win_pct": round(diff_mm_vs_st["baseline_win_pct"], 1),
            "persistence_beats_mm_count": diff_pers_vs_mm["candidate_wins"],
            "persistence_beats_mm_pct": round(diff_pers_vs_mm["candidate_win_pct"], 1),
            "autocorrelation_rho_1": round(ess["rho_1"], 4),
            "effective_sample_size": ess["n_eff"],
            "diebold_mariano_rectangular_hln": {
                "bandwidth_lags": dm_rect["bandwidth"],
                "dm_stat_raw": round(dm_rect["dm_stat_raw"], 4),
                "hln_factor": round(dm_rect["hln_factor"], 4),
                "dm_stat_hln": round(dm_rect["dm_stat_hln"], 4),
                "degrees_of_freedom": dm_rect["df"],
                "p_two_sided": round(dm_rect["p_two_sided"], 5),
                "p_one_sided": round(dm_rect["p_one_sided"], 5)
            },
            "diebold_mariano_bartlett_hln": {
                "bandwidth_lags": dm_bart["bandwidth"],
                "dm_stat_raw": round(dm_bart["dm_stat_raw"], 4),
                "hln_factor": round(dm_bart["hln_factor"], 4),
                "dm_stat_hln": round(dm_bart["dm_stat_hln"], 4),
                "degrees_of_freedom": dm_bart["df"],
                "p_two_sided": round(dm_bart["p_two_sided"], 5),
                "p_one_sided": round(dm_bart["p_one_sided"], 5)
            },
            "moving_block_bootstrap": {
                str(b): {
                    "se": round(boot_res[b]["bootstrap_se"], 5),
                    "ci_95_percentile": [round(x, 5) for x in boot_res[b]["ci_percentile_95"]],
                    "ci_95_basic": [round(x, 5) for x in boot_res[b]["ci_basic_95"]],
                    "p_value_empirical": round(boot_res[b]["p_value_empirical"], 5)
                }
                for b in [1, 2, 3, 4, 5, 6]
            }
        }

        audit_results["horizons"][f"+{h_str}"] = h_data

        print(f"\n--- Horizon +{h_str} ---")
        print(f"  MAE Persistence:     {diff_pers_vs_mm['mae_candidate']:.4f} m")
        print(f"  MAE Stage Ridge:     {diff_mm_vs_st['mae_baseline']:.4f} m")
        print(f"  MAE Multimodal Ridge:{diff_mm_vs_st['mae_candidate']:.4f} m")
        print(f"  Delta MAE (MM - ST): {diff_mm_vs_st['delta_mae']:.5f} m ({diff_mm_vs_st['pct_gain']:+.2f}%)")
        print(f"  MM Wins: {diff_mm_vs_st['candidate_wins']}/30 ({diff_mm_vs_st['candidate_win_pct']:.1f}%) | Persistence Beats MM: {diff_pers_vs_mm['candidate_wins']}/30 ({diff_pers_vs_mm['candidate_win_pct']:.1f}%)")
        print(f"  rho_1: {ess['rho_1']:.3f} | N_eff: {ess['n_eff']}")
        print(f"  DM-HLN Rect (k={dm_rect['bandwidth']}): DM*={dm_rect['dm_stat_hln']:.4f}, p_2sided={dm_rect['p_two_sided']:.5f}")
        print(f"  DM-HLN Bart (k=3): DM*={dm_bart['dm_stat_hln']:.4f}, p_2sided={dm_bart['p_two_sided']:.5f}")
        for b in [1, 2, 3, 4, 5, 6]:
            br = boot_res[b]
            print(f"    Bootstrap b={b}: SE={br['bootstrap_se']:.5f}, 95% CI={br['ci_percentile_95']}, p={br['p_value_empirical']:.5f}")

    # Multiple Testing Corrections
    adj_rect = compute_multiple_testing_adjustments(p_vals_rect)
    adj_bart = compute_multiple_testing_adjustments(p_vals_bart)
    audit_results["multiple_testing"]["rectangular_hln"] = adj_rect
    audit_results["multiple_testing"]["bartlett_hln"] = adj_bart

    print("\nMULTIPLE TESTING CORRECTIONS:")
    for h_key in ["+1h", "+2h", "+3h"]:
        print(f"  {h_key}:")
        print(f"    Rectangular: raw p={adj_rect[h_key]['p_raw']:.5f}, Bonferroni adj={adj_rect[h_key]['bonferroni_p_adj']:.5f}, Holm adj={adj_rect[h_key]['holm_p_adj']:.5f}")
        print(f"    Bartlett:    raw p={adj_bart[h_key]['p_raw']:.5f}, Bonferroni adj={adj_bart[h_key]['bonferroni_p_adj']:.5f}, Holm adj={adj_bart[h_key]['holm_p_adj']:.5f}")

    with open(OUT_AUDIT_JSON, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nSaved statistical audit results to: {OUT_AUDIT_JSON}")

if __name__ == "__main__":
    run_audit()
