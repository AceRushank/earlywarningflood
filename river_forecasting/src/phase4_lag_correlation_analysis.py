"""
river_forecasting/src/phase4_lag_correlation_analysis.py
=========================================================
Phase 4: Comprehensive Rainfall-Stage Lag Cross-Correlation Reassessment.

Requirements:
1. Recompute correlations between hourly catchment rainfall depth and hourly stage change
   for lags of 0–72 hours (step 1h and 3h).
2. For every lag, report:
   - Correlation coefficient (Pearson r)
   - Valid pair count (N)
   - Rainfall timestamp range
   - Stage-change timestamp range
   - Missing-data treatment
3. Plot the full lag-correlation curve (0 to 72h).
4. Test sensitivity to influential rainfall observations (top-1, top-2 rainfall leverage).
5. Assess serial autocorrelation using AR(1) and ACF.
6. Estimate effective degrees of freedom (N_eff) using Bretherton / Chelton method.
7. Distinguish exploratory association from validated physical travel time.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNED_CSV = REPO_ROOT / "river_forecasting" / "outputs" / "event1_aligned_rainfall_stage.csv"
OUT_PLOT_PNG = REPO_ROOT / "river_forecasting" / "outputs" / "event1_lag_correlation_curve.png"
OUT_RESULTS_CSV = REPO_ROOT / "river_forecasting" / "outputs" / "event1_lag_correlations.csv"


def bretherton_effective_n(x: np.ndarray, y: np.ndarray, max_k: int = 10) -> float:
    """
    Estimates effective sample size (N_eff) accounting for serial autocorrelation
    using the Bretherton et al. (1999) / Chelton (1983) method:
    N_eff = N / (1 + 2 * sum_{k=1}^M r_k^x * r_k^y)
    Bounded between 2 and N.
    """
    n = len(x)
    if n <= 3:
        return float(n)

    # Demean
    xd = x - np.mean(x)
    yd = y - np.mean(y)

    var_x = np.sum(xd ** 2)
    var_y = np.sum(yd ** 2)
    if var_x == 0 or var_y == 0:
        return float(n)

    sum_terms = 0.0
    for k in range(1, min(max_k + 1, n // 3)):
        rx_k = np.sum(xd[:-k] * xd[k:]) / var_x
        ry_k = np.sum(yd[:-k] * yd[k:]) / var_y
        prod = rx_k * ry_k
        if prod < 0:
            # Standard truncation when correlation product turns negative
            break
        sum_terms += prod

    neff = n / (1.0 + 2.0 * sum_terms)
    return max(3.0, min(float(n), neff))


def run_phase4_lag_analysis():
    print("=" * 90)
    print("PHASE 4: REASSESS RAINFALL-STAGE LAG CORRELATIONS (0 to 72 Hours)")
    print("=" * 90)

    if not ALIGNED_CSV.exists():
        raise FileNotFoundError(f"Aligned dataset not found: {ALIGNED_CSV}")

    df = pd.read_csv(ALIGNED_CSV)
    df["dt_ist"] = pd.to_datetime(df["timestamp_ist"])
    df.sort_values("dt_ist", inplace=True)
    df.reset_index(drop=True, inplace=True)

    stage_diff = df["stage_change_m"]  # dH(t) = H(t) - H(t-1)
    rain = df["hourly_depth_mm"]

    print(f"Total aligned rows: {len(df)}")
    print(f"Valid stage differences: {stage_diff.notna().sum()}")
    print(f"Valid rainfall hours:    {rain.notna().sum()}")

    # Autocorrelation of raw series
    valid_diff = stage_diff.dropna()
    valid_rain = rain.dropna()
    ac1_stage = valid_diff.autocorr(lag=1)
    ac1_rain = valid_rain.autocorr(lag=1)
    print(f"\nSerial Autocorrelation (Lag-1):")
    print(f"  Stage Difference (dH) Lag-1 Autocorrelation: {ac1_stage:.3f}")
    print(f"  Catchment Rainfall Depth Lag-1 Autocorrelation: {ac1_rain:.3f}")

    # Compute correlation across all lags from 0 to 72 hours
    records = []
    print("\n" + "-" * 110)
    print(f"{'Lag(h)':<6} | {'N':<4} | {'N_eff':<6} | {'r (Pearson)':<11} | {'p-value':<9} | {'95% CI':<16} | {'Rainfall Window (IST)':<35} | {'Stage Window (IST)'}")
    print("-" * 110)

    # Prepare Complete-Only and Scaled series
    rain_complete_only = rain.copy()
    rain_complete_only[df["completeness"] != "COMPLETE"] = np.nan

    rain_scaled = rain.copy()
    part_mask = df["completeness"] == "PARTIAL_INCOMPLETE"
    rain_scaled[part_mask] = df.loc[part_mask, "mean_rain_rate_mm_hr"]

    for lag in range(0, 73):
        # Shift rainfall forward by lag hours: rain at (t - lag) aligns with stage change at t
        shifted_rain = rain.shift(lag)
        mask = shifted_rain.notna() & stage_diff.notna()
        n = int(mask.sum())

        if n >= 10:
            x = shifted_rain[mask].to_numpy()
            y = stage_diff[mask].to_numpy()

            # Complete-only subset
            s_comp = rain_complete_only.shift(lag)
            m_comp = s_comp.notna() & stage_diff.notna()
            r_comp = float(stats.pearsonr(s_comp[m_comp], stage_diff[m_comp])[0]) if m_comp.sum() >= 10 else np.nan

            # Scaled subset
            s_scale = rain_scaled.shift(lag)
            m_scale = s_scale.notna() & stage_diff.notna()
            r_scale = float(stats.pearsonr(s_scale[m_scale], stage_diff[m_scale])[0]) if m_scale.sum() >= 10 else np.nan

            if np.std(x) > 1e-6 and np.std(y) > 1e-6:
                r, p_raw = stats.pearsonr(x, y)
                neff = bretherton_effective_n(x, y, max_k=8)

                # Adjusted p-value using N_eff (accounting for autocorrelation ONLY)
                t_stat = r * np.sqrt((neff - 2) / max(1e-6, 1 - r**2))
                p_adj = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, neff - 2)))
                # Multi-lag Bonferroni p-value across 73 candidate lags
                p_bonferroni = min(1.0, p_adj * 73.0)

                # Fisher z CI
                z = np.arctanh(np.clip(r, -0.999, 0.999))
                se_z = 1.0 / np.sqrt(max(1.0, neff - 3.0))
                z_low, z_high = z - 1.96 * se_z, z + 1.96 * se_z
                r_low, r_high = np.tanh(z_low), np.tanh(z_high)

                r_start = df.loc[df.index[mask] - lag, "timestamp_ist"].iloc[0]
                r_end = df.loc[df.index[mask] - lag, "timestamp_ist"].iloc[-1]
                s_start = df.loc[df.index[mask], "timestamp_ist"].iloc[0]
                s_end = df.loc[df.index[mask], "timestamp_ist"].iloc[-1]

                records.append({
                    "lag_hours": lag,
                    "n_pairs": n,
                    "n_eff": round(neff, 1),
                    "pearson_r": round(r, 4),
                    "r_complete_only": round(r_comp, 4) if not np.isnan(r_comp) else None,
                    "r_rate_scaled": round(r_scale, 4) if not np.isnan(r_scale) else None,
                    "p_value_raw": round(p_raw, 5),
                    "p_value_autocorr_adj": round(p_adj, 5),
                    "p_value_bonferroni_73lags": round(p_bonferroni, 4),
                    "ci_95_low": round(r_low, 3),
                    "ci_95_high": round(r_high, 3),
                    "rain_window_start": r_start,
                    "rain_window_end": r_end,
                    "stage_window_start": s_start,
                    "stage_window_end": s_end,
                    "partial_hours_treatment": "raw_has_30m_accumulation; r_complete_only_excludes_partials"
                })

                if lag % 6 == 0 or lag in [24, 66]:
                    ci_str = f"[{r_low:+.2f}, {r_high:+.2f}]"
                    print(f"{lag:<6} | {n:<4} | {neff:<6.1f} | {r:<+11.4f} | {p_adj:<9.4f} | {ci_str:<16} | {r_start} to {r_end[:10]} | {s_start} to {s_end[:10]}")

    df_corrs = pd.DataFrame(records)
    df_corrs.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\nWrote full lag correlation table: {OUT_RESULTS_CSV}")

    # Identify Peak Lags
    max_row = df_corrs.loc[df_corrs["pearson_r"].idxmax()]
    print(f"\nPeak Positive Correlation:")
    print(f"  Lag:        {int(max_row['lag_hours'])} hours")
    print(f"  Pearson r:  {max_row['pearson_r']:+.4f} (95% CI: [{max_row['ci_95_low']:+.3f}, {max_row['ci_95_high']:+.3f}])")
    print(f"  N:          {int(max_row['n_pairs'])} pairs (N_eff: {max_row['n_eff']:.1f})")
    print(f"  p (autocorr adj): {max_row['p_value_autocorr_adj']:.4f}")
    print(f"  p (Bonferroni 73 lags): {max_row['p_value_bonferroni_73lags']:.4f}")
    print(f"  r (Complete-only): {max_row['r_complete_only']}")
    print(f"  r (Rate-scaled):   {max_row['r_rate_scaled']}")

    # Sensitivity Analysis on Influential Points for Key Lags (e.g. lag 24 and lag 66)
    print("\n" + "=" * 60)
    print("SENSITIVITY & LEVERAGE ANALYSIS")
    print("=" * 60)

    for test_lag in [24, 66, int(max_row['lag_hours'])]:
        if test_lag in df_corrs["lag_hours"].values:
            s_rain = rain.shift(test_lag)
            m = s_rain.notna() & stage_diff.notna()
            sub = pd.DataFrame({
                "rain": s_rain[m],
                "stage_diff": stage_diff[m]
            })
            r_full = np.corrcoef(sub["rain"], sub["stage_diff"])[0, 1]

            # Drop top 1 rainfall point
            sub_drop1 = sub.drop(sub["rain"].idxmax())
            r_drop1 = np.corrcoef(sub_drop1["rain"], sub_drop1["stage_diff"])[0, 1]

            # Drop top 2 rainfall points
            sub_drop2 = sub_drop1.drop(sub_drop1["rain"].idxmax())
            r_drop2 = np.corrcoef(sub_drop2["rain"], sub_drop2["stage_diff"])[0, 1]

            # Spearman rank correlation (robust to outliers)
            rho, _ = stats.spearmanr(sub["rain"], sub["stage_diff"])

            print(f"Lag {test_lag}h Sensitivity:")
            print(f"  Full Pearson r (N={len(sub)}):               {r_full:+.4f}")
            print(f"  Without highest rain observation (N={len(sub)-1}):  {r_drop1:+.4f} (Delta={r_drop1 - r_full:+.4f})")
            print(f"  Without top 2 rain observations (N={len(sub)-2}):   {r_drop2:+.4f} (Delta={r_drop2 - r_full:+.4f})")
            print(f"  Spearman rank correlation (rho):           {rho:+.4f}")

    # Plot Full Lag-Correlation Curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_corrs["lag_hours"], df_corrs["pearson_r"], color="#1f77b4", lw=2.5, marker="o", ms=4, label="Pearson Correlation r(lag)")
    ax.fill_between(df_corrs["lag_hours"], df_corrs["ci_95_low"], df_corrs["ci_95_high"], color="#1f77b4", alpha=0.18, label="95% CI (Bretherton N_eff)")

    ax.axhline(0, color="gray", ls="--", alpha=0.7)
    ax.axvline(24, color="orange", ls=":", lw=1.5, label="24h lag")
    ax.axvline(66, color="green", ls=":", lw=1.5, label="66h lag")

    ax.set_xlabel("Rainfall Lag (Hours)", fontsize=11)
    ax.set_ylabel("Correlation with Hourly Stage Change (dH)", fontsize=11)
    ax.set_title("Full Lag-Correlation Profile: Catchment Rainfall vs River Stage Rise (0-72h)\n[Exploratory Cross-Correlation - Single Flood Event]", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.5, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(OUT_PLOT_PNG, dpi=200)
    plt.close(fig)
    print(f"\nSaved lag correlation plot: {OUT_PLOT_PNG}")


if __name__ == "__main__":
    run_phase4_lag_analysis()
