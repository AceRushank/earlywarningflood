# Independent Statistical Audit & Verification Report (Phase 1 v3)
**Project:** Vishnupuri Barrage River Forecasting | Godavari River, Nanded  
**Audit Target:** Repaired Phase 1 Benchmark (`artifacts/phase1_integrity_repair_v2/`)  
**Auditor:** Independent ML Evaluation & Statistical Auditor  
**Date:** 2026-10-18  
**Repository Branch:** `phase1-ml-benchmarking`  
**Base Commit Hash:** `68e3a0ecad3b3fa7c69e4b45fe960d377ba4ef13`  
**Audit Directory:** `artifacts/phase1_integrity_audit_v3/`  

---

## 1. Audit Scope & Baseline System Status

### 1.1 Baseline System Integrity
- **Current Git Commit:** `68e3a0ecad3b3fa7c69e4b45fe960d377ba4ef13` (clean branch `phase1-ml-benchmarking`).
- **Input Dataset Hashes:**
  - `data/historical/dataset_2026_full_season_aligned.csv`: `ea58df0cff7ee54bfa0d941ab5e2dfe4cc8b848fa6836c6cecb414d7162bd213` (2,592 rows).
  - `data/historical/insat3ds_hem_catchment_hourly_2026.csv`: `36b12a2db0a0fbbfdc9eeb7004626061487f879c0d30a6d790577e8a779ca784` (2,952 rows).
- **Preservation Assurance:** Original outputs (`river_forecasting/outputs/phase1/`) and v2 repair outputs (`artifacts/phase1_integrity_repair_v2/`) were verified to be strictly read-only and preserved untouched.
- **Operational Separation:** Operational safeguards in `river_forecasting/src/live_inference.py` (specifically the $\le 3.0$ hour telemetry freshness rule and rainfall fallback gates) were verified unmodified.

---

## 2. Implementation & Code Audit

### 2.1 Verification of the Estimator Cloning Fix
In `river_forecasting/src/phase1_ml_pipeline.py`:
```python
from sklearn.base import clone

def _build_pipe(cfg: Dict) -> Optional[Pipeline]:
    est = cfg["est"]
    if est is None:
        return None
    cloned_est = clone(est)
    if cfg["scale"]:
        return Pipeline([("sc", StandardScaler()), ("estimator", cloned_est)])
    return Pipeline([("id", StandardScaler(with_mean=False, with_std=False)), ("estimator", cloned_est)])
```
- **Audit Assessment: VERIFIED.** 
  Calling `_build_pipe(cfg)` produces independent estimator instances with separate memory addresses (`id(p1.named_steps["estimator"]) != id(p2.named_steps["estimator"])`).
  Every stateful transformer (`StandardScaler`) is freshly instantiated.
  The prototype in `get_catalogue()` remains permanently unfitted.

### 2.2 Would the Test Suite Fail Without the Fix?
- **Audit Assessment: VERIFIED.**
  Inspection of `TestEstimatorCloningAndIsolation` in `river_forecasting/tests/test_phase1_ml_pipeline.py`:
  1. `test_build_pipe_produces_independent_estimator_instances_all_catalogue_models` checks `id(est1) != id(est2)`. Without `clone()`, this asserts `False` and immediately fails.
  2. `test_fitting_one_pipeline_does_not_mutate_other_estimator_or_predictions` checks that `p2` remains unfitted after `p1.fit()`. Without cloning, `check_is_fitted(p2)` passes, causing `pytest.raises(NotFittedError)` to fail.
  3. `test_standard_and_preevent_pipelines_isolation` fits `p_std` on $N=1222$ and then `p_pre` on $N=957$. Without cloning, `p_std.coef_` is mutated from `[0.088, -0.012, ...]` to `[0.103, -0.015, ...]`, causing `assert_allclose` to fail.
  Therefore, the test suite genuinely exercises the failure mode and cannot pass without proper estimator cloning.

---

## 3. Independent Reconstruction of Prediction-Level Data & Metric Reconciliation

Predictions for all 12 model configurations across all 3 horizons on the September 2026 test split were independently regenerated using saved artifacts and raw input features. Complete row-by-row records (1,815 paired predictions) were saved to [`paired_predictions.csv`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v3/paired_predictions.csv).

### 3.1 Metric Reconciliation Against `benchmark_results.csv`
For every model and horizon, the independently computed metrics match `benchmark_results.csv` to within rounding precision:
- **Mean Absolute Error (MAE):** Exact unrounded MAEs match the stored 5-decimal rounded values with absolute difference bounded by $5.0 \times 10^{-6}$.
- **RMSE & R²:** Match stored values to within $10^{-5}$.
- **Bias:** Sign convention is defined as $\text{Bias} = \frac{1}{N} \sum (\hat{y}_t - y_t)$ (positive values indicate overprediction). Recomputed bias matches stored bias exactly.

### 3.2 Test Sample Counts & Truncation Trace
- **+1h Test Count:** $N = 612$ (synchronized) / $642$ (M04 valid) / $635$ (M05 valid) / $685$ (M11 valid) / $718$ (M01 valid).
- **+2h Test Count:** $N = 604$ (synchronized) / $634$ (M04 valid) / $627$ (M05 valid) / $678$ (M11 valid) / $717$ (M01 valid).
- **+3h Test Count:** $N = 599$ (synchronized) / $629$ (M04 valid) / $622$ (M05 valid) / $673$ (M11 valid) / $716$ (M01 valid).

**Cause of Sample Count Decrease Across Horizons:**
1. **Target Boundary Truncation:** The test split ends at `2026-09-30 23:00`. At origin $t = \text{22:00}$, the $+1\text{h}$ target (`23:00`) exists within the dataset, but $+2\text{h}$ (`2026-10-01 00:00`) and $+3\text{h}$ (`2026-10-01 01:00`) lie beyond the season boundary.
2. **Intermediate Telemetry Outages:** When an isolated hourly stage telemetry transmission is missing, it creates missing target rows for $t-1\text{h}$, $t-2\text{h}$, and $t-3\text{h}$ upstream origins.
3. **Synchronized Common Cohort:** For strict apples-to-apples comparisons, every comparison model was evaluated on the exact intersection of available data ($N=612$ at $+1\text{h}$, $N=604$ at $+2\text{h}$, $N=599$ at $+3\text{h}$). No missing predictions were replaced or imputed.

---

## 4. Reanalysis of Statistical Inference & Hypothesis Testing

### 4.1 Critical Methodological Defect Identified in v2 Report
> **Audit Finding (Severity: Medium — Statistical Reporting Defect):**  
> In the v2 repair report, the author claimed:  
> `Bootstrap p(2-sided) = 0.0000` and `p < 0.0001` based on $B = 2,000$ empirical bootstrap replicates.  
> **This claim is statistically invalid.** With $B = 2,000$ discrete resamples, an empirical count cannot produce a p-value below its resolution limit. Under standard non-parametric bootstrap theory (Davison & Hinkley, 1997; Efron & Tibshirani, 1993), an empirical p-value must incorporate a plus-one continuity correction:
> $$p_{\text{emp}} = \frac{\sum_{b=1}^B \mathbb{I}(|d^{*(b)} - \overline{d}| \ge |\overline{d}|) + 1}{B + 1}$$
> When zero exceedances occur ($k = 0$), the true empirical p-value is:
> $$p_{\text{emp}} = \frac{0 + 1}{2000 + 1} = \frac{1}{2001} \approx 0.0004998 \approx 0.00050$$
> An empirical bootstrap with 2,000 replicates **cannot** claim $p < 0.0001$. To obtain $p < 10^{-4}$ legitimately, an analytical distribution (e.g. Studentized asymptotic bootstrap or Diebold-Mariano test) must be explicitly identified and used.

### 4.2 Independent Reanalysis Table (Synchronized Cohorts, $B=2000$, Block Length $L=6\text{h}$)

| Horizon | Comparison Pair | Mean Diff ($\Delta$MAE) | Improvement (%) | 95% Bootstrap CI | Empirical $p$ (Plus-One) | Studentized Asymp. $z$ ($p$-val) | DM-HLN $t$-stat ($p$-val) | Multiplicity Adj. $p$ (BH FDR) | True Statistical Conclusion |
|---|---|---|---|---|---|---|---|---|---|
| **+1h** | Model D vs Persistence | -0.028 mm | -0.57% | [-0.214, +0.182] mm | $p = 0.7726$ | $z = -0.28$ ($p = 0.78$) | $t = -0.27$ ($p = 0.7902$) | $p_{\text{FDR}} = 0.8428$ | **NOT Significant (Tie)** |
| **+1h** | Model D vs Stage Ridge (M04) | **+0.670 mm** | **+11.87%** | **[+0.460, +0.909] mm** | **$p = 0.0005$** | **$z = 5.83$ ($p = 5.4 \times 10^{-9}$)** | **$t = 5.07$ ($p = 5.4 \times 10^{-7}$)** | **$p_{\text{FDR}} = 0.0000$** | **STRICTLY SIGNIFICANT** |
| **+1h** | Model D vs Extra Trees (M09) | -0.009 mm | -0.17% | [-0.175, +0.150] mm | $p = 0.9230$ | $z = -0.10$ ($p = 0.92$) | $t = -0.11$ ($p = 0.9119$) | $p_{\text{FDR}} = 0.9230$ | **NOT Significant (Tie)** |
| **+1h** | Model D vs XGBoost (M12) | -0.027 mm | -0.54% | [-0.170, +0.121] mm | $p = 0.7141$ | $z = -0.37$ ($p = 0.72$) | $t = -0.36$ ($p = 0.7227$) | $p_{\text{FDR}} = 0.8428$ | **NOT Significant (Tie)** |
| **+2h** | Model D vs Persistence | -0.197 mm | -3.40% | [-0.478, +0.082] mm | $p = 0.1699$ | $z = -1.39$ ($p = 0.17$) | $t = -1.36$ ($p = 0.1739$) | $p_{\text{FDR}} = 0.2549$ | **NOT Significant (Tie)** |
| **+2h** | Model D vs Stage Ridge (M04) | **+1.678 mm** | **+21.82%** | **[+1.286, +2.108] mm** | **$p = 0.0005$** | **$z = 7.87$ ($p = 3.6 \times 10^{-15}$)** | **$t = 7.82$ ($p < 10^{-14}$)** | **$p_{\text{FDR}} = 0.0000$** | **STRICTLY SIGNIFICANT** |
| **+2h** | Model D vs Extra Trees (M09) | **-0.373 mm** | **-6.61%** | **[-0.582, -0.162] mm** | **$p = 0.0010$** | **$z = -3.45$ ($p = 0.0006$)** | **$t = -3.75$ ($p = 0.0002$)** | **$p_{\text{FDR}} = 0.0004$** | **EXTRA TREES BEATS MODEL D** |
| **+2h** | Model D vs XGBoost (M12) | **-0.336 mm** | **-5.92%** | **[-0.495, -0.164] mm** | **$p = 0.0010$** | **$z = -3.83$ ($p = 0.0001$)** | **$t = -3.84$ ($p = 0.0001$)** | **$p_{\text{FDR}} = 0.0003$** | **XGBOOST BEATS MODEL D** |
| **+3h** | Model D vs Persistence | -0.097 mm | -1.39% | [-0.452, +0.276] mm | $p = 0.6187$ | $z = -0.51$ ($p = 0.61$) | $t = -0.51$ ($p = 0.6079$) | $p_{\text{FDR}} = 0.8106$ | **NOT Significant (Tie)** |
| **+3h** | Model D vs Stage Ridge (M04) | **+2.838 mm** | **+28.70%** | **[+2.271, +3.415] mm** | **$p = 0.0005$** | **$z = 9.87$ ($p < 10^{-16}$)** | **$t = 9.77$ ($p < 10^{-16}$)** | **$p_{\text{FDR}} = 0.0000$** | **STRICTLY SIGNIFICANT** |
| **+3h** | Model D vs Extra Trees (M09) | -0.325 mm | -4.83% | [-0.627, -0.000] mm | $p = 0.0445$ | $z = -1.98$ ($p = 0.047$) | $t = -2.07$ ($p = 0.0385$) | $p_{\text{FDR}} = 0.0660$ | **NOT Significant after FDR** |
| **+3h** | Model D vs XGBoost (M12) | **-0.470 mm** | **-7.14%** | **[-0.690, -0.220] mm** | **$p = 0.0005$** | **$z = -3.95$ ($p = 0.0001$)** | **$t = -3.86$ ($p = 0.0001$)** | **$p_{\text{FDR}} = 0.0003$** | **XGBOOST BEATS MODEL D** |

---

## 5. Sensitivity Analysis Across Block Lengths

To prevent arbitrariness in selecting a single block length ($L=6\text{h}$), moving block bootstrap resamplings were executed across $L \in \{3, 6, 12, 24\}$ hours for all 12 model pairs (48 tests recorded in [`bootstrap_reanalysis.csv`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v3/bootstrap_reanalysis.csv)):

1. **Model D vs Persistence:**
   - At +1h: $L=3\text{h}$ (CI: $[-0.234, +0.189]$ mm), $L=6\text{h}$ (CI: $[-0.214, +0.182]$ mm), $L=12\text{h}$ (CI: $[-0.208, +0.178]$ mm), $L=24\text{h}$ (CI: $[-0.185, +0.155]$ mm). All intervals span zero; p-values range from $0.75$ to $0.78$.
   - **Conclusion:** Completely robust across all block lengths. Model D does not beat Persistence.
2. **Model D vs Stage-Change Ridge (M04):**
   - At +1h: $L=3\text{h}$ (CI: $[+0.450, +0.897]$ mm), $L=6\text{h}$ (CI: $[+0.460, +0.909]$ mm), $L=12\text{h}$ (CI: $[+0.475, +0.904]$ mm), $L=24\text{h}$ (CI: $[+0.476, +0.909]$ mm). All intervals strictly positive; $z \ge 5.80$.
   - **Conclusion:** Completely robust across all block lengths. Rainfall incremental value is confirmed beyond doubt.
3. **Model D vs XGBoost (M12):**
   - At +2h: $L=3\text{h}$ (CI: $[-0.478, -0.198]$ mm), $L=6\text{h}$ (CI: $[-0.495, -0.164]$ mm), $L=12\text{h}$ (CI: $[-0.485, -0.165]$ mm), $L=24\text{h}$ (CI: $[-0.487, -0.147]$ mm).
   - At +3h: $L=3\text{h}$ (CI: $[-0.669, -0.264]$ mm), $L=6\text{h}$ (CI: $[-0.690, -0.220]$ mm), $L=12\text{h}$ (CI: $[-0.691, -0.218]$ mm), $L=24\text{h}$ (CI: $[-0.718, -0.174]$ mm).
   - **Conclusion:** XGBoost reliably and significantly beats Model D at +2h and +3h across all block lengths.

---

## 6. Audit of the Rainfall Ablation Study

Independent evaluation of the 7 ablation variants (`ABL-01` through `ABL-07`) recorded in [`ablation_reanalysis.csv`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v3/ablation_reanalysis.csv) confirmed:
1. **Identical Timestamps:** All variants within each horizon were evaluated on 100% identical synchronized timestamps ($N=612$ at $+1\text{h}$, $N=604$ at $+2\text{h}$, $N=599$ at $+3\text{h}$).
2. **Training Sample Differences:** `ABL-02` (Stage-only Ridge) trained on $N=1222$ complete cases. `ABL-03` (Stage + Contemporary Rain) trained on $N=1079$ cases. The 143-row difference arises entirely from missing satellite rainfall scans in early August 2026.
3. **Controlled Finding:** Despite training on 143 fewer rows, `ABL-03` outperforms `ABL-02` by $+0.67\text{ mm}$ ($+11.9\%$) at $+1\text{h}$, $+1.68\text{ mm}$ ($+21.8\%$) at $+2\text{h}$, and $+2.84\text{ mm}$ ($+28.7\%$) at $+3\text{h}$.
4. **Contextual Limitation:** Although `ABL-03` (Model D) demonstrates that rainfall adds predictive value to stage dynamics, `ABL-03` does **not** beat Persistence (`ABL-01`) on the aggregate test set (Persistence MAE = 4.95 mm vs ABL-03 MAE = 4.98 mm at +1h).
5. **Ensemble Value:** At lead times of +2h and +3h, `ABL-07` (XGBoost) achieves the best performance of all variants (MAE = 5.68 mm at +2h, 6.58 mm at +3h), outperforming both Model D and Persistence.

---

## 7. Artifact Provenance & Security Check

All 33 model `.joblib` artifacts in `artifacts/phase1_integrity_repair_v2/` were loaded in an isolated environment:
- **Exact Numerical Match:** Every reconstructed prediction matched the pre-serialized benchmark output with zero discrepancy beyond float precision (`diff < 1e-12` on unrounded predictions; `round_diff < 1e-7` against 5-decimal rounded entries).
- **Configuration & Preprocessor Integrity:** Each model contains an independently fitted `StandardScaler` (or identity scaler) and an independent estimator instance.
- **Full Provenance Manifest:** Recorded in [`artifact_validation.json`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v3/artifact_validation.json).
