# Phase 1 Benchmark Integrity Repair Report
**Project:** Vishnupuri Barrage / Godavari River Machine Learning Forecasting  
**Author:** Independent ML Evaluation & Integrity Auditor  
**Date:** 2026-10-18  
**Repository Branch:** `phase1-ml-benchmarking`  
**Base Commit Hash:** `68e3a0ecad3b3fa7c69e4b45fe960d377ba4ef13`  
**Repair Artifact Directory:** `artifacts/phase1_integrity_repair_v2/`  

---

## 1. Executive Overview & Root Cause Analysis

### 1.1 The Bug Hypothesis & Empirical Confirmation
The audit verified the presence of an **estimator-reuse defect** in `river_forecasting/src/phase1_ml_pipeline.py`. 

Specifically, `get_catalogue()` defined mutable scikit-learn estimator instances (such as `Ridge(alpha=10.0)`, `RandomForestRegressor(...)`, and `xgb.XGBRegressor(...)`) in a static list of dictionaries. In `_build_pipe(cfg)`, the pipeline constructor passed `cfg["est"]` directly into the scikit-learn `Pipeline`:

```python
# DEFECTIVE ORIGINAL CODE (river_forecasting/src/phase1_ml_pipeline.py lines 334-341)
def _build_pipe(cfg: Dict) -> Optional[Pipeline]:
    est = cfg["est"]
    if est is None:
        return None
    if cfg["scale"]:
        return Pipeline([("sc", StandardScaler()), ("estimator", est)])
    return Pipeline([("id", StandardScaler(with_mean=False, with_std=False)), ("estimator", est)])
```

Because `cfg["est"]` was not cloned, multiple calls to `_build_pipe(cfg)` within the same execution session returned distinct `Pipeline` containers that shared the **exact same mutable underlying estimator instance in memory** (`id(pipe1.named_steps["estimator"]) == id(pipe2.named_steps["estimator"])`).

### 1.2 The In-Place Mutation Sequence in `Phase1Runner.run()`
In `Phase1Runner.run()`, each configuration was evaluated on standard training ($N=1222$ for stage-only models, $N=1079$ for stage+rain models) and pre-event training ($N=957$ for stage-only models, $N=944$ for stage+rain models):

```python
# IN-PLACE MUTATION CALL SITE
p_std = _build_pipe(cfg)
fitted = _fit_model(p_std, tr_c[feats], tr_c[tgt].values, ...) # 1. Trained on standard data (N=1222)

if len(pre_c) > 0:
    p_pre = _build_pipe(cfg)                                   # 2. Re-used identical estimator object!
    p_pre.fit(pre_c[feats], pre_c[tgt].values)                 # 3. OVERWROTE estimator inside fitted (p_std)!
    fitted_pre = p_pre

art = _save_artifact(fitted, cfg, h, tgt, len(tr_c))          # 4. Saved fitted, which now held pre-event weights!
```

### 1.3 Consequence on Prior Benchmark Results
1. **Invalidated On-Disk Model Artifacts:** All 33 `.joblib` files originally written to `river_forecasting/outputs/phase1/` did not contain the standard full-season training weights as intended; they contained the pre-event training weights (fitted on $N=957$ rows).
2. **Evaluation Inconsistency:** Standard test evaluation in `Phase1Runner.run()` evaluated `fitted`, but because `fitted` had been mutated by `p_pre.fit()`, test evaluation was run on pre-event weights rather than full-season weights.
3. **Discrepancy with `rainfall_ablation()`:** The ablation study script constructed clean `Ridge` pipelines independently. Consequently, `rainfall_ablation()` evaluated full-season weights ($N=1222$), while the primary benchmark evaluated pre-event weights ($N=957$), causing the reported M04 test MAE discrepancy (4.88 mm in benchmark vs 5.65 mm in ablation).

---

## 2. Code Modifications & Estimator Isolation Fix

### 2.1 Applied Code Changes
The defect was resolved by importing `clone` from `sklearn.base` and creating independent estimator instances in `_build_pipe(cfg)`:

```python
from sklearn.base import clone

def _build_pipe(cfg: Dict) -> Optional[Pipeline]:
    est = cfg["est"]
    if est is None:
        return None
    cloned_est = clone(est)
    if cfg["scale"]:
        return Pipeline([("sc", StandardScaler()), ("estimator", cloned_est)])
    # Tree models: identity scaler for uniform param naming in tuning
    return Pipeline([("id", StandardScaler(with_mean=False, with_std=False)), ("estimator", cloned_est)])
```

In addition, `Phase1Runner`, `_save_artifact`, `data_quality_report`, `rainfall_ablation`, and `run_phase1` were modified to accept an optional `out_dir: Optional[Path] = None` parameter and a corresponding CLI flag `--out-dir`. This allows running the benchmark directly into the new versioned directory `artifacts/phase1_integrity_repair_v2/` without touching, deleting, or overwriting the original artifacts in `river_forecasting/outputs/phase1/`.

### 2.2 Automated Regression Test Suite
A new test class `TestEstimatorCloningAndIsolation` was added to `river_forecasting/tests/test_phase1_ml_pipeline.py` with 5 automated test cases:
1. `test_build_pipe_produces_independent_estimator_instances_all_catalogue_models`: Verifies `id(p1.named_steps["estimator"]) != id(p2.named_steps["estimator"])` across all 11 parametric models (M02–M12).
2. `test_fitting_one_pipeline_does_not_mutate_other_estimator_or_predictions`: Proves that fitting `p1` leaves `p2` completely unfitted (`NotFittedError`), and fitting `p2` does not change `p1` predictions or coefficients ($< 10^{-12}$ diff).
3. `test_standard_and_preevent_pipelines_isolation`: Re-enacts the `Phase1Runner` sequence for M04 and M05, asserting that `p_std.coef_` and intercept remain unaffected by `p_pre.fit()`.
4. `test_rebuilding_pipeline_is_deterministic`: Confirms deterministic reproducibility for tree-based models (Random Forest) under fixed random seeds.
5. `test_catalogue_template_estimators_remain_unfitted`: Proves that prototype estimator instances in `get_catalogue()` remain completely unfitted after arbitrary pipeline usage.

All 58 tests in `test_phase1_ml_pipeline.py` and all 158 tests across the entire repository test suite passed cleanly.

---

## 3. Data Integrity, Timestamps, and Leakage Audit

Prior to executing the rebuild, the input dataset and feature generation pipeline were audited:

### 3.1 Data Provenance and Integrity Fingerprints
- **Primary Aligned Dataset:** `data/historical/dataset_2026_full_season_aligned.csv`
  - SHA256: `ea58df0cff7ee54bfa0d941ab5e2dfe4cc8b848fa6836c6cecb414d7162bd213`
  - Row Count: 2,592 hourly rows (100% complete hourly temporal index, no missing timestamps).
  - Time Range: `2026-06-15 00:00:00` to `2026-09-30 23:00:00` IST (naive, Asia/Kolkata).
- **Extended Rainfall Archive:** `data/historical/insat3ds_hem_catchment_hourly_2026.csv`
  - SHA256: `36b12a2db0a0fbbfdc9eeb7004626061487f879c0d30a6d790577e8a779ca784`
  - Row Count: 2,952 hourly rows (`2026-06-01 00:00:00` to `2026-09-30 23:00:00` IST). Used strictly for API 7-day antecedent precipitation spinup prior to the June 15 start date.
  - Provable Source: Real INSAT-3DS HEM satellite catchment extractions from ISRO MOSDAC. No Open-Meteo or synthetic substitution.

### 3.2 Temporal Boundary Partitioning
The chronological partition boundaries are strictly non-overlapping and ordered:
- **TRAIN:** `2026-06-15 00:00:00` to `2026-08-09 23:00:00` IST (1,344 rows).
- **VAL:** `2026-08-10 00:00:00` to `2026-08-31 23:00:00` IST (528 rows).
- **TEST:** `2026-09-01 00:00:00` to `2026-09-30 23:00:00` IST (720 rows).
- **PHASE A (Historical Flood Event):** `2026-07-30 00:00:00` to `2026-07-31 05:00:00` IST (30 rows). Pre-event training cutoff: `2026-07-29 23:00:00` IST.

### 3.3 Leakage and Missingness Controls
- **Target Construction:** Forecast targets $H_{t+1\text{h}}$, $H_{t+2\text{h}}$, $H_{t+3\text{h}}$ are shifted forward in time (`shift(-1)`, `shift(-2)`, `shift(-3)`).
- **Causal Features:** All predictors (stage lags, stage velocity $\Delta H_{\text{1h}}$, acceleration, rainfall accumulators $1\text{h}, 3\text{h}, 6\text{h}, 12\text{h}, 24\text{h}$, and rainfall lags $1\text{h}, 2\text{h}, 3\text{h}$) are strictly causal at origin $t$.
- **Preprocessing Isolation:** `StandardScaler` is fitted exclusively on training splits (`tr_c[feats]`). No test or validation information enters feature transformation.
- **Missing Value Handling:** Missing rainfall observations (5.6% overall) are preserved as `NaN`; they are never imputed with zero or substituted with NWP values. Complete-case analysis is enforced for evaluation.

---

## 4. Repaired Benchmark Results & Before/After Comparison

The full benchmark was executed with the cloned-estimator fix into `artifacts/phase1_integrity_repair_v2/`.

### 4.1 Comparison of M04 and M05 Models Before and After Repair
| Horizon | Model | Training Cohort | Un-Repaired Test MAE (Old Buggy Run) | Repaired Test MAE (Clean Full Train) | Repaired Synchronized Test MAE ($N_{\text{sync}}$) |
|---|---|---|---|---|---|
| **+1h** | M04 (Stage-Change Ridge, no rain) | Full Train ($N=1222$) | 4.88 mm *(actually pre-event $N=957$)* | **5.60 mm** | **5.65 mm** ($N=612$) |
| **+1h** | M05 (Model D: Stage + Contemp. Rain) | Full Train ($N=1079$) | 4.76 mm *(actually pre-event $N=944$)* | **4.97 mm** | **4.98 mm** ($N=612$) |
| **+2h** | M04 (Stage-Change Ridge, no rain) | Full Train ($N=1218$) | 6.84 mm *(actually pre-event $N=955$)* | **7.59 mm** | **7.69 mm** ($N=604$) |
| **+2h** | M05 (Model D: Stage + Contemp. Rain) | Full Train ($N=1075$) | 5.86 mm *(actually pre-event $N=942$)* | **6.01 mm** | **6.01 mm** ($N=604$) |
| **+3h** | M04 (Stage-Change Ridge, no rain) | Full Train ($N=1217$) | 8.89 mm *(actually pre-event $N=954$)* | **9.82 mm** | **9.89 mm** ($N=599$) |
| **+3h** | M05 (Model D: Stage + Contemp. Rain) | Full Train ($N=1074$) | 6.95 mm *(actually pre-event $N=941$)* | **7.08 mm** | **7.05 mm** ($N=599$) |

### 4.2 Key Insight from the Repair
The un-repaired run severely flattered M04's test performance because fitting on pre-event data ($N=957$) happened to produce small coefficients that performed unusually well during dry September periods. Once M04 is properly trained on the entire training set ($N=1222$), its true test error is **5.60 mm to 5.65 mm**, not 4.88 mm.

Conversely, Model D (M05) performs stably around **4.97 mm to 4.98 mm** under full training. Consequently, the true incremental value of rainfall over stage-change Ridge is **much larger and unambiguous** than previously reported.

---

## 5. Statistical Inference & Paired Hypothesis Testing

Paired differences $d_t = |y_t - \hat{y}_{\text{base},t}| - |y_t - \hat{y}_{\text{test},t}|$ were evaluated on **100% identical synchronized timestamps** using a 2,000-replicate Moving Block Bootstrap ($L=6\text{h}$) and the Diebold-Mariano test with Newey-West HAC standard errors.

### 5.1 Model D vs Persistence Baseline (M01)
*Scientific Question: Does Model D outperform a simple persistence forecast across all test timestamps?*

| Horizon | Common $N$ | Model D MAE | Persistence MAE | Error Difference ($\Delta$MAE) | Improvement | 95% Bootstrap CI | Bootstrap $p$ (two-sided) | DM $z$-stat ($p$-value) | Conclusion |
|---|---|---|---|---|---|---|---|---|---|
| **+1h** | 612 | 4.978 mm | 4.949 mm | -0.028 mm | -0.57% | [-0.214, +0.182] mm | $p = 0.7725$ | $z = -0.27$ ($p = 0.79$) | **No significant difference** |
| **+2h** | 604 | 6.012 mm | 5.815 mm | -0.197 mm | -3.40% | [-0.478, +0.082] mm | $p = 0.1695$ | $z = -1.36$ ($p = 0.17$) | **No significant difference** |
| **+3h** | 599 | 7.049 mm | 6.952 mm | -0.097 mm | -1.39% | [-0.452, +0.276] mm | $p = 0.6185$ | $z = -0.52$ ($p = 0.61$) | **No significant difference** |

**Finding:** Model D does **not** statistically outperform Persistence across the aggregate test cohort. In a low-energy, slowly-varying river reach during non-flood periods, persistence is extremely competitive.

### 5.2 Model D vs Stage-Change Ridge Baseline (M04) — Rainfall Value
*Scientific Question: Does contemporary catchment rainfall add measurable forecasting skill over stage dynamics alone?*

| Horizon | Common $N$ | Model D MAE | Stage Ridge MAE | Error Difference ($\Delta$MAE) | Improvement | 95% Bootstrap CI | Bootstrap $p$ (two-sided) | DM $z$-stat ($p$-value) | Conclusion |
|---|---|---|---|---|---|---|---|---|---|
| **+1h** | 612 | 4.978 mm | 5.648 mm | **+0.670 mm** | **+11.87%** | **[+0.460, +0.909] mm** | **$p < 0.0001$** | $z = +5.93$ ($p < 0.0001$) | **Highly Significant** |
| **+2h** | 604 | 6.012 mm | 7.690 mm | **+1.678 mm** | **+21.82%** | **[+1.286, +2.108] mm** | **$p < 0.0001$** | $z = +8.22$ ($p < 0.0001$) | **Highly Significant** |
| **+3h** | 599 | 7.049 mm | 9.886 mm | **+2.838 mm** | **+28.70%** | **[+2.271, +3.415] mm** | **$p < 0.0001$** | $z = +9.81$ ($p < 0.0001$) | **Highly Significant** |

**Finding:** Rainfall provides a decisive, statistically supported improvement over stage dynamics alone at all three horizons (+11.9% at +1h, +21.8% at +2h, +28.7% at +3h), with zero confidence interval overlap across zero. The earlier "inconclusive" $p=0.068$ result was purely an artifact of the estimator-reuse bug.

### 5.3 Model D vs Top Machine Learning Competitors
*Scientific Question: How does Model D compare to nonlinear multimodal tree ensembles (XGBoost M12 and Extra Trees M09)?*

- **At +1h:** Model D (4.978 mm), XGBoost (4.951 mm), and Extra Trees (4.969 mm) are statistical ties ($p > 0.70$).
- **At +2h:** XGBoost (5.676 mm, $p = 0.0005$) and Extra Trees (5.639 mm, $p = 0.0005$) significantly outperform Model D (6.012 mm). Both also beat Persistence (5.815 mm).
- **At +3h:** XGBoost (6.579 mm, $p < 0.0001$) and Extra Trees (6.723 mm, $p = 0.044$) significantly outperform Model D (7.049 mm). XGBoost also beats Persistence (6.952 mm).

---

## 6. Independent Artifact Reload & Manifest Verification

All 33 retrained model pipelines in `artifacts/phase1_integrity_repair_v2/` were loaded into a clean Python process and validated against freshly engineered test features:
- **Verification Rule:** `abs(round(reloaded_mae, 5) - in_mem_mae) < 1e-7`.
- **Outcome:** 33 / 33 artifacts verified with exact parity.
- **Manifest:** Full mapping of model ID, horizon, configuration, feature lists, data fingerprints, and SHA256 hashes generated in `artifacts/phase1_integrity_repair_v2/artifact_manifest.json`.

---

## 7. Limitations and Operational Governance

1. **Single Monsoon Season:** The benchmark is constructed on the 2026 monsoon season (June 15 – September 30, 2026). Inter-annual generalization requires multi-season validation.
2. **Operational Safeguards Preserved:** Offline benchmark repair does not alter live inference safeguards. The 3-hour telemetry freshness gate, rainfall compatibility validation, and barrage gate coordination protocols in `live_inference.py` remain strictly active and unmodified.
3. **No Automatic Promotion:** Neither Model D nor XGBoost has been promoted to production. Any operational transition requires shadow deployment and formal validation against physical barrage constraints.
