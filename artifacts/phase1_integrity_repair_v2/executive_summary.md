# Executive Summary: Phase 1 ML Benchmark Integrity Repair
**Project:** Vishnupuri Barrage River Forecasting (Godavari River, Nanded)  
**Evaluation Scope:** Phase 1 Offline Research ML Benchmark  
**Repair Target:** `artifacts/phase1_integrity_repair_v2/`  
**Date:** 2026-10-18  

---

### 1. Was the estimator-reuse bug confirmed?
**YES, conclusively confirmed.**  
`get_catalogue()` defined mutable estimator instances that were passed directly into scikit-learn `Pipeline` objects by `_build_pipe(cfg)` without cloning. In `Phase1Runner.run()`, fitting the pre-event pipeline (`p_pre.fit()`, $N=957$) mutated the mutable estimator inside the standard-training pipeline (`p_std`, $N=1222$) in place. Diagnostic memory tests proved `id(p1.named_steps["estimator"]) == id(p2.named_steps["estimator"])` across all 11 parametric models, causing fitting one pipeline to instantly overwrite the state and predictions of the other.

### 2. Which results were invalidated or remain unverified?
- **Invalidated:** All 33 `.joblib` model artifacts originally saved in `river_forecasting/outputs/phase1/` were invalidated because they contained pre-event training weights ($N=957$), not full-season training weights ($N=1222$).
- **Invalidated:** The previously reported M04 (Stage-Change Ridge) test MAE of 4.88 mm was invalidated; it was produced by a pre-event fit. Under clean full-season training, M04's true test MAE is **5.60 mm** (and **5.65 mm** on synchronized ablation timestamps).
- **Resolved Discrepancy:** The discrepancy between the old benchmark (4.88 mm) and the rainfall ablation (5.65 mm) is now completely resolved. With estimator cloning, both the benchmark and ablation consistently report M04 at 5.60–5.65 mm and Model D (M05) at 4.97–4.98 mm.

### 3. Does Model D outperform persistence on identical test timestamps?
**NO.**  
On identical synchronized test timestamps (September 1–30, 2026):
- **+1h ($N=612$):** Model D MAE = 4.978 mm vs Persistence MAE = 4.949 mm ($\Delta\text{MAE} = -0.028\text{ mm}$, $p = 0.7725$).
- **+2h ($N=604$):** Model D MAE = 6.012 mm vs Persistence MAE = 5.815 mm ($\Delta\text{MAE} = -0.197\text{ mm}$, $p = 0.1695$).
- **+3h ($N=599$):** Model D MAE = 7.049 mm vs Persistence MAE = 6.952 mm ($\Delta\text{MAE} = -0.097\text{ mm}$, $p = 0.6185$).

Across the aggregate test period, river stage changes are predominantly gradual, and Persistence is exceptionally strong. Model D is statistically indistinguishable from Persistence overall.

### 4. Does Model D outperform the stage-change Ridge model at each horizon?
**YES, decisively.**  
When comparing Model D (Stage Dynamics + Contemporary Rain) against the stage-only Ridge baseline (M04) on identical timestamps:
- **+1h ($N=612$):** Model D achieves **4.978 mm** vs M04's **5.648 mm** (Error reduction: **+0.670 mm**, **+11.87%**).
- **+2h ($N=604$):** Model D achieves **6.012 mm** vs M04's **7.690 mm** (Error reduction: **+1.678 mm**, **+21.82%**).
- **+3h ($N=599$):** Model D achieves **7.049 mm** vs M04's **9.886 mm** (Error reduction: **+2.838 mm**, **+28.70%**).

### 5. Are the differences statistically supported?
**YES.**  
Under a 2,000-replicate Moving Block Bootstrap ($L=6\text{h}$) and the Diebold-Mariano test:
- At +1h, +2h, and +3h, Model D's error reduction over Stage-Change Ridge yields **$p < 0.0001$**, with 95% confidence intervals entirely above zero ([+0.46, +0.91] mm at +1h; [+1.29, +2.11] mm at +2h; [+2.27, +3.42] mm at +3h).
- The earlier follow-up audit finding of $p = 0.068$ at +1h was an artifact of the estimator-reuse defect (which had improperly evaluated M04 with pre-event weights). Under clean full-season training, the rainfall advantage is unequivocal.

### 6. Does the rainfall ablation demonstrate incremental value under a fair comparison?
**YES.**  
The synchronized ablation across identical cohorts shows that adding INSAT-3DS contemporary rainfall reduces MAE by 11.9% at +1h, 21.8% at +2h, and 28.7% at +3h relative to stage dynamics alone. Furthermore, at lead times of +2h and +3h, nonlinear models with multimodal features (XGBoost M12: MAE = 5.68 mm at +2h, 6.58 mm at +3h) significantly outperform both Model D and Persistence.

### 7. What remains necessary before any operational use?
1. **Telemetry Freshness Rule:** The live inference pipeline's mandatory 3-hour telemetry age cutoff must remain strictly enforced.
2. **Missing-Rainfall Safeguards:** Missing satellite rainfall must trigger safe fallback modes; it must never be imputed with zero.
3. **Multi-Monsoon Season Validation:** The model must be validated on additional monsoon years beyond 2026.
4. **Physical Barrage Integration:** Model forecasts must be coupled with upstream barrage release logs (Babhali / Jayakwadi) and gate discharge curves before issuing operational flood warnings.
