# Revised Executive Summary & Final Decision Table (Phase 1 v3)
**Project:** Vishnupuri Barrage River Forecasting | Godavari River, Nanded  
**Audit Scope:** Independent Verification of Repaired Phase 1 Benchmark (`artifacts/phase1_integrity_repair_v2/`)  
**Auditor:** Independent Statistical and ML Integrity Auditor  
**Date:** 2026-10-18  

---

### Core Questions & Direct Answers

#### 1. Does Model D beat persistence at any horizon?
**NO.** On the full September 2026 test set across all three forecast horizons (+1h, +2h, +3h), Model D **does not outperform Persistence**:
- **+1h:** Model D MAE = 4.978 mm vs Persistence MAE = 4.949 mm ($\Delta\text{MAE} = -0.028\text{ mm}$, $-0.57\%$, $p = 0.7726$).
- **+2h:** Model D MAE = 6.012 mm vs Persistence MAE = 5.815 mm ($\Delta\text{MAE} = -0.197\text{ mm}$, $-3.40\%$, $p = 0.1699$).
- **+3h:** Model D MAE = 7.049 mm vs Persistence MAE = 6.952 mm ($\Delta\text{MAE} = -0.097\text{ mm}$, $-1.39\%$, $p = 0.6187$).

All 95% bootstrap confidence intervals span zero ($[-0.21, +0.18]$ mm at +1h; $[-0.48, +0.08]$ mm at +2h; $[-0.45, +0.28]$ mm at +3h). River stage variation is predominantly gradual during non-flood periods; naive persistence is statistically indistinguishable from Model D overall.

#### 2. Does Model D beat M04 (Stage-Change Ridge without rainfall) at each horizon?
**YES, decisively.** When compared against the stage-only Ridge baseline on identical timestamps:
- **+1h ($N=612$):** Model D achieves **4.978 mm** vs M04's **5.648 mm** (Error reduction: **+0.670 mm**, **+11.87%**).
- **+2h ($N=604$):** Model D achieves **6.012 mm** vs M04's **7.690 mm** (Error reduction: **+1.678 mm**, **+21.82%**).
- **+3h ($N=599$):** Model D achieves **7.049 mm** vs M04's **9.886 mm** (Error reduction: **+2.838 mm**, **+28.70%**).

#### 3. Are the differences statistically supported after dependence handling and multiplicity correction?
**YES for Model D vs M04; NO for Model D vs Persistence.**
- For Model D vs M04, the error reduction is highly statistically significant across all 4 tested block lengths ($L \in \{3, 6, 12, 24\}$ hours). Under the Diebold-Mariano test with Harvey-Leybourne-Newey small sample correction, $p = 5.4 \times 10^{-7}$ at +1h, and $p < 10^{-14}$ at +2h and +3h. After full Benjamini-Hochberg FDR and Bonferroni corrections across the family of 12 hypothesis tests, $p_{\text{adj}} < 0.001$.
- *P-value Correction Note:* The previously reported empirical bootstrap claim of $p < 0.0001$ from 2,000 replicates has been corrected to the true plus-one empirical minimum ($p = 0.00050$), while noting that the analytical Studentized and DM-HLN tests confirm asymptotic $p < 10^{-6}$.

#### 4. Does the ablation isolate rainfall's incremental contribution?
**YES.** Under synchronized common-cohort evaluation ($N=612, 604, 599$), adding contemporary INSAT-3DS catchment rainfall to stage dynamics reduces error by 11.9% at +1h, 21.8% at +2h, and 28.7% at +3h. Even though rainfall missingness reduced the training cohort by 143 rows ($N=1079$ vs $1222$), the model with rainfall decisively outperforms the stage-only model trained on the larger cohort.

#### 5. Do Extra Trees or XGBoost outperform Model D?
**YES at +2h and +3h.**
- **At +1h:** Model D (4.978 mm), XGBoost (4.951 mm), and Extra Trees (4.969 mm) are in a statistical tie ($p > 0.70$).
- **At +2h:** Both Extra Trees (5.639 mm, $p_{\text{FDR}} = 0.0004$) and XGBoost (5.676 mm, $p_{\text{FDR}} = 0.0003$) significantly outperform Model D (6.012 mm) and beat Persistence (5.815 mm).
- **At +3h:** XGBoost (6.579 mm, $p_{\text{FDR}} = 0.0003$) significantly outperforms Model D (7.049 mm) and beats Persistence (6.952 mm). Extra Trees' edge ($p_{\text{raw}} = 0.0385$) becomes non-significant after FDR correction ($p_{\text{FDR}} = 0.0660$).

#### 6. Are the saved artifacts scientifically and computationally valid?
**YES.** All 33 model artifacts in `artifacts/phase1_integrity_repair_v2/` reproduce predictions with zero numerical discrepancy beyond float precision (`round_diff < 1e-7`). Preprocessing pipelines and estimators are completely isolated and free from the earlier in-place estimator-reuse bug.

#### 7. Which claims are safe to publish, and which must be removed or qualified?
- **SAFE TO PUBLISH:** Catchment rainfall provides proven, statistically significant incremental skill over stage dynamics alone in autoregressive Ridge forecasting (+11.9% to +28.7%). At longer lead times (+2h and +3h), tree ensembles (XGBoost) beat both linear Ridge and Persistence.
- **MUST BE REMOVED / QUALIFIED:** Any claim that Model D is the "overall best forecaster" or beats Persistence on the aggregate test set is false and must be retracted. Model D does not beat Persistence overall.

---

### Final Decision Table

| Claim | Original Reported Result | Independently Reproduced Result | Evidence Source | Status | Implication for Research Manuscript |
|---|---|---|---|---|---|
| **Estimator-Sharing Bug Fixed** | Estimator reuse fixed via `clone()` | Rebuilding produces distinct instances (`id1 != id2`); prototype remains unfitted; tests fail without fix | Source code inspection, `TestEstimatorCloningAndIsolation` | **VERIFIED** | Manuscript can state that pipeline isolation was verified by regression tests. |
| **All Test Suite Tests Pass** | 158 passed | 158 passed, 0 failed, 11 warnings in 35.1s | `audit_test_results.txt` | **VERIFIED** | Computational test integrity confirmed. |
| **Artifact Parity & Reloading** | 33 / 33 exact parity | 33 / 33 exact parity (`diff < 1e-12`, `round_diff < 1e-7`) | `artifact_validation.json` | **VERIFIED** | Saved model artifacts are reproducible and structurally sound. |
| **Model D vs Persistence (+1h)** | Indistinguishable ($p = 0.77$) | Model D 4.978 mm vs Pers 4.949 mm ($\Delta = -0.028$ mm, $p = 0.7726$) | `paired_predictions.csv`, `bootstrap_reanalysis.csv` | **VERIFIED** | Must state clearly that Model D does not beat persistence on aggregate hourly data. |
| **Model D vs Persistence (+2h, +3h)** | Indistinguishable ($p = 0.17, 0.62$) | +2h: $\Delta = -0.197$ mm ($p = 0.1699$); +3h: $\Delta = -0.097$ mm ($p = 0.6187$) | `paired_predictions.csv`, `bootstrap_reanalysis.csv` | **VERIFIED** | Must qualify that persistence is strongly competitive in non-flood baseflow regimes. |
| **Model D vs M04 (+1h)** | Model D beats M04 by +11.87% | Model D 4.978 mm vs M04 5.648 mm ($\Delta = +0.670$ mm, +11.87%) | `paired_predictions.csv`, `multiple_comparison_results.csv` | **VERIFIED** | Primary scientific finding: contemporary rainfall significantly improves stage-change forecasting. |
| **Model D vs M04 (+2h, +3h)** | Model D beats M04 by +21.82%, +28.70% | +2h: $\Delta = +1.678$ mm (+21.82%); +3h: $\Delta = +2.838$ mm (+28.70%) | `paired_predictions.csv`, `multiple_comparison_results.csv` | **VERIFIED** | Rainfall value expands with increasing forecast horizon. |
| **Empirical Bootstrap $p < 0.0001$** | $p < 0.0001$ claimed from 2,000 resamples | Resolution limit with $B=2000$ is $(0+1)/(2001) \approx 0.00050$. Asymptotic Studentized $p = 5.4 \times 10^{-9}$; DM $p = 5.4 \times 10^{-7}$ | `bootstrap_reanalysis.csv`, `diebold_mariano_reanalysis.csv` | **CORRECTED** | Replace empirical $p < 0.0001$ with empirical plus-one $p = 0.00050$, and report DM-HLN / asymptotic z-test for sub-$10^{-4}$ tails. |
| **XGBoost & Extra Trees at +2h** | Extra Trees and XGBoost beat Model D | Extra Trees beats Model D by $+0.373$ mm ($p_{\text{FDR}} = 0.0004$); XGBoost beats Model D by $+0.336$ mm ($p_{\text{FDR}} = 0.0003$) | `paired_predictions.csv`, `multiple_comparison_results.csv` | **VERIFIED** | Nonlinear multimodal tree ensembles are superior at +2h. |
| **Extra Trees at +3h** | Extra Trees beats Model D ($p = 0.044$) | Raw $p = 0.0385$; after Benjamini-Hochberg FDR correction, $p_{\text{FDR}} = 0.0660$ | `multiple_comparison_results.csv` | **QUALIFIED** | Extra Trees edge at +3h is marginally insignificant after multiplicity correction. |
| **XGBoost at +3h** | XGBoost beats Model D | XGBoost beats Model D by $+0.470$ mm (+7.14%), $p_{\text{DM}} = 0.0001$, $p_{\text{FDR}} = 0.0003$ | `multiple_comparison_results.csv` | **VERIFIED** | XGBoost is the definitively superior model at +3h. |
