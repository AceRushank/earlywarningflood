# Phase 1 Audit v4: Controlled Rainfall Ablation & Statistical Reconciliation Report

**Project:** Vishnupuri Barrage River Forecasting | Godavari River, Nanded  
**Audit Phase:** Audit v4 (Final Verification & Controlled Ablation)  
**Author:** Independent Statistical & ML Integrity Auditor  
**Date:** 2026-10-18  
**Repository Branch:** `phase1-ml-benchmarking`  
**Base Commit Hash:** `68e3a0ecad3b3fa7c69e4b45fe960d377ba4ef13`  
**Output Directory:** `artifacts/phase1_integrity_audit_v4/`  

---

## 1. Executive Summary & Audit Background

This audit resolves the confounding factors identified in earlier Phase 1 benchmark evaluations. In the prior v2 and v3 audits:
1. Model M04 (Stage-Change Ridge) was trained on $N = 1,222$ complete cases for stage features.
2. Model M05 (Model D: Stage-Change Ridge + Contemporary Rainfall) was trained on $N = 1,079$ complete cases because 143 observations in early August 2026 had missing satellite rainfall scans.
3. Consequently, the previously reported performance difference between M04 and M05 (+11.87% at +1h, +21.82% at +2h, +28.70% at +3h) conflated two distinct effects:
   - The effect of **training on a different temporal subset** (1,222 rows vs 1,079 rows).
   - The effect of **adding rainfall predictor features**.

To eliminate this confounding and isolate the true predictive contribution of rainfall, Audit v4 executed a **strictly matched controlled ablation** alongside a complete reconciliation of the inferential test families and statistical testing methodologies.

---

## 2. Controlled Rainfall Ablation Experiment

### 2.1 Experimental Protocol
For each forecast horizon ($+1\text{h}$, $+2\text{h}$, $+3\text{h}$), four distinct models were trained under identical algorithmic settings (`Ridge(alpha=10.0, random_state=42)`):
1. **M04-all:** Stage-only features (`FEAT_STAGE_DYN`), trained on all eligible rows where stage features and target are non-null ($N = 1,222$ at +1h).
2. **M05-all:** Stage-plus-rainfall features (`FEAT_STAGE_DYN + FEAT_RAIN_CONT`), trained on all rows where stage and rainfall features are non-null ($N = 1,079$ at +1h).
3. **M04-matched:** Stage-only features (`FEAT_STAGE_DYN`), trained on the **exact same 1,079 rows** used by M05-all.
4. **M05-matched:** Stage-plus-rainfall features, trained on those identical 1,079 rows (identical to M05-all).

**Controlled Invariants:**
- M04-matched and M05-matched share **100% identical training timestamps and targets**.
- Both pipelines fit an independent `StandardScaler()` exclusively on their respective training features.
- Both models are evaluated on the exact same synchronized test timestamps ($N = 612$ at +1h, $N = 604$ at +2h, $N = 599$ at +3h).
- **The only material difference between M04-matched and M05-matched is the presence of the rainfall features.**

### 2.2 Decomposition of Predictive Performance

The total apparent error reduction decomposes into:
$$\text{Total Apparent Gain} = \text{Cohort Effect (M04-all } - \text{ M04-matched)} + \text{Controlled Rainfall Gain (M04-matched } - \text{ M05-matched)}$$

Results on the synchronized September 2026 test cohort:

| Horizon | Training Rows (All / Matched) | M04-all MAE | M04-matched MAE | M05-matched MAE | Total Apparent Gain | Training Cohort Effect (% of Total) | Controlled Rainfall Effect (% of Total) | Controlled Rainfall Improvement (%) |
|---|---|---|---|---|---|---|---|---|
| **+1h** | 1,222 / 1,079 | 5.648 mm | 5.132 mm | 4.978 mm | +0.670 mm | **+0.516 mm (76.9%)** | **+0.154 mm (23.1%)** | **+3.01%** |
| **+2h** | 1,218 / 1,075 | 7.690 mm | 6.450 mm | 6.012 mm | +1.678 mm | **+1.240 mm (73.9%)** | **+0.438 mm (26.1%)** | **+6.80%** |
| **+3h** | 1,217 / 1,074 | 9.886 mm | 7.672 mm | 7.049 mm | +2.838 mm | **+2.214 mm (78.0%)** | **+0.624 mm (22.0%)** | **+8.13%** |

### 2.3 Statistical Inference on Controlled Rainfall Contribution (M04-matched vs M05-matched)

| Horizon | Controlled $\Delta$MAE | 95% Bootstrap CI ($L=6\text{h}$) | Empirical $p$ (Plus-One) | Diebold–Mariano $t$-stat | DM-HLN $p$-value | Statistical Significance ($\alpha = 0.05$) |
|---|---|---|---|---|---|---|
| **+1h** | **+0.154 mm (+3.01%)** | **[+0.070, +0.243] mm** | **$p = 0.0010$** | $t = 3.10$ | **$p = 0.0020$** | **STRICTLY SIGNIFICANT** |
| **+2h** | **+0.438 mm (+6.80%)** | **[+0.298, +0.586] mm** | **$p = 0.0005$** | $t = 5.52$ | **$p = 5.1 \times 10^{-8}$** | **STRICTLY SIGNIFICANT** |
| **+3h** | **+0.624 mm (+8.13%)** | **[+0.432, +0.827] mm** | **$p = 0.0005$** | $t = 5.68$ | **$p = 2.1 \times 10^{-8}$** | **STRICTLY SIGNIFICANT** |

### 2.4 Scientific Takeaways from Controlled Ablation
1. **Rainfall Incremental Value is Real and Verified:** Even when the training cohort is strictly held identical, adding contemporary rainfall significantly reduces stage prediction error at all three horizons ($p < 0.005$, all 95% CIs strictly positive).
2. **Prior Effect Size Was Overstated by 3–4$\times$:** In previous reports, the reported +11.87% to +28.70% gains were heavily confounded by training cohort differences. In reality, **over 70% of the apparent error gap was caused by training cohort shift** ($N=1222 \to 1079$), because the 143 rows with missing rainfall happened to introduce noisy training dynamics.
3. **True Incremental Predictive Value:** The isolated incremental predictive contribution of rainfall under controlled conditions is **+3.01% (+0.154 mm) at +1h**, **+6.80% (+0.438 mm) at +2h**, and **+8.13% (+0.624 mm) at +3h**.
4. **Epistemic Attribution:** This contribution represents *incremental predictive value* under this specific evaluation design and feature representation. It must not be labeled as an isolated causal hydrological relationship.

---

## 3. Reconciliation of the Statistical Test Family

### 3.1 Confirmatory Family vs Sensitivity Analyses
The v3 audit output recorded 48 bootstrap evaluations (4 comparisons $\times$ 3 horizons $\times$ 4 block lengths) alongside a statement that multiplicity correction was applied to a family of 12 tests. Audit v4 explicitly reconciles this structure:

1. **Unique Scientific Hypotheses ($M = 12$):**
   There are exactly 12 primary confirmatory scientific hypotheses, corresponding to:
   - 4 Model Pair Comparisons: (1) Model D vs Persistence, (2) Model D vs M04, (3) Model D vs Extra Trees, (4) Model D vs XGBoost.
   - 3 Forecast Horizons: $+1\text{h}, +2\text{h}, +3\text{h}$.
2. **Primary Confirmatory Tests ($L = 6\text{h}$):**
   The primary confirmatory test for each of the 12 hypotheses is pre-specified at block length $L = 6$ hours, representing the characteristic autocorrelation timescale of river stage dynamics at Vishnupuri Barrage.
   Multiplicity adjustments (Bonferroni and Benjamini–Hochberg False Discovery Rate) are rigorously applied across these **12 confirmatory hypotheses**.
3. **Non-Independent Sensitivity Checks (36 Tests):**
   The additional evaluations at $L \in \{3, 12, 24\}$ hours are secondary sensitivity analyses testing whether conclusions are fragile to block length tuning. They do not constitute independent scientific discoveries and must not be pooled into the confirmatory family.

### 3.2 Confirmatory Multiple Testing Table ($M = 12$)

| Hypothesis | Horizon | Comparison | Raw DM-HLN $p$ | Raw Boot $p$ (Plus-One) | Bonferroni $p$ (DM) | Benjamini–Hochberg FDR $p$ (DM) | Final Confirmatory Verdict |
|---|---|---|---|---|---|---|---|
| $H_{1,\text{Pers}}$ | +1h | Model D vs Persistence | 0.7902 | 0.7726 | 1.0000 | 0.8621 | **NOT SIGNIFICANT (Tie)** |
| $H_{1,\text{M04}}$ | +1h | Model D vs Stage Ridge (M04) | $5.4 \times 10^{-7}$ | 0.0005 | $6.5 \times 10^{-6}$ | **$< 0.0001$** | **SIGNIFICANT (Model D Superior)** |
| $H_{1,\text{M09}}$ | +1h | Model D vs Extra Trees | 0.9119 | 0.9230 | 1.0000 | 0.9119 | **NOT SIGNIFICANT (Tie)** |
| $H_{1,\text{M12}}$ | +1h | Model D vs XGBoost | 0.7227 | 0.7141 | 1.0000 | 0.8621 | **NOT SIGNIFICANT (Tie)** |
| $H_{2,\text{Pers}}$ | +2h | Model D vs Persistence | 0.1739 | 0.1699 | 1.0000 | 0.2608 | **NOT SIGNIFICANT (Tie)** |
| $H_{2,\text{M04}}$ | +2h | Model D vs Stage Ridge (M04) | $< 10^{-14}$ | 0.0005 | $< 10^{-13}$ | **$< 0.0001$** | **SIGNIFICANT (Model D Superior)** |
| $H_{2,\text{M09}}$ | +2h | Model D vs Extra Trees | 0.0002 | 0.0010 | 0.0023 | **0.0004** | **SIGNIFICANT (Extra Trees Superior)** |
| $H_{2,\text{M12}}$ | +2h | Model D vs XGBoost | 0.0001 | 0.0010 | 0.0016 | **0.0003** | **SIGNIFICANT (XGBoost Superior)** |
| $H_{3,\text{Pers}}$ | +3h | Model D vs Persistence | 0.6079 | 0.6187 | 1.0000 | 0.8106 | **NOT SIGNIFICANT (Tie)** |
| $H_{3,\text{M04}}$ | +3h | Model D vs Stage Ridge (M04) | $< 10^{-15}$ | 0.0005 | $< 10^{-14}$ | **$< 0.0001$** | **SIGNIFICANT (Model D Superior)** |
| $H_{3,\text{M09}}$ | +3h | Model D vs Extra Trees | 0.0385 | 0.0445 | 0.4621 | **0.0660** | **NOT SIGNIFICANT after FDR (Tie)** |
| $H_{3,\text{M12}}$ | +3h | Model D vs XGBoost | 0.0001 | 0.0005 | 0.0015 | **0.0003** | **SIGNIFICANT (XGBoost Superior)** |

---

## 4. Methodological Reconciliation of Statistical Testing

### 4.1 Resolution Limit of the Non-Parametric Bootstrap
A key correction in Audit v4 addresses empirical bootstrap p-value reporting. When resampling $B = 2,000$ blocks under the centered null hypothesis $H_0: \mathbb{E}[d_t] = 0$:
- If zero exceedances occur ($k = 0$), the raw fraction $k/B = 0/2000 = 0.0$.
- Under established bootstrap theory (Davison & Hinkley 1997), the valid empirical p-value must be reported using the plus-one estimator:
  $$p_{\text{plus-one}} = \frac{k + 1}{B + 1} = \frac{0 + 1}{2001} \approx 0.0004998 \approx 0.00050$$
- It is mathematically invalid to claim an empirical bootstrap $p < 0.0001$ from 2,000 resamples without fitting a parametric tail model.
- However, when the asymptotic Studentized distribution is evaluated ($z = \text{diff} / \text{SE}_{\text{boot}}$), the resulting asymptotic tail probability is $p = 5.4 \times 10^{-9}$ at +1h.
- Similarly, the Diebold–Mariano test with Harvey–Leybourne–Newey small-sample modification yields $t = 5.07$ ($p = 5.4 \times 10^{-7}$).
- **Resolution:** Audit v4 reports the exact plus-one empirical p-value ($p = 0.00050$) and explicitly identifies analytical p-values when sub-$10^{-4}$ tails are reported.

### 4.2 Diebold–Mariano Test Verification
- **Loss Differential:** Defined symmetrically as $d_t = |e_{\text{baseline},t}| - |e_{\text{candidate},t}|$. Positive values indicate that the candidate model has smaller absolute forecast error.
- **Autocovariance Estimation:** Evaluated using a rectangular lag window of $h-1$ lags (Diebold & Mariano 1995) and Bartlett weighting to account for serial correlation induced by multi-step forecast horizons.
- **Small-Sample Correction:** Evaluated using the Harvey, Leybourne, and Newey (1997) factor:
  $$\text{DM}^* = \text{DM} \times \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}$$
  with $p$-values calculated from Student's $t$ distribution with $n-1$ degrees of freedom.
- **Agreement:** In all 12 confirmatory comparisons, the Diebold–Mariano HLN test and the moving block bootstrap ($L=6\text{h}$) reach identical statistical conclusions.

---

## 5. Artifact Provenance & Test Suite Verification

1. **Artifact Deserialization & Prediction Parity:**
   All 33 model `.joblib` files in `artifacts/phase1_integrity_repair_v2/` were loaded into a clean Python session. Predictions were reconstructed from the original feature pipeline. Maximum observed prediction discrepancy across all models and horizons was $< 10^{-12}$ on raw floats and $< 10^{-7}$ against 5-decimal rounded benchmark tables.
2. **Regression Test Integrity:**
   The regression test suite in `river_forecasting/tests/test_phase1_ml_pipeline.py` (`TestEstimatorCloningAndIsolation`) was audited. It was confirmed that without `clone()`, the tests fail on object identity (`id(est1) != id(est2)`), fitted state inspection (`check_is_fitted(est2)`), and coefficient mutation (`assert_allclose(p_std.coef_, std_coef_initial)`).
3. **Repository Test Results:**
   Full execution of `pytest river_forecasting/tests/` passed 158 tests, 0 failed, 11 warnings in 35.1 seconds (logged in `audit_test_results.txt`).

---

## 6. Deliverable Files in `artifacts/phase1_integrity_audit_v4/`

- [`controlled_ablation.csv`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v4/controlled_ablation.csv): Exact sample counts, MAEs, RMSEs, biases, cohort effects, and isolated rainfall gains for +1h, +2h, +3h.
- [`final_paired_comparisons.csv`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v4/final_paired_comparisons.csv): 12 confirmatory hypotheses evaluated with unrounded prediction errors, moving-block bootstrap CIs, DM-HLN tests, Bonferroni, and Benjamini–Hochberg FDR adjustments.
- [`statistical_family_reconciliation.csv`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v4/statistical_family_reconciliation.csv): Reconciliation of 12 confirmatory tests vs 36 sensitivity checks across block lengths $L \in \{3, 6, 12, 24\}$ hours.
- [`final_claims_matrix.md`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v4/final_claims_matrix.md): Definitive classification of every claim as VERIFIED, QUALIFIED, UNSUPPORTED, or UNRESOLVED.
- [`audit_v4_report.md`](file:///c:/Users/aceru/floodsystem/artifacts/phase1_integrity_audit_v4/audit_v4_report.md): This report.
