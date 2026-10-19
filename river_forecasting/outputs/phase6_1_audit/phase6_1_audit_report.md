# Phase 6.1 — Forensic Audit Report: Medium-Range Upstream Telemetry Experiment

**Audit Target:** Phase 6 Medium-Range Upstream Signal and Flood-Wave Routing Experiment  
**Target Station:** Vishnupuri Barrage, Godavari River, Nanded, Maharashtra (`sw_006_0016`)  
**Upstream Station:** Mudgal Barrage, Godavari River, Pathri/Manwath, Parbhani (`sw_006_0017`)  
**Horizons Audited:** $+18\text{h}$, $+24\text{h}$, and $+36\text{h}$  
**Audit Directory:** [`river_forecasting/outputs/phase6_1_audit/`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/)  
**Test Suite:** [`river_forecasting/tests/test_phase6_1_audit.py`](file:///c:/Users/aceru/floodsystem/river_forecasting/tests/test_phase6_1_audit.py) (6/6 tests passing)

---

## Executive Summary & Audit Verdict

### Final Decision: **Implementation Defect in Feature Differencing Pipeline, Compounded by Evaluation Regime Decoupling; Yet the Underlying Hydrological Rejection of Raw Upstream Stage Telemetry Remains Valid.**

This forensic audit was commissioned to determine why upstream-augmented Ridge regression models in Phase 6 reported catastrophic errors of **268.68 mm to 436.79 mm MAE**, whether those errors reflect genuine physical dynamics or pipeline bugs, and whether the reported metrics, statistical tests, and hydrological conclusions are reproducible.

The audit reveals two intertwined mechanisms:

1. **The Primary Implementation Defect (Raw Gauge Telemetry Differencing Without Spike Filtering or Clamping):**
   - At timestamp `2026-09-08 02:00:00`, authentic CWC/MWRD hourly telemetry at Mudgal Barrage recorded an isolated single-hour gauge glitch of **393.634 m MSL** (an unphysical instantaneous jump of $+10.35\text{ m}$ from 383.28 m at 01:00 and returning to 383.28 m at 03:00).
   - In `run_phase6_medium_range.py`, multi-hour differences were constructed via simple series shifting (`dH_m_6h = stage_m_lag1 - stage_m_lag7`) without physical rate-of-change validation, despiking, or robust clipping.
   - When the differencing window passed across `2026-09-08 02:00:00` (e.g. at origin `2026-09-08 09:00:00`), `dH_m_6h` equaled **$-10.37\text{ m}$**. In the training data, the standard deviation of `dH_m_6h` was only $0.0357\text{ m}$ (35.7 mm). Consequently, this single feature reached an extreme standardized value of **$z = -290.6$ standard deviations**.
   - Multiplied by the fitted Ridge coefficient ($+0.0348$), this single feature injected **$-10.10\text{ m}$** into the predicted stage change, producing an unconstrained point prediction of $\Delta H = -10.13\text{ m}$ at $+18\text{h}$, $-14.52\text{ m}$ at $+24\text{h}$, and $-17.57\text{ m}$ at $+36\text{h}$.
   - Because linear Ridge regression contains no physical bounds or soft clamping, these isolated spikes inflated the reported mean absolute error by **$\approx 90\text{ mm}$ at $+18\text{h}$, $\approx 113\text{ mm}$ at $+24\text{h}$, and $\approx 144\text{ mm}$ at $+36\text{h}$**.

2. **The Systematic Phenomenon (Why Upstream Models Still Fail Even Without the Glitch):**
   - When all rows corrupted by the single-hour telemetry glitch are completely removed, Ridge Local+Upstream (Config D) **still reports an MAE of 178.91 mm at $+18\text{h}$, 224.30 mm at $+24\text{h}$, and 292.87 mm at $+36\text{h}$** (vastly worse than Persistence at 16.52 mm and Local Stage at 21.34 mm).
   - Furthermore, **XGBoost (Config H)**, which operates on tree splits that naturally saturate and are **completely immune** to the $-290\sigma$ outlier, **still reports an MAE of 76.68 mm to 89.40 mm** (which is **$3.3\times$ to $4.6\times$ worse than persistence**).
   - The root cause is a genuine physical regime mismatch: On September 18, Mudgal experienced an upstream flood rise of $+0.80\text{ m}$. In July (monsoon training), high open-river discharge meant an upstream rise always produced a massive downstream rise (+0.5 to +1.2 m). But in September, Vishnupuri was maintained virtually flat at Full Reservoir Level (354.50 m MSL) across a tiny 27.8 cm band. The models predicted a $+0.88\text{ m}$ surge (XGBoost) and $+1.5\text{ to }+2.4\text{ m}$ surge (Ridge) that **never materialized downstream**.
   - Therefore, while the 268–437 mm error numbers were exaggerated by an implementation defect in telemetry differencing, **the substantive conclusion of Phase 6 remains valid**: raw upstream stage telemetry degrades medium-range forecasts during controlled reservoir regimes and provides zero useful incremental value over local persistence.

---

## 1. Independent Recomputation & Reconciliation

All metrics reported in Phase 6 were recalculated directly from the raw row-level predictions stored in `river_forecasting/outputs/phase6_medium_range/row_level_predictions.csv` ($N = 2,592$ hourly grid points).

### Reconciliation Results

Every metric in `horizon_metrics.csv` and `feature_ablation_results.csv` was verified against raw row calculations:
- **Discrepancies Found:** **0** (Zero discrepancies).
- **Target Unit:** Millimetres of river stage difference ($\Delta H_v(t + h) = H_v(t + h) - H_v(t)$).
- **Prediction Unit:** Millimetres of predicted stage change.
- **Sample Counts:** Exactly match ($N = 188$ matched valid origins for Synchronized Test Cohort at $+18\text{h}$ and $+24\text{h}$; $N = 176$ origins at $+36\text{h}$).
- Complete reconciliation audit logged in [`river_forecasting/outputs/phase6_1_audit/recomputed_metrics.csv`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/recomputed_metrics.csv).

### Summary Recomputed Table (Synchronized Test Cohort, Sept 8–22, 2026)

| Horizon | Config | Recomputed MAE (mm) | Reported MAE (mm) | Diff (mm) | Recomputed RMSE (mm) | Recomputed Bias (mm) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+18h** | Config_A (Persistence) | 16.63 | 16.63 | 0.00 | 23.33 | +1.22 | EXACT MATCH |
| | Config_B (Local Stage) | 21.56 | 21.56 | 0.00 | 28.53 | -14.47 | EXACT MATCH |
| | Config_C (Local + Rain) | 24.35 | 24.35 | 0.00 | 32.55 | -16.48 | EXACT MATCH |
| | Config_D (Local + Upstream) | 268.68 | 268.68 | 0.00 | 363.85 | +237.11 | EXACT MATCH |
| | Config_E (Local + Rain + Upstream) | 267.05 | 267.05 | 0.00 | 361.64 | +235.62 | EXACT MATCH |
| | Config_F (Lagged Proxy) | 20.30 | 20.30 | 0.00 | 27.53 | -10.95 | EXACT MATCH |
| | Config_G (Ridge Upstream) | 268.68 | 268.68 | 0.00 | 363.85 | +237.11 | EXACT MATCH |
| | Config_H (XGBoost Upstream) | 76.68 | 76.68 | 0.00 | 98.71 | +44.75 | EXACT MATCH |
| **+24h** | Config_A (Persistence) | 19.98 | 19.98 | 0.00 | 27.67 | +2.34 | EXACT MATCH |
| | Config_B (Local Stage) | 22.69 | 22.69 | 0.00 | 29.80 | -15.42 | EXACT MATCH |
| | Config_C (Local + Rain) | 25.12 | 25.12 | 0.00 | 33.45 | -17.20 | EXACT MATCH |
| | Config_D (Local + Upstream) | 337.58 | 337.58 | 0.00 | 449.27 | +306.90 | EXACT MATCH |
| | Config_H (XGBoost Upstream) | 87.25 | 87.25 | 0.00 | 114.71 | +47.53 | EXACT MATCH |
| **+36h** | Config_A (Persistence) | 27.45 | 27.45 | 0.00 | 36.75 | +5.31 | EXACT MATCH |
| | Config_B (Local Stage) | 27.44 | 27.44 | 0.00 | 37.05 | -18.23 | EXACT MATCH |
| | Config_C (Local + Rain) | 28.53 | 28.53 | 0.00 | 38.38 | -19.46 | EXACT MATCH |
| | Config_D (Local + Upstream) | 436.79 | 436.79 | 0.00 | 569.21 | +407.96 | EXACT MATCH |
| | Config_H (XGBoost Upstream) | 89.40 | 89.40 | 0.00 | 118.06 | +46.88 | EXACT MATCH |

---

## 2. Tracing the Prediction Pipeline & Root Cause of Extreme Predictions

### Model Target Specification
All models in Phase 6 predict the **stage change** $\Delta H_v(t + h) = H_v(t + h) - H_v(t)$ over horizon $h$. Absolute future stage is reconstructed at inference time via:
$$\hat{H}_v(t + h) = H_v(t) + \Delta\hat{H}_v(t + h)$$
This formulation was verified in `run_phase6_medium_range.py` lines 116, 279–280.

### Mathematical Anatomy of the Ridge Error Blowout

Inspection of the fitted Ridge pipeline (`StandardScaler` + `Ridge(alpha=100.0)`) revealed the following coefficients and training scalers for `Config_D_Local_Stage_Upstream__18h`:
- Intercept: $+0.0509\text{ m}$ ($+50.9\text{ mm}$)
- `stage_vishnupuri`: mean = $351.57\text{ m}$, scale = $1.545\text{ m}$, coef = $-0.0378$
- `dH_m_6h`: mean = $+0.0039\text{ m}$, scale = **$0.0357\text{ m}$** (35.7 mm), coef = **$+0.0348$**

#### The Sept 8 Telemetry Spike Breakdown (Row 2049: `2026-09-08 09:00:00`)
1. **Raw Telemetry Glitch:**  
   At $t = \text{2026-09-08 02:00:00}$, Mudgal stage spiked to **393.634 m MSL** (a $+10.353\text{ m}$ instantaneous jump over 01:00).
2. **Lag Calculation at Origin `2026-09-08 09:00:00`:**  
   `stage_m_lag1` (at $t-1\text{h} = \text{08:00}$) was $383.264\text{ m}$.  
   `stage_m_lag7` (at $t-7\text{h} = \text{02:00}$) was $393.634\text{ m}$.  
   Therefore, $\text{dH\_m\_6h} = 383.264 - 393.634 = \mathbf{-10.370\text{ m}}$!
3. **Z-Score Explosion:**  
   $$z(\text{dH\_m\_6h}) = \frac{-10.370 - 0.0039}{0.0357} = \mathbf{-290.608\text{ standard deviations}}$$
4. **Contribution to Prediction:**  
   $$\text{contrib} = z \times \beta = -290.608 \times 0.0348 = \mathbf{-10.101\text{ meters}}$$
5. **Final Prediction:**  
   $$\Delta\hat{H}_{\text{Ridge}} = -10.132\text{ meters}$$  
   (Predicting that Vishnupuri stage would drop from 354.69 m to 344.56 m).  
   Because actual $\Delta H$ was $-0.003\text{ m}$, the absolute error for this single row was **$10,129\text{ mm}$**!

Across $+18\text{h}$, $+24\text{h}$, and $+36\text{h}$, exactly 4 rows were corrupted as each differencing lag passed through `2026-09-08 02:00:00`.

### Purified Cohort Analysis (Removing Telemetry Spike Rows)

To determine whether the failure of upstream telemetry was solely an artifact of this single spike, we audited a purified test cohort excluding the 4 corrupted rows:

| Horizon | Model | Full Test Cohort MAE (mm) | Purified Cohort MAE (mm) | Persistence MAE (mm) | Ratio vs Persistence |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **+18h** | Persistence (Config A) | 16.63 | **16.52** | 16.52 | $1.0\times$ |
| | Local Stage (Config B) | 21.56 | **21.34** | 16.52 | $1.3\times$ |
| | Ridge Local+Upstream (Config D) | 268.68 | **178.91** | 16.52 | **$10.8\times$** |
| | XGBoost Local+Upstream (Config H) | 76.68 | **77.24** | 16.52 | **$4.7\times$** |
| **+24h** | Persistence (Config A) | 19.98 | **19.89** | 19.89 | $1.0\times$ |
| | Local Stage (Config B) | 22.69 | **22.66** | 19.89 | $1.1\times$ |
| | Ridge Local+Upstream (Config D) | 337.58 | **224.30** | 19.89 | **$11.3\times$** |
| | XGBoost Local+Upstream (Config H) | 87.25 | **87.80** | 19.89 | **$4.4\times$** |
| **+36h** | Persistence (Config A) | 27.45 | **27.44** | 27.44 | $1.0\times$ |
| | Local Stage (Config B) | 27.44 | **27.24** | 27.44 | $1.0\times$ |
| | Ridge Local+Upstream (Config D) | 436.79 | **292.87** | 27.44 | **$10.7\times$** |
| | XGBoost Local+Upstream (Config H) | 89.40 | **89.99** | 27.44 | **$3.3\times$** |

**Crucial Finding:** Even after completely eliminating the telemetry glitch, **Ridge Local+Upstream remains $>10\times$ worse than persistence**, and **XGBoost remains $3.3\times$ to $4.7\times$ worse than persistence**. The rejection of upstream telemetry is therefore **hydrologically robust** and not an illusion caused by gauge noise.

---

## 3. Temporal Integrity & Boundary Audit

The audit verified five fundamental temporal integrity constraints:

1. **Origin-to-Target Alignment:**  
   Every row in `row_level_predictions.csv` has target timestamp strictly equal to $t_{\text{origin}} + H$. Zero misalignments across 2,399 evaluated pairs.
2. **Operational 1-Hour Latency Buffer:**  
   All upstream Mudgal features are strictly lagged by $\ge 1\text{ hour}$ (`stage_m_lag1` is at $t-1\text{h}$; differences use $t-2\text{h}$ to $t-25\text{h}$). Zero contemporaneous or future upstream stage values enter the feature vectors.
3. **Chronological Split Boundary Separation:**  
   - Training split ended at `2026-08-10 23:00 - H` (e.g. `2026-08-10 05:00` for $+18\text{h}$).
   - Validation split began at `2026-08-11 00:00`.
   - The buffer separation was $\ge H + 1$ hours, guaranteeing zero overlap between training targets and validation origins.
4. **Duplicate Timestamps & Grid Uniformity:**  
   The hourly grid has exactly 2,592 hourly steps with zero duplicate timestamps.
5. **Missing Interval Handling:**  
   Missing gauge measurements produce NaN target differences; no interpolation or zero-filling was used.
   
Complete audit logged in [`river_forecasting/outputs/phase6_1_audit/temporal_integrity_audit.csv`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/temporal_integrity_audit.csv).

---

## 4. Statistical Testing & Significance Audit

### Sign Convention of $\Delta\text{MAE}$
In `paired_comparisons.csv`, the reported metric is:
$$\Delta\text{MAE} = \text{MAE}_{\text{baseline}} - \text{MAE}_{\text{candidate}}$$
- **Positive $\Delta\text{MAE}$:** The candidate model achieves lower error (improvement).
- **Negative $\Delta\text{MAE}$:** The candidate model achieves higher error (degradation).

### Moving-Block Bootstrap Recomputation ($B=1,000, b=24\text{h}$)
Because hourly errors exhibit significant serial dependence (lag-1 autocorrelation $\approx 0.64$), standard i.i.d. bootstrap or paired t-tests are invalid. A moving-block bootstrap with block length $b = 24\text{ hours}$ was recomputed:

- **Config D (Ridge Upstream) vs Config B (Local Stage):**
  - $+18\text{h}$: $\Delta\text{MAE} = -247.12\text{ mm}$ ($95\%\text{ CI: } [-288.72, -206.13]$, $p < 0.001$).
  - $+24\text{h}$: $\Delta\text{MAE} = -314.89\text{ mm}$ ($95\%\text{ CI: } [-368.79, -263.26]$, $p < 0.001$).
  - $+36\text{h}$: $\Delta\text{MAE} = -409.35\text{ mm}$ ($95\%\text{ CI: } [-480.05, -342.31]$, $p < 0.001$).
- **Config H (XGBoost Upstream) vs Persistence Baseline:**
  - $+18\text{h}$: $\Delta\text{MAE} = -60.05\text{ mm}$ ($95\%\text{ CI: } [-81.42, -41.20]$, $p < 0.001$).
  - $+24\text{h}$: $\Delta\text{MAE} = -67.27\text{ mm}$ ($95\%\text{ CI: } [-90.15, -46.88]$, $p < 0.001$).
  - $+36\text{h}$: $\Delta\text{MAE} = -61.96\text{ mm}$ ($95\%\text{ CI: } [-82.94, -43.10]$, $p < 0.001$).

**Zero** paired comparisons demonstrate any positive incremental value from upstream telemetry. Complete recomputations logged in [`river_forecasting/outputs/phase6_1_audit/statistical_bootstrap_audit.csv`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/statistical_bootstrap_audit.csv).

---

## 5. Physical Claims Audit: Separating Measured Data from Inferences

A critical requirement of Phase 6.1 is separating **empirically measured telemetry** from **inferred operational explanations**:

1. **What is Empirically Measured:**
   - In July (monsoon training), cross-correlation between Mudgal upstream rate of change and Vishnupuri downstream rate of change peaked at $r = +0.771$ at $\tau = 15\text{ hours}$.
   - In September (test set), the overall correlation collapsed to $r = +0.11$.
   - On September 18, Mudgal stage rose by $+0.80\text{ m}$ (from 383.02 m to 384.04 m MSL).
   - At Vishnupuri, stage over that entire period varied by only **27.8 cm** (354.42 m to 354.70 m MSL). During the 18–36 hours following Mudgal's rise, Vishnupuri stage changed by only $-0.003\text{ m}$ to $-0.021\text{ m}$.
   - The test set contained only 26 single-hour rising-stage blips, with a maximum rise of 3.9 cm. There were zero major flood waves passing Vishnupuri in September.
2. **What is an Inferred Explanation:**
   - The claim that "barrage radial gates were throttled or closed to impound post-monsoon storage" is a **plausible operational inference** based on standard Maharashtra Water Resources Department (MWRD) barrage guidelines for Full Reservoir Level maintenance (354.50 m MSL).
   - However, **no direct log of radial gate openings (aperture heights in metres) or spillway discharge schedules was included in the dataset**.
   - Therefore, scientific integrity requires stating that the **telemetry decoupling is an empirical fact**, while the gate closure explanation is a **physically consistent hypothesis**, not a measured telemetry variable.

---

## 6. Recommendations for Engineering Standards

To prevent unconstrained linear models from blowing up on raw telemetry in future research phases:

1. **Mandatory Input Despiking / Rate-of-Change Filter:**  
   Implement a physical sanity gate on raw stage inputs: $|\Delta H_{\text{hourly}}| \le 2.0\text{ m/h}$. Any reading exceeding physical Godavari rate-of-change limits must be flagged as an instrument glitch and routed to fallback.
2. **Prediction Bounding / Soft Clamping:**  
   Clamp medium-range stage change predictions to physical maximum limits: e.g. $|\Delta H_{18h}| \le 2.5\text{ m}$. No linear extrapolation should be permitted to predict $-17\text{ m}$ in a 5-metre deep reservoir pool.
3. **Preserve Operational RA-AG v2 Architecture:**  
   Maintain RA-AG v2 Hard (local stage + rainfall) as the sole champion model for live warnings. Upstream stage telemetry from Mudgal Barrage must **not** be integrated into operational forecasting without coupled real-time gate aperture and reservoir storage telemetry.

---

## Deliverables Index

All Phase 6.1 deliverables are preserved and available in [`river_forecasting/outputs/phase6_1_audit/`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/):
1. [`phase6_1_audit_report.md`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/phase6_1_audit_report.md): This comprehensive forensic report.
2. [`recomputed_metrics.csv`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/recomputed_metrics.csv): Full reconciliation table of all 48 metric combinations.
3. [`extreme_prediction_audit.csv`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/extreme_prediction_audit.csv): Row-level mathematical trace of extreme predictions.
4. [`pipeline_coefficients_audit.csv`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/pipeline_coefficients_audit.csv): Fitted Ridge coefficients, intercepts, and standardization parameters.
5. [`statistical_bootstrap_audit.csv`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/statistical_bootstrap_audit.csv): Recomputed moving-block bootstrap hypothesis tests and 95% CIs.
6. [`temporal_integrity_audit.csv`](file:///c:/Users/aceru/floodsystem/river_forecasting/outputs/phase6_1_audit/temporal_integrity_audit.csv): Verification of timestamp alignment, latency, and split boundaries.
7. [`test_phase6_1_audit.py`](file:///c:/Users/aceru/floodsystem/river_forecasting/tests/test_phase6_1_audit.py): Reproducible automated pytest suite (6/6 passing).
