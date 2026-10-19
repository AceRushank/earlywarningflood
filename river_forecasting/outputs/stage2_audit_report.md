# Godavari River Stage Forecasting — Stage 2 Audit Report

**Date:** 2026-10-09  
**Target Station:** Vishnupuri Barrage (Shankar Sagar Jalashay), Nanded, Maharashtra (`19.126944° N, 77.280833° E`)  
**Scope:** Methodological and Numerical Audit of Stage 2 Feature Engineering, Anti-Leakage Guarantees, Baseline Evaluations, and Rapid-Rise Regimes.

---

## 1. Direct Recalculation & Metric Verification

All baseline metrics were independently recalculated directly from the out-of-sample predictions file [`river_forecasting/outputs/baseline_predictions_test.csv`](baseline_predictions_test.csv) across all models and horizons.

### Recalculated Verification Table:

| Model | Horizon | Valid Pairs ($N$) | MAE (m) | RMSE (m) | Max AE (m) | NSE | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | **1-hour** | 163 | **0.0383** | **0.0706** | **0.4110** | **0.9248** | Verified (Exact Match) |
| **Water-Level-Only (Ridge)** | **1-hour** | 163 | **0.0466** | **0.0843** | **0.4784** | **0.8926** | Verified (Corrected MaxAE) |
| **Water-Level-Only (GB)** | **1-hour** | 163 | **0.0412** | **0.0730** | **0.4851** | **0.9196** | Verified (Exact Match) |
| **Persistence Baseline** | **2-hour** | 163 | **0.0592** | **0.1099** | **0.5250** | **0.8159** | Verified (Exact Match) |
| **Water-Level-Only (Ridge)** | **2-hour** | 163 | **0.0702** | **0.1272** | **0.6466** | **0.7534** | Verified (Exact Match) |
| **Water-Level-Only (GB)** | **2-hour** | 163 | **0.0686** | **0.1184** | **0.7003** | **0.7864** | Verified (Exact Match) |
| **Persistence Baseline** | **3-hour** | 160 | **0.0705** | **0.1279** | **0.6450** | **0.7513** | Verified (Exact Match) |
| **Water-Level-Only (Ridge)** | **3-hour** | 160 | **0.0771** | **0.1480** | **0.7696** | **0.6671** | Verified (Exact Match) |
| **Water-Level-Only (GB)** | **3-hour** | 160 | **0.0803** | **0.1382** | **0.7530** | **0.7096** | Verified (Exact Match) |

*Note on Corrected Metric:* In the initial markdown summary, Ridge 1h Maximum Absolute Error was displayed as `0.4079` m due to a table formatting discrepancy; the exact mathematical value computed directly from `baseline_predictions_test.csv` is **0.4784 m** (occurring at timestamp `2024-08-16 00:00:00` where stage rose from 354.513 m to 354.924 m).

---

## 2. Temporal Leakage & Split Audit

### 2.1 Feature Availability at Issue Time $t$
- **Autoregressive Lags:** In [`river_forecasting/src/feature_pipeline.py`](../src/feature_pipeline.py), stage lags ($H_{t-1}, H_{t-2}, H_{t-3}, H_{t-4}, H_{t-6}$) are constructed via positive shifts (`.shift(lag)`). Row index $i$ only accesses rows $i - \text{lag}$.
- **Rolling Statistics:** Windows ($3h, 6h, 12h$) are computed using backward-looking windows strictly ending at issue time $t$.
- **No Future Lookahead:** Verified via unit test [`test_no_future_leakage_in_features`](../tests/test_pipeline.py): modifying future observations after time $t$ results in zero change to any feature vector $\le t$.
- **Feature Scaling:** StandardScaler is fitted strictly on the training partition (`train_df`) and subsequently applied via `.transform()` to validation and test data. No test statistics leak into model normalization.

### 2.2 Missing-Hour Handling
- **Origin Timestamps:** If an observation is missing at origin $t$ (`is_observed == False`), the row is marked `is_valid_forecast_origin = False` and excluded from issuing forecasts.
- **Antecedent History:** Forward-fill (`.ffill(limit=3)`) is restricted to historical antecedent lags strictly prior to $t$ so that transient 1-2 hour network dropouts do not invalidate subsequent issue times. Gaps exceeding 3 hours leave features as `NaN`, disqualifying the origin.
- **Target Observations:** Future targets ($Y_{t+h} = H_{t+h}$) are strictly ground truth observations without imputation. Missing future target hours evaluate to `NaN` and are strictly excluded from evaluation error denominators ($N=163$ for 1h/2h, $N=160$ for 3h).

### 2.3 Split Boundary Leakage & Embargo Buffer
- **Identified Risk:** In standard chronological splitting without a buffer, the final training samples (e.g. at 21:00 and 22:00) have multi-hour targets ($t+2, t+3$) that fall into the validation time span (23:00 and 00:00).
- **Corrective Mitigation:** An **embargo buffer** (`horizon_embargo_hours = 3`) was implemented in [`chronological_split()`](../src/feature_pipeline.py).
  - Train origins now end at `2024-08-07 20:00:00`, so the 3h target reaches `2024-08-07 23:00:00` (the first validation origin) without crossing into validation.
  - Validation origins end at `2024-08-15 04:00:00`, so the 3h target reaches `2024-08-15 07:00:00` (the first test origin) without crossing into test.
  - This completely eliminates cross-split target overlap.

---

## 3. Investigation of the "Rapid-Rise" Improvement

### 3.1 Methodological Problem Identified
The initial Stage 2 summary reported a "63.8% error reduction during rapid rises" ($N=11$). An in-depth audit revealed two distinct methodological issues:
1. **Inclusion of Recessions:** The condition used `feat_stage_diff_1h.abs() >= 0.15`. Using absolute value selected **6 steep falls/recessions** ($\le -0.18$ m/hr) and only **5 steep rises** ($\ge +0.15$ m/hr). Calling this set "rapid rise" was mislabeled; it was actually **antecedent volatility / rapid change**.
2. **Antecedent vs. Future Surge Distinction:**
   - **Antecedent Rise ($H_t - H_{t-1} \ge +0.15$ m/hr):** The river was *already surging* prior to issue time $t$. This information **is available at prediction time**.
   - **Future Surge Onset ($H_{t+1} - H_t \ge +0.15$ m):** The river *will surge* between $t$ and $t+1$. Conditioning an evaluation on this set requires knowing the future ground truth, which is a post-hoc diagnostic, not an online operational filter.

### 3.2 Breakdown of Regimes in Test Data

#### Regime A: Antecedent Rapid Rise ($H_t - H_{t-1} \ge +0.15$ m/hr, $N=5$)
*All features available at forecast issue time $t$.*
- **Timestamps:**
  1. `2024-08-16 01:00:00` ($H_t = 354.924$ m, $\Delta H_t = +0.411$ m/hr, Target $H_{t+1} = 355.038$ m)
  2. `2024-08-21 01:00:00` ($H_t = 354.267$ m, $\Delta H_t = +0.252$ m/hr, Target $H_{t+1} = 354.480$ m)
  3. `2024-08-21 02:00:00` ($H_t = 354.480$ m, $\Delta H_t = +0.213$ m/hr, Target $H_{t+1} = 354.660$ m)
  4. `2024-08-21 03:00:00` ($H_t = 354.660$ m, $\Delta H_t = +0.180$ m/hr, Target $H_{t+1} = 354.816$ m)
  5. `2024-08-21 04:00:00` ($H_t = 354.816$ m, $\Delta H_t = +0.156$ m/hr, Target $H_{t+1} = 354.618$ m)
- **Model Errors on Regime A (1h Horizon):**
  - **Persistence Baseline:** $\text{MAE} = 0.1722\text{ m}$, $\text{RMSE} = 0.1757\text{ m}$, $\text{MaxAE} = 0.2130\text{ m}$
  - **Ridge:** $\text{MAE} = 0.2085\text{ m}$, $\text{RMSE} = 0.2219\text{ m}$, $\text{MaxAE} = 0.3130\text{ m}$
  - **Gradient Boosting:** $\mathbf{MAE = 0.0574\text{ m}}$, $\mathbf{RMSE = 0.0837\text{ m}}$, $\mathbf{MaxAE = 0.1775\text{ m}}$
- **Finding:** When an ongoing surge is already evident at the gauge, Gradient Boosting achieves a **66.7% MAE reduction** over persistence by learning momentum continuity ($\Delta H_{t-1}, \Delta H_{t-2}$).

#### Regime B: Antecedent Volatility / Rapid Change ($|H_t - H_{t-1}| \ge 0.15$ m/hr, $N=11$)
*Includes the 5 rises above plus 6 steep recessions (e.g. gates opening or surge passing).*
- **Timestamps:**
  `2024-08-15 22:00` (-0.180m), `2024-08-16 01:00` (+0.411m), `2024-08-16 03:00` (-0.186m), `2024-08-16 21:00` (-0.231m), `2024-08-17 04:00` (-0.210m), `2024-08-19 16:00` (-0.216m), `2024-08-21 01:00` (+0.252m), `2024-08-21 02:00` (+0.213m), `2024-08-21 03:00` (+0.180m), `2024-08-21 04:00` (+0.156m), `2024-08-21 05:00` (-0.198m).
- **Model Errors on Regime B (1h Horizon):**
  - **Persistence:** $\text{MAE} = 0.1143\text{ m}$, $\text{RMSE} = 0.1294\text{ m}$
  - **Ridge:** $\text{MAE} = 0.1531\text{ m}$, $\text{RMSE} = 0.1716\text{ m}$
  - **Gradient Boosting:** $\mathbf{MAE = 0.0414\text{ m}}$, $\mathbf{RMSE = 0.0647\text{ m}}$

#### Regime C: Future Rapid Surge Onset ($H_{t+1} - H_t \ge +0.15$ m, $N=5$)
*Post-hoc diagnostic evaluating the ability to predict surges before they appear at the station.*
- **Timestamps:**
  `2024-08-16 00:00` (before +0.411m rise), `2024-08-21 00:00` (before +0.252m rise), `2024-08-21 01:00`, `2024-08-21 02:00`, `2024-08-21 03:00`.
- **Model Errors on Regime C (1h Horizon):**
  - **Persistence:** $\text{MAE} = 0.2424\text{ m}$, $\text{RMSE} = 0.2587\text{ m}$, $\text{MaxAE} = 0.4110\text{ m}$
  - **Ridge:** $\text{MAE} = 0.2657\text{ m}$, $\text{RMSE} = 0.2972\text{ m}$, $\text{MaxAE} = 0.4784\text{ m}$
  - **Gradient Boosting:** $\text{MAE} = 0.1718\text{ m}$, $\text{RMSE} = 0.2537\text{ m}$, $\text{MaxAE} = 0.4803\text{ m}$
- **Critical Hydrological Insight:** When the river is quiescent right before a flood wave arrives (e.g. at `2024-08-16 00:00` and `2024-08-21 00:00`), **all water-level-only models fail to anticipate the sudden rise**, suffering maximum errors $>0.41$ m. This demonstrates why **upstream catchment rainfall is essential** to provide lead-time for true flood forecasting.

---

## 4. Scope & Terminology Clarification

- **Current Capability:** The current pipeline predicts **river stage (water level in meters)** over 1- to 3-hour lead times.
- **Flood Forecasting Restriction:** The current ground telemetry dataset from Maharashtra State Water (`rwl_tel_hr_maharashtra_sw_006_2021_2025.csv`) does **not** provide verified CWC flood-warning thresholds (e.g. Warning Level, Danger Level, or High Flood Level [HFL] in meters above MSL).
- **Rule Enforced:** The project will not claim to forecast "floods" or issue warning classifications until official station thresholds are verified and integrated.

---

## 5. Next Implementation Blockers

The modeling pipeline is unblocked for water-level autoregression. The next project milestones require two external data assets:

1. **Catchment Boundary Acquisition:**
   - **Requirement:** Upper Godavari contributing catchment polygon delineated to pour point `(19.126944 N, 77.280833 E)` matching the CWC benchmark of **51,390.78 km²** ($\pm 3\%$).
   - **Resolution:** Download official Upper Godavari layer from India-WRIS or delineate from HydroSHEDS Asia (`hybas_as_lev06_v1c.shp`) using the specification in [`river_forecasting/src/catchment_boundary_spec.py`](../src/catchment_boundary_spec.py). Save to `data/spatial/vishnupuri_upstream_catchment.geojson`.
2. **Historical INSAT-3DS HEM Rainfall Download:**
   - **Requirement:** ~2,346 half-hourly HDF5 files for product `3SIMG_L2B_HEM_V01R00` covering `2024-07-04` through `2024-08-23` (~22.4 GB).
   - **Resolution:** Place order via MOSDAC web catalog (`https://mosdac.gov.in/catalog/`) and download staged SFTP files using `scripts/download_mosdac_hdf.py`.
   - **Current Status:** Rainfall-runoff model [`RainfallRunoffForecaster`](../src/baseline_models.py#L118-L148) is implemented and ready to ingest real precipitation features as soon as files are staged.
