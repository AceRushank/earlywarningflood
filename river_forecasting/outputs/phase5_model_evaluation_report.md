# Phase 5: Leakage-Free Model Evaluation & Benchmark Comparison

## 1. Methodological Integrity & Leakage Elimination

* **Evaluation Period:** `2024-08-05 05:00:00` to `2024-08-05 23:00:00` IST
* **Test Set Origins:** `19` points
* **Hydrological Regime:** **POST_PEAK_TAIL_REGIME (Begins 9h after Aug 4 20:00 peak of 354.798m; initial drop to 353.958m trough then mild fluctuation to 354.183m; zero rising-limb coverage)**
* **Imputation Policy:** Strict complete-origins-only evaluation. No future-dependent imputation (`.bfill()` eliminated).
* **Split Protocol:** Chronological 70% Train (79 origins) / 15% Validation (14 origins) / 15% Test (19 origins) with a **3-hour embargo buffer** to eliminate boundary horizon leakage.
* **StandardScaler:** Fitted strictly on training data; test data strictly transformed using training parameters.
* **Feature Availability:** All autoregressive stage features and antecedent rainfall features use observations strictly $\le t$.

## 2. Benchmark Comparison Results

| Model | Horizon | N Samples | MAE (m) | RMSE (m) | Max AE (m) | NSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Persistence Baseline | 1h | 18 | 0.0200 | 0.0248 | 0.0510 | 0.8635 |
| Persistence Baseline | 2h | 17 | 0.0318 | 0.0392 | 0.0870 | 0.6163 |
| Persistence Baseline | 3h | 16 | 0.0375 | 0.0449 | 0.0810 | 0.3465 |
| Water-Level-Only ML (Ridge) | 1h | 18 | 0.0216 | 0.0273 | 0.0488 | 0.8343 |
| Water-Level-Only ML (Ridge) | 2h | 17 | 0.0425 | 0.0493 | 0.1048 | 0.3909 |
| Water-Level-Only ML (Ridge) | 3h | 16 | 0.0556 | 0.0653 | 0.1278 | -0.3837 |
| Rainfall-Enhanced ML (Ridge) | 1h | 18 | 0.0217 | 0.0269 | 0.0529 | 0.8393 |
| Rainfall-Enhanced ML (Ridge) | 2h | 17 | 0.0434 | 0.0499 | 0.1052 | 0.3757 |
| Rainfall-Enhanced ML (Ridge) | 3h | 16 | 0.0600 | 0.0703 | 0.1387 | -0.6050 |

## 3. Findings and Critical Qualifications

* **Persistence Benchmark Dominance:** Persistence ($H_{t+h} = H_t$) outperforms both machine learning models across all horizons:
  * **1-Hour Horizon:** Persistence achieves MAE = 0.0200 m (vs 0.0216 m for WL-only and 0.0217 m for Rainfall-enhanced).
  * **2-Hour Horizon:** Persistence achieves MAE = 0.0318 m and NSE = 0.6163 (vs MAE = 0.0425 m, NSE = 0.3909 for WL-only).
  * **3-Hour Horizon:** Persistence achieves MAE = 0.0375 m and positive NSE = 0.3465, whereas both ML models yield negative NSE (-0.3837 and -0.6050).
* **Hydrological Context (Post-Peak Tail Regime Limitation):**
  * **Crucial Qualification:** Event 1 reached its hydrological crest on **August 4, 2024 at 20:00 IST at 354.798 m** (following a +1.410 m rise from 353.388 m on Aug 3 10:00).
  * The test set begins on **August 5, 2024 at 05:00 IST (9 hours after the peak)** at 354.009 m, drops to a local trough of 353.958 m at 07:00 IST, and then fluctuates mildly up to 354.183 m at 23:00 IST.
  * It is therefore a **post-peak tail regime** (tail of the recession plus minor post-event stage rebound). It contains **zero observations from the rising limb** (Aug 3 10:00 to Aug 4 20:00).
  * Consequently, these test results evaluate post-peak stage tracking and **cannot establish early-warning performance during a rising flood**.
* **Physical Realism:** Short-term (1–3h) rainfall accumulations provide negligible predictive skill because the physical catchment travel time from upstream tributaries exceeds 24–48 hours.
