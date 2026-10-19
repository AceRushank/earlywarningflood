# Vishnupuri Barrage Historical River-Rise Events & Data Quality Audit

**Station:** Vishnupuri Barrage, Nanded, Maharashtra (`19.126944°N, 77.280833°E`)  
**Data Source:** `river_forecasting/outputs/stage_observations.csv`  
**Temporal Span:** `2024-07-04 00:00:00 IST` through `2024-08-22 22:00:00 IST`  
**Total Observations:** 1,154 hourly records (out of 1,199 possible chronological hours; 45 missing hours total, zero duplicate timestamps).

---

## 1. Top 3 Candidate Historical River-Rise Events

### Event 1: August 3–4, 2024 (Sharpest & Largest Sustained Surge)
* **Start (Trough):** `2024-08-03 10:00:00 IST` at **`353.388 m`**
* **End (Peak):** `2024-08-04 20:00:00 IST` at **`354.798 m`**
* **Rise Magnitude:** **`+1.410 m`**
* **Rise Duration:** **`34.0 hours`** (Mean rate of rise: `+0.0415 m/hr`, peak hourly rate: `+0.165 m/hr`)
* **Observations During Rise:** **35 consecutive hours** without a single missing reading (100% data completeness).
* **Prior Baseline History (48h preceding trough):** 47 observations present out of 48 hours (`2024-08-01 10:00:00` to `2024-08-03 09:00:00 IST`), baseline mean stage `354.248 m`. Only 1 missing hour (`2024-08-01 19:00:00 IST`).
* **Evaluation Value:** Exceptional benchmark event. Maximum continuous rise in the entire 50-day telemetry record with flawless data continuity.

### Event 2: July 26–28, 2024 (Major Mid-Season Flood Wave)
* **Start (Trough):** `2024-07-26 01:00:00 IST` at **`353.625 m`**
* **End (Peak):** `2024-07-28 12:00:00 IST` at **`354.715 m`**
* **Rise Magnitude:** **`+1.090 m`**
* **Rise Duration:** **`59.0 hours`** (Mean rate of rise: `+0.0185 m/hr`)
* **Observations During Rise:** **60 consecutive hours** without a single missing reading (100% data completeness).
* **Prior Baseline History (48h preceding trough):** 47 observations present out of 48 hours (`2024-07-24 01:00:00` to `2024-07-26 00:00:00 IST`), baseline mean stage `354.002 m`. Only 1 missing hour (`2024-07-25 04:00:00 IST`).
* **Evaluation Value:** Classic multi-day monsoon catchment flood wave exhibiting steady, volume-driven rise.

### Event 3: August 21–22, 2024 (Late August Spate)
* **Start (Trough):** `2024-08-21 00:00:00 IST` at **`354.015 m`**
* **End (Peak):** `2024-08-22 16:00:00 IST` at **`354.831 m`**
* **Rise Magnitude:** **`+0.816 m`**
* **Rise Duration:** **`40.0 hours`** (Mean rate of rise: `+0.0204 m/hr`)
* **Observations During Rise:** 35 observations present out of 41 hours (6 missing hours intermittently: Aug 21 12:00, 17:00, 19:00; Aug 22 00:00, 02:00, 09:00 IST).
* **Prior Baseline History (48h preceding trough):** **48 observations out of 48 hours (100% complete)** from `2024-08-19 00:00:00` to `2024-08-20 23:00:00 IST`, baseline mean stage `354.313 m`.
* **Evaluation Value:** Coincides with the end of the operational test window. Stage rises cleanly from 354.015 m to 354.831 m.

---

## 2. Data-Quality Audit: Gaps & Suspicious Readings

1. **Telemetry Sensor Artifacts / Step Discontinuities:**
   * **Index 429 (`2024-07-22 19:00:00 IST`):** Stage abruptly jumps by **`+1.004 m`** in one hour (from `353.496 m` to `354.500 m`), flagged as `TRANSIENT_1M_SPIKE`.
   * **Index 430 (`2024-07-22 20:00:00 IST`):** Stage abruptly drops by **`-1.004 m`** in the very next hour (from `354.500 m` back to `353.496 m`), flagged as `DISCONTINUITY_STEP`.
   * **Assessment:** Classic 1-meter integer/register transmission glitch. Flagged correctly by existing QC rules. Preserved unchanged in raw records.
2. **Missing Timestamps:**
   * Total 45 missing hourly slots across 50 days (3.75% missingness).
   * Most gaps are isolated 1-hour drops (26 occurrences) or 2-hour drops (10 occurrences).
   * Largest gaps: Two 3-hour drops (`2024-07-06 20:00–23:00`, `2024-07-29 18:00–21:00`) and two 4-hour drops (`2024-07-04 19:00–23:00`, `2024-08-22 16:00–20:00`).
   * No missing hours occur during the critical rising limbs of **Event 1** or **Event 2**.

---

## 3. Proposed Rainfall Retrieval Windows (With Hydrological Lead Time)

Because the upstream Godavari catchment commands **$51,768\text{ km}^2$** extending over $400\text{ km}$ upstream, runoff concentration times range from:
* Local tributaries (immediate sub-basin): **$6\text{ to }18\text{ hours}$**
* Mid-basin reach (Jalna / Ahmednagar / Paithan): **$24\text{ to }48\text{ hours}$**
* Distal Western Ghats headwaters: **$48\text{ to }72+\text{ hours}$**

To capture antecedent catchment saturation and routing lags, each event requires a **72-hour antecedent rainfall window**:

| Event | Proposed Satellite Download Window (IST) | Equivalent Window (UTC) | MOSDAC Search Window | Est. Half-Hourly Files | Est. Volume |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Event 1** | `2024-07-31 00:00` to `2024-08-05 12:00` | `2024-07-30 18:30` to `2024-08-05 06:30` | `2024-07-31` to `2024-08-05` (6 days) | $\sim 288$ files | $\sim 2.8\text{ GB}$ |
| **Event 2** | `2024-07-23 00:00` to `2024-07-29 00:00` | `2024-07-22 18:30` to `2024-07-28 18:30` | `2024-07-23` to `2024-07-28` (6 days) | $\sim 288$ files | $\sim 2.8\text{ GB}$ |
| **Event 3** | `2024-08-18 00:00` to `2024-08-22 23:59` | `2024-08-17 18:30` to `2024-08-22 18:30` | `2024-08-18` to `2024-08-22` (5 days) | $\sim 240$ files | $\sim 2.4\text{ GB}$ |

---

## 4. Audit of Existing INSAT-3DS HEM Granules

Two authentic `V01R00` granules are currently downloaded and verified in `river_forecasting/data/mosdac_raw/3SIMG_L2B_HEM/2024/04JUL/`:

1. **`3SIMG_04JUL2024_2300_L2B_HEM_V01R00.h5`**
   * **UTC Timestamp:** `2024-07-04 23:00:00 UTC`
   * **IST Timestamp:** `2024-07-05 04:30:00 IST`
   * **Basin-Average Rainfall:** `0.1697 mm/hr` (Coverage: 1.0, 2,869 valid pixels)
2. **`3SIMG_04JUL2024_2330_L2B_HEM_V01R00.h5`**
   * **UTC Timestamp:** `2024-07-04 23:30:00 UTC`
   * **IST Timestamp:** `2024-07-05 05:00:00 IST`
   * **Basin-Average Rainfall:** `0.1519 mm/hr` (Coverage: 1.0, 2,869 valid pixels)

* **Stage Context at Observation Time:**
  * At `2024-07-05 04:00:00 IST`: `350.613 m`
  * At `2024-07-05 05:00:00 IST`: `350.617 m`
  * At `2024-07-05 06:00:00 IST`: `350.634 m`
* **Correlation Disclaimer:** No statistical rainfall–stage correlation can or should be computed from only two consecutive 30-minute samples ($N=2$) spanning 1 hour of baseflow telemetry.

---

## 5. Recommended Next Step

* **Target Event 1 (`2024-07-31` through `2024-08-05`)** as the primary satellite rainfall experiment:
  1. It features the steepest surge ($+1.41\text{ m}$), zero missing hours during the rise, and continuous preceding baseline observations.
  2. The download requirement is modest ($\sim 288$ files, $2.8\text{ GB}$), easily managed by our validated `mosdac_retriever.py` with automatic `V01R00` filtering and rate-limit backoff.
  3. Processing this single targeted event will yield an authentic 144-timestep rainfall sequence to evaluate rainfall routing lags ($R_{t-6\text{h}}$ to $R_{t-48\text{h}}$) before committing to a full 43 GB historical ingestion.
