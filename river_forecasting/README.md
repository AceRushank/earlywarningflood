# Godavari River Flood Forecasting — System Documentation

This repository houses the **Godavari River Flood Forecasting** pipeline isolated inside `river_forecasting/`. The system focuses on flood prediction at **Vishnupuri Barrage (Shankar Sagar Jalashay)**, Nanded, Maharashtra (`19.126944° N, 77.280833° E`) on the mainstem Godavari River.

---

## 1. Station & Hydrological Context

| Parameter | Specification | Source / Verification |
| :--- | :--- | :--- |
| **Station Name** | `Vishnupuri` | Maharashtra State Water Telemetry Network |
| **Project Name** | Vishnupuri Prakalp (Shankar Sagar) | Godavari Basin Headquarters, Nanded |
| **River / Basin** | Godavari / Godavari Basin | Mainstem Godavari River |
| **District / State** | Nanded, Maharashtra | LGD Code: 485 |
| **Pour Point Coordinates** | `19.126944° N, 77.280833° E` | Exact barrage crest axis |
| **Gauge Datum** | `RL_of_zeroGauge = 0.0 m` | Verified constant across all telemetry records |
| **Reference Datum** | `MeanSeaLevel = 366.50 m` | Verified constant across all telemetry records |
| **Drainage Basin Area** | **51,390.78 km²** | Official CWC & Maharashtra Water Resources Dept |
| **Target Historical Window** | `2024-07-04 00:00` to `2024-08-22 22:00 IST` | Continuous peak monsoon flood wave (49.9 days) |

---

## 2. Catchment Boundary Investigation & Specification

Detailed in [`river_forecasting/src/catchment_boundary_spec.py`](src/catchment_boundary_spec.py).

### Hydrological Drainage Benchmark
- **Statutory Contributing Area:** **51,390.78 km²** draining the Upper Godavari basin (from source at Trimbakeshwar, Nashik, through Ahmednagar, Aurangabad, Jalna, Parbhani, Beed, to Vishnupuri in Nanded).
- **Validation Tolerance:** Evaluated boundaries must satisfy $49,000 \le \text{Area} \le 53,000\text{ km²}$ ($\pm 3\%$).

### Authoritative Data Sources:
1. **Official Government Source (India-WRIS / Central Water Commission):**
   - **Portal:** `https://indiawris.gov.in/wris/#/download`
   - **Layer:** Godavari Basin -> Upper Godavari Sub-basin & contributing watersheds.
   - **License:** Government Open Data License - India (GODL-India).
   - **CRS:** `EPSG:4326` (WGS84).
2. **Global Hydro-conditioned DEM (HydroSHEDS / HydroBASINS):**
   - **Portal:** `https://www.hydrosheds.org/products/hydrobasins`
   - **Product:** `hybas_as_lev06_v1c.shp` (Pfafstetter Level 6 sub-basins for Asia).
   - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
   - **Delineation Method:** Trace upstream topological hierarchy (`HYBAS_ID` $\to$ `NEXT_DOWN`) draining to the sub-basin containing pour point `(19.126944 N, 77.280833 E)`.

### Strict Hydrological Prohibitions:
- **Bounding Box (BBOX) approximations are strictly prohibited:** A rectangular bbox captures non-contributing basins (e.g. portions of Tapi, Bhima, or Penganga) and distorts rainfall mass balance.
- **Single pixel queries are strictly prohibited:** A single ~4 km satellite pixel at the dam wall ignores >51,000 km² of upstream storm runoff.

---

## 3. Historical MOSDAC Rainfall Acquisition Plan

Detailed in [`river_forecasting/src/mosdac_order_spec.py`](src/mosdac_order_spec.py).

### Specifications & Volume:
- **Satellite & Sensor:** INSAT-3DS Imager.
- **Product Identifier:** `3SIMG_L2B_HEM_V01R00` (Level-2B Hydro-Estimator Precipitation).
- **Dataset:** `/HEM` (Precipitation rate, $mm/hr$, 30-minute interval).
- **Target Window:** `2024-07-04 00:00 UTC` to `2024-08-23 23:59 UTC` (51 days).
- **Expected File Count:** ~2,346 HDF5 files (48 slots/day minus known 17:30 and 19:00 UTC eclipse/calibration windows).
- **Storage Requirement:**
  - Full-disk HDF5 files (~9.76 MB each): **~22.4 GB**.
  - Godavari spatial subset (if clipped): **~1.8 GB**.
- **Processing Time:** ~28.2 seconds total extraction time using our cached curvilinear pixel mask in [`CatchmentRainfallExtractor`](src/catchment_extractor.py).

### Acquisition Workflow:
1. Log into the official MOSDAC catalog at `https://mosdac.gov.in/catalog/`.
2. Select Mission: `INSAT-3DS`, Sensor: `Imager`, Product: `3SIMG_L2B_HEM_V01R00`.
3. Set Date Range: `2024-07-04` to `2024-08-23`.
4. Submit Cart Order. Once staged on `download.mosdac.gov.in:22`, download with:
   ```powershell
   python scripts/download_mosdac_hdf.py --start-date 2024-07-04 --end-date 2024-08-23
   ```

---

## 4. Modeling Pipeline & Anti-Leakage Guarantees

Implemented in [`river_forecasting/src/feature_pipeline.py`](src/feature_pipeline.py).

### Data Integrity & Leakage Prevention:
1. **Regular Hourly Grid Alignment:** Resamples observations into a continuous 1-hour grid. Gaps and network outages are documented; true ground observations are **never altered, deleted, or fabricated**.
2. **Zero Future Lookahead:** At forecast issue time $t$, features strictly draw upon past records $\le t$:
   - Current stage: $H_t$
   - Autoregressive lags: $H_{t-1}, H_{t-2}, H_{t-3}, H_{t-4}, H_{t-6}$
   - Hourly rates of change: $\Delta H_{t-1}, \Delta H_{t-2}, \Delta H_{t, 3h}, \Delta H_{t, 6h}$
   - Rolling statistics over preceding windows: mean, standard deviation, min, max ($3h, 6h, 12h$)
   - Quality tracking: anomaly indicator flag (tracks transient spike on 2024-07-22)
3. **Chronological Splitting (No Shuffling):**
   - **Train (70%):** 803 hourly samples (`2024-07-04 06:00` to `2024-08-07 22:00 IST`)
   - **Validation (15%):** 172 hourly samples (`2024-08-07 23:00` to `2024-08-15 06:00 IST`)
   - **Test (15%):** 173 hourly samples (`2024-08-15 07:00` to `2024-08-22 22:00 IST`, contains the season's flood peak at 355.038 m)

---

## 5. Baseline Experiment Results (Real Data Evaluation)

Evaluated on the authentic test split across 1-, 2-, and 3-hour horizons ($t+1, t+2, t+3$).

### Overall Test Set Performance:

| Model | Horizon | MAE (m) | RMSE (m) | Max AE (m) | NSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | **1-hour** | **0.0383** | **0.0706** | **0.4110** | **0.9248** |
| **Water-Level-Only (Ridge)** | **1-hour** | **0.0466** | **0.0843** | **0.4784** | **0.8926** |
| **Water-Level-Only (Gradient Boosting)** | **1-hour** | **0.0412** | **0.0730** | **0.4851** | **0.9196** |
| **Persistence Baseline** | **2-hour** | **0.0592** | **0.1099** | **0.5250** | **0.8159** |
| **Water-Level-Only (Ridge)** | **2-hour** | **0.0702** | **0.1272** | **0.6466** | **0.7534** |
| **Water-Level-Only (Gradient Boosting)** | **2-hour** | **0.0686** | **0.1184** | **0.7003** | **0.7864** |
| **Persistence Baseline** | **3-hour** | **0.0705** | **0.1279** | **0.6450** | **0.7513** |
| **Water-Level-Only (Ridge)** | **3-hour** | **0.0771** | **0.1480** | **0.7696** | **0.6671** |
| **Water-Level-Only (Gradient Boosting)** | **3-hour** | **0.0803** | **0.1382** | **0.7530** | **0.7096** |

### High-Stage and Surge Regime Evaluations:
- **Peak-Stage Events ($H \ge 354.80$ m, $N=42$):**
  - Persistence 1h: $\text{MAE} = 0.0572\text{ m}$, $\text{RMSE} = 0.1000\text{ m}$
  - Gradient Boosting 1h: $\text{MAE} = 0.0718\text{ m}$, $\text{RMSE} = 0.1101\text{ m}$
- **Antecedent Rapid Rise ($H_t - H_{t-1} \ge +0.15$ m/hr, $N=5$):**
  - Persistence 1h: $\text{MAE} = 0.1722\text{ m}$, $\text{RMSE} = 0.1757\text{ m}$
  - **Gradient Boosting 1h:** $\mathbf{MAE = 0.0574\text{ m}}$, $\mathbf{RMSE = 0.0837\text{ m}}$ (66.7% error reduction over persistence by projecting ongoing surge momentum).
- **Future Rapid Surge Onset ($H_{t+1} - H_t \ge +0.15$ m, $N=5$):**
  - Persistence 1h: $\text{MAE} = 0.2424\text{ m}$, $\text{RMSE} = 0.2587\text{ m}$
  - Gradient Boosting 1h: $\text{MAE} = 0.1718\text{ m}$, $\text{RMSE} = 0.2537\text{ m}$
  - *Methodological Finding:* All water-level-only models fail to anticipate surge onsets prior to their appearance at the station gauge. Upstream catchment rainfall observations are strictly necessary for flood early warning.

### Water-Level + Catchment-Rainfall Model:
- Architecture implemented in [`RainfallRunoffForecaster`](src/baseline_models.py).
- Status: **BLOCKED_AWAITING_AUTHENTIC_RAINFALL**.
- In strict adherence to scientific integrity rules, **zero synthetic metrics are reported** until real rainfall is acquired.

---

## 6. Architecture & File Manifest

```
river_forecasting/
├── README.md                              # Complete system specification
├── __init__.py
├── outputs/
│   ├── stage_observations.csv             # 1,154 verified ground stage records
│   ├── baseline_predictions_test.csv      # Out-of-sample predictions vs ground truth
│   ├── baseline_experiment_results.json   # Machine-readable experiment & audit metrics
│   ├── stage2_audit_report.md             # Dedicated Stage 2 audit report
│   ├── data_quality_report.json           # Stage 1 quality control report
│   └── implementation_status.md           # Implementation status breakdown
├── src/
│   ├── __init__.py
│   ├── stage_extractor.py                 # Telemetry ingestion, QC, and discontinuity analysis
│   ├── mosdac_audit.py                    # Local HDF archive & spatial asset audit
│   ├── catchment_extractor.py             # Spatial rainfall extraction engine & mask cache
│   ├── catchment_boundary_spec.py         # Catchment specification & area validation
│   ├── mosdac_order_spec.py               # Historical order planner & storage estimation
│   ├── feature_pipeline.py                # Leakage-free alignment, lag features & splits
│   ├── baseline_models.py                 # Persistence, Ridge, Gradient Boosting & Rainfall models
│   ├── evaluate.py                        # Hydrological metrics (MAE, RMSE, MaxAE, NSE)
│   ├── run_pipeline.py                    # Master Stage 1 execution runner
│   └── run_stage2_pipeline.py             # Master Stage 2 experiment runner
└── tests/
    ├── __init__.py
    └── test_pipeline.py                   # Automated unit test suite (13 test cases)
```

---

## 7. How to Reproduce All Results

### Run Unit Tests:
```powershell
python -m unittest river_forecasting/tests/test_pipeline.py
```

### Run Stage 1 Data Foundation:
```powershell
python -m river_forecasting.src.run_pipeline
```

### Run Stage 2 Baseline Model Experiments:
```powershell
python -m river_forecasting.src.run_stage2_pipeline
```
