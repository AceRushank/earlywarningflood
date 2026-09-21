# Machine Learning-Based Flood Forecasting for Early Warning at Microlevel

## Mumbai Ward-Level Hybrid Early-Warning Framework

> **Scientific Transparency Notice:** This system is an uncalibrated research prototype combining an ML-derived spatial susceptibility model with real-time forward numerical weather forecasts. It is **not** an end-to-end supervised flood event classifier and does **not** output calibrated flood probabilities. Alert messages are in-process simulations.

---

## 1. System Architecture

The framework couples verified historical remote sensing data with live forward meteorological forcing:

```
Historical Sentinel-1-derived flood exposure (Jalem et al. 2026)
                      +
     Static Physiographic Features (HydroSHEDS, ESA WorldCover, JRC)
                      ↓
  Ridge Spatial Susceptibility Model (Target: Inundation Ratio)
                      +
     Real-Time Forward NWP Forecasts (Open-Meteo API: 6h, 12h, 24h)
                      ↓
  Hydrometeorological Decision Layer (IMD Categories & BRIMSTOWAD Benchmark)
                      ↓
  Hybrid Risk Index (Dimensionless relative ranking in [0.0, 1.0])
                      ↓
  Internal System Decision Categories (LOW, MODERATE, HIGH, CRITICAL)
                      ↓
  Simulated Multi-Channel Early Warning Advisories (SMS, IVR, Push, Dashboard)
```

---

## 2. Scientific Boundaries & What the Model Represents

- **What the ML Model Does:**  
  The machine learning model estimates **spatial flood susceptibility / historical inundation exposure** ($S \in [0, 1]$). It predicts the expected normalized historical inundation ratio:
  $$\text{inundation\_ratio} = \frac{\text{flooded\_area\_km}^2}{\text{ward\_area\_km}^2}$$
  using verified multi-year Sentinel-1 SAR observations (2018–2025) and static physiographic features.
- **What the Real-Time Forecast Layer Does:**  
  Provides forward meteorological forcing ($H_{\text{rain}}$) using live hourly precipitation forecasts from Open-Meteo across 6-hour, 12-hour, and 24-hour windows.
- **What the Final System Outputs:**  
  A **`hybrid_risk_index`** in $[0.0, 1.0]$ representing a relative decision priority ranking.
- **What the System Does NOT Do:**  
  - It does **not** predict the occurrence of specific storm-by-storm flood events because verified historical event-level flood labels are not available.
  - It does **not** output a calibrated probability (e.g., an index of $0.73$ does **not** mean a "73% chance of flooding").
  - It does **not** simulate hydrodynamic pipe flow, overland wave routing, or tidal backwater locking.
  - It does **not** generate official IMD alerts or dispatch live telecommunications.

---

## 3. Spatial Resolution: The "Microlevel" Definition

The spatial resolution supported by this repository is the **BMC Administrative Ward level ($10^0\text{--}10^1\text{ km}^2$, averaging $\approx 18\text{ km}^2$)** across the 24 municipal wards of Mumbai.

- **Why this is sub-city / ward-level:** Conventional meteorological bulletins treat Mumbai as one or two monolithic points (Santacruz and Colaba). Operating at 24 administrative polygons provides spatial disaggregation and localized prioritization.
- **Limitation:** This is **not** street-level, building-level, or drainage-inlet-level prediction. Within an $18\text{ km}^2$ ward, local topography creates micro-variations that ward-averaged polygons cannot resolve.

---

## 4. Machine Learning Model & LOOCV Benchmark

### Target Variable
$$\text{inundation\_ratio} = \frac{\text{mean flooded km}^2\text{ (2018–2025)}}{\text{ward surface area km}^2\text{ (projected UTM Zone 43N)}}$$

### Leave-One-Out Cross-Validation ($N = 23$ Wards)
Ward `G/S` (Worli / Lower Parel) has no corresponding locality in the published SAR study (*Jalem et al., 2026*) and was dropped from training. The remaining 23 wards were evaluated under strict Leave-One-Out Cross-Validation (LOOCV):

| Model | LOOCV RMSE | LOOCV MAE | LOOCV $R^2$ | Status vs Baseline |
|---|---|---|---|---|
| Baseline (Dummy Mean) | 0.037598 | 0.029550 | -0.092975 | Reference |
| **Ridge Regression ($\alpha=1.0$)** | **0.034567** | **0.028506** | **+0.076144** | **Selected Winner** |
| Ridge Regression ($\alpha=10.0$) | 0.034765 | 0.027487 | +0.065533 | Outperforms baseline |
| ElasticNet ($\alpha=0.1, \ell_1=0.5$) | 0.037598 | 0.029550 | -0.092975 | Equal to baseline |
| Random Forest ($d=2, n=100$) | 0.038131 | 0.029710 | -0.124154 | Inferior to baseline |
| Random Forest ($d=4, n=200$) | 0.041943 | 0.032033 | -0.360170 | Overfitted |

### Scientific Interpretation of Metrics
- **Ridge Regression ($\alpha=1.0$)** achieves an $8.06\%$ reduction in RMSE and a positive out-of-sample $R^2 = +0.0761$.
- **Signal Interpretation:** $R^2 \approx +0.076$ represents a **modest directional spatial signal**. It indicates that static physiography accounts for approximately $7.6\%$ of the variance in spatial flood exposure. It does **not** imply high predictive certainty.

### Ridge Feature Coefficients (Fitted on all 23 Wards)
1. `water_pct` ($-0.027190$): Permanent water presence proxy.
2. `elevation_mean` ($-0.019619$): Lower elevation strongly correlates with higher inundation ratio.
3. `dist_to_water_mean` ($-0.017626$): Proximity to tidal water increases exposure.
4. `slope_mean` ($-0.012024$): Flatter terrain impedes natural drainage.
5. `vegetation_pct` ($+0.011980$): Reflects peripheral mangrove/wetland vegetation zones.
6. `builtup_pct` ($+0.002421$): Urban imperviousness.

---

## 5. Hydrometeorological Decision Layer & Risk Math

### Rainfall Hazard Formulation ($H_{\text{rain}}$)
Rainfall forcing distinguishes accumulation volume and instantaneous peak intensity across forecast horizons:
- **For 24-Hour Horizon (`next_24h`):**
  $$H_{\text{accum}} = \min\left(\frac{R_{\text{24h}}}{204.4\text{ mm}}, 1.0\right)$$
  $$H_{\text{intensity}} = \min\left(\frac{I_{\text{peak}}}{50.0\text{ mm/hr}}, 1.0\right)$$
  $$H_{\text{rain}} = \max(H_{\text{accum}}, H_{\text{intensity}})$$
- **For Shorter Horizons (`next_6h`, `next_12h`):**
  To avoid applying a 24-hour meteorological accumulation denominator to short windows or inventing unverified 6h/12h accumulation thresholds, rainfall hazard relies transparently on the peak-hourly-intensity benchmark:
  $$H_{\text{rain}} = H_{\text{intensity}} = \min\left(\frac{I_{\text{peak}}}{50.0\text{ mm/hr}}, 1.0\right)$$
  Actual 6h and 12h forecast accumulation totals remain fully exposed in the API as informational quantities.

- **$204.4\text{ mm}$ Benchmark:** India Meteorological Department (IMD) 24-hour threshold for "Extremely Heavy Rain". Used strictly as a 24-hour meteorological accumulation reference, **not** an empirical flood trigger and **not** applied to 6h/12h windows.
- **$50.0\text{ mm/hr}$ Benchmark:** BMC BRIMSTOWAD storm drainage design planning capacity. Used as an urban drainage engineering capacity benchmark comparison, **not** a hydraulic flood simulation.
- **Exposed Indicator:** $\text{drainage\_intensity\_ratio} = \frac{I_{\text{peak}}}{50.0\text{ mm/hr}}$.

### Hybrid Risk Index Formula
$$\text{hybrid\_risk_index} = H_{\text{rain}} \cdot (0.50 + 0.50 \cdot S_{\text{norm}})$$
where $S_{\text{norm}} = \text{minmax}(\hat{y}) \in [0.0, 1.0]$ is the normalized ML susceptibility.

> **Methodological Disclaimer:** This formula is an **uncalibrated prototype decision heuristic**. It is not an empirically fitted hydraulic damage function or a probability model.

### Zero-Rain Safety Guarantee
$$\text{If } H_{\text{rain}} = 0 \implies \text{hybrid\_risk\_index} = 0.0 \quad (\text{early\_warning} = \text{False})$$
Dry forecasts will **never** trigger a flood warning, regardless of how susceptible the ward is.

### Weather Failure Safety
If the Open-Meteo API fails (timeout, network error), the system sets `weather_available = False`, `hybrid_risk_index = None`, and `risk_level = "WEATHER_UNAVAILABLE"`. It **never** silently defaults to 0 mm.

### Internal System Decision Categories
- `LOW`: $[0.00, 0.25)$ — Negligible risk under current forecast conditions.
- `MODERATE`: $[0.25, 0.50)$ — Elevated risk in vulnerable areas.
- `HIGH`: $[0.50, 0.75)$ — Substantial risk in susceptible zones. Early warning flag activated.
- `CRITICAL`: $[0.75, 1.00]$ — Severe meteorological forcing and widespread risk. Early warning flag activated.

*These categories are internal prototype decision states, NOT official IMD warnings.*

---

## 6. API Specification

### Endpoint: `GET /wards/risk`

#### Query Parameters:
- `horizon` (optional, default: `"next_24h"`): Supported windows: `"next_6h"`, `"next_12h"`, `"next_24h"`.
- `simulate_rain_mm` (optional, default: `None`): Test override. When provided, sets `data_mode = "simulation"` for deterministic verification.

#### Example Response Object:
```json
{
  "ward_name": "L",
  "bmc_ward_code": "L",
  "ward_area_km2": 15.681,
  "lat": 19.087553,
  "lon": 72.886668,
  "historical_flood_score": 1.7225,
  "inundation_ratio": 0.109846,
  "forecast_horizon": "next_24h",
  "forecast_rainfall_mm": 142.5,
  "next_6h_rainfall_mm": 28.0,
  "next_12h_rainfall_mm": 65.0,
  "next_24h_rainfall_mm": 142.5,
  "peak_hourly_mm": 38.0,
  "imd_24h_rainfall_category": "VERY_HEAVY",
  "imd_intensity_category": "VERY_INTENSE_SPELL",
  "ml_susceptibility": 0.0894,
  "susceptibility_norm": 0.924,
  "drainage_intensity_ratio": 0.760,
  "rainfall_hazard": 0.760,
  "hybrid_risk_index": 0.7311,
  "risk_level": "HIGH",
  "early_warning": true,
  "weather_available": true,
  "forecast_timestamp": "2026-09-21T13:43:37+00:00",
  "source_model": "ECMWF / GFS Seamless (Open-Meteo NWP)",
  "data_mode": "live",
  "alert_channels": {
    "sms_draft": "[Ward L] SYSTEM RISK: HIGH. Rain 142mm (next_24h), peak 38mm/h. Avoid waterlogged subways. (Prototype early-warning assessment — NOT an official IMD alert)",
    "ivr_script": "Namaste. Yeh Mumbai Flood Early Warning research prototype ka alert hai...",
    "push_notification": {
      "title": "🟠 Ward L Flood Advisory: HIGH",
      "body": "Forecast: 142.5mm over next_24h (Peak: 38.0mm/hr, VERY_INTENSE_SPELL)..."
    },
    "dashboard_entry": "WARD=L | RISK=HIGH | HYBRID_INDEX=0.731 | HORIZON=next_24h..."
  }
}
```

---

## 7. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and validate ML model (updates model artifact, CSV, and chart)
python train_ward_susceptibility.py

# 3. Run test suites
python test_loader.py
python test_robustness.py

# 4. Start FastAPI server
python -m uvicorn app.main:app --port 8000

# 5. Run API integration tests
python test_api.py
python test_simulate.py
```

---

## 8. Summary of Known Scientific & Engineering Limitations

1. **No Supervised Event Training:** Lacks storm-by-storm ground truth; the alert decision boundary is an adapted hydrometeorological heuristic, not a trained classifier.
2. **Coarse Spatial Boundaries:** 24 administrative polygons average $18\text{ km}^2$, meaning intra-ward micro-topography and street-level waterlogging cannot be differentiated.
3. **No Tidal Interaction:** Coastal astronomical tides and Mithi River flap-gate closures are unmodeled due to absence of open hydrodynamic sensor telemetry.
4. **Simulated Dissemination:** SMS, IVR, Push, and Dashboard outputs are in-process string renderings; no third-party telecommunications carrier is connected.
