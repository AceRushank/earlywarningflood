# Stage 1 Data Source & Hydrological Provenance Audit

**Project:** Vishnupuri Barrage River Stage Forecasting (+1h, +2h, +3h)  
**Location:** Godavari River, Nanded, Maharashtra, India  
**Phase:** Phase 5 — Upstream Telemetry Integration and Out-of-Sample Surge Forecasting  
**Authoritative Provenance:** Maharashtra Water Resources Department (MWRD) / Central Water Commission (CWC) / National Water Informatics Centre (NWIC)  

---

## 1. Executive Summary

This audit evaluates authoritative, real upstream telemetry stations in the Godavari basin to determine whether authentic upstream river-stage observations improve +1h, +2h, and +3h river-stage forecasts at Vishnupuri Barrage.

In accordance with strict scientific integrity guidelines:
- **Zero synthetic upstream measurements** were created or evaluated.
- **Zero inferred or simulated gauge heights** are presented as observations.
- Only verified, authentic hourly telemetry records from the official NWIC/MWRD surface-water telemetry feed (`rwl_tel_hr_maharashtra_sw_006_2026_2030.csv`) are ingested.
- Provenance, transmission latency, and missingness are rigorously preserved.

---

## 2. Target Location & Hydrological Context

- **Target Station:** Vishnupuri Barrage (Asarjan / Nanded)
- **Official Station Name:** Vishnupuri
- **Coordinates:** Lat 19.12694° N, Lon 77.28083° E
- **River:** Godavari (Mainstem)
- **District / State:** Nanded, Maharashtra
- **Catchment Area:** ~51,000 km² at Vishnupuri
- **Operating Agency:** Maharashtra Water Resources Department (MWRD) / CWC SW Network
- **Full Reservoir Level (FRL):** 355.00 m MSL
- **Crest Level:** 345.00 m MSL
- **Observed Monsoon 2026 Range:** 350.54 m to 355.09 m MSL (active barrage regulation window)
- **Available Parameters:** Hourly River Water Level (meters MSL). Observed discharge is unavailable (`Is_DischargeDataAvailable: No`).

---

## 3. Comprehensive Inventory of Upstream & Basin Stations

| Station Name | Basin / River | District | Coordinates (Lat, Lon) | Upstream Distance / Relation | Elevation Range (m MSL) | Valid Monsoon Coverage | Discharge Available | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vishnupuri** | Godavari (Mainstem) | Nanded | 19.1269°N, 77.2808°E | 0 km (Target Barrage) | 350.54 – 355.09 | 93.2% (2417/2592 h) | No | Target Forecast Location |
| **Mudgal Barrage** | Godavari (Mainstem) | Parbhani | 19.0889°N, 76.4917°E | ~105 km direct upstream | 380.50 – 393.63 | 95.7% (2481/2592 h) | No | **Primary Upstream Mainstem Station** |
| **Siddeshwar Dam** | Purna (Godavari Trib.) | Hingoli | 19.6128°N, 76.9719°E | ~95 km upstream tributary | 408.89 – 411.26 | 89.5% (2321/2592 h) | No | Upstream Tributary Storage |
| **Majalgaon** | Sindphana (Godavari Trib.) | Beed | 19.1508°N, 76.1869°E | ~140 km upstream tributary | 427.36 – 428.32 | 95.7% (2481/2592 h) | No | Upstream Tributary Dam |
| **Lower Dudhana Dam** | Dudhana (Godavari Trib.) | Parbhani | 19.5308°N, 76.3928°E | ~120 km upstream tributary | 422.10 – 422.74 | 91.1% (2361/2592 h) | No | Upstream Tributary Dam |
| **Raher** | Godavari (Mainstem) | Nanded | 18.8969°N, 77.6772°E | ~55 km downstream | 328.95 – 332.15 | 95.7% (2480/2592 h) | No | Downstream Reference Point |
| **Babhali Barrage** | Godavari (Mainstem) | Nanded | 18.8536°N, 76.7794°E | ~70 km downstream | 327.16 – 330.44 | 91.2% (2364/2592 h) | No | Downstream Boundary Check |

---

## 4. Hydrological Travel-Time and Wave Celerity Analysis

### 4.1 Hydraulic Wave Speed on the Godavari Mainstem
- Reach: Mudgal Barrage to Vishnupuri Barrage
- Approximate river channel distance: $L \approx 105\text{ km}$
- Bed slope: $S_0 \approx 0.00035$ (alluvial low-gradient valley)
- Mean bankfull velocity: $u \approx 1.2 - 1.6\text{ m/s}$ ($4.3 - 5.8\text{ km/h}$)
- Theoretical kinematic wave celerity: $c = \frac{5}{3} u \approx 2.0 - 2.7\text{ m/s}$ ($7.2 - 9.7\text{ km/h}$)
- **Theoretical Flood Wave Travel Time:**
  $$T_{\text{travel}} = \frac{L}{c} = \frac{105\text{ km}}{7.2\text{ to }9.7\text{ km/h}} \approx 11\text{ to }15\text{ hours}$$
- Under low-to-moderate baseflow conditions, travel times extend to $18 - 24\text{ hours}$.

### 4.2 Cross-Correlation Empirical Evidence
Empirical cross-correlation analysis over the entire 2026 monsoon (2,592 hours) confirms these physical hydrodynamics:
1. **Dynamic Rate-of-Change ($\Delta H_{1h}$):**
   - At lead times $t - 1\text{h}$, $t - 2\text{h}$, and $t - 3\text{h}$, correlation between Mudgal $\Delta H$ and Vishnupuri $\Delta H$ is statistically indistinguishable from zero ($r = 0.0197, -0.0087, 0.0256$).
   - A sudden rise or gate operation at Mudgal Barrage has **zero physical or empirical manifestation** at Vishnupuri within 1 to 3 hours.
2. **Absolute Seasonal Stage Level ($H$):**
   - Stage level correlation is high ($r \approx 0.9017$) because both stations reflect macro-basin monsoonal filling and seasonal catchment saturation.
3. **Purna River Tributary (Siddeshwar Dam):**
   - Peak rate-of-change cross-correlation occurs at lag $t - 18\text{h}$ ($r = 0.3129$), matching tributary transit times.

---

## 5. Telemetry Availability, Provenance, and Latency Constraints

### 5.1 Reporting Protocol & Telemetry Latency
- Field telemetry stations record hourly water surface elevations at the top of the hour (`HH:00 IST`).
- Data packets are transmitted via GPRS/GSM cellular networks or INSAT satellite telemetry to the state gateway.
- While local barrage data at Vishnupuri is available on-site immediately, remote upstream gauges experience a typical operational latency buffer of **1 to 2 hours** before data is verified and posted to the central database.
- **Leakage Prevention Rule:** In our controlled experiments, upstream features evaluated at forecast origin $t$ are lagged by at least 1 hour ($t-1$) and 2 hours ($t-2$) to reflect realistic telemetry availability at issuance time. Never use contemporaneous ($t-0$) upstream observations without proving zero transmission delay.

### 5.2 Outage and Missing Data Characteristics
- **Whole Season Availability:** Mudgal Barrage achieves 95.72% valid records (2,481/2,592 hours), with a maximum continuous gap of 58 hours (during early August network downtime).
- **Synchronized Test Period (Sept 8–22):** Mudgal Barrage achieves 96.86% availability (339/350 hours), with a maximum gap of only 4 hours, aligning closely with Vishnupuri (97.14%).
- **Tributary Outages:** Siddeshwar Dam suffered a severe 63-hour telemetry failure during mid-September, dropping its availability during the test subwindow to 49.71%. This makes Siddeshwar unsuitable as an operational real-time feature.
- **Decision:** `Mudgal Barrage` is selected as the sole verified, authentic upstream telemetry candidate for feature ablation.
