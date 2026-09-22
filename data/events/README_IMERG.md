# NASA GPM IMERG V07B Rainfall Pipeline for Mumbai Microlevel Flood Forecasting

## Overview
This pipeline implements **Phase 2** of the Mumbai microlevel flood forecasting dataset construction. In accordance with the scientific directives:
- **No synthetic labels or synthetic rainfall** are permitted.
- **NASA GPM IMERG V07B** serves as the empirical rainfall observation source while MOSDAC institutional access approval is pending.
- **Modular abstraction** is maintained so MOSDAC or other radar/reanalysis sources can later be ingested for sensitivity and comparative analysis without modifying the Sentinel-1 SAR flood labels or ML target.

---

## 1. Data Source Specifications
- **Product Name**: NASA GPM IMERG Final Precipitation L3 Half Hourly 0.1° × 0.1° V07B (`NASA/GPM_L3/IMERG_V07`)
- **Temporal Resolution**: 30-minute intervals (48 frames per 24 hours, 144 frames per 72 hours)
- **Spatial Resolution**: 0.1° × 0.1° (~11 km grid cells)
- **Physical Band**: `precipitation` (calibrated rainfall rate in mm/hr)
- **Depth Conversion**: Each 30-minute frame is multiplied by `0.5 hr` to obtain precipitation depth in mm:
  $$\text{Depth}_{\text{frame}} (\text{mm}) = \text{Rate} (\text{mm/hr}) \times 0.5\,\text{hr}$$
  $$\text{Rainfall}_{24\text{h}} = \sum_{t=T-24\text{h}}^{T} \text{Depth}_t, \quad \text{Rainfall}_{72\text{h}} = \sum_{t=T-72\text{h}}^{T} \text{Depth}_t$$

---

## 2. Strict Temporal Causality & Anti-Leakage Windows
To prevent temporal data leakage, rainfall is strictly integrated over periods occurring **BEFORE or AT** the exact Sentinel-1 post-event acquisition timestamp $T$. No post-event rainfall is included.

| Event ID | Event Type | Sentinel-1 Post Timestamp ($T$) | 24-Hour Integration Window ($T-24\text{h} \to T$) | 72-Hour Integration Window ($T-72\text{h} \to T$) | Frames (24h / 72h) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EV_2018_07_19** | Flood | `2018-07-19 01:02:53 UTC` | `2018-07-18 01:00:00` to `2018-07-19 01:00:00 UTC` | `2018-07-16 01:00:00` to `2018-07-19 01:00:00 UTC` | 48 / 144 |
| **EV_2018_08_24** | Flood | `2018-08-24 01:02:55 UTC` | `2018-08-23 01:00:00` to `2018-08-24 01:00:00 UTC` | `2018-08-21 01:00:00` to `2018-08-24 01:00:00 UTC` | 48 / 144 |
| **EV_2019_09_24** | Flood | `2019-09-24 01:03:02 UTC` | `2019-09-23 01:00:00` to `2019-09-24 01:00:00 UTC` | `2019-09-21 01:00:00` to `2019-09-24 01:00:00 UTC` | 48 / 144 |
| **EV_2023_07_29** | Flood | `2023-07-29 01:03:31 UTC` | `2023-07-28 01:00:00` to `2023-07-29 01:00:00 UTC` | `2023-07-26 01:00:00` to `2023-07-29 01:00:00 UTC` | 48 / 144 |
| **CTRL_2024_03_12** | Control | `2024-03-13 01:03:29 UTC` | `2024-03-12 01:00:00` to `2024-03-13 01:00:00 UTC` | `2024-03-10 01:00:00` to `2024-03-13 01:00:00 UTC` | 48 / 144 |

---

## 3. Spatial Aggregation Methodology
- The Sentinel-1 flood labels operate on a high-resolution **500 m × 500 m grid** (6,752 cells covering Mumbai).
- NASA GPM IMERG has an intrinsically coarser grid resolution of **0.1° (~11 km)**.
- **Scientific Integrity Rule**: We explicitly acknowledge the resolution difference and do **not** interpolate or invent artificial 500 m spatial precipitation patterns.
- The spatial aggregation computes the study area areal mean across the Mumbai bounding box (`[72.75, 18.85, 73.05, 19.32]`).
- The event-level precipitation forcing is linked to all 6,752 cells for that event, with spatial resolution metadata (`rainfall_spatial_resolution = "0.1_degree"`) explicitly documented in every row.

---

## 4. Execution Workflow
1. **Google Earth Engine Extraction**:
   - Open [gee_imerg_rainfall_extractor.js](file:///c:/Users/aceru/flood/gee_imerg_rainfall_extractor.js) in the GEE Code Editor.
   - Click **Run**.
   - The exact intermediate event rainfall table will be printed to the Console and exported to Google Drive as `mumbai_imerg_rainfall_events.csv`.
2. **Dataset Merging & QA Audit**:
   - Place `mumbai_imerg_rainfall_events.csv` into `data/events/`.
   - Run:
     ```bash
     python data/events/build_imerg_rainfall.py
     ```
   - This script runs the 10-point Quality Assurance audit and generates `mumbai_real_500m_event_dataset.csv` (33,760 rows).
