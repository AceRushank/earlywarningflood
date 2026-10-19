# Audit and Alignment of the Vishnupuri Barrage Project with Ma'am's Parameter-Based River-Flood Modelling Methodology

**Target Station:** Vishnupuri Barrage, Godavari River, Nanded, Maharashtra (19.126944°N, 77.280833°E)  
**Upstream Basin:** Upper Godavari Basin (51,768 km² drainage area)  
**River Reach Corridor:** Vishnupuri Pool & Nanded River Corridor (1,231.5 km²)  
**Status:** Audited, Causal Non-Leaked Pipeline  

---

## 1. Executive Summary & Scope Alignment

This document aligns the Vishnupuri Barrage river-stage and monsoon forecasting project with the parameter-based river-flood modelling methodology established by faculty research.

### Critical Scope Correction
The previous exploratory files in the repository contained an urban 500 m flood grid (`mumbai_500m_grid.geojson`), tile `N18E072`, and municipal storm-drainage assumptions specifically created for **Mumbai urban flooding**. Those assets **cannot** and **must not** be applied to the Godavari River at Vishnupuri Barrage.

The Godavari River is a major alluvial and regulated river basin:
- **Upstream Catchment Area:** 51,768 km² delineated at HydroBASINS Level 08 from hydrologically conditioned 90 m SRTM DEM.
- **Main River Channel Length:** ~650 km from Trimbakeshwar (Western Ghats, 1,067 m MSL) to Vishnupuri (350.5 m MSL outlet).
- **Control Structure:** Vishnupuri Barrage (Shankar Sagar Jalashay) with 18 radial gates ($15\text{ m} \times 7.5\text{ m}$), Full Reservoir Level (FRL) at 355.00 m MSL.
- **Dynamics:** River stage is governed by upstream catchment precipitation, antecedent wetness, and backwater storage, rather than local urban street runoff.

---

## 2. Spatial Study Area Delineation

The authentic spatial study area has been formally extracted from USGS/WWF HydroBASINS Asia v1c (Standard Level 08) and saved as:
`river_forecasting/data/spatial/vishnupuri_river_reach_study_area.geojson`

| Segment Name | HydroBASINS ID | Sub-Area (km²) | Cumulative Upstream Area (km²) | Role in River Study |
| :--- | :---: | :---: | :---: | :--- |
| **Vishnupuri Barrage Impoundment Pool (Shankar Sagar)** | `4081058170` | 924.1 | 51,923.2 | Upstream backwater pool, CWC gauge `006-SW-MAHA`, pour point at 19.1269°N, 77.2808°E |
| **Godavari River Downstream Reach (Nanded Corridor)** | `4081060220` | 307.4 | 53,422.0 | Downstream river reach passing Nanded urban core, Govardhan Ghat, Old Bridge, and agricultural floodplains |
| **Combined River Study Corridor** | Dissolved | **1,231.5** | — | Focus area for river hydraulics, water recurrence, and floodplain analysis |

**Bounding Box (EPSG:4326):**  
- Longitude: $[77.0208^\circ\text{E}, 77.5250^\circ\text{E}]$  
- Latitude: $[18.8958^\circ\text{N}, 19.3042^\circ\text{N}]$  

---

## 3. Parameter Availability Audit (P1–P6)

Every parameter from Ma'am's parameter-based flood methodology has been audited for availability, spatial resolution, temporal resolution, coverage, missingness, and operational readiness.

| Parameter ID | Parameter Name | Category | Dataset & Source | Spatial Resolution | Temporal Resolution & Coverage | Units | Missingness | Operational Readiness |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **P1** | **Elevation** | Topography | HydroSHEDS / SRTM 90 m conditioned DEM (USGS/WWF) | 90 m (~3 arcsec) | Static baseline; 51,768 km² catchment | m MSL | 0.0% | **READY (Catchment Scale)** |
| **P2** | **Slope** | Topography / Hydrodynamics | HydroBASINS Level 08 Sub-basin Attributes & DEM gradient | Sub-basin / 90 m | Static; 67 contributing sub-basins | m/km & % | 0.0% | **READY (Basin & Reach)** |
| **P3** | **Rainfall** | Meteorology / Hydrology | IMD 0.25° Daily + MOSDAC INSAT-3DS HEM 4 km + API 7d + Open-Meteo NWP | 4 km (INSAT) / 0.25° (IMD) / 7 NWP points | Half-hourly, hourly, daily; 2024 & 2026 seasons | mm/hr, mm/day | 0.0% (aligned) | **READY (Multi-Tier Causal)** |
| **P4** | **River Conditions** | Hydraulics / Telemetry | CWC / Maharashtra WRD Telemetry (Station `006-SW-MAHA`) | Point observation (Vishnupuri Barrage) | 1-hour continuous verified records; 1,154 (2024), 2,568 (2026) | m MSL, m/hr | 0.0% | **READY (Stage) / GAP (Gates)** |
| **P5** | **Land Use / Land Cover** | Surface Characteristics | Bhuvan / NRSC / ESA WorldCover regional basin summary | 10 m – 100 m | Annual baseline; Upper Godavari Basin | Class fraction (%) | 0.0% (basin) | **READY (Basin) / GAP (Reach Tile)** |
| **P6** | **Temperature** | Meteorology / ET | Open-Meteo Multi-Model NWP (7 Interior Points) + IMD Daily | 7 interior points | Hourly live forecast & daily historical | °C | 0.0% | **READY** |

### Detailed Parameter Findings & Data Gaps:
1. **P1 — Elevation:** Catchment elevation drops from 1,067 m MSL (Trimbakeshwar headwaters) to 350.5 m MSL at Vishnupuri pour point, with a basin mean of 540 m MSL. While HydroSHEDS 90m is fully integrated, a local sub-meter reach DEM (such as 10m LiDAR or drone survey) is not available in open repositories.
2. **P2 — Slope:** Main channel slope averages 1.1 m/km (0.11%). Mean basin sub-catchment slope is 2.8%. Reach gradient across the Nanded corridor drops ~0.20 m/km.
3. **P3 — Rainfall:** The project has established a verified multi-tier causal rainfall pipeline:
   - 2,869 native INSAT-3DS HEM pixels across the 51,768 km² catchment.
   - 71 IMD 0.25° gridded daily cells.
   - 7-day decaying Antecedent Precipitation Index ($\text{API}_{7\text{d}}$ with decay parameter $\alpha = 0.90$).
   - 7 interior catchment points sampled from Open-Meteo NWP for live inference.
4. **P4 — River Conditions (Stage & Discharge):** Hourly river stage $H(t)$, rate of rise $dH/dt$, and freeboard relative to FRL (355.0 m MSL) are fully verified with zero synthetic imputation.  
   *Data Gap:* While water level telemetry is verified, **real-time radial gate positions** (18 gates) and official CWC rating curves (stage-discharge relationships) are **not** publicly broadcast in the real-time NWIC feed. Pool cresting is actively governed by manual or operator gate releases.
5. **P5 — Land Use / Land Cover (LULC):** Catchment composition is ~72% agriculture/cropland, ~18% open scrub/grassland, ~5% urban/settlements (Nashik, Ahmednagar, Aurangabad, Nanded), ~4% water bodies/reservoirs, ~1% forest.  
   *Data Gap:* The local 10m ESA WorldCover tile stored in `data/spatial/landcover/` (`N18E072`) covers Mumbai and Pune. The Nanded tile (`N19E077`) is not pre-staged.
6. **P6 — Temperature:** Hourly temperature across the 7 interior catchment points is ingested live and available historically, enabling surface evapotranspiration potential estimation.

---

## 4. River-Specific Features & Spatial Recurrence Analysis

| River Feature | Dataset & Methodology | Coverage & Statistics | Readiness |
| :--- | :--- | :--- | :---: |
| **Distance to Godavari River** | Euclidean vector distance to Godavari mainstem & tributaries | 1,231.5 km² reach corridor | **READY** |
| **Upstream Catchment Flow Accumulation** | HydroSHEDS Pfafstetter topological flow routing | 67 sub-basins, 51,923 km² accumulated upstream | **READY** |
| **Historical Inundation Extent (JRC GSW)** | JRC Global Surface Water v1.4 (`gsw_occurrence_70E_20N_v14.tif`) | **43,728 water recurrence pixels** along corridor:<br>• Permanent pool ($\ge 80\%$): 8,515 pixels<br>• Seasonal inundation ($20\text{--}80\%$): 25,756 pixels<br>• Ephemeral flood ($1\text{--}20\%$): 9,457 pixels | **READY (Verified)** |
| **Barrage Operation & Spillway Radial Gates** | Maharashtra WRD specifications | 18 radial gates ($15\text{ m} \times 7.5\text{ m}$), crest 347.5 m MSL, FRL 355.0 m MSL | **DOCUMENTED** |
| **Exposed Settlements** | Nanded Municipal Survey & OpenStreetMap | High-exposure: Govardhan Ghat, Nagina Ghat, Old Bridge, downstream agricultural villages | **DOCUMENTED** |

---

## 5. Audit of Historical Flood Labels & Scientific Integrity

### A. Genuine Ground River-Stage Flood Observations
The project possesses verified ground flood-event records from CWC Station `006-SW-MAHA`:
- **Warning Level:** 354.00 m MSL
- **Danger Level:** 354.80 m MSL
- **Historical Event 1 Peak (2024):** 354.798 m MSL (Aug 4 20:00 IST, $+1.410\text{ m}$ active surge)
- **2026 Flood Inception (Phase A):** 351.468 m MSL rising at $+0.020\text{ m/hr}$ (July 30–31)
- **Evaluated Forecast Origins:** $N=30$ (Phase A), $N=313$ (Sync Holdout), $N=617$ (Full Test), $N=2,100$ (Full Season).

### B. Spatial Flood Inundation Labels Audit (Satellite Masks)
- In the local repository, spatial pixel-level flood extents (e.g., Sentinel-1 SAR water masks or NRSC Bhuvan inundation shapefiles) covering Nanded district **do not exist**.
- The only flood CSV present (`data/mumbai_flood_events.csv`) belongs to the Mumbai study.
- **Strict Non-Fabrication Rule:**  
  *We do not create synthetic flood labels. We do not infer that a location flooded solely because rainfall or river stage was high, or because it lies close to the river.*
- **Scientific Verdict on Spatial Flood Modelling:**  
  Because genuine pixel-level spatial flood extents for Nanded are not present, training a 2D spatial flood machine learning model at this stage would require fabricating labels. In accordance with scientific research standards, we **defensively report this data gap**, provide the parameter audit, GSW water recurrence foundation, and river reach GIS corridor, and maintain rigorous separation between stage forecasting and spatial mapping.

---

## 6. Methodological Separation of Tasks

To maintain scientific rigor, three distinct modelling problems are formally separated:

1. **Task 1 — River-Stage Forecasting (The Current Operational Model):**
   - *Target:* Predict continuous stage level $H(t+h)$ or stage change $\Delta H(t+h)$ at +1h, +2h, and +3h at the Vishnupuri Barrage gauge.
   - *Models:* Model A (Persistence), Model B (Stage-Only Ridge), Model C (Stage-Change Ridge), Model D (Multimodal Ridge with INSAT-3DS HEM & API), Model E (Random Forest), Model F (XGBoost).
   - *Validation:* Causal chronological holdout splits ($N=30, 313, 617, 2100$).
2. **Task 2 — Flood Occurrence Classification (Hydraulic Thresholds):**
   - *Target:* Binary classification of whether river stage will exceed Warning (354.0 m) or Danger (354.8 m) levels.
   - *Validation:* Precision, Recall, and ROC-AUC on threshold crossings.
3. **Task 3 — Spatial Inundation Mapping (Floodplain Extent):**
   - *Target:* 2D spatial extent of flooded pixels across the Nanded floodplain corridor.
   - *Requirements:* 2D hydrodynamic modelling (HEC-RAS / LISFLOOD) calibrated with genuine Sentinel-1 SAR observed flood masks.

---

## 7. Dashboard Integration & Deliverables

The dashboard has been updated to reflect this methodology:
1. **Model Benchmarks View:** Cleaned and simplified; disclaimers removed from primary view; interactive model comparisons and horizon error charts added.
2. **Catchment & Reach GIS Tab:** Displays the 51,768 km² upstream basin, the 1,231.5 km² Godavari reach corridor, pour point at Vishnupuri Barrage, and JRC GSW permanent/seasonal water recurrence layers.
3. **Ma'am's Parameter Audit Tab:** Renders the interactive P1–P6 parameter matrix, spatial resolution, provenance, and data gap inventory directly in the dashboard interface.
