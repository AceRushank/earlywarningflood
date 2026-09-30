# Mumbai Micro-Level Early Warning Flood Forecasting System

A high-resolution (500m grid) hydrometeorological and spatial machine learning framework for micro-level flood forecasting and risk evaluation in Mumbai, India.

---

## 📌 System Overview

Standard flood alerts for Mumbai often rely on coarse administrative or city-wide warnings (e.g., Santacruz vs. Colaba point stations). This system transitions flood risk estimation to a **500m micro-grid foundation**, integrating:

1. **High-Resolution Topographic & Spatial Foundations (500m AOI Grid)**:
   - Copernicus DEM 30m topographic parameters (elevation, slope, terrain morphometry).
   - JRC Global Surface Water (GSW) long-term water occurrence and inundation baselines.
2. **Satellite Hydrometeorological Forcing**:
   - ISRO MOSDAC INSAT-3D / INSAT-3DR Hydro-Estimator Method (HEM) half-hourly rainfall observations and short-term forecasting.
   - IMD 0.25° gridded daily precipitation series matched with multi-decade historical flood events.
3. **Historical Event Calibration & Risk Profiling**:
   - Comprehensive inventory of documented Mumbai flood incidents (1980–2023).
   - Multi-day antecedent rainfall index and spatial rainfall variation analysis.
4. **Interactive Geospatial Visualization**:
   - Self-contained interactive Leaflet-based spatial exploration interface (`outputs/mumbai_spatial_features_map.html`).

---

## 🗂️ Repository Structure

```text
├── data/
│   ├── historical/
│   │   ├── mumbai_flood_events.csv                 # Curated historical flood inventory (1980-2023)
│   │   ├── mumbai_daily_rainfall_imd.csv           # IMD 0.25° gridded daily rainfall series
│   │   └── mumbai_flood_events_with_rainfall.csv   # Event inventory matched with cumulative rainfall
│   ├── mosdac/
│   │   ├── mumbai_rainfall_30min.csv               # Extracted half-hourly INSAT-3D/3DR rainfall series
│   │   ├── mosdac_forecast_dataset.csv             # Feature-engineered forecast dataset
│   │   └── hdf_sample/                             # Sample rainfall extraction references
│   └── spatial/
│       ├── mumbai_500m_grid.geojson                # Unified 500m micro-grid geometry
│       └── mumbai_500m_grid_features.csv           # Elevation, slope, and GSW water percentage per cell
├── outputs/
│   ├── historical_analysis/                        # Visual distributions and event timelines
│   │   ├── antecedent_rainfall_distribution.png
│   │   ├── major_flood_event_timelines.png
│   │   ├── rainfall_event_distribution.png
│   │   └── rainfall_spatial_variation.png
│   ├── mumbai_spatial_features_map.html            # Interactive Leaflet 500m grid map
│   ├── mosdac_forecast_predictions.csv             # Model evaluation predictions
│   └── *_qc.txt                                    # Automated Quality Control & audit logs
├── scripts/
│   ├── build_spatial_foundation.py                 # Generates 500m grid & extracts DEM/GSW features
│   ├── build_spatial_map.py                        # Generates interactive HTML visualization
│   ├── download_mosdac_hdf.py                      # SFTP/HTTP pipeline for MOSDAC INSAT-3D HDF5 archives
│   ├── extract_mosdac_hdf.py                       # Extracts Mumbai rainfall pixels from raw HDF5 files
│   ├── forecast_mosdac_rainfall.py                 # Gap-validated ML precipitation forecasting pipeline
│   ├── match_historical_rainfall.py                # Matches IMD gridded rainfall to flood events
│   ├── historical_flood_analysis.py                # Historical flood event & antecedent rainfall analysis
│   └── process_flood_inventory.py                  # Standardizes and audits flood event records
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note on Raw Satellite Data:**
> In accordance with repository storage guidelines, heavy raw raster files (`.tif`, `.nc`, `.h5`) exceeding 100MB (including raw DEM mosaics and multi-gigabyte HDF5 archives) are excluded from version control via `.gitignore`. Processed feature matrices, GeoJSON grids, and time series CSVs are fully tracked.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- GDAL / PROJ (standard geospatial dependencies)

### 2. Install Dependencies
```bash
git clone https://github.com/AceRushank/earlywarningflood.git
cd earlywarningflood
pip install -r requirements.txt
```

---

## 🛠️ Usage Workflows

### 1. Spatial Foundation & Interactive Map
Generate or re-extract spatial features across the 500m grid and generate the interactive web viewer:
```bash
python scripts/build_spatial_map.py
```
Open `outputs/mumbai_spatial_features_map.html` in any browser to inspect elevation, slope, and water occurrence across all cells.

### 2. Historical Flood & Rainfall Analysis
Match historical flood events with IMD gridded daily rainfall and generate analysis figures:
```bash
python scripts/match_historical_rainfall.py
python scripts/historical_flood_analysis.py
```

### 3. MOSDAC Satellite Rainfall Forecasting
Train and evaluate short-term (1h, 2h, 3h lead time) rainfall forecasting models using gap-corrected time series validation:
```bash
python scripts/forecast_mosdac_rainfall.py
```
Model evaluation results are exported to `outputs/mosdac_forecast_results.txt` and `outputs/mosdac_forecast_predictions.csv`.

---

## 📊 Quality Control & Validation

All processing steps output transparent audit trails:
- `outputs/mumbai_500m_grid_qc.txt`: Grid geometry and spatial consistency audits.
- `outputs/spatial_features_qc.txt`: Topographic attribute statistics and boundary checks.
- `outputs/historical_rainfall_matching_qc.txt`: IMD grid-to-event alignment verification.
- `outputs/mosdac_forecast_data_qc.txt`: Temporal gap audits and sample completeness logs.

---

## ⚖️ License
Research and educational use. Satellite data courtesy of ISRO MOSDAC and IMD.
