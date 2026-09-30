# -*- coding: utf-8 -*-
"""
extract_gsw_to_grid.py
Reprojects JRC Global Surface Water occurrence tile to UTM43N,
extracts water_pct per 500m grid cell, and updates:
  data/spatial/mumbai_500m_grid.geojson
  data/spatial/mumbai_500m_grid_features.csv
  outputs/spatial_features_qc.txt
  outputs/spatial_foundation_summary.txt
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from pyproj import CRS

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPATIAL_DIR = os.path.join(BASE_DIR, "data", "spatial")
GSW_DIR     = os.path.join(SPATIAL_DIR, "gsw")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")

GRID_GEOJSON    = os.path.join(SPATIAL_DIR, "mumbai_500m_grid.geojson")
GRID_CSV        = os.path.join(SPATIAL_DIR, "mumbai_500m_grid_features.csv")
FEAT_QC_TXT     = os.path.join(OUT_DIR, "spatial_features_qc.txt")
SUMMARY_TXT     = os.path.join(OUT_DIR, "spatial_foundation_summary.txt")

GSW_PATH        = os.path.join(GSW_DIR, "gsw_occurrence_70E_20N_v14.tif")
gsw_clip_path   = os.path.join(GSW_DIR, "gsw_occurrence_aoi_clip.tif")
gsw_reproj_path = os.path.join(GSW_DIR, "mumbai_gsw_occurrence_utm43n.tif")

EPSG_WGS84 = 4326
EPSG_UTM   = 32643
CRS_UTM    = CRS.from_epsg(EPSG_UTM)

AOI_LON_MIN, AOI_LAT_MIN = 72.70, 18.85
AOI_LON_MAX, AOI_LAT_MAX = 73.10, 19.30

print("Loading 500m grid...")
grid_wgs = gpd.read_file(GRID_GEOJSON)
grid     = grid_wgs.to_crs(EPSG_UTM)
n_cells  = len(grid)
print("  Cells: %d" % n_cells)

# ── Clip GSW to AOI ─────────────────────────────────────────────────────────
print("Clipping GSW tile to Mumbai AOI...")
aoi_box = box(AOI_LON_MIN - 0.1, AOI_LAT_MIN - 0.1,
              AOI_LON_MAX + 0.1, AOI_LAT_MAX + 0.1)

with rasterio.open(GSW_PATH) as src:
    clipped, clip_transform = rio_mask(
        src, [mapping(aoi_box)], crop=True, filled=True, nodata=255)
    clip_meta = src.meta.copy()
    clip_meta.update({
        "height": clipped.shape[1], "width": clipped.shape[2],
        "transform": clip_transform, "nodata": 255,
    })
    with rasterio.open(gsw_clip_path, "w", **clip_meta) as tmp:
        tmp.write(clipped)
print("  Clipped: %d x %d pixels" % (clipped.shape[2], clipped.shape[1]))

# ── Reproject to UTM43N ──────────────────────────────────────────────────────
print("Reprojecting GSW to UTM43N (30m)...")
with rasterio.open(gsw_clip_path) as src2:
    transform_utm, w2, h2 = calculate_default_transform(
        src2.crs, CRS_UTM, src2.width, src2.height, *src2.bounds)
    meta_utm = src2.meta.copy()
    meta_utm.update({
        "crs": CRS_UTM, "transform": transform_utm,
        "width": w2, "height": h2, "nodata": 255, "compress": "lzw",
    })
    with rasterio.open(gsw_reproj_path, "w", **meta_utm) as dst2:
        reproject(
            source=rasterio.band(src2, 1),
            destination=rasterio.band(dst2, 1),
            src_crs=src2.crs, dst_crs=CRS_UTM,
            resampling=Resampling.nearest,
        )
print("  Reprojected: %s" % os.path.basename(gsw_reproj_path))

# ── Extract water_pct per cell ───────────────────────────────────────────────
print("Extracting water_pct to 500m grid (%d cells)..." % n_cells)
water_pct = np.full(n_cells, np.nan)

with rasterio.open(gsw_reproj_path) as src3:
    grid_src_crs = grid.to_crs(src3.crs)
    for i, geom in enumerate(grid_src_crs.geometry):
        if i % 1000 == 0:
            print("  ... cell %d / %d" % (i, n_cells))
        try:
            out_img, _ = rio_mask(
                src3, [mapping(geom)], crop=True,
                nodata=255, filled=True, all_touched=True)
            data = out_img[0].astype(float)
            valid = data[data != 255]
            valid = valid[np.isfinite(valid)]
            if len(valid) > 0:
                # Fraction of pixels with occurrence >= 10
                # (GSW occurrence: 0=never, 100=always, 255=nodata)
                water_pct[i] = float(np.sum(valid >= 10)) / len(valid)
        except Exception:
            pass

n_valid_water = int(np.sum(~np.isnan(water_pct)))
print("  water_pct: %d valid / %d total" % (n_valid_water, n_cells))
print("  Cells with water_pct > 0  : %d" % int(np.sum(water_pct > 0)))
print("  Cells with water_pct > 0.1: %d" % int(np.sum(water_pct > 0.1)))
print("  Max water_pct             : %.4f" % float(np.nanmax(water_pct)))

# ── Update grid files ─────────────────────────────────────────────────────────
print("Updating grid GeoJSON and CSV...")
grid_wgs["water_pct_gsw"] = np.round(water_pct, 4)
grid_wgs.to_file(GRID_GEOJSON, driver="GeoJSON")
print("  GeoJSON updated: %s" % GRID_GEOJSON)

feat_df = pd.read_csv(GRID_CSV)
feat_df["water_pct_gsw"] = np.round(water_pct, 4)
feat_df.to_csv(GRID_CSV, index=False)
print("  CSV updated: %s" % GRID_CSV)

# ── Update QC report ─────────────────────────────────────────────────────────
print("Updating QC and summary reports...")

# Read existing QC and replace GSW section
with open(FEAT_QC_TXT, "r", encoding="utf-8") as f:
    qc_text = f.read()

gsw_section_old = "       Status    : PENDING (download failed)"
gsw_section_new = "       Status    : DOWNLOADED AND EXTRACTED"
qc_text = qc_text.replace(gsw_section_old, gsw_section_new, 1)

# Update water_pct stats
water_valid = water_pct[~np.isnan(water_pct)]
stats_new = (
    "     Status      : OK\n"
    "     n_valid     : %d / %d\n"
    "     n_missing   : %d\n"
    "     min / max   : %.4f / %.4f\n"
    "     mean        : %.4f\n"
) % (n_valid_water, n_cells, n_cells - n_valid_water,
     float(water_valid.min()), float(water_valid.max()), float(water_valid.mean()))

# Replace the ALL MISSING placeholder for water_pct_gsw
old_water_status = (
    "     Status      : ALL MISSING (PENDING)\n"
    "     n_valid     : 0 / %d\n"
    "     n_missing   : %d\n"
    "     min / max   : None / None\n"
    "     mean        : None\n"
) % (n_cells, n_cells)
qc_text = qc_text.replace(old_water_status, stats_new, 1)

with open(FEAT_QC_TXT, "w", encoding="utf-8") as f:
    f.write(qc_text)
print("  Feature QC updated.")

# Update summary
with open(SUMMARY_TXT, "r", encoding="utf-8") as f:
    sum_text = f.read()

sum_text = sum_text.replace(
    "       Status    : PENDING (download failed)",
    "       Status    : DOWNLOADED AND EXTRACTED", 1)
sum_text = sum_text.replace(
    "   water_pct_gsw     : 0 / %d cells with valid values" % n_cells,
    "   water_pct_gsw     : %d / %d cells with valid values" % (n_valid_water, n_cells), 1)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(sum_text)
print("  Summary updated.")

print("\nGSW extraction complete.")
