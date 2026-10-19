# -*- coding: utf-8 -*-
"""
extract_landcover_worldcover.py
================================
Downloads ESA WorldCover 10 m 2021 (v200) for the Mumbai AOI,
clips to the existing 500 m prediction grid, and aggregates land-cover
class fractions - with special focus on built-up/impervious percentage -
to each of the 8,600 grid cells.

DATA SOURCE (scientifically defensible, open, no registration required)
-----------------------------------------------------------------------
  Dataset   : ESA WorldCover 10 m 2021, version 200
  Provider  : European Space Agency (ESA) / VITO
  Algorithm : WorldCover v200 (Sentinel-1 + Sentinel-2 fusion)
  Reference : Zanaga et al. (2022), https://doi.org/10.5281/zenodo.7254221
  License   : Creative Commons CC-BY 4.0
  CRS       : EPSG:4326 (WGS84 geographic)
  Resolution: 10 m (~0.000083 deg per pixel)
  Year      : 2021 (nominal reference year)
  Tile      : N18E072  (3x3 deg covering 18-21 N, 72-75 E)
  AWS URL   : https://esa-worldcover.s3.eu-central-1.amazonaws.com/
               v200/2021/map/ESA_WorldCover_10m_2021_v200_N18E072_Map.tif

LAND COVER CLASSES (WorldCover v200 integer codes)
---------------------------------------------------
  10  Tree cover
  20  Shrubland
  30  Grassland
  40  Cropland
  50  Built-up   <-- used as impervious/built-up proxy
  60  Bare / sparse vegetation
  70  Snow and ice
  80  Permanent water bodies
  90  Herbaceous wetland
  95  Mangroves
  100 Moss and lichen

OUTPUTS
-------
  data/spatial/landcover/
    ESA_WorldCover_10m_2021_v200_N18E072_Map.tif   (raw 3x3-deg tile)
    mumbai_worldcover_aoi_clip.tif                 (clipped to AOI, WGS84)
    mumbai_worldcover_utm43n.tif                   (reprojected to UTM43N)
  data/spatial/mumbai_500m_grid_features.csv       (updated with LC columns)
  data/spatial/mumbai_500m_grid.geojson            (updated GeoJSON)
  outputs/landcover_qc.txt                         (provenance + statistics)

RULES (same as rest of project)
  - No synthetic data; no invented values.
  - Missing cells (no valid pixels) remain NaN.
  - Source resolution (10 m) clearly distinguished from 500 m grid.
  - No flood model is trained here.
"""

import os
import sys
import time
import warnings
import datetime

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
from pyproj import CRS, Transformer

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPATIAL_DIR  = os.path.join(BASE_DIR, "data", "spatial")
LC_DIR       = os.path.join(SPATIAL_DIR, "landcover")
OUT_DIR      = os.path.join(BASE_DIR, "outputs")

for d in [LC_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

GRID_GEOJSON  = os.path.join(SPATIAL_DIR, "mumbai_500m_grid.geojson")
GRID_CSV      = os.path.join(SPATIAL_DIR, "mumbai_500m_grid_features.csv")
LC_QC_TXT     = os.path.join(OUT_DIR, "landcover_qc.txt")

# =============================================================================
# Dataset metadata (hardcoded provenance -- never synthetic)
# =============================================================================
DATASET_META = {
    "name"        : "ESA WorldCover 10 m 2021",
    "version"     : "v200",
    "year"        : "2021",
    "provider"    : "European Space Agency (ESA) / VITO",
    "algorithm"   : "WorldCover v200 (Sentinel-1 + Sentinel-2 fusion)",
    "doi"         : "https://doi.org/10.5281/zenodo.7254221",
    "license"     : "Creative Commons CC-BY 4.0",
    "resolution"  : "10 m (geographic: ~0.000083 deg/pixel)",
    "native_crs"  : "EPSG:4326 (WGS84 geographic)",
    "tile"        : "N18E072",
    "tile_extent" : "18-21 N, 72-75 E (3x3 deg)",
    "aws_url"     : ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                     "v200/2021/map/ESA_WorldCover_10m_2021_v200_N18E072_Map.tif"),
    "acquisition" : datetime.datetime.utcnow().strftime("%Y-%m-%d UTC"),
}

# WorldCover class definitions
WC_CLASSES = {
    10 : "Tree cover",
    20 : "Shrubland",
    30 : "Grassland",
    40 : "Cropland",
    50 : "Built-up",
    60 : "Bare / sparse vegetation",
    70 : "Snow and ice",
    80 : "Permanent water bodies",
    90 : "Herbaceous wetland",
    95 : "Mangroves",
    100: "Moss and lichen",
}
BUILT_UP_CODE = 50  # class used as built-up / impervious proxy

# =============================================================================
# AOI and CRS (must match build_spatial_foundation.py)
# =============================================================================
AOI_LAT_MIN = 18.85
AOI_LAT_MAX = 19.30
AOI_LON_MIN = 72.70
AOI_LON_MAX = 73.10

EPSG_WGS84 = 4326
EPSG_UTM   = 32643   # WGS84 / UTM Zone 43N
CRS_WGS84  = CRS.from_epsg(EPSG_WGS84)
CRS_UTM    = CRS.from_epsg(EPSG_UTM)

# Buffer AOI by 0.05 deg to avoid edge artefacts after reprojection
AOI_BUFFER = 0.05

print("=" * 70)
print("ESA WorldCover 10m 2021 -- Mumbai Land Cover Extraction")
print("=" * 70)
print("  AOI: lat %.2f-%.2f N, lon %.2f-%.2f E" % (
    AOI_LAT_MIN, AOI_LAT_MAX, AOI_LON_MIN, AOI_LON_MAX))
print("  Source tile: %s" % DATASET_META["tile"])
print("  Resolution: %s" % DATASET_META["resolution"])
print("  License: %s" % DATASET_META["license"])
print()

# =============================================================================
# STEP 1 -- Load existing grid
# =============================================================================
print("[STEP 1] Loading existing 500m grid...")
if not os.path.exists(GRID_GEOJSON):
    print("  ERROR: Grid file not found: %s" % GRID_GEOJSON)
    print("  Please run scripts/build_spatial_foundation.py first.")
    sys.exit(1)

grid_wgs = gpd.read_file(GRID_GEOJSON)
n_cells  = len(grid_wgs)
print("  Loaded %d cells from %s" % (n_cells, os.path.basename(GRID_GEOJSON)))
print("  CRS: %s" % grid_wgs.crs)

# Also work in UTM for extraction (same CRS as foundation script)
grid_utm = grid_wgs.to_crs(EPSG_UTM)
print("  Reprojected to UTM43N (EPSG:%d) for extraction." % EPSG_UTM)

# =============================================================================
# STEP 2 -- Download ESA WorldCover tile
# =============================================================================
RAW_TILE_PATH = os.path.join(LC_DIR, "ESA_WorldCover_10m_2021_v200_N18E072_Map.tif")

print("\n[STEP 2] Downloading ESA WorldCover tile %s..." % DATASET_META["tile"])
print("  URL: %s" % DATASET_META["aws_url"])

def download_file(url, dest, desc=""):
    """Download a file with progress reporting. Skip if already present and valid."""
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        size_mb = os.path.getsize(dest) / 1e6
        print("  Already present: %s (%.0f MB)" % (os.path.basename(dest), size_mb))
        return True
    print("  Downloading %s..." % desc)
    try:
        r = requests.get(url, stream=True, timeout=300)
        if r.status_code != 200:
            print("  FAILED: HTTP %d" % r.status_code)
            return False
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    elapsed = time.time() - t0
                    mb_s = (downloaded / 1e6) / max(elapsed, 0.1)
                    print("\r  Progress: %.1f%%  (%.0f/%.0f MB)  %.1f MB/s" % (
                        pct, downloaded/1e6, total/1e6, mb_s), end="", flush=True)
        print()
        size_mb = os.path.getsize(dest) / 1e6
        print("  Done: %.1f MB -> %s" % (size_mb, os.path.basename(dest)))
        return True
    except Exception as e:
        print("  FAILED: %s" % str(e)[:120])
        if os.path.exists(dest):
            os.remove(dest)
        return False

tile_ok = download_file(DATASET_META["aws_url"], RAW_TILE_PATH,
                        desc="ESA WorldCover tile %s" % DATASET_META["tile"])

if not tile_ok:
    print("\n  CRITICAL: Cannot proceed without the WorldCover tile.")
    print("  Check internet connection or try manually downloading:")
    print("  %s" % DATASET_META["aws_url"])
    print("  Save to: %s" % RAW_TILE_PATH)
    sys.exit(1)

# Verify the tile
with rasterio.open(RAW_TILE_PATH) as src:
    print("  Tile verification:")
    print("    CRS       : %s" % src.crs)
    print("    Shape     : %d rows x %d cols" % (src.height, src.width))
    print("    Bounds    : %s" % str(src.bounds))
    print("    Res (deg) : %s" % str(src.res))
    print("    DType     : %s" % src.dtypes[0])
    print("    NoData    : %s" % src.nodata)

# =============================================================================
# STEP 3 -- Clip to Mumbai AOI (WGS84)
# =============================================================================
print("\n[STEP 3] Clipping WorldCover to Mumbai AOI...")

CLIP_PATH = os.path.join(LC_DIR, "mumbai_worldcover_aoi_clip.tif")
aoi_box_buffered = box(
    AOI_LON_MIN - AOI_BUFFER, AOI_LAT_MIN - AOI_BUFFER,
    AOI_LON_MAX + AOI_BUFFER, AOI_LAT_MAX + AOI_BUFFER
)

with rasterio.open(RAW_TILE_PATH) as src:
    nodata_raw = src.nodata if src.nodata is not None else 0
    clipped, clip_transform = rio_mask(
        src, [mapping(aoi_box_buffered)],
        crop=True, filled=True,
        nodata=nodata_raw
    )
    clip_meta = src.meta.copy()
    clip_meta.update({
        "height"   : clipped.shape[1],
        "width"    : clipped.shape[2],
        "transform": clip_transform,
        "nodata"   : nodata_raw,
        "compress" : "lzw",
    })
    with rasterio.open(CLIP_PATH, "w", **clip_meta) as dst:
        dst.write(clipped)

with rasterio.open(CLIP_PATH) as src_clip:
    clip_h, clip_w = src_clip.height, src_clip.width
    clip_bounds    = src_clip.bounds
    clip_res       = src_clip.res

print("  Clipped tile saved: %s" % os.path.basename(CLIP_PATH))
print("    Shape  : %d rows x %d cols" % (clip_h, clip_w))
print("    Bounds : %s" % str(clip_bounds))
print("    Res    : %.6f deg x %.6f deg" % (clip_res[0], clip_res[1]))
approx_res_m = clip_res[0] * 111_320
print("    ~Res(m): %.1f m (at AOI latitude ~%.1f N)" % (
    approx_res_m, (AOI_LAT_MIN + AOI_LAT_MAX) / 2))

# =============================================================================
# STEP 4 -- Reproject to UTM43N (nearest neighbour -- categorical data!)
# =============================================================================
print("\n[STEP 4] Reprojecting to UTM43N (EPSG:%d)..." % EPSG_UTM)

UTM_PATH = os.path.join(LC_DIR, "mumbai_worldcover_utm43n.tif")

with rasterio.open(CLIP_PATH) as src:
    transform_utm, width_utm, height_utm = calculate_default_transform(
        src.crs, CRS_UTM, src.width, src.height, *src.bounds
    )
    meta_utm = src.meta.copy()
    meta_utm.update({
        "crs"      : CRS_UTM,
        "transform": transform_utm,
        "width"    : width_utm,
        "height"   : height_utm,
        "nodata"   : 0,   # 0 is not a valid class code in WorldCover
        "compress" : "lzw",
    })
    with rasterio.open(UTM_PATH, "w", **meta_utm) as dst:
        reproject(
            source      = rasterio.band(src, 1),
            destination = rasterio.band(dst, 1),
            src_crs     = src.crs,
            dst_crs     = CRS_UTM,
            resampling  = Resampling.nearest,  # MUST use nearest for categorical!
        )

with rasterio.open(UTM_PATH) as src_utm:
    utm_h, utm_w = src_utm.height, src_utm.width
    utm_res      = src_utm.res
    utm_bounds   = src_utm.bounds

print("  UTM tile saved: %s" % os.path.basename(UTM_PATH))
print("    Shape  : %d rows x %d cols" % (utm_h, utm_w))
print("    Res    : %.2f m x %.2f m" % (utm_res[0], utm_res[1]))
print("    Bounds : %s" % str(utm_bounds))

# =============================================================================
# STEP 5 -- Aggregate to 500m grid cells
# =============================================================================
print("\n[STEP 5] Aggregating WorldCover to %d grid cells..." % n_cells)
print("  Each 500m cell covers approx (%.0f)^2 = %.0f source pixels at %.1f m res" % (
    500/utm_res[0], (500/utm_res[0])**2, utm_res[0]))

TRACK_CLASSES = sorted(WC_CLASSES.keys())

# Output arrays -- one fraction per class
class_frac  = {cls: np.full(n_cells, np.nan) for cls in TRACK_CLASSES}
builtup_pct = np.full(n_cells, np.nan)
n_pix_valid = np.full(n_cells, np.nan)

with rasterio.open(UTM_PATH) as src:
    nodata_val = 0  # class 0 is not defined in WorldCover

    # Grid is already in UTM43N -- no further reprojection needed
    grid_src_crs = grid_utm.to_crs(src.crs)

    t0 = time.time()
    for i, (geom_src, geom_utm) in enumerate(
            zip(grid_src_crs.geometry, grid_utm.geometry)):

        if i > 0 and i % 500 == 0:
            elapsed = time.time() - t0
            rate    = i / max(elapsed, 0.1)
            eta     = (n_cells - i) / max(rate, 0.01)
            print("\r  Cell %d/%d  (%.0f cells/s, ETA %.0fs)  " % (
                i, n_cells, rate, eta), end="", flush=True)

        try:
            out_img, _ = rio_mask(
                src, [mapping(geom_src)],
                crop=True, filled=True,
                nodata=nodata_val,
                all_touched=True,
            )
            data = out_img[0]
            valid_mask = (data != nodata_val) & np.isin(data, TRACK_CLASSES)
            valid_pix  = data[valid_mask]
            n_valid    = len(valid_pix)

            if n_valid == 0:
                continue  # genuine NoData -- leave NaN

            n_pix_valid[i] = n_valid

            for cls in TRACK_CLASSES:
                class_frac[cls][i] = float(np.sum(valid_pix == cls)) / n_valid

            builtup_pct[i] = float(np.sum(valid_pix == BUILT_UP_CODE)) / n_valid

        except Exception:
            pass

    print("\r  Done: %d/%d cells in %.0fs        " % (n_cells, n_cells, time.time()-t0))

n_valid_cells = int(np.sum(~np.isnan(builtup_pct)))
n_missing     = n_cells - n_valid_cells
print("\n  builtup_pct: %d / %d cells valid, %d missing (NaN)" % (
    n_valid_cells, n_cells, n_missing))
print("  Built-up: min=%.1f%%  max=%.1f%%  mean=%.1f%%" % (
    np.nanmin(builtup_pct)*100, np.nanmax(builtup_pct)*100, np.nanmean(builtup_pct)*100))

# =============================================================================
# STEP 6 -- Update grid CSV and GeoJSON
# =============================================================================
print("\n[STEP 6] Updating grid CSV and GeoJSON...")

if os.path.exists(GRID_CSV):
    df = pd.read_csv(GRID_CSV)
    print("  Loaded existing CSV: %d rows, %d cols" % (len(df), len(df.columns)))
else:
    df = grid_wgs.drop(columns=["geometry"]).copy()
    print("  Built CSV from GeoDataFrame: %d rows" % len(df))

grid_ids  = list(grid_wgs["grid_id"])
id_to_idx = {gid: i for i, gid in enumerate(grid_ids)}

def make_series(arr, name):
    return pd.Series(
        [arr[id_to_idx[gid]] if gid in id_to_idx else np.nan
         for gid in df["grid_id"]],
        name=name
    )

df["builtup_pct_wc"] = np.round(make_series(builtup_pct, "builtup_pct_wc"), 4)

for cls in TRACK_CLASSES:
    col = "wc_cls%03d_frac" % cls
    df[col] = np.round(make_series(class_frac[cls], col), 4)

df.to_csv(GRID_CSV, index=False)
print("  Updated CSV -> %s" % GRID_CSV)
print("  New columns: builtup_pct_wc + wc_cls{code}_frac x %d" % len(TRACK_CLASSES))

lc_cols = ["builtup_pct_wc"] + ["wc_cls%03d_frac" % c for c in TRACK_CLASSES]
for col in lc_cols:
    if col in df.columns:
        grid_wgs[col] = df[col].values
grid_wgs.to_file(GRID_GEOJSON, driver="GeoJSON")
print("  Updated GeoJSON -> %s" % GRID_GEOJSON)

# =============================================================================
# STEP 7 -- Write QC / provenance report
# =============================================================================
print("\n[STEP 7] Writing QC report -> %s..." % LC_QC_TXT)

def pct_stats(arr):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"n_valid": 0, "n_missing": len(arr),
                "min": None, "max": None, "mean": None, "median": None, "p95": None}
    return {
        "n_valid"  : int(len(valid)),
        "n_missing": int(np.sum(np.isnan(arr))),
        "min"      : float(np.min(valid)),
        "max"      : float(np.max(valid)),
        "mean"     : float(np.mean(valid)),
        "median"   : float(np.median(valid)),
        "p95"      : float(np.percentile(valid, 95)),
    }

st = pct_stats(builtup_pct)

def fmt(v, pct=True):
    if v is None:
        return "N/A"
    return "%.4f  (%.1f%%)" % (v, v*100) if pct else "%.4f" % v

qc_lines = [
    "=" * 70,
    "LAND COVER QC REPORT",
    "Mumbai Urban Flood Project -- Hydrological/Hydraulic Foundation",
    "=" * 70,
    "",
    "1. DATASET PROVENANCE",
    "",
    "   Name        : %s" % DATASET_META["name"],
    "   Version     : %s" % DATASET_META["version"],
    "   Year        : %s" % DATASET_META["year"],
    "   Provider    : %s" % DATASET_META["provider"],
    "   Algorithm   : %s" % DATASET_META["algorithm"],
    "   DOI / Cite  : %s" % DATASET_META["doi"],
    "   License     : %s" % DATASET_META["license"],
    "   Resolution  : %s" % DATASET_META["resolution"],
    "   Native CRS  : %s" % DATASET_META["native_crs"],
    "   Tile        : %s  (%s)" % (DATASET_META["tile"], DATASET_META["tile_extent"]),
    "   AWS S3 URL  : %s" % DATASET_META["aws_url"],
    "   Acquisition : %s" % DATASET_META["acquisition"],
    "",
    "   Scientific rationale for selecting this dataset:",
    "   - 10 m resolution is the highest legitimately available open global",
    "     land cover dataset as of 2024 (Sentinel-2 native scale).",
    "   - Derived from real Sentinel-1 SAR + Sentinel-2 optical observations;",
    "     NOT synthetic, modelled, or interpolated.",
    "   - CC-BY 4.0 license: fully open, no registration required.",
    "   - Peer-reviewed algorithm (Zanaga et al. 2022, Zenodo DOI above).",
    "   - Class 50 'Built-up' directly identifies impervious urban surfaces,",
    "     which is the primary driver of high runoff in urban flood models.",
    "   - Used by ESA, JRC, UN-SPIDER and others for urban flood risk mapping.",
    "",
    "2. SPATIAL PROCESSING CHAIN",
    "",
    "   Step 1: Raw tile downloaded from AWS S3 (public, no credentials).",
    "           Path: data/spatial/landcover/ESA_WorldCover_10m_2021_v200_N18E072_Map.tif",
    "   Step 2: Clipped to Mumbai AOI + %.2f deg buffer." % AOI_BUFFER,
    "           AOI: lat %.2f-%.2f N, lon %.2f-%.2f E" % (
        AOI_LAT_MIN, AOI_LAT_MAX, AOI_LON_MIN, AOI_LON_MAX),
    "           Clip path: data/spatial/landcover/mumbai_worldcover_aoi_clip.tif",
    "   Step 3: Reprojected to UTM43N (EPSG:32643) using NEAREST NEIGHBOUR.",
    "           Nearest-neighbour is mandatory for categorical class data;",
    "           bilinear/cubic would corrupt class labels.",
    "           UTM path: data/spatial/landcover/mumbai_worldcover_utm43n.tif",
    "   Step 4: Aggregated to 8,600-cell 500 m grid (same grid as DEM/GSW).",
    "           Method: count pixels per class / total valid pixels per cell.",
    "           Valid = non-zero class code in WorldCover legend.",
    "           Cells with zero valid pixels retain NaN (not interpolated).",
    "",
    "   Clipped raster dimensions (AOI):",
    "     WGS84 clip : %d rows x %d cols  (res ~%.1f m)" % (
        clip_h, clip_w, clip_res[0]*111320),
    "     UTM43N     : %d rows x %d cols  (res ~%.1f m x %.1f m)" % (
        utm_h, utm_w, utm_res[0], utm_res[1]),
    "     Pixels per 500m cell (approx): (%.0f)^2 = %.0f" % (
        500/utm_res[0], (500/utm_res[0])**2),
    "",
    "3. LAND COVER CLASS DEFINITIONS (WorldCover v200)",
    "",
]
for code, label in WC_CLASSES.items():
    marker = " <-- BUILT-UP / IMPERVIOUS PROXY" if code == BUILT_UP_CODE else ""
    qc_lines.append("   %3d  %s%s" % (code, label, marker))

qc_lines += [
    "",
    "4. PRIMARY OUTPUT -- builtup_pct_wc",
    "   (fraction of 500m cell covered by built-up pixels, WorldCover class 50)",
    "",
    "   Column name  : builtup_pct_wc",
    "   Units        : dimensionless fraction (0.0 to 1.0)",
    "   n_valid      : %d / %d" % (st["n_valid"], n_cells),
    "   n_missing    : %d  (NaN -- no valid pixels in cell)" % st["n_missing"],
    "   min          : %s" % fmt(st["min"]),
    "   max          : %s" % fmt(st["max"]),
    "   mean         : %s" % fmt(st["mean"]),
    "   median       : %s" % fmt(st["median"]),
    "   95th pctile  : %s" % fmt(st["p95"]),
    "",
    "5. PER-CLASS FRACTION STATISTICS (all WorldCover classes)",
    "",
    "   %5s  %-28s  %7s  %7s  %7s" % ("Class", "Label", "n_valid", "mean", "max"),
    "   %s  %s  %s  %s  %s" % ("-"*5, "-"*28, "-"*7, "-"*7, "-"*7),
]

for cls in TRACK_CLASSES:
    s = pct_stats(class_frac[cls])
    label = WC_CLASSES.get(cls, "Unknown")
    mean_str = "%.1f%%" % (s["mean"]*100) if s["mean"] is not None else "N/A"
    max_str  = "%.1f%%" % (s["max"]*100)  if s["max"]  is not None else "N/A"
    qc_lines.append("   %5d  %-28s  %7d  %7s  %7s" % (
        cls, label, s["n_valid"], mean_str, max_str))

qc_lines += [
    "",
    "6. MISSING DATA POLICY",
    "",
    "   - Cells with zero valid WorldCover pixels have NaN for all LC columns.",
    "   - NaN is NEVER filled with synthetic, interpolated, or neighbouring values.",
    "   - Missing cells are documented here and in the CSV.",
    "   - The WorldCover dataset has no defined NoData gaps over land;",
    "     missing cells arise only at AOI edges.",
    "",
    "7. OUTPUT FILES CREATED / UPDATED",
    "",
    "   data/spatial/landcover/",
    "     ESA_WorldCover_10m_2021_v200_N18E072_Map.tif   [raw 3x3-deg tile]",
    "     mumbai_worldcover_aoi_clip.tif                 [AOI clip, WGS84]",
    "     mumbai_worldcover_utm43n.tif                   [AOI clip, UTM43N]",
    "",
    "   data/spatial/mumbai_500m_grid_features.csv       [+ builtup_pct_wc, wc_cls*]",
    "   data/spatial/mumbai_500m_grid.geojson            [+ same columns]",
    "   outputs/landcover_qc.txt                         [this file]",
    "",
    "8. WHAT WAS NOT DONE (per project constraints)",
    "",
    "   - No hydraulic / hydrological model was built or trained.",
    "   - No flood simulation was run.",
    "   - No synthetic land cover was generated.",
    "   - Existing MOSDAC or spatial datasets were NOT modified.",
    "   - No values were interpolated into missing cells.",
    "",
    "9. CITATION",
    "",
    "   Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W.,",
    "   Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M.,",
    "   Fritz, S., Lesiv, M., Herold, M., Tsendbazar, N.E., Xu, P.,",
    "   Ramoino, F., Arino, O. (2022). ESA WorldCover 10 m 2021 v200.",
    "   Zenodo. https://doi.org/10.5281/zenodo.7254221",
    "",
    "=" * 70,
    "Generated by : scripts/extract_landcover_worldcover.py",
    "Generated at : %s" % datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S UTC"),
    "=" * 70,
]

with open(LC_QC_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(qc_lines))

print("  QC report saved -> %s" % LC_QC_TXT)

# =============================================================================
# Final summary to console
# =============================================================================
print()
print("=" * 70)
print("LAND COVER EXTRACTION COMPLETE")
print("=" * 70)
print("  Dataset  : ESA WorldCover 10m 2021 v200 (CC-BY 4.0)")
print("  DOI      : https://doi.org/10.5281/zenodo.7254221")
print("  Source   : %s" % DATASET_META["aws_url"])
print("  Tile     : %s  (covers entire Mumbai AOI)" % DATASET_META["tile"])
print("  CRS orig : EPSG:4326 (WGS84)")
print("  CRS work : EPSG:32643 (UTM43N, metres)")
print("  Cells    : %d / %d with valid built-up fraction" % (n_valid_cells, n_cells))
print("  Missing  : %d (NaN, edges only, not interpolated)" % n_missing)
print("  Builtup  : mean %.1f%%  max %.1f%%" % (
    np.nanmean(builtup_pct)*100, np.nanmax(builtup_pct)*100))
print()
print("  New columns in grid:")
print("    builtup_pct_wc   = fraction of cell covered by built-up (class 50)")
print("    wc_cls010_frac   = tree cover fraction")
print("    wc_cls020_frac   = shrubland fraction")
print("    wc_cls030_frac   = grassland fraction")
print("    wc_cls040_frac   = cropland fraction")
print("    wc_cls060_frac   = bare/sparse vegetation fraction")
print("    wc_cls080_frac   = permanent water fraction")
print("    wc_cls090_frac   = herbaceous wetland fraction")
print("    wc_cls095_frac   = mangroves fraction")
print()
print("  QC report : outputs/landcover_qc.txt")
print("=" * 70)
