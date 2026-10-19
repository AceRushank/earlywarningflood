# -*- coding: utf-8 -*-
"""
scripts/build_river_reach_study_area.py
========================================
Establishes the river-flood study area for Vishnupuri Barrage & Godavari River Reach:
1. Delineates the authentic river corridor using HydroBASINS Asia Level 08 (WWF/USGS):
   - Subbasin 4081058170 (Vishnupuri Barrage Shankar Sagar pool)
   - Subbasin 4081060220 (Downstream Godavari reach through Nanded city)
2. Extracts JRC Global Surface Water (GSW) recurrence patterns from 
   data/spatial/gsw/gsw_occurrence_70E_20N_v14.tif.
3. Exports GeoJSON and spatial metadata for dashboard integration.
"""

import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.windows import from_bounds
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HYDROBASINS_SHP = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "hydrobasins_raw" / "lev08" / "hybas_as_lev08_v1c.shp"
GSW_TIF = REPO_ROOT / "data" / "spatial" / "gsw" / "gsw_occurrence_70E_20N_v14.tif"
OUT_GEOJSON = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "vishnupuri_river_reach_study_area.geojson"
OUT_QC = REPO_ROOT / "river_forecasting" / "outputs" / "godavari_reach_spatial_qc.json"


def run():
    print("=" * 70)
    print("DELINEATING VISHNUPURI BARRAGE & GODAVARI RIVER REACH STUDY AREA")
    print("=" * 70)

    gdf = gpd.read_file(HYDROBASINS_SHP)
    reach_subbasins = gdf[gdf["HYBAS_ID"].isin([4081058170, 4081060220])].copy()

    pool_subbasin = reach_subbasins[reach_subbasins["HYBAS_ID"] == 4081058170].iloc[0]
    downstream_subbasin = reach_subbasins[reach_subbasins["HYBAS_ID"] == 4081060220].iloc[0]
    unified_geom = reach_subbasins.union_all()

    total_corridor_area = float(reach_subbasins["SUB_AREA"].sum())
    bounds = unified_geom.bounds

    print(f"Pool Sub-basin (4081058170) Area: {pool_subbasin['SUB_AREA']:.1f} km²")
    print(f"Downstream Reach (4081060220) Area: {downstream_subbasin['SUB_AREA']:.1f} km²")
    print(f"Total Corridor Area: {total_corridor_area:.1f} km²")
    print(f"Corridor Bounds: {bounds}")

    # Build GeoJSON FeatureCollection
    features = [
        {
            "type": "Feature",
            "properties": {
                "name": "Vishnupuri Barrage & Godavari River Reach Corridor",
                "segment": "Combined Study Area",
                "river": "Godavari",
                "district": "Nanded",
                "state": "Maharashtra",
                "area_sq_km": round(total_corridor_area, 2),
                "upstream_basin_area_sq_km": 51768.25,
                "barrage_pour_point_lat": 19.126944,
                "barrage_pour_point_lon": 77.280833,
                "barrage_frl_m_msl": 355.0,
                "barrage_danger_m_msl": 354.8,
                "barrage_warning_m_msl": 354.0
            },
            "geometry": mapping(unified_geom)
        },
        {
            "type": "Feature",
            "properties": {
                "name": "Vishnupuri Barrage Impoundment Pool (Shankar Sagar)",
                "segment": "Upstream Pool Reach",
                "hybas_id": 4081058170,
                "sub_area_sq_km": float(pool_subbasin["SUB_AREA"]),
                "upstream_area_sq_km": float(pool_subbasin["UP_AREA"])
            },
            "geometry": mapping(pool_subbasin.geometry)
        },
        {
            "type": "Feature",
            "properties": {
                "name": "Godavari River Downstream Reach (Nanded Urban Corridor)",
                "segment": "Downstream River Reach",
                "hybas_id": 4081060220,
                "sub_area_sq_km": float(downstream_subbasin["SUB_AREA"]),
                "upstream_area_sq_km": float(downstream_subbasin["UP_AREA"])
            },
            "geometry": mapping(downstream_subbasin.geometry)
        }
    ]

    geojson_data = {
        "type": "FeatureCollection",
        "name": "vishnupuri_godavari_reach_study_area",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }

    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)
    print(f"Successfully saved study area GeoJSON: {OUT_GEOJSON}")

    # Extract GSW statistics
    min_lon, min_lat, max_lon, max_lat = bounds
    gsw_stats = {}
    if GSW_TIF.exists():
        with rasterio.open(GSW_TIF) as src:
            win = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
            data = src.read(1, window=win)
            valid = data[data <= 100]
            water = data[(data > 0) & (data <= 100)]
            perm_water = data[data >= 80]
            seasonal_water = data[(data >= 20) & (data < 80)]
            ephemeral_water = data[(data > 0) & (data < 20)]

            gsw_stats = {
                "source": "JRC Global Surface Water (GSW) v1.4 (1984-2021)",
                "tile": "gsw_occurrence_70E_20N_v14.tif",
                "grid_shape": list(data.shape),
                "total_valid_pixels": int(len(valid)),
                "water_detected_pixels": int(len(water)),
                "water_fraction_pct": round(float(len(water) / len(valid) * 100), 2) if len(valid) > 0 else 0,
                "permanent_water_pixels_ge80pct": int(len(perm_water)),
                "seasonal_water_pixels_20_to_80pct": int(len(seasonal_water)),
                "ephemeral_inundation_pixels_1_to_20pct": int(len(ephemeral_water)),
                "mean_water_occurrence_pct": round(float(np.mean(water)), 2) if len(water) > 0 else 0,
                "max_water_occurrence_pct": int(np.max(water)) if len(water) > 0 else 0
            }
            print("GSW Water Recurrence Stats:", gsw_stats)

    qc_data = {
        "study_area": {
            "name": "Vishnupuri Barrage & Godavari River Reach Study Area",
            "river": "Godavari",
            "location": "Nanded, Maharashtra",
            "coordinates": {"lat": 19.126944, "lon": 77.280833},
            "crs": "EPSG:4326 (WGS84)",
            "bounding_box": {
                "min_lon": bounds[0], "min_lat": bounds[1],
                "max_lon": bounds[2], "max_lat": bounds[3]
            },
            "corridor_area_sq_km": total_corridor_area,
            "upstream_catchment_sq_km": 51768.25
        },
        "gsw_surface_water": gsw_stats
    }

    OUT_QC.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_QC, "w", encoding="utf-8") as f:
        json.dump(qc_data, f, indent=2)
    print(f"Saved QC metadata to: {OUT_QC}")


if __name__ == "__main__":
    run()
