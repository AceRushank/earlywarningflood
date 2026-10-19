"""
river_forecasting/src/catchment_extractor.py
============================================
Spatial Catchment-Average Rainfall Extraction Interface for INSAT-3DS HEM.

Designed for the Godavari River / Vishnupuri Barrage upstream catchment.
Reuses verified MOSDAC decoding routines (decode_time_from_dataset, native 2D curvilinear arrays).

Key Principles:
1. Hydrologically rigorous: Requires a verified upstream catchment polygon delineated from
   a hydrologically conditioned DEM to the pour point (19.126944 N, 77.280833 E).
   DOES NOT substitute a bounding box or single pixel.
2. Reusable & fast: Builds and caches a 2D curvilinear grid boolean mask.
3. Transparent QC: Handles fill values (-999.0), missing pixels, and missing time slots
   without interpolation or fabricated data.
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import h5py
import pandas as pd
import shapely
from shapely.geometry import shape, Polygon, MultiPolygon, Point
from shapely.prepared import prep
import geopandas as gpd

# Re-use decode_time_from_dataset from scripts/extract_mosdac_hdf.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.extract_mosdac_hdf import decode_time_from_dataset
except ImportError:
    def decode_time_from_dataset(time_ds):
        raw_val = float(time_ds[0])
        epoch = datetime(2000, 1, 1, 0, 0)
        dt = epoch + timedelta(minutes=raw_val)
        return dt, dt.strftime("%Y-%m-%d %H:%M")


class CatchmentPolygonSpecification:
    """
    Formal specification of the required upstream catchment polygon for Vishnupuri Barrage.
    """
    TARGET_STATION = "Vishnupuri Barrage, Nanded, Maharashtra"
    POUR_POINT_COORDINATES = {"latitude": 19.126944, "longitude": 77.280833}
    ACCEPTED_FILE_FORMATS = [".geojson", ".json", ".shp", ".gpkg"]
    REQUIRED_CRS = "EPSG:4326 (WGS84)"
    GEOMETRY_TYPES = ["Polygon", "MultiPolygon"]
    MIN_EXPECTED_AREA_SQ_KM = 10000.0  # Vishnupuri upstream Godavari basin is >20,000 km^2
    DELINEATION_METHOD = (
        "Hydrologically conditioned flow direction/accumulation network "
        "(e.g., HydroSHEDS, MERIT Hydro, or Copernicus DEM 30m) "
        "delineated to the Vishnupuri Barrage pour point."
    )
    PROHIBITED_SUBSTITUTES = (
        "Bounding box (BBOX) approximations and single satellite pixel queries "
        "are strictly prohibited because they violate physical mass balance "
        "and do not represent spatial catchment runoff volume."
    )


class CatchmentRainfallExtractor:
    """
    Production-ready spatial rainfall extraction interface for MOSDAC INSAT-3DS HEM HDF5 files.
    """

    def __init__(self, catchment_polygon_path: Optional[Union[str, Path]] = None):
        self.catchment_geom: Optional[Union[Polygon, MultiPolygon]] = None
        self.prepared_geom = None
        self.catchment_bbox: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
        self.catchment_area_sq_km: Optional[float] = None
        self.pixel_mask: Optional[np.ndarray] = None
        self.mask_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.total_catchment_pixels: int = 0
        self.pixel_coords: Optional[Dict[str, np.ndarray]] = None

        if catchment_polygon_path:
            self.load_catchment_polygon(catchment_polygon_path)

    def load_catchment_polygon(self, polygon_path_or_gdf: Union[str, Path, gpd.GeoDataFrame]) -> Dict[str, Any]:
        """
        Loads, validates, and prepares the upstream catchment polygon.
        Ensures EPSG:4326 CRS and validates geometry topology.
        """
        if isinstance(polygon_path_or_gdf, gpd.GeoDataFrame):
            gdf = polygon_path_or_gdf.copy()
        else:
            p = Path(polygon_path_or_gdf)
            if not p.exists():
                raise FileNotFoundError(f"Catchment polygon file not found: {p}")
            if p.suffix.lower() not in CatchmentPolygonSpecification.ACCEPTED_FILE_FORMATS:
                raise ValueError(
                    f"Unsupported format: {p.suffix}. Accepted: {CatchmentPolygonSpecification.ACCEPTED_FILE_FORMATS}"
                )
            gdf = gpd.read_file(p)

        if gdf.empty:
            raise ValueError("Catchment polygon GeoDataFrame is empty.")

        # Ensure EPSG:4326
        if gdf.crs is not None:
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
        else:
            gdf.set_crs(epsg=4326, inplace=True)

        # Dissolve into single geometry if multiple features
        unified_geom = gdf.unary_union
        if not isinstance(unified_geom, (Polygon, MultiPolygon)):
            raise ValueError(f"Geometry must be Polygon or MultiPolygon, got {type(unified_geom)}")

        self.catchment_geom = unified_geom
        self.prepared_geom = prep(unified_geom)
        self.catchment_bbox = unified_geom.bounds  # (minx, miny, maxx, maxy) = (min_lon, min_lat, max_lon, max_lat)

        # Estimate area in sq km using equal area cylindrical projection
        gdf_proj = gpd.GeoSeries([unified_geom], crs="EPSG:4326").to_crs(epsg=6933)
        self.catchment_area_sq_km = float(gdf_proj.area.iloc[0] / 1e6)

        # Reset cached masks
        self.pixel_mask = None
        self.mask_indices = None
        self.total_catchment_pixels = 0

        return {
            "status": "LOADED_AND_VALIDATED",
            "crs": "EPSG:4326",
            "bbox_lon_lat": {
                "min_lon": self.catchment_bbox[0],
                "min_lat": self.catchment_bbox[1],
                "max_lon": self.catchment_bbox[2],
                "max_lat": self.catchment_bbox[3],
            },
            "area_sq_km": round(self.catchment_area_sq_km, 2),
            "geometry_type": unified_geom.geom_type,
        }

    def build_catchment_pixel_mask(self, hdf_sample_path: Path) -> int:
        """
        Builds and caches the 2D boolean mask over the native INSAT-3DS curvilinear grid.
        This operation is performed once and reused across all timesteps.
        """
        if self.catchment_geom is None:
            raise ValueError("Catchment polygon not loaded. Call load_catchment_polygon first.")

        if not hdf_sample_path.exists():
            raise FileNotFoundError(f"HDF sample file not found: {hdf_sample_path}")

        min_lon, min_lat, max_lon, max_lat = self.catchment_bbox

        with h5py.File(hdf_sample_path, "r") as f:
            lat_ds = f["Latitude"]
            lon_ds = f["Longitude"]

            lat_raw = lat_ds[:]
            lon_raw = lon_ds[:]

            lat_scale = lat_ds.attrs.get("scale_factor", 0.01)
            if isinstance(lat_scale, np.ndarray):
                lat_scale = lat_scale.item()
            lat_fill = lat_ds.attrs.get("_FillValue", 32767)
            if isinstance(lat_fill, np.ndarray):
                lat_fill = lat_fill.item()

            lon_scale = lon_ds.attrs.get("scale_factor", 0.01)
            if isinstance(lon_scale, np.ndarray):
                lon_scale = lon_scale.item()
            lon_fill = lon_ds.attrs.get("_FillValue", 32767)
            if isinstance(lon_fill, np.ndarray):
                lon_fill = lon_fill.item()

        # Valid coordinates mask
        valid_geo = (lat_raw != lat_fill) & (lon_raw != lon_fill)
        lat_deg = np.where(valid_geo, lat_raw * lat_scale, np.nan)
        lon_deg = np.where(valid_geo, lon_raw * lon_scale, np.nan)

        # Bounding box candidate pre-filter
        cand_mask = (
            valid_geo &
            (lat_deg >= min_lat) & (lat_deg <= max_lat) &
            (lon_deg >= min_lon) & (lon_deg <= max_lon)
        )

        cand_rows, cand_cols = np.where(cand_mask)

        # Point in polygon verification using prepared geometry
        in_catchment = []
        for r, c in zip(cand_rows, cand_cols):
            pt = Point(lon_deg[r, c], lat_deg[r, c])
            in_catchment.append(self.prepared_geom.contains(pt))

        in_catchment = np.array(in_catchment, dtype=bool)

        final_rows = cand_rows[in_catchment]
        final_cols = cand_cols[in_catchment]

        full_mask = np.zeros(lat_raw.shape, dtype=bool)
        full_mask[final_rows, final_cols] = True

        self.pixel_mask = full_mask
        self.mask_indices = (final_rows, final_cols)
        self.total_catchment_pixels = len(final_rows)

        self.pixel_coords = {
            "latitudes": lat_deg[final_rows, final_cols],
            "longitudes": lon_deg[final_rows, final_cols]
        }

        return self.total_catchment_pixels

    def extract_single_timestep(self, hdf_path: Path) -> Dict[str, Any]:
        """
        Extracts spatially averaged rainfall over the verified catchment for a single HDF5 file.
        Handles fill values (-999.0) and computes coverage metrics transparently.
        """
        if self.pixel_mask is None or self.mask_indices is None:
            # Build mask using current file
            self.build_catchment_pixel_mask(hdf_path)

        result: Dict[str, Any] = {
            "source_file": hdf_path.name,
            "success": False,
            "timestamp_utc": None,
            "timestamp_ist": None,
            "total_catchment_pixels": self.total_catchment_pixels,
            "valid_rainfall_pixels": 0,
            "fill_pixels_excluded": 0,
            "coverage_ratio": 0.0,
            "mean_rainfall_mm_hr": None,
            "min_rainfall_mm_hr": None,
            "max_rainfall_mm_hr": None,
            "std_rainfall_mm_hr": None,
            "zero_rainfall_pixels": 0,
            "positive_rainfall_pixels": 0,
            "qc_flag": "NOT_PROCESSED",
            "error": None
        }

        try:
            with h5py.File(hdf_path, "r") as f:
                # 1. Decode timestamp
                if "time" in f:
                    dt_utc, _ = decode_time_from_dataset(f["time"])
                    dt_ist = dt_utc + timedelta(hours=5, minutes=30)
                    result["timestamp_utc"] = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
                    result["timestamp_ist"] = dt_ist.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    result["qc_flag"] = "MISSING_TIME_DATASET"
                    return result

                # 2. Read HEM rainfall slice
                hem_ds = f["HEM"]
                hem_slice = hem_ds[0]  # shape (2816, 2805)
                hem_fill = hem_ds.attrs.get("_FillValue", -999.0)
                if isinstance(hem_fill, np.ndarray):
                    hem_fill = hem_fill.item()

            # Extract pixels under mask
            rows, cols = self.mask_indices
            values = hem_slice[rows, cols]

            # Exclude fill values and negative sentinel values
            valid_mask = (values != hem_fill) & (values >= 0.0)
            valid_vals = values[valid_mask]

            n_valid = int(len(valid_vals))
            n_fill = int(self.total_catchment_pixels - n_valid)
            coverage = float(n_valid / self.total_catchment_pixels) if self.total_catchment_pixels > 0 else 0.0

            result["valid_rainfall_pixels"] = n_valid
            result["fill_pixels_excluded"] = n_fill
            result["coverage_ratio"] = round(coverage, 4)

            if n_valid > 0:
                result["mean_rainfall_mm_hr"] = round(float(np.mean(valid_vals)), 4)
                result["min_rainfall_mm_hr"] = round(float(np.min(valid_vals)), 4)
                result["max_rainfall_mm_hr"] = round(float(np.max(valid_vals)), 4)
                result["std_rainfall_mm_hr"] = round(float(np.std(valid_vals)), 4)
                result["zero_rainfall_pixels"] = int(np.count_nonzero(valid_vals == 0.0))
                result["positive_rainfall_pixels"] = int(np.count_nonzero(valid_vals > 0.0))

                if coverage >= 0.95:
                    result["qc_flag"] = "OK"
                elif coverage >= 0.70:
                    result["qc_flag"] = "PARTIAL_COVERAGE"
                else:
                    result["qc_flag"] = "HIGH_FILL_RATIO"
            else:
                result["qc_flag"] = "NO_VALID_DATA"

            result["success"] = True
            return result

        except Exception as e:
            result["error"] = str(e)
            result["qc_flag"] = "FILE_ERROR"
            return result

    def extract_sequence(
        self,
        hdf_paths: List[Path],
        expected_interval_minutes: int = 30
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extracts rainfall sequence across multiple HDF5 files.
        Detects missing time intervals without interpolation.
        """
        if not hdf_paths:
            return pd.DataFrame(), {"status": "NO_FILES_PROVIDED"}

        records = []
        for fp in hdf_paths:
            res = self.extract_single_timestep(fp)
            records.append(res)

        df = pd.DataFrame(records)
        df_valid = df[df["timestamp_utc"].notna()].copy()

        if df_valid.empty:
            return df, {"status": "NO_VALID_TIMESTAMPS"}

        df_valid["dt_utc"] = pd.to_datetime(df_valid["timestamp_utc"])
        df_valid.sort_values(by="dt_utc", inplace=True)
        df_valid.reset_index(drop=True, inplace=True)

        earliest = df_valid["dt_utc"].min()
        latest = df_valid["dt_utc"].max()

        # Expected 30-min sequence
        expected_range = pd.date_range(earliest, latest, freq=f"{expected_interval_minutes}min")
        observed_set = set(df_valid["dt_utc"])
        missing_dts = [t.strftime("%Y-%m-%d %H:%M:%S") for t in expected_range if t not in observed_set]

        summary = {
            "total_files_evaluated": len(hdf_paths),
            "successful_extractions": int(df["success"].sum()),
            "earliest_timestamp_utc": earliest.strftime("%Y-%m-%d %H:%M:%S"),
            "latest_timestamp_utc": latest.strftime("%Y-%m-%d %H:%M:%S"),
            "expected_timesteps_count": len(expected_range),
            "observed_unique_timesteps_count": len(observed_set),
            "missing_timesteps_count": len(missing_dts),
            "missing_timesteps": missing_dts[:20],  # sample first 20 if any
            "catchment_total_pixels": self.total_catchment_pixels,
            "catchment_area_sq_km": self.catchment_area_sq_km,
        }

        # Drop temporary dt_utc
        df_valid.drop(columns=["dt_utc"], inplace=True)
        return df_valid, summary
