#!/usr/bin/env python3
"""
scripts/inspect_mosdac_hdf.py

MOSDAC INSAT-3DS HDF File Structure & Metadata Inspection Script.
Detects format (HDF5 vs HDF4), recursively walks the internal structure,
dumps all global and per-dataset attributes verbatim, and generates
a structured inspection report with explicit Findings and Could Not Determine sections.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def format_attr_value(val: Any) -> str:
    """Formats an attribute value verbatim into a clean string representation."""
    if isinstance(val, (bytes, np.bytes_)):
        try:
            return repr(val.decode("utf-8"))
        except UnicodeDecodeError:
            return repr(val)
    elif isinstance(val, np.ndarray):
        if val.ndim == 0:
            return format_attr_value(val.item())
        if val.size == 1:
            item = val.flat[0]
            if isinstance(item, (bytes, np.bytes_)):
                try:
                    return repr(item.decode("utf-8"))
                except UnicodeDecodeError:
                    return repr(item)
            return str(item)
        return np.array2string(val, separator=", ", threshold=20)
    elif isinstance(val, (np.integer, np.floating)):
        return str(val.item())
    return repr(val)


def inspect_hdf5(filepath: Path) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Attempts to inspect the file using h5py (HDF5)."""
    try:
        import h5py
    except ImportError as e:
        return False, f"h5py not installed: {e}", None

    try:
        with h5py.File(filepath, "r") as f:
            # Successfully opened with h5py!
            global_attrs = {}
            for k in f.attrs.keys():
                global_attrs[k] = f.attrs[k]

            groups = []
            datasets = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    ds_attrs = {}
                    for k in obj.attrs.keys():
                        ds_attrs[k] = obj.attrs[k]
                    datasets.append({
                        "name": "/" + name,
                        "base_name": name.split("/")[-1],
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "attrs": ds_attrs,
                    })
                elif isinstance(obj, h5py.Group):
                    grp_attrs = {}
                    for k in obj.attrs.keys():
                        grp_attrs[k] = obj.attrs[k]
                    groups.append({
                        "name": "/" + name,
                        "attrs": grp_attrs,
                    })

            f.visititems(visitor)

            return True, None, {
                "format": "HDF5",
                "library": f"h5py (version {h5py.__version__})",
                "global_attrs": global_attrs,
                "groups": groups,
                "datasets": datasets,
            }
    except Exception as e:
        return False, str(e), None


def inspect_hdf4(filepath: Path) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Attempts to inspect the file using pyhdf.SD (HDF4)."""
    try:
        from pyhdf.SD import SD, SDC
    except ImportError as e:
        return False, f"pyhdf not installed: {e}", None

    try:
        sd = SD(str(filepath), SDC.READ)
        try:
            global_attrs = sd.attributes()
            datasets_meta = sd.datasets()  # dict: {name: (shape, type, sds_type, index)}
            datasets = []
            for name, meta in datasets_meta.items():
                sds = sd.select(name)
                try:
                    sds_attrs = sds.attributes()
                finally:
                    sds.endaccess()
                datasets.append({
                    "name": name,
                    "base_name": name,
                    "shape": meta[0],
                    "dtype": str(meta[1]),
                    "attrs": sds_attrs,
                })
            return True, None, {
                "format": "HDF4",
                "library": "pyhdf.SD",
                "global_attrs": global_attrs,
                "groups": [],
                "datasets": datasets,
            }
        finally:
            sd.end()
    except Exception as e:
        return False, str(e), None


def analyze_structure(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Analyzes datasets and attributes to answer the 7 specific findings questions.
    Records missing items strictly in 'could_not_determine'.
    Does not assume, extrapolate, or infer.
    """
    findings = {}
    could_not_determine = []

    datasets = data["datasets"]
    global_attrs = data["global_attrs"]

    # 1. Rainfall / Rain-rate variable
    rain_ds = None
    for ds in datasets:
        name_lower = ds["base_name"].lower()
        long_name = str(ds["attrs"].get("long_name", "")).lower()
        std_name = str(ds["attrs"].get("standard_name", "")).lower()
        if "hem" in name_lower or "rain" in name_lower or "precip" in long_name or "precip" in std_name:
            rain_ds = ds
            break

    if rain_ds is not None:
        findings["rainfall_variable"] = rain_ds["name"]
        findings["rainfall_shape"] = rain_ds["shape"]
        findings["rainfall_dtype"] = rain_ds["dtype"]
    else:
        could_not_determine.append("Rainfall/rain-rate dataset: Could not identify rainfall dataset from metadata")

    # 2. Units for rainfall dataset
    if rain_ds is not None:
        unit_val = None
        for k in ["units", "unit", "UNITS", "Unit"]:
            if k in rain_ds["attrs"]:
                unit_val = format_attr_value(rain_ds["attrs"][k])
                break
        if unit_val is not None:
            findings["rainfall_units"] = unit_val
        else:
            findings["rainfall_units"] = "not found in metadata"
            could_not_determine.append("Rainfall units: Literal 'units' or 'unit' attribute not found on rainfall dataset")
    else:
        could_not_determine.append("Rainfall units: Rainfall dataset not identified")

    # 3. Fill value / NoData value on rainfall dataset
    if rain_ds is not None:
        fill_val = None
        for k in ["_FillValue", "fill_value", "FillValue", "missing_value", "nodata", "NoData"]:
            if k in rain_ds["attrs"]:
                fill_val = format_attr_value(rain_ds["attrs"][k])
                break
        if fill_val is not None:
            findings["fill_value"] = fill_val
        else:
            findings["fill_value"] = "not found in metadata"
            could_not_determine.append("Fill value: No _FillValue or fill_value attribute found on rainfall dataset")
    else:
        could_not_determine.append("Fill value: Rainfall dataset not identified")

    # 4. Scale factor and offset on rainfall dataset
    if rain_ds is not None:
        scale_val = None
        offset_val = None
        for k in ["scale_factor", "Scale_Factor", "scale"]:
            if k in rain_ds["attrs"]:
                scale_val = format_attr_value(rain_ds["attrs"][k])
                break
        for k in ["add_offset", "Add_Offset", "offset"]:
            if k in rain_ds["attrs"]:
                offset_val = format_attr_value(rain_ds["attrs"][k])
                break

        if scale_val is not None or offset_val is not None:
            findings["scale_factor"] = scale_val if scale_val is not None else "not present"
            findings["add_offset"] = offset_val if offset_val is not None else "not present"
        else:
            findings["scale_factor"] = "not present"
            findings["add_offset"] = "not present"
            # Note: explicit reporting as "not present" without assuming scale=1/offset=0
    else:
        could_not_determine.append("Scale factor / add_offset: Rainfall dataset not identified")

    # 5. Timestamp representation
    timestamp_sources = []
    # Check separate dataset
    for ds in datasets:
        if ds["base_name"].lower() in ["time", "datetime", "timestamp"]:
            t_units = format_attr_value(ds["attrs"].get("units", "no units attribute"))
            timestamp_sources.append(f"Dedicated dataset '{ds['name']}' (shape: {ds['shape']}, dtype: {ds['dtype']}, units: {t_units})")

    # Check global attributes
    time_attrs = []
    for k in [
        "Acquisition_Date", "Acquisition_Time_in_GMT", "Acquisition_Start_Time",
        "Acquisition_End_Time", "Product_Creation_Time", "time", "date"
    ]:
        if k in global_attrs:
            time_attrs.append(f"{k} = {format_attr_value(global_attrs[k])}")

    if time_attrs:
        timestamp_sources.append("Global attributes: " + "; ".join(time_attrs))

    if timestamp_sources:
        findings["timestamp_representation"] = " | ".join(timestamp_sources)
    else:
        findings["timestamp_representation"] = "not found in metadata"
        could_not_determine.append("Timestamp representation: Neither a dedicated time dataset nor timestamp attributes found in metadata")

    # 6. Latitude / Longitude / Geolocation representation
    geo_sources = []
    lat_ds = None
    lon_ds = None
    for ds in datasets:
        b_name = ds["base_name"].lower()
        if b_name in ["latitude", "lat"]:
            lat_ds = ds
        elif b_name in ["longitude", "lon"]:
            lon_ds = ds

    if lat_ds and lon_ds:
        lat_scale = format_attr_value(lat_ds["attrs"].get("scale_factor", "none"))
        lat_fill = format_attr_value(lat_ds["attrs"].get("_FillValue", "none"))
        lat_units = format_attr_value(lat_ds["attrs"].get("units", "none"))
        lon_scale = format_attr_value(lon_ds["attrs"].get("scale_factor", "none"))
        lon_fill = format_attr_value(lon_ds["attrs"].get("_FillValue", "none"))
        lon_units = format_attr_value(lon_ds["attrs"].get("units", "none"))
        geo_sources.append(
            f"Separate 2D geolocation datasets: "
            f"'{lat_ds['name']}' (shape: {lat_ds['shape']}, dtype: {lat_ds['dtype']}, scale_factor: {lat_scale}, _FillValue: {lat_fill}, units: {lat_units}) and "
            f"'{lon_ds['name']}' (shape: {lon_ds['shape']}, dtype: {lon_ds['dtype']}, scale_factor: {lon_scale}, _FillValue: {lon_fill}, units: {lon_units})"
        )

    # Check dimension scales GeoX, GeoY
    dim_scales = [ds["name"] for ds in datasets if ds["base_name"] in ["GeoX", "GeoY", "x", "y"]]
    if dim_scales:
        geo_sources.append(f"Dimension scales present: {', '.join(dim_scales)}")

    # Check global bounding attributes
    bounds_attrs = []
    for k in ["left_longitude", "right_longitude", "lower_latitude", "upper_latitude",
              "Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude", "Nominal_Altitude(km)"]:
        if k in global_attrs:
            bounds_attrs.append(f"{k} = {format_attr_value(global_attrs[k])}")
    if bounds_attrs:
        geo_sources.append("Global geographic boundary attributes: " + "; ".join(bounds_attrs))

    if geo_sources:
        findings["geolocation_representation"] = "\n     - ".join(geo_sources)
    else:
        findings["geolocation_representation"] = "not found in metadata"
        could_not_determine.append("Geolocation representation: No latitude/longitude datasets or geolocation attributes found")

    # 7. Spatial dimensions / resolution
    if rain_ds is not None:
        dim_info = f"Rainfall array shape: {rain_ds['shape']} (bands/time: {rain_ds['shape'][0]}, Y/rows: {rain_ds['shape'][1]}, X/cols: {rain_ds['shape'][2]})"
        # Check if pixel spacing / resolution is explicitly present in metadata
        res_info = None
        for k in ["spatial_resolution", "pixel_resolution", "resolution", "nominal_resolution", "pixel_size"]:
            if k in global_attrs:
                res_info = f"Global attribute '{k}' = {format_attr_value(global_attrs[k])}"
                break
            if k in rain_ds["attrs"]:
                res_info = f"Dataset attribute '{k}' = {format_attr_value(rain_ds['attrs'][k])}"
                break

        if res_info is not None:
            findings["spatial_dimensions_and_resolution"] = f"{dim_info} | Pixel spacing: {res_info}"
        else:
            findings["spatial_dimensions_and_resolution"] = f"{dim_info} | Pixel spacing: not explicitly specified as an attribute in metadata (stored as explicit 2D Latitude/Longitude grids of shape {rain_ds['shape'][1:]})"
            could_not_determine.append("Pixel spacing / resolution: No explicit 'resolution' attribute in metadata (pixel coordinates are provided by explicit Latitude/Longitude arrays)")
    else:
        could_not_determine.append("Spatial dimensions: Rainfall dataset not identified")

    return findings, could_not_determine


def generate_report(data: Dict[str, Any], filepath: Path, findings: Dict[str, Any], could_not_determine: List[str]) -> str:
    """Generates the structured inspection report."""
    lines = []
    lines.append("=" * 80)
    lines.append("MOSDAC INSAT-3DS HDF FILE INSPECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"File inspected: {filepath.name}")
    lines.append(f"File path:      {filepath.resolve()}")
    lines.append(f"File size:      {filepath.stat().st_size / (1024 * 1024):.2f} MB")
    lines.append("")

    lines.append("1. FILE FORMAT DETECTION")
    lines.append(f"   Detected format: {data['format']}")
    lines.append(f"   Library used:    {data['library']}")
    if "Output_Format" in data["global_attrs"]:
        lines.append(f"   Metadata tag 'Output_Format': {format_attr_value(data['global_attrs']['Output_Format'])}")
    lines.append("")

    lines.append("2. FULL INTERNAL STRUCTURE DUMP")
    if data["groups"]:
        lines.append(f"   Groups ({len(data['groups'])}):")
        for grp in data["groups"]:
            lines.append(f"     [GROUP] {grp['name']}")
    else:
        lines.append("   Groups: None (flat root hierarchy)")

    lines.append(f"   Datasets ({len(data['datasets'])}):")
    for ds in data["datasets"]:
        lines.append(f"     [DATASET] {ds['name']:<15} | Shape: {str(ds['shape']):<18} | Dtype: {ds['dtype']}")
    lines.append("")

    lines.append("3. VERBATIM METADATA & ATTRIBUTES DUMP")
    lines.append(f"   A. Global Attributes ({len(data['global_attrs'])}):")
    for k, v in sorted(data["global_attrs"].items()):
        lines.append(f"      - {k:<55} = {format_attr_value(v)}")
    lines.append("")

    lines.append("   B. Per-Dataset Attributes:")
    for ds in data["datasets"]:
        lines.append(f"      Dataset: {ds['name']} (shape={ds['shape']}, dtype={ds['dtype']})")
        if ds["attrs"]:
            for k, v in sorted(ds["attrs"].items()):
                lines.append(f"        - {k:<25} = {format_attr_value(v)}")
        else:
            lines.append("        (No attributes)")
        lines.append("")

    lines.append("4. FINDINGS SECTION")
    lines.append(f"   a. Rainfall / rain-rate dataset name: {findings.get('rainfall_variable', 'not found')}")
    lines.append(f"   b. Rainfall units:                    {findings.get('rainfall_units', 'not found in metadata')}")
    lines.append(f"   c. Fill value / NoData value:         {findings.get('fill_value', 'not found in metadata')}")
    lines.append(f"   d. Scale factor / Add offset:         scale_factor={findings.get('scale_factor', 'not present')}, add_offset={findings.get('add_offset', 'not present')}")
    lines.append(f"   e. Timestamp representation:          {findings.get('timestamp_representation', 'not found in metadata')}")
    lines.append(f"   f. Geolocation representation:        \n     - {findings.get('geolocation_representation', 'not found in metadata')}")
    lines.append(f"   g. Spatial dimensions & resolution:   {findings.get('spatial_dimensions_and_resolution', 'not found')}")
    lines.append("")

    lines.append("5. COULD NOT DETERMINE (Strictly Unfilled / Not Explicit in Metadata)")
    if could_not_determine:
        for item in could_not_determine:
            lines.append(f"   - {item}")
    else:
        lines.append("   - None. All target metadata fields were explicitly identified in metadata.")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    sample_dir = base_dir / "data" / "mosdac" / "hdf_sample"
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Locate sample file
    sample_files = list(sample_dir.glob("*.h5")) + list(sample_dir.glob("*.hdf")) + list(sample_dir.glob("*.he5"))
    if not sample_files:
        print(f"ERROR: No HDF sample file (*.h5, *.hdf, *.he5) found in {sample_dir}")
        sys.exit(1)

    target_file = sample_files[0]
    print(f"Target HDF file: {target_file.name}")
    print("=" * 80)
    print("Step 1: Detecting HDF format...")

    # Format detection: Try HDF5 first, then HDF4
    hdf5_success, hdf5_err, data = inspect_hdf5(target_file)
    if not hdf5_success:
        print(f"HDF5 open attempt failed: {hdf5_err}")
        print("Trying HDF4 (pyhdf.SD)...")
        hdf4_success, hdf4_err, data = inspect_hdf4(target_file)
        if not hdf4_success:
            print("=" * 80)
            print(f"FATAL ERROR: Failed to open file with both HDF5 (h5py) and HDF4 (pyhdf):")
            print(f"  h5py error:  {hdf5_err}")
            print(f"  pyhdf error: {hdf4_err}")
            print("=" * 80)
            sys.exit(1)
        else:
            print("Successfully opened with HDF4 (pyhdf.SD)!")
    else:
        print("Successfully opened with HDF5 (h5py)!")

    # Step 2: Analyze structure
    findings, could_not_determine = analyze_structure(data)

    # Step 3: Generate and write report
    report_text = generate_report(data, target_file, findings, could_not_determine)
    report_path = outputs_dir / "mosdac_hdf_inspection.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Successfully generated inspection report: {report_path}")
    print("=" * 80)

    # Print summary to console
    print(report_text)


if __name__ == "__main__":
    main()
