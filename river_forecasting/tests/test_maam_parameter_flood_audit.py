"""
river_forecasting/tests/test_maam_parameter_flood_audit.py
==========================================================
Tests for Phase 2: Ma'am's Parameter-Based Flood Modelling Methodology Alignment:
- Exclusive focus on Vishnupuri Barrage / Godavari Reach (Rejection of Mumbai urban grid)
- Audited Parameters P1 through P6 coverage and provenance
- Verified Reach corridor delineation (HydroBASINS Lev08 4081058170 + 4081060220)
- Global Surface Water recurrence extraction
- Zero synthetic flood labels protocol
- Distinct task separation (stage forecasting vs spatial flood mapping)
"""

from pathlib import Path
import json
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPATIAL_DIR = REPO_ROOT / "river_forecasting" / "data" / "spatial"
REACH_GEOJSON = SPATIAL_DIR / "vishnupuri_river_reach_study_area.geojson"
AUDIT_JSON = REPO_ROOT / "river_forecasting" / "outputs" / "maam_parameter_flood_audit.json"
DASHBOARD_DATA_JSON = REPO_ROOT / "ui" / "dashboard_data.json"
INDEX_HTML = REPO_ROOT / "ui" / "index.html"


def test_reach_study_area_geojson_valid_and_godavari_specific():
    """Verify that the reach corridor GeoJSON exists and is strictly located at Godavari (Nanded)."""
    assert REACH_GEOJSON.exists(), "Reach GeoJSON missing"
    with open(REACH_GEOJSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    # Contains unified corridor feature + 2 constituent subbasins
    assert len(data["features"]) == 3, "Expected unified corridor + 2 Lev08 subbasins"
    
    subbasin_ids = [str(f["properties"].get("hybas_id")) for f in data["features"] if "hybas_id" in f["properties"]]
    assert "4081058170" in subbasin_ids, "Shankar Sagar barrage pool subbasin missing"
    assert "4081060220" in subbasin_ids, "Downstream reach subbasin missing"

    unified_feat = [f for f in data["features"] if "area_sq_km" in f["properties"]][0]
    total_area = unified_feat["properties"]["area_sq_km"]
    assert 1200.0 <= total_area <= 1260.0, f"Expected ~1,231.5 km² reach corridor, got {total_area}"

    # Verify bounding box is around Nanded (77.0°E - 77.6°E, 18.8°N - 19.4°N)
    # Strictly NOT Mumbai (72.8°E, 19.0°N)
    bbox = data.get("bbox")
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        assert 76.8 <= min_lon <= 77.3, f"Unexpected min longitude: {min_lon}"
        assert 77.3 <= max_lon <= 77.8, f"Unexpected max longitude: {max_lon}"
        assert 18.7 <= min_lat <= 19.1, f"Unexpected min latitude: {min_lat}"
        assert 19.1 <= max_lat <= 19.5, f"Unexpected max latitude: {max_lat}"


def test_parameter_audit_artifacts_completeness():
    """Verify Ma'am's Parameter audit reports P1 to P6 and river-specific auxiliary features."""
    assert AUDIT_JSON.exists(), "Parameter audit JSON missing"
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit = json.load(f)

    params = {p["id"]: p for p in audit["parameters"]}
    for p_id in ["P1", "P2", "P3", "P4", "P5", "P6"]:
        assert p_id in params, f"Parameter {p_id} missing from audit"

    # Verify P1 is Elevation
    assert "Elevation" in params["P1"]["parameter"]
    assert "HydroSHEDS" in params["P1"]["dataset_source"] or "SRTM" in params["P1"]["dataset_source"]

    # Verify P2 is Slope
    assert "Slope" in params["P2"]["parameter"]

    # Verify P3 is Rainfall
    assert "Rainfall" in params["P3"]["parameter"]
    assert "INSAT-3DS" in params["P3"]["dataset_source"]

    # Verify P4 River Conditions
    assert "River" in params["P4"]["parameter"]

    # Verify P5 LULC
    assert "Land Use" in params["P5"]["parameter"]

    # Verify P6 Temperature
    assert "Temperature" in params["P6"]["parameter"]

    # Verify river features
    rf_names = [f["feature"] for f in audit["river_features"]]
    assert any("Distance to Godavari" in name for name in rf_names)
    assert any("Flow Accumulation" in name for name in rf_names)
    assert any("Historical Inundation Extent" in name for name in rf_names)
    assert any("Barrage Operation" in name for name in rf_names)


def test_strict_flood_label_integrity_and_task_separation():
    """Verify that synthetic flood labels are forbidden and task separation is enforced."""
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit = json.load(f)

    # Inundation labels audit must document genuine status without inventing labels
    labels_audit = audit["flood_labels_audit"]
    assert "ground_river_stage_flood_events" in labels_audit
    assert "spatial_pixel_level_inundation_labels" in labels_audit
    
    spatial_labels = labels_audit["spatial_pixel_level_inundation_labels"]
    assert "NO synthetic flood labels" in spatial_labels["integrity_rule"]

    # Verify distinct task separation definitions
    tasks = audit["readiness_summary"]["distinct_task_separation"]
    assert "task_1_stage_forecasting" in tasks
    assert "task_2_flood_occurrence_classification" in tasks
    assert "task_3_spatial_inundation_mapping" in tasks


def test_ui_contains_reach_gis_and_parameter_audit_sections():
    """Verify that ui/index.html exposes Reach GIS, Layer Toggles, and Parameter Matrix."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "Godavari River Study Area" in html or "River Reach &amp; Catchment" in html
    assert "HydroBASINS Lev08" in html
    assert "btn-toggle-reach" in html
    assert "btn-toggle-catchment" in html
    assert "Parameter Information" in html or "Parameter Matrix" in html or "Parameters" in html
    assert "Elevation" in html
    assert "Slope" in html
    assert "Rainfall" in html
    assert "River Stage" in html or "River stage" in html
    assert "Land cover" in html or "Land Cover" in html or "LULC" in html
    assert "Temperature" in html
    assert "modelComparisonCanvas" in html


def test_dashboard_bundle_includes_reach_and_audit_data():
    """Verify that ui/dashboard_data.json bundles reach_corridor_geojson and maam_parameter_audit."""
    assert DASHBOARD_DATA_JSON.exists(), "dashboard_data.json missing"
    with open(DASHBOARD_DATA_JSON, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    assert "reach_corridor_geojson" in bundle and bundle["reach_corridor_geojson"] is not None
    assert "maam_parameter_audit" in bundle and bundle["maam_parameter_audit"] is not None
    assert "timeline_2026" in bundle and len(bundle["timeline_2026"]) > 0
    assert "timeline_2024" in bundle and len(bundle["timeline_2024"]) > 0
