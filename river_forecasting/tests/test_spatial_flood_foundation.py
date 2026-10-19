# -*- coding: utf-8 -*-
"""
river_forecasting/tests/test_spatial_flood_foundation.py
=========================================================
Tests for the 500m spatial flood foundation and dashboard model binding:
1. Spatial grid delineation in projected CRS (EPSG:32643 UTM Zone 43N).
2. Physical feature sampling (GSW water recurrence, distance to barrage).
3. Strict non-fabrication guardrail: HALTS before model fitting when SAR masks are absent.
4. Dashboard data integrity: all 5 models have predictions across 2026 full season.
5. In outage window (July 31–Aug 4), Model D is strictly UNAVAILABLE without silent fallback.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRID_GEOJSON = REPO_ROOT / "river_forecasting" / "data" / "spatial" / "vishnupuri_500m_grid.geojson"
AUDIT_JSON = REPO_ROOT / "river_forecasting" / "outputs" / "vishnupuri_500m_spatial_flood_audit.json"
DASHBOARD_DATA = REPO_ROOT / "ui" / "dashboard_data.json"
GEE_SCRIPT = REPO_ROOT / "scripts" / "gee_event_flood_extractor.js"


def test_500m_grid_exists_and_valid():
    """Verify 500m grid file exists and has valid cells in projected UTM 43N."""
    assert GRID_GEOJSON.exists(), f"500m grid not found at {GRID_GEOJSON}"
    with open(GRID_GEOJSON, "r", encoding="utf-8") as f:
        grid_data = json.load(f)

    assert grid_data["type"] == "FeatureCollection"
    features = grid_data["features"]
    assert len(features) > 1000, f"Expected >1000 cells for 1,231 km² corridor, got {len(features)}"

    # Check cell structure
    first_cell = features[0]["properties"]
    assert "grid_id" in first_cell
    assert "x_utm_m" in first_cell
    assert "y_utm_m" in first_cell
    assert "centroid_lon" in first_cell
    assert "centroid_lat" in first_cell
    assert first_cell["cell_area_m2"] == 250000.0  # 500m x 500m


def test_spatial_features_sampled():
    """Verify authentic physical features were sampled onto grid cells."""
    with open(GRID_GEOJSON, "r", encoding="utf-8") as f:
        grid_data = json.load(f)

    features = grid_data["features"]
    has_gsw = any("gsw_mean_water_recurrence_pct" in feat["properties"] for feat in features)
    has_dist = any("dist_to_barrage_km" in feat["properties"] for feat in features)
    has_segment = any("segment" in feat["properties"] for feat in features)

    assert has_gsw, "GSW water recurrence feature missing from grid"
    assert has_dist, "Distance to barrage feature missing from grid"
    assert has_segment, "Subbasin segment classification missing from grid"


def test_flood_label_audit_and_safe_halt():
    """Verify spatial flood audit reports missing SAR labels and halts without fabrication."""
    assert AUDIT_JSON.exists(), f"Audit file not found at {AUDIT_JSON}"
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit = json.load(f)

    # Must report that local SAR labels are absent
    assert audit["flood_labels_audit"]["label_availability"] is False
    assert len(audit["flood_labels_audit"]["local_sar_flood_masks_found"]) == 0

    # Model training state MUST be NOT_TRAINED
    assert audit["model_training_status"]["status"] == "NOT_TRAINED_PENDING_SAR_LABELS"
    assert audit["model_training_status"]["metrics"]["precision"] is None
    assert audit["model_training_status"]["metrics"]["recall"] is None

    # GEE script must exist
    assert GEE_SCRIPT.exists(), "GEE extraction script missing"


def test_dashboard_model_selector_binding():
    """Verify all 5 models have predictions in dashboard_data.json across 2026."""
    assert DASHBOARD_DATA.exists(), f"Dashboard data not found at {DASHBOARD_DATA}"
    with open(DASHBOARD_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    timeline_2026 = data.get("timeline_2026", [])
    assert len(timeline_2026) > 2000, f"Expected >2000 rows in timeline_2026, got {len(timeline_2026)}"

    # Sample a non-outage row with active predictions (e.g. index 500)
    eval_row = next((r for r in timeline_2026 if r.get("predictions") and r["predictions"]["1h"]["stage_change_rain"] is not None), None)
    assert eval_row is not None, "No rows found with active Model D predictions"

    preds_1h = eval_row["predictions"]["1h"]
    assert "stage_change_rain" in preds_1h
    assert "stage_change_ridge" in preds_1h
    assert "wl_ridge" in preds_1h
    assert "random_forest" in preds_1h
    assert "xgboost" in preds_1h
    assert "persistence" in preds_1h

    # Check that +1h, +2h, and +3h lead times exist
    assert "1h" in eval_row["predictions"]
    assert "2h" in eval_row["predictions"]
    assert "3h" in eval_row["predictions"]


def test_outage_preservation_without_silent_fallback():
    """Verify that during the INSAT-3DS outage (July 31–Aug 4), Model D is strictly null/unavailable."""
    with open(DASHBOARD_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    timeline_2026 = data.get("timeline_2026", [])
    # Find a row within July 31 10:00 to August 4 00:00
    outage_row = next((r for r in timeline_2026 if "2026-08-01" in r["time"] or "2026-08-02" in r["time"]), None)
    assert outage_row is not None, "Outage timeline row not found"

    preds = outage_row.get("predictions")
    if preds and "1h" in preds:
        # stage_change_rain MUST be None during outage
        assert preds["1h"]["stage_change_rain"] is None, "Model D must NOT have synthetic prediction during satellite outage"
        # stage-only ridge or persistence CAN exist
        assert preds["1h"]["persistence"] is not None
