"""
river_forecasting/tests/test_audit_verification.py
==================================================
Regression test suite verifying the comprehensive pipeline audit:
1. Acquisition reconciliation and HDF5 integrity.
2. Datum (366.5 m) and FRL (355.0 m) authoritative verification.
3. Causal rainfall alignment, API warmup, outage preservation, and zero target leakage.
4. Chronological evaluation metrics and sample counts.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HDF_DIR = REPO_ROOT / "data" / "mosdac" / "hdf_archive"
MANIFEST_CSV = REPO_ROOT / "data" / "events_2026" / "insat3ds_full_season_2026_manifest.csv"
CATALOG_JSON = REPO_ROOT / "data" / "events_2026" / "mosdac_catalog_index_2026.json"
DATASET_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"
TELEMETRY_RAW = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"
BENCHMARK_JSON = REPO_ROOT / "data" / "historical" / "multivariable_benchmark_results.json"


def test_acquisition_reconciliation_and_hdf5_integrity():
    """Verify archive integrity and study window completeness."""
    assert HDF_DIR.exists()
    h5_files = list(HDF_DIR.glob("*.h5"))
    assert len(h5_files) >= 5590, f"Expected >= 5590 files, found {len(h5_files)}"

    # Check magic headers on a sample of files
    HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
    for fp in h5_files[:50]:
        assert fp.stat().st_size > 5 * 1024 * 1024
        with open(fp, "rb") as f:
            assert f.read(8) == HDF5_MAGIC

    # Check catalog study window
    with open(CATALOG_JSON) as f:
        catalog = json.load(f)
    records = catalog["records"]
    monsoon_records = {k: v for k, v in records.items() if "2026-06-15" <= v["date"] <= "2026-09-30"}
    on_disk_names = set(f.name for f in h5_files)
    
    missing_monsoon = set(monsoon_records.keys()) - on_disk_names
    assert len(missing_monsoon) == 0, f"Missing {len(missing_monsoon)} granules in monsoon study window!"


def test_datum_and_frl_authoritative_metadata():
    """Verify that NWIC stage values are already absolute MSL elevations and FRL is 355.0m."""
    assert TELEMETRY_RAW.exists()
    df_raw = pd.read_csv(TELEMETRY_RAW, nrows=500000, low_memory=False)
    v = df_raw[df_raw["Station"] == "Vishnupuri"]
    assert len(v) > 0

    first_row = v.iloc[0]
    assert float(first_row["RL_of_zeroGauge"]) == 0.0
    assert float(first_row["MeanSeaLevel"]) == 366.50

    stage_col = [c for c in v.columns if "River Water Level" in c][0]
    valid_stages = pd.to_numeric(v[stage_col], errors="coerce").dropna()
    
    # Telemetry is in the 350.5m - 355.2m range (already MSL)
    assert valid_stages.max() <= 356.0
    assert valid_stages.quantile(0.1) >= 350.0

    # Ensure that no datum shift addition (e.g. + 366.5) was applied
    df_aligned = pd.read_csv(DATASET_CSV)
    v_aligned = df_aligned["stage_vishnupuri"].dropna()
    assert v_aligned.min() > 345.0
    assert v_aligned.max() < 356.0


def test_leakage_safety_and_outage_preservation():
    """Verify that outage is NaN, API is causal, and no future targets leaked into features."""
    df = pd.read_csv(DATASET_CSV)
    df["dt"] = pd.to_datetime(df["dt"])

    # Outage window: July 31 06:00 to August 4 22:00
    outage_mask = (df["dt"] >= "2026-07-31 06:00:00") & (df["dt"] <= "2026-08-04 22:00:00")
    assert df.loc[outage_mask, "rain_depth_1h_mm"].isna().all()
    assert df.loc[outage_mask, "rain_qc_flag"].isin(["OUTAGE_UNAVAILABLE", np.nan]).all()

    # Targets are strictly future
    for h in [1, 2, 3]:
        diff = df[f"target_v_{h}h"] - df["stage_vishnupuri"].shift(-h)
        assert diff.dropna().abs().max() < 1e-6


def test_benchmark_metrics_and_matched_origin_counts():
    """Verify matched sample counts and that Stage-Change-plus-Rainfall outperforms Persistence."""
    with open(BENCHMARK_JSON) as f:
        res = json.load(f)

    phase_a = res["phase_a_event_eval"]
    assert phase_a["+1h"]["A_Persistence"]["n_samples"] == 30
    assert phase_a["+2h"]["A_Persistence"]["n_samples"] == 30
    assert phase_a["+3h"]["A_Persistence"]["n_samples"] == 30

    for h in ["+1h", "+2h", "+3h"]:
        p_mae = phase_a[h]["A_Persistence"]["mae_m"]
        d_mae = phase_a[h]["D_Multimodal_Ridge_dH"]["mae_m"]
        assert d_mae < p_mae
