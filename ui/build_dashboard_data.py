"""
ui/build_dashboard_data.py
==========================
Bundles all verified data, model predictions, and catchment metadata
for both the 2024 baseline and the 2026 audited full-monsoon multivariable
pipeline into a unified, optimized JSON for the Monsoon Timeline UI.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "river_forecasting" / "outputs"
SPATIAL_DIR = REPO_ROOT / "river_forecasting" / "data" / "spatial"
BENCHMARK_2026_JSON = REPO_ROOT / "data" / "historical" / "multivariable_benchmark_results.json"
OUT_JSON = REPO_ROOT / "ui" / "dashboard_data.json"


def build_data():
    print("=" * 70)
    print("BUILDING UNIFIED DASHBOARD DATA BUNDLE (2024 & 2026 PIPELINES)")
    print("=" * 70)

    # 1. Catchment GeoJSON & Physiography
    with open(SPATIAL_DIR / "vishnupuri_upstream_catchment.geojson", "r") as f:
        catchment_geojson = json.load(f)

    with open(OUTPUTS_DIR / "catchment_physiographic_parameters.json", "r") as f:
        physio_params = json.load(f)

    with open(OUTPUTS_DIR / "rainfall_sources_metadata.json", "r") as f:
        sources_metadata = json.load(f)

    with open(OUTPUTS_DIR / "event1_rising_limb_results.json", "r") as f:
        rising_limb_results = json.load(f)

    # 2. Build 2024 Historical Baseline Timeline
    print("\n[1/3] Building 2024 Event 1 timeline...")
    df_stage_2024 = pd.read_csv(OUTPUTS_DIR / "stage_observations.csv")
    df_stage_2024["dt"] = pd.to_datetime(df_stage_2024["timestamp_ist"])
    df_stage_2024.sort_values("dt", inplace=True)
    df_stage_2024.drop_duplicates(subset=["dt"], keep="first", inplace=True)

    df_aligned_2024 = pd.read_csv(OUTPUTS_DIR / "event1_aligned_rainfall_stage.csv")
    df_aligned_2024["dt"] = pd.to_datetime(df_aligned_2024["dt_ist"])
    df_aligned_2024.sort_values("dt", inplace=True)
    df_aligned_2024.drop_duplicates(subset=["dt"], keep="first", inplace=True)
    sat_rain_map = df_aligned_2024.set_index("dt_ist")[["hourly_depth_mm", "mean_rain_rate_mm_hr", "completeness", "rain_accum_24h_mm"]].to_dict(orient="index")

    df_imd_2024 = pd.read_csv(OUTPUTS_DIR / "historical_catchment_rainfall_daily.csv")
    imd_rain_map = df_imd_2024.set_index("date")[["rainfall_1d_mm", "rainfall_3d_accum_mm", "rainfall_7d_accum_mm"]].to_dict(orient="index")

    df_rise_preds = pd.read_csv(OUTPUTS_DIR / "event1_rising_limb_predictions.csv")
    rise_preds_map = {}
    for _, row in df_rise_preds.iterrows():
        key = f"{row['issue_timestamp_ist']}_{row['horizon_hours']}h"
        if key not in rise_preds_map:
            rise_preds_map[key] = {
                "target_time": row["target_timestamp_ist"],
                "actual_m": float(row["actual_stage_m"])
            }
        model_key = "persistence" if "Persistence" in row["model"] else "wl_ridge"
        rise_preds_map[key][model_key] = float(row["predicted_stage_m"])
        rise_preds_map[key][f"{model_key}_err"] = float(row["absolute_error_m"])

    df_post_preds = pd.read_csv(OUTPUTS_DIR / "event1_model_predictions_test.csv")
    post_preds_map = {}
    for _, row in df_post_preds.iterrows():
        t = row["timestamp_ist"]
        post_preds_map[t] = {}
        for h in [1, 2, 3]:
            post_preds_map[t][f"{h}h"] = {
                "actual_m": float(row[f"actual_stage_{h}h_m"]) if pd.notna(row[f"actual_stage_{h}h_m"]) else None,
                "persistence": float(row[f"pred_persistence_{h}h_m"]) if pd.notna(row[f"pred_persistence_{h}h_m"]) else None,
                "wl_ridge": float(row[f"pred_wl_only_{h}h_m"]) if pd.notna(row[f"pred_wl_only_{h}h_m"]) else None,
                "rain_enhanced": float(row[f"pred_rain_enhanced_{h}h_m"]) if pd.notna(row[f"pred_rain_enhanced_{h}h_m"]) else None
            }

    min_dt_2024 = df_stage_2024["dt"].min()
    max_dt_2024 = df_stage_2024["dt"].max()
    full_idx_2024 = pd.date_range(min_dt_2024, max_dt_2024, freq="1h")

    stage_lookup_2024 = df_stage_2024.set_index("dt")["stage_m"].to_dict()
    stage_rate_lookup_2024 = df_stage_2024.set_index("dt")["rate_of_change_m_per_hr"].to_dict()

    timeline_2024 = []
    for dt_val in full_idx_2024:
        t_str = dt_val.strftime("%Y-%m-%d %H:%M:%S")
        date_str = dt_val.strftime("%Y-%m-%d")

        stage_val = stage_lookup_2024.get(dt_val)
        stage_rate = stage_rate_lookup_2024.get(dt_val)

        sat_info = sat_rain_map.get(t_str, {})
        sat_depth = sat_info.get("hourly_depth_mm")
        sat_completeness = sat_info.get("completeness", "NO_OBSERVATION")
        sat_accum_24h = sat_info.get("rain_accum_24h_mm")

        imd_info = imd_rain_map.get(date_str, {})
        imd_1d = imd_info.get("rainfall_1d_mm")
        imd_3d = imd_info.get("rainfall_3d_accum_mm")
        imd_7d = imd_info.get("rainfall_7d_accum_mm")

        preds = {}
        for h in [1, 2, 3]:
            rise_key = f"{t_str}_{h}h"
            if rise_key in rise_preds_map:
                p_item = rise_preds_map[rise_key]
                preds[f"{h}h"] = {
                    "regime": "rising_limb",
                    "target_time": p_item["target_time"],
                    "actual_m": p_item["actual_m"],
                    "persistence": p_item.get("persistence"),
                    "wl_ridge": p_item.get("wl_ridge"),
                    "stage_change_rain": None,
                    "rain_enhanced_status": "UNAVAILABLE"
                }
            elif t_str in post_preds_map and f"{h}h" in post_preds_map[t_str]:
                p_item = post_preds_map[t_str][f"{h}h"]
                if p_item["actual_m"] is not None:
                    preds[f"{h}h"] = {
                        "regime": "post_peak_tail",
                        "target_time": (dt_val + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"),
                        "actual_m": p_item["actual_m"],
                        "persistence": p_item.get("persistence"),
                        "wl_ridge": p_item.get("wl_ridge"),
                        "stage_change_rain": p_item.get("rain_enhanced"),
                        "rain_enhanced_status": "AVAILABLE_POST_PEAK"
                    }

        pt = {
            "time": t_str,
            "stage_m": round(float(stage_val), 3) if pd.notna(stage_val) else None,
            "dh_dt": round(float(stage_rate), 4) if pd.notna(stage_rate) else None,
            "sat_rain_mm": round(float(sat_depth), 3) if pd.notna(sat_depth) else None,
            "sat_completeness": sat_completeness,
            "sat_accum_24h_mm": round(float(sat_accum_24h), 2) if pd.notna(sat_accum_24h) else None,
            "api_7d_mm": None,
            "imd_1d_mm": round(float(imd_1d), 2) if pd.notna(imd_1d) else None,
            "imd_3d_mm": round(float(imd_3d), 2) if pd.notna(imd_3d) else None,
            "imd_7d_mm": round(float(imd_7d), 2) if pd.notna(imd_7d) else None,
            "predictions": preds if preds else None
        }
        timeline_2024.append(pt)

    # 3. Build 2026 Audited Multivariable Timeline
    print("\n[2/3] Building 2026 audited multivariable timeline & predictions...")
    from river_forecasting.src.multivariable_pipeline import (
        prepare_multivariable_dataset,
        MultivariableModelComparator
    )

    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    with open(BENCHMARK_2026_JSON) as f:
        bench_2026 = json.load(f)

    df_2026 = prepare_multivariable_dataset()
    comparator = MultivariableModelComparator(df_2026, train_end_ist="2026-07-29 23:00:00")
    trained_2026 = {h: comparator.train_models(h) for h in ["1h", "2h", "3h"]}

    # Train Random Forest models on multimodal features
    f_b = comparator.feature_sets["stage_base"]
    f_c = comparator.feature_sets["stage_dyn_only"]
    f_d = comparator.feature_sets["stage_change_multimodal"]

    rf_models = {}
    for h in ["1h", "2h", "3h"]:
        h_int = int(h.replace("h", ""))
        tgt_dh = f"target_v_dH_{h}"
        valid_train = (df_2026["dt"] >= "2026-06-15 00:00:00") & \
                      ((df_2026["dt"] + pd.Timedelta(hours=h_int)) <= pd.to_datetime("2026-07-29 23:00:00"))
        tr = df_2026[valid_train].dropna(subset=f_d + [tgt_dh])
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(tr[f_d], tr[tgt_dh])
        rf_models[h] = rf

    phase_a_mask = (df_2026["dt"] >= "2026-07-30 00:00:00") & (df_2026["dt"] <= "2026-07-31 05:00:00")
    sep_sync_mask = (df_2026["dt"] >= "2026-09-08 06:00:00") & (df_2026["dt"] <= "2026-09-22 19:00:00")

    b_mask = df_2026[f_b].notna().all(axis=1) & df_2026["stage_vishnupuri"].notna()
    c_mask = df_2026[f_c].notna().all(axis=1) & df_2026["stage_vishnupuri"].notna()
    d_mask = df_2026[f_d].notna().all(axis=1) & df_2026["stage_vishnupuri"].notna()

    vec_preds = {}
    for h in ["1h", "2h", "3h"]:
        vec_preds[h] = {
            "b": np.full(len(df_2026), np.nan),
            "c": np.full(len(df_2026), np.nan),
            "d": np.full(len(df_2026), np.nan),
            "rf": np.full(len(df_2026), np.nan),
            "xgb": np.full(len(df_2026), np.nan),
        }
        vec_preds[h]["b"][b_mask] = trained_2026[h]["m_B"].predict(df_2026.loc[b_mask, f_b])
        vec_preds[h]["c"][c_mask] = df_2026.loc[c_mask, "stage_vishnupuri"].values + trained_2026[h]["m_D_st"].predict(df_2026.loc[c_mask, f_c])
        vec_preds[h]["d"][d_mask] = df_2026.loc[d_mask, "stage_vishnupuri"].values + trained_2026[h]["m_D"].predict(df_2026.loc[d_mask, f_d])
        vec_preds[h]["rf"][d_mask] = df_2026.loc[d_mask, "stage_vishnupuri"].values + rf_models[h].predict(df_2026.loc[d_mask, f_d])
        vec_preds[h]["xgb"][d_mask] = df_2026.loc[d_mask, "stage_vishnupuri"].values + trained_2026[h]["m_E"].predict(df_2026.loc[d_mask, f_d])

    timeline_2026 = []
    for idx, row in df_2026.iterrows():
        t_dt = row["dt"]
        t_str = t_dt.strftime("%Y-%m-%d %H:%M:%S")
        stage_val = row["stage_vishnupuri"]
        dh_val = row["dH_v_1h"]
        r_val = row["rain_depth_1h_mm"]
        r_qc = row["rain_qc_flag"] if pd.notna(row["rain_qc_flag"]) else "MISSING_SCAN"
        r_24h = row["rain_24h_mm"]
        api_val = row["api_7d_mm"]

        preds = {}
        has_stage = pd.notna(stage_val)
        has_any_pred = b_mask.iloc[idx] or c_mask.iloc[idx] or d_mask.iloc[idx]

        if has_stage and has_any_pred:
            if phase_a_mask.iloc[idx]:
                regime_label = "phase_a_inception"
            elif sep_sync_mask.iloc[idx]:
                regime_label = "september_holdout"
            elif str(r_qc) == "OUTAGE_UNAVAILABLE":
                regime_label = "payload_outage"
            else:
                regime_label = "monsoon_monitored"

            for h in [1, 2, 3]:
                h_str = f"{h}h"
                target_col = f"target_v_{h_str}"
                actual_val = row[target_col] if pd.notna(row[target_col]) else None
                has_d = d_mask.iloc[idx]
                has_b = b_mask.iloc[idx]
                has_c = c_mask.iloc[idx]

                preds[h_str] = {
                    "regime": regime_label,
                    "target_time": (t_dt + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S"),
                    "actual_m": round(float(actual_val), 3) if actual_val is not None else None,
                    "persistence": round(float(stage_val), 3),
                    "wl_ridge": round(float(vec_preds[h_str]["b"][idx]), 3) if has_b and not np.isnan(vec_preds[h_str]["b"][idx]) else None,
                    "stage_change_ridge": round(float(vec_preds[h_str]["c"][idx]), 3) if has_c and not np.isnan(vec_preds[h_str]["c"][idx]) else None,
                    "stage_change_rain": round(float(vec_preds[h_str]["d"][idx]), 3) if has_d and not np.isnan(vec_preds[h_str]["d"][idx]) else None,
                    "random_forest": round(float(vec_preds[h_str]["rf"][idx]), 3) if has_d and not np.isnan(vec_preds[h_str]["rf"][idx]) else None,
                    "xgboost": round(float(vec_preds[h_str]["xgb"][idx]), 3) if has_d and not np.isnan(vec_preds[h_str]["xgb"][idx]) else None,
                    "rain_enhanced_status": "EXPERIMENTAL_CANDIDATE" if has_d else ("OUTAGE_RAIN_UNAVAILABLE" if str(r_qc) == "OUTAGE_UNAVAILABLE" else "UNAVAILABLE_RAIN_MISSING")
                }

        pt = {
            "time": t_str,
            "stage_m": round(float(stage_val), 3) if pd.notna(stage_val) else None,
            "dh_dt": round(float(dh_val), 4) if pd.notna(dh_val) else None,
            "sat_rain_mm": round(float(r_val), 3) if pd.notna(r_val) else None,
            "sat_completeness": str(r_qc),
            "sat_accum_24h_mm": round(float(r_24h), 2) if pd.notna(r_24h) else None,
            "api_7d_mm": round(float(api_val), 2) if pd.notna(api_val) else None,
            "imd_1d_mm": None,
            "imd_3d_mm": None,
            "predictions": preds if preds else None
        }
        timeline_2026.append(pt)

    # 4. Annotations
    annotations_2024 = [
        {
            "time": "2024-07-31 18:00:00",
            "title": "Full Window Maximum",
            "desc": "Antecedent spate crest from late-July rain reaching 354.888 m MSL.",
            "type": "warning"
        },
        {
            "time": "2024-08-03 10:00:00",
            "title": "Event 1 Rising Limb Onset",
            "desc": "Stage trough at 353.388 m MSL before steep convective inflow surge.",
            "type": "info"
        },
        {
            "time": "2024-08-03 14:00:00",
            "title": "Rapid Surge Initiation",
            "desc": "River stage begins rapid ascent (+0.291 m in 1 hour).",
            "type": "surge"
        },
        {
            "time": "2024-08-04 20:00:00",
            "title": "Event 1 Crest (Peak)",
            "desc": "True flood crest of Event 1 reaches 354.798 m MSL (+1.410 m rise over 34h).",
            "type": "danger"
        },
        {
            "time": "2024-08-05 05:00:00",
            "title": "Post-Peak Test Start",
            "desc": "Recession tail evaluation begins at 354.009 m MSL (9h after peak).",
            "type": "info"
        }
    ]

    annotations_2026 = [
        {
            "time": "2026-06-15 00:00:00",
            "title": "2026 Season Start",
            "desc": "Pre-event flat baseline pool at ~350.8 m MSL. API initialized with 15-day warmup.",
            "type": "info"
        },
        {
            "time": "2026-07-30 00:00:00",
            "title": "Phase A Flood Inception",
            "desc": "Monsoon flood wave inception onset (+1.41 m surge). 100% complete satellite rain.",
            "type": "surge"
        },
        {
            "time": "2026-07-31 05:30:00",
            "title": "Satellite Payload Outage",
            "desc": "INSAT-3DS operational payload offline (113 consecutive hours). Preserved strictly as NaN.",
            "type": "warning"
        },
        {
            "time": "2026-08-01 08:00:00",
            "title": "Flood Crest (Radial Gates Open)",
            "desc": "Barrage stage crests at ~355.18 m MSL. Regulated by unmonitored radial gate discharge.",
            "type": "danger"
        },
        {
            "time": "2026-08-04 23:00:00",
            "title": "Satellite Scans Resumed",
            "desc": "INSAT-3DS operational payload restored with valid scans.",
            "type": "info"
        },
        {
            "time": "2026-09-08 06:00:00",
            "title": "September Holdout Start",
            "desc": "Synchronized held-out test subwindow begins under flat-pool quiescent conditions.",
            "type": "info"
        },
        {
            "time": "2026-09-22 19:00:00",
            "title": "September Holdout End",
            "desc": "Evaluation subwindow ends. Stage-Change+Rain strictly matches persistence within <0.4mm.",
            "type": "info"
        }
    ]

    # 5. Assemble Bundle
    print("\n[3/3] Assembling combined dashboard JSON bundle...")

    # Load Ma'am's Parameter Audit & Reach GeoJSON
    reach_geojson_file = SPATIAL_DIR / "vishnupuri_river_reach_study_area.geojson"
    reach_corridor_geojson = None
    if reach_geojson_file.exists():
        with open(reach_geojson_file, "r", encoding="utf-8") as f:
            reach_corridor_geojson = json.load(f)

    gsw_geojson_file = SPATIAL_DIR / "vishnupuri_gsw_occurrence_reach.geojson"
    gsw_geojson = None
    if gsw_geojson_file.exists():
        with open(gsw_geojson_file, "r", encoding="utf-8") as f:
            gsw_geojson = json.load(f)

    audit_file = OUTPUTS_DIR / "maam_parameter_flood_audit.json"
    maam_parameter_audit = None
    if audit_file.exists():
        with open(audit_file, "r", encoding="utf-8") as f:
            maam_parameter_audit = json.load(f)

    comp_bench_file = REPO_ROOT / "data" / "historical" / "comprehensive_model_comparison_results.json"
    benchmarks_comp = None
    if comp_bench_file.exists():
        with open(comp_bench_file, "r", encoding="utf-8") as f:
            benchmarks_comp = json.load(f)

    bundle = {
        "metadata": {
            "title": "Godavari River Early Warning Flood Forecasting System",
            "station": "Vishnupuri Barrage, Nanded, Maharashtra",
            "coordinates": {"lat": 19.126944, "lon": 77.280833},
            "disclaimer": "RESEARCH PROTOTYPE: Validated scientific demonstration for Vishnupuri Barrage. Evaluates causal forecast skill under strict non-leakage protocols.",
            "sources": sources_metadata["sources"]
        },
        "physiography": physio_params,
        "evaluation_metrics": {
            "rising_limb": rising_limb_results["models"],
            "rising_limb_paired": rising_limb_results["paired_comparisons_vs_persistence"],
            "audited_2026": bench_2026,
            "experimental_disclaimer": (
                "EXPERIMENTAL CANDIDATE: Stage-Change + Rainfall (dH Model D) reduces MAE by 5.5%–6.8% "
                "over persistence during early flood inception, but matches persistence on flat pool and is unproven "
                "during peak cresting due to unmonitored spillway gate operations."
            )
        },
        "benchmarks_comprehensive": benchmarks_comp,
        "catchment_geojson": catchment_geojson,
        "reach_corridor_geojson": reach_corridor_geojson,
        "gsw_geojson": gsw_geojson,
        "maam_parameter_audit": maam_parameter_audit,
        "annotations_2024": annotations_2024,
        "annotations_2026": annotations_2026,
        "annotations": annotations_2024,
        "timeline_2024": timeline_2024,
        "timeline_2026": timeline_2026,
        "timeline": timeline_2024
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(bundle, f)

    sz_kb = OUT_JSON.stat().st_size / 1024
    print(f"\nSaved dashboard bundle to {OUT_JSON}")
    print(f"  2024 Timeline: {len(timeline_2024)} steps")
    print(f"  2026 Timeline: {len(timeline_2026)} steps")
    print(f"  Total Bundle Size: {sz_kb:.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    build_data()
