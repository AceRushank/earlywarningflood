"""
river_forecasting/src/multivariable_pipeline.py
================================================
Comprehensive multivariable pipeline for Vishnupuri Barrage stage forecasting.

Implements:
1. Feature availability audit metadata.
2. Strictly causal feature engineering (stage dynamics, rainfall accumulations,
   antecedent precipitation index, FRL freeboard, operating regime indicators,
   and upstream/downstream gauge dynamics).
3. Candidate Model Implementations:
   - Model A: Persistence
   - Model B: Regularized Autoregressive Stage-Only (Level)
   - Model C: Stage + Rainfall (Level)
   - Model D: Stage Change Increment Model (dH)
   - Model E: Conservative Non-Linear XGBoost (dH)
   - Model F: Regime-Aware Model (dH)
4. Comprehensive chronological evaluation and ablation suite.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"
TELEMETRY_RAW_CSV = REPO_ROOT / "data" / "rwl_tel_hr_maharashtra_sw_006_2026_2030.csv"

FRL_ELEVATION_M = 355.00
REGIME_HIGH_POOL_THRESHOLD_M = 354.00


def get_feature_audit_matrix() -> List[Dict[str, Any]]:
    """
    Returns the comprehensive feature availability and provenance audit table.
    """
    return [
        {
            "feature_name": "stage_vishnupuri",
            "units": "m MSL",
            "source": "NWIC Godavari Telemetry (Maharashtra SW 006)",
            "temporal_resolution": "1 hour",
            "season_coverage_pct": 93.2,
            "missing_pct": 6.8,
            "available_at_t": True,
            "used_in_models": "A, B, C, D, E, F",
            "notes": "Current river stage at Vishnupuri Barrage. Baseline elevation ~350.5m, FRL=355.0m."
        },
        {
            "feature_name": "stage_v_lag1, stage_v_lag2, stage_v_lag3",
            "units": "m MSL",
            "source": "Derived from Vishnupuri Telemetry",
            "temporal_resolution": "1 hour (t-1, t-2, t-3)",
            "season_coverage_pct": 93.1,
            "missing_pct": 6.9,
            "available_at_t": True,
            "used_in_models": "B, C, D, E, F",
            "notes": "Autoregressive stage memory at 1h, 2h, 3h lags."
        },
        {
            "feature_name": "dH_v_1h, dH_v_3h, dH_v_6h",
            "units": "m / window",
            "source": "Derived from Vishnupuri Telemetry",
            "temporal_resolution": "1 hour (t - t-k)",
            "season_coverage_pct": 93.0,
            "missing_pct": 7.0,
            "available_at_t": True,
            "used_in_models": "B, C, D, E, F",
            "notes": "Recent rate-of-rise and hydrograph trend over 1h, 3h, 6h."
        },
        {
            "feature_name": "accel_v_1h",
            "units": "m / hr^2",
            "source": "Derived from Vishnupuri Telemetry",
            "temporal_resolution": "1 hour (dH_1h(t) - dH_1h(t-1))",
            "season_coverage_pct": 92.8,
            "missing_pct": 7.2,
            "available_at_t": True,
            "used_in_models": "D, E, F",
            "notes": "Second difference representing curvature / acceleration of rising limb."
        },
        {
            "feature_name": "rain_depth_1h_mm",
            "units": "mm",
            "source": "ISRO MOSDAC INSAT-3DS HEM (Catchment-integrated)",
            "temporal_resolution": "1 hour",
            "season_coverage_pct": 94.4,
            "missing_pct": 5.6,
            "available_at_t": True,
            "used_in_models": "C, D, E, F",
            "notes": "Hourly spatial mean depth over 51,768 km² catchment. Outage July 31–Aug 4 preserved as NaN."
        },
        {
            "feature_name": "rain_3h_mm, rain_6h_mm, rain_12h_mm, rain_24h_mm",
            "units": "mm",
            "source": "Derived from INSAT-3DS HEM",
            "temporal_resolution": "Rolling backward sums",
            "season_coverage_pct": 93.8,
            "missing_pct": 6.2,
            "available_at_t": True,
            "used_in_models": "C, D, E, F",
            "notes": "Backward accumulations capturing antecedent catchment rainfall."
        },
        {
            "feature_name": "api_7d_mm",
            "units": "mm equivalent",
            "source": "Derived Antecedent Precipitation Index (decay k=0.9/day)",
            "temporal_resolution": "1 hour",
            "season_coverage_pct": 94.4,
            "missing_pct": 5.6,
            "available_at_t": True,
            "used_in_models": "D, E, F",
            "notes": "Catchment wetness proxy accounting for soil saturation memory."
        },
        {
            "feature_name": "freeboard_frl_m",
            "units": "m",
            "source": "Hydrological specification (355.0m - stage)",
            "temporal_resolution": "1 hour",
            "season_coverage_pct": 93.2,
            "missing_pct": 6.8,
            "available_at_t": True,
            "used_in_models": "F",
            "notes": "Available storage buffer remaining before Full Reservoir Level (355.0m MSL)."
        },
        {
            "feature_name": "is_high_pool",
            "units": "binary {0, 1}",
            "source": "Regime definition (stage >= 354.0m MSL)",
            "temporal_resolution": "1 hour",
            "season_coverage_pct": 93.2,
            "missing_pct": 6.8,
            "available_at_t": True,
            "used_in_models": "F",
            "notes": "Distinguishes filling regime (gates closed) from crest regulation regime (gates operating)."
        },
        {
            "feature_name": "stage_mudgal, dH_m_1h",
            "units": "m MSL, m/hr",
            "source": "NWIC Godavari Telemetry (Mudgal Barrage, ~120km upstream)",
            "temporal_resolution": "1 hour",
            "season_coverage_pct": 94.8,
            "missing_pct": 5.2,
            "available_at_t": True,
            "used_in_models": "Ablation candidate",
            "notes": "Upstream mainstem stage. Hydrodynamic travel time ~20–30h limits 1–3h lead time utility."
        },
        {
            "feature_name": "stage_raher, stage_babhali",
            "units": "m MSL",
            "source": "NWIC Godavari Telemetry (Raher & Babhali Barrage)",
            "temporal_resolution": "1 hour",
            "season_coverage_pct": 94.8,
            "missing_pct": 5.2,
            "available_at_t": True,
            "used_in_models": "Ablation candidate",
            "notes": "Downstream gauges. Causal direction runs from Vishnupuri downstream, not reverse."
        },
        {
            "feature_name": "ground_rain_imd",
            "units": "mm",
            "source": "IMD automatic weather stations (Nanded / catchment)",
            "temporal_resolution": "UNAVAILABLE",
            "season_coverage_pct": 0.0,
            "missing_pct": 100.0,
            "available_at_t": False,
            "used_in_models": "None (UNAVAILABLE)",
            "notes": "No ground telemetry in active 2026 archive. Limits ground-truth validation."
        },
        {
            "feature_name": "nwp_rainfall_forecasts",
            "units": "mm/hr",
            "source": "IMD GFS / NCMRWF numerical weather prediction grids",
            "temporal_resolution": "UNAVAILABLE",
            "season_coverage_pct": 0.0,
            "missing_pct": 100.0,
            "available_at_t": False,
            "used_in_models": "None (UNAVAILABLE)",
            "notes": "Archived NWP forecast grids not available in local dataset."
        },
        {
            "feature_name": "gate_openings_and_discharge",
            "units": "m (gate height), m^3/s (discharge)",
            "source": "Vishnupuri Barrage operational logs (WRD Maharashtra)",
            "temporal_resolution": "UNAVAILABLE",
            "season_coverage_pct": 0.0,
            "missing_pct": 100.0,
            "available_at_t": False,
            "used_in_models": "None (UNAVAILABLE)",
            "notes": "NWIC lists Is_DischargeDataAvailable='No'. Major unobservable for downstream warnings."
        },
        {
            "feature_name": "soil_moisture_in_situ",
            "units": "m^3/m^3",
            "source": "In-situ / SMAP / ASCAT satellite soil moisture",
            "temporal_resolution": "UNAVAILABLE",
            "season_coverage_pct": 0.0,
            "missing_pct": 100.0,
            "available_at_t": False,
            "used_in_models": "None (Proxied by API)",
            "notes": "Not archived for 2026. Proxied by 7-day Antecedent Precipitation Index (API)."
        }
    ]


def prepare_multivariable_dataset(
    dataset_csv: Path = DATASET_CSV,
    telemetry_raw_csv: Path = TELEMETRY_RAW_CSV
) -> pd.DataFrame:
    """
    Loads base dataset, merges upstream Mudgal telemetry, and engineers causal features.
    """
    df = pd.read_csv(dataset_csv)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    # 1. Merge Upstream Mudgal Barrage Telemetry
    if telemetry_raw_csv.exists():
        df_raw = pd.read_csv(
            telemetry_raw_csv,
            usecols=["Station", "Data Acquisition Time", "River Water Level Telemetry Hourly (meter)"]
        )
        mudgal = df_raw[df_raw["Station"].str.contains("Mudgal", case=False, na=False)].copy()
        mudgal["dt"] = pd.to_datetime(mudgal["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce")
        mudgal["stage_mudgal"] = pd.to_numeric(mudgal["River Water Level Telemetry Hourly (meter)"], errors="coerce")
        mudgal = mudgal.drop_duplicates(subset=["dt"]).sort_values("dt")
        df = pd.merge(df, mudgal[["dt", "stage_mudgal"]], on="dt", how="left")
    else:
        df["stage_mudgal"] = np.nan

    # 2. Stage Dynamics & Trends
    df["dH_v_6h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(6)
    df["dH_v_12h"] = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(12)
    df["accel_v_1h"] = df["dH_v_1h"] - df["dH_v_1h"].shift(1)

    # 3. Upstream Mudgal Dynamics
    df["stage_m_lag1"] = df["stage_mudgal"].shift(1)
    df["dH_m_1h"] = df["stage_mudgal"] - df["stage_m_lag1"]

    # 4. FRL and Operating Regime Features
    df["freeboard_frl_m"] = FRL_ELEVATION_M - df["stage_vishnupuri"]
    df["is_high_pool"] = (df["stage_vishnupuri"] >= REGIME_HIGH_POOL_THRESHOLD_M).astype(float)

    # 5. Causal Antecedent Precipitation Index (API, 7-day memory with May 31 antecedent warmup)
    rain_hourly_path = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_hourly_2026.csv"
    if rain_hourly_path.exists():
        df_full_rain = pd.read_csv(rain_hourly_path)
        df_full_rain["dt"] = pd.to_datetime(df_full_rain["timestamp_ist"])
        df_full_rain = df_full_rain.sort_values("dt").reset_index(drop=True)
        
        api = 0.0
        decay_hr = 0.9 ** (1.0 / 24.0)
        api_series = []
        for r in df_full_rain["hourly_rainfall_depth_mm"]:
            if pd.isna(r):
                api = api * decay_hr
                api_series.append(np.nan)
            else:
                api = api * decay_hr + float(r)
                api_series.append(api)
        df_full_rain["api_7d_mm"] = api_series
        
        # Merge pre-warmed API into main dataframe
        df = pd.merge(df, df_full_rain[["dt", "api_7d_mm"]], on="dt", how="left")
    else:
        api = 0.0
        api_list = []
        decay_hr = 0.9 ** (1.0 / 24.0)
        for r in df["rain_depth_1h_mm"]:
            if pd.isna(r):
                api = api * decay_hr
                api_list.append(np.nan)
            else:
                api = api * decay_hr + float(r)
                api_list.append(api)
        df["api_7d_mm"] = api_list

    return df


class MultivariableModelComparator:
    """
    Builds, trains, and compares candidate models (A, B, C, D, E, F)
    using strictly causal features and chronological validation.
    """

    def __init__(self, df: pd.DataFrame, train_end_ist: str = "2026-07-29 23:00:00"):
        self.df = df.copy()
        self.train_end_ist = pd.to_datetime(train_end_ist)
        self.feature_sets = self._define_feature_sets()

    def _define_feature_sets(self) -> Dict[str, List[str]]:
        stage_base = ["stage_vishnupuri", "stage_v_lag1", "stage_v_lag2", "dH_v_1h", "dH_v_3h"]
        stage_dyn = ["dH_v_1h", "dH_v_3h", "dH_v_6h", "accel_v_1h"]
        rain = ["rain_depth_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm", "api_7d_mm"]
        regime = ["freeboard_frl_m", "is_high_pool"]
        return {
            "stage_base": stage_base,
            "multimodal_level": stage_base + ["rain_depth_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm"],
            "stage_dyn_only": stage_dyn,
            "stage_change_multimodal": stage_dyn + rain,
            "regime_aware": stage_dyn + rain + regime,
            "all_features": stage_dyn + rain + regime + ["dH_m_1h", "dH_r_1h", "dH_b_1h"]
        }

    def train_models(self, h_str: str) -> Dict[str, Any]:
        """
        Trains all model candidates for horizon h_str with zero target leakage.
        """
        h_int = int(h_str.replace("h", ""))
        target_lvl = f"target_v_{h_str}"
        target_dh = f"target_v_dH_{h_str}"

        # Causal pre-event training cohort
        valid_train = (self.df["dt"] >= "2026-06-15 00:00:00") & \
                      ((self.df["dt"] + pd.Timedelta(hours=h_int)) <= self.train_end_ist)
        tr = self.df[valid_train].copy()

        trained = {}

        # Model B: Regularized Autoregressive Stage-Only (Level)
        f_b = self.feature_sets["stage_base"]
        tr_b = tr.dropna(subset=f_b + [target_lvl])
        trained["m_B"] = Ridge(alpha=1.0).fit(tr_b[f_b], tr_b[target_lvl])

        # Model C: Stage + Rainfall (Level)
        f_c = self.feature_sets["multimodal_level"]
        tr_c = tr.dropna(subset=f_c + [target_lvl])
        trained["m_C"] = Ridge(alpha=1.0).fit(tr_c[f_c], tr_c[target_lvl])

        # Model D: Future Stage Change Model (dH)
        f_d = self.feature_sets["stage_change_multimodal"]
        tr_d = tr.dropna(subset=f_d + [target_dh])
        trained["m_D"] = Ridge(alpha=10.0).fit(tr_d[f_d], tr_d[target_dh])

        # Model D_stage_only: Stage dynamics only without rain (for ablation)
        f_d_st = self.feature_sets["stage_dyn_only"]
        tr_d_st = tr.dropna(subset=f_d_st + [target_dh])
        trained["m_D_st"] = Ridge(alpha=10.0).fit(tr_d_st[f_d_st], tr_d_st[target_dh])

        # Model E: Conservative Non-Linear XGBoost (dH)
        tr_e = tr.dropna(subset=f_d + [target_dh])
        m_e = xgb.XGBRegressor(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42
        )
        m_e.fit(tr_e[f_d], tr_e[target_dh])
        trained["m_E"] = m_e

        # Model F: Regime-Aware Model (dH)
        f_f = self.feature_sets["regime_aware"]
        tr_f = tr.dropna(subset=f_f + [target_dh])
        trained["m_F"] = Ridge(alpha=10.0).fit(tr_f[f_f], tr_f[target_dh])

        return trained

    def evaluate_test_cohort(
        self,
        df_test: pd.DataFrame,
        h_str: str,
        trained_models: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates Candidates A through F on a given test partition.
        """
        target_lvl = f"target_v_{h_str}"
        target_dh = f"target_v_dH_{h_str}"

        # Complete-case rows for multimodal evaluation
        f_all = self.feature_sets["regime_aware"]
        valid_eval = df_test.dropna(subset=f_all + [target_lvl, "stage_vishnupuri"]).copy()

        if len(valid_eval) == 0:
            return {}

        y_true = valid_eval[target_lvl].values
        current_h = valid_eval["stage_vishnupuri"].values

        # Predictions
        preds = {
            "A_Persistence": current_h,
            "B_StageOnly_Ridge_Level": trained_models["m_B"].predict(valid_eval[self.feature_sets["stage_base"]]),
            "C_Multimodal_Ridge_Level": trained_models["m_C"].predict(valid_eval[self.feature_sets["multimodal_level"]]),
            "D_StageDynamicsOnly_dH": current_h + trained_models["m_D_st"].predict(valid_eval[self.feature_sets["stage_dyn_only"]]),
            "D_Multimodal_Ridge_dH": current_h + trained_models["m_D"].predict(valid_eval[self.feature_sets["stage_change_multimodal"]]),
            "E_Conservative_XGBoost_dH": current_h + trained_models["m_E"].predict(valid_eval[self.feature_sets["stage_change_multimodal"]]),
            "F_RegimeAware_Ridge_dH": current_h + trained_models["m_F"].predict(valid_eval[self.feature_sets["regime_aware"]])
        }

        results = {}
        for name, p_arr in preds.items():
            mae = float(mean_absolute_error(y_true, p_arr))
            rmse = float(np.sqrt(mean_squared_error(y_true, p_arr)))
            bias = float(np.mean(p_arr - y_true))
            peak_err = float(np.max(np.abs(p_arr - y_true)))

            results[name] = {
                "mae_m": round(mae, 4),
                "rmse_m": round(rmse, 4),
                "bias_m": round(bias, 4),
                "peak_abs_err_m": round(peak_err, 4),
                "n_samples": len(valid_eval)
            }

        return results
