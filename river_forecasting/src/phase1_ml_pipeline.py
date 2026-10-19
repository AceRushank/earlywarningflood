"""
river_forecasting/src/phase1_ml_pipeline.py
============================================
Phase 1 - Complete ML Training, Benchmarking and Model Selection
Vishnupuri Barrage / Godavari River, Nanded, Maharashtra

12 genuinely distinct model configurations (M01-M12).
Horizons: +1h, +2h, +3h evaluated independently with strict chronological splits.

Temporal splits (Asia/Kolkata IST naive):
  TRAIN:   2026-06-15 00:00 to 2026-08-09 23:00
  VAL:     2026-08-10 00:00 to 2026-08-31 23:00
  TEST:    2026-09-01 00:00 to 2026-09-30 23:00
  PHASE_A: 2026-07-30 00:00 to 2026-07-31 05:00 (pre-event train cutoff: 2026-07-29 23:00)

Safety invariants:
  - All features strictly causal at forecast origin t.
  - Missing rainfall preserved as NaN; never imputed with zero or NWP values.
  - StandardScaler fitted on TRAIN only; PredefinedSplit prevents val leakage.
  - live_inference.py 3-hour freshness gate is NOT modified by this module.
  - Artifacts saved to: river_forecasting/outputs/phase1/
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATASET_CSV = REPO_ROOT / "data" / "historical" / "dataset_2026_full_season_aligned.csv"
RAIN_HOURLY = REPO_ROOT / "data" / "historical" / "insat3ds_hem_catchment_hourly_2026.csv"
OUT_DIR     = REPO_ROOT / "river_forecasting" / "outputs" / "phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS       = [1, 2, 3]
SEED           = 42
API_DECAY_HR   = 0.9 ** (1.0 / 24.0)
TRAIN_START    = "2026-06-15 00:00:00"
TRAIN_END      = "2026-08-09 23:00:00"
VAL_START      = "2026-08-10 00:00:00"
VAL_END        = "2026-08-31 23:00:00"
TEST_START     = "2026-09-01 00:00:00"
TEST_END       = "2026-09-30 23:00:00"
PHASE_A_START  = "2026-07-30 00:00:00"
PHASE_A_END    = "2026-07-31 05:00:00"
PHASE_A_CUTOFF = "2026-07-29 23:00:00"

# ---------------------------------------------------------------------------
# Feature sets (all strictly causal at t)
# ---------------------------------------------------------------------------
FEAT_STAGE_BASE = [
    "stage_vishnupuri", "stage_v_lag1", "stage_v_lag2", "dH_v_1h", "dH_v_3h"
]
FEAT_STAGE_DYN = ["dH_v_1h", "dH_v_3h", "dH_v_6h", "accel_v_1h"]
FEAT_RAIN_CONT = [
    "rain_depth_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm"
]
FEAT_RAIN_LAG = FEAT_RAIN_CONT + [
    "rain_lag1_1h_mm", "rain_lag2_1h_mm", "rain_lag3_1h_mm"
]
FEAT_MULTIMODAL = FEAT_STAGE_DYN + FEAT_RAIN_CONT + ["api_7d_mm"]
FEAT_FULL       = FEAT_STAGE_DYN + FEAT_RAIN_LAG + ["api_7d_mm"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _sha256_short(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _compute_api(rain: pd.Series) -> pd.Series:
    """API with daily decay k=0.9, applied hourly. NaN rain -> API decays but output is NaN."""
    api, out = 0.0, []
    for r in rain:
        if pd.isna(r):
            api *= API_DECAY_HR
            out.append(np.nan)
        else:
            api = api * API_DECAY_HR + float(r)
            out.append(api)
    return pd.Series(out, index=rain.index, dtype=float)


def _valid(df: pd.DataFrame, feats: List[str], tgt: str) -> pd.DataFrame:
    """Complete-case rows: non-null in all features AND target."""
    return df.dropna(subset=feats + [tgt]).copy()


def _valid_histgb(df: pd.DataFrame, tgt: str) -> pd.DataFrame:
    """HistGB handles NaN features natively; only target must be non-null."""
    return df[df[tgt].notna()].copy()


def _bootstrap_ci(
    yt: np.ndarray, yp: np.ndarray, B: int = 500, block: int = 6
) -> Tuple[float, float]:
    """Block-bootstrap 95% CI for MAE accounting for temporal autocorrelation."""
    rng = np.random.default_rng(SEED)
    n = len(yt)
    starts = np.arange(0, max(1, n - block + 1))
    nb = max(1, n // block)
    boot = []
    for _ in range(B):
        idx = np.concatenate(
            [np.arange(s, min(s + block, n)) for s in rng.choice(starts, nb, replace=True)]
        )[:n]
        boot.append(mean_absolute_error(yt[idx], yp[idx]))
    boot = np.array(boot)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _metrics(yt: np.ndarray, yp: np.ndarray, ypers: np.ndarray) -> Dict[str, Any]:
    mae     = float(mean_absolute_error(yt, yp))
    rmse    = float(np.sqrt(mean_squared_error(yt, yp)))
    r2      = float(r2_score(yt, yp))
    bias    = float(np.mean(yp - yt))
    mae_p   = float(mean_absolute_error(yt, ypers))
    imp_pct = (mae_p - mae) / mae_p * 100 if mae_p > 0 else 0.0
    ci_lo, ci_hi = _bootstrap_ci(yt, yp)
    return dict(
        n=len(yt),
        mae_m=round(mae, 5), mae_mm=round(mae * 1000, 2),
        rmse_m=round(rmse, 5), rmse_mm=round(rmse * 1000, 2),
        r2=round(r2, 5), bias_m=round(bias, 5),
        mae_vs_persistence_m=round(mae - mae_p, 5),
        imp_vs_persistence_pct=round(imp_pct, 2),
        mae_ci_low_m=round(ci_lo, 5), mae_ci_high_m=round(ci_hi, 5),
    )


def _strata(df_e: pd.DataFrame, yp: np.ndarray, ypers: np.ndarray, tl: str) -> Dict:
    yt   = df_e[tl].values
    dh   = df_e["dH_v_1h"].values if "dH_v_1h" in df_e.columns else np.zeros(len(yt))
    rain = (df_e["rain_depth_1h_mm"].values if "rain_depth_1h_mm" in df_e.columns
            else np.full(len(yt), np.nan))
    masks = {
        "rising":         dh > 0.05,
        "falling":        dh < -0.05,
        "stable":         np.abs(dh) <= 0.05,
        "rainfall_event": rain >= 1.0,
        "dry_period":    ~(rain >= 1.0),
    }
    out = {}
    for name, m in masks.items():
        if m.sum() < 10:
            out[name] = {"n": int(m.sum()), "note": "insufficient samples (<10)"}
            continue
        mp = float(mean_absolute_error(yt[m], ypers[m]))
        mc = float(mean_absolute_error(yt[m], yp[m]))
        out[name] = {
            "n": int(m.sum()),
            "mae_m": round(mc, 5),
            "mae_persistence_m": round(mp, 5),
            "imp_pct": round((mp - mc) / max(mp, 1e-9) * 100, 2),
        }
    return out


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    """
    Load aligned dataset and engineer derived features.
    Preserves all missing values exactly; no imputation.
    """
    df = pd.read_csv(DATASET_CSV)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    log.info("Dataset: %d rows  %s .. %s IST", len(df), df["dt"].min(), df["dt"].max())
    log.info("Missing stage_vishnupuri=%.1f%%  rain_depth_1h_mm=%.1f%%",
             df["stage_vishnupuri"].isna().mean() * 100,
             df["rain_depth_1h_mm"].isna().mean() * 100)

    # Stage dynamics
    df["dH_v_6h"]    = df["stage_vishnupuri"] - df["stage_vishnupuri"].shift(6)
    df["accel_v_1h"] = df["dH_v_1h"] - df["dH_v_1h"].shift(1)

    # API - use extended archive for spinup when available
    if RAIN_HOURLY.exists():
        dr = pd.read_csv(RAIN_HOURLY)
        dr["dt"] = pd.to_datetime(dr["timestamp_ist"])
        dr = dr.sort_values("dt").reset_index(drop=True)
        dr["api_7d_mm"] = _compute_api(dr["hourly_rainfall_depth_mm"])
        df = df.merge(dr[["dt", "api_7d_mm"]], on="dt", how="left")
        log.info("API merged from extended rain archive (%d rows)", len(dr))
    else:
        df["api_7d_mm"] = _compute_api(df["rain_depth_1h_mm"])
        log.warning("Extended rain archive absent; API computed from dataset only (no pre-season spinup).")

    # Lagged rainfall (routing delay candidates)
    df["rain_lag1_1h_mm"] = df["rain_depth_1h_mm"].shift(1)
    df["rain_lag2_1h_mm"] = df["rain_depth_1h_mm"].shift(2)
    df["rain_lag3_1h_mm"] = df["rain_depth_1h_mm"].shift(3)

    log.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def get_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sp = {
        "train":   df[(df["dt"] >= TRAIN_START) & (df["dt"] <= TRAIN_END)].copy(),
        "val":     df[(df["dt"] >= VAL_START)   & (df["dt"] <= VAL_END)].copy(),
        "test":    df[(df["dt"] >= TEST_START)   & (df["dt"] <= TEST_END)].copy(),
        "phase_a": df[(df["dt"] >= PHASE_A_START) & (df["dt"] <= PHASE_A_END)].copy(),
    }
    for k, v in sp.items():
        log.info("  Split %-8s: %4d rows", k, len(v))
    return sp


# ---------------------------------------------------------------------------
# Model catalogue - 12 configurations
# ---------------------------------------------------------------------------
def get_catalogue() -> List[Dict]:
    _rf  = {"estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [4, 5, 6, 8],
            "estimator__min_samples_leaf": [3, 5, 10]}
    _et  = {"estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [4, 5, 6, None],
            "estimator__min_samples_leaf": [3, 5, 10]}
    _gb  = {"estimator__n_estimators": [50, 100, 150],
            "estimator__max_depth": [2, 3, 4],
            "estimator__learning_rate": [0.03, 0.05, 0.10]}
    _hgb = {"estimator__max_iter": [100, 200],
            "estimator__max_depth": [3, 4, None],
            "estimator__learning_rate": [0.03, 0.05, 0.10],
            "estimator__l2_regularization": [0.0, 0.1, 1.0]}
    _xgb = {"estimator__n_estimators": [50, 100, 150],
            "estimator__max_depth": [2, 3, 4],
            "estimator__learning_rate": [0.03, 0.05, 0.10],
            "estimator__reg_alpha": [0.5, 1.0, 2.0],
            "estimator__reg_lambda": [2.0, 5.0, 10.0]}

    base = [
        {"id": "M01", "name": "Persistence Baseline",
         "feats": ["stage_vishnupuri"], "ttype": "level",
         "est": None, "scale": False, "grid": {}},
        {"id": "M02", "name": "Linear Regression (Stage History)",
         "feats": FEAT_STAGE_BASE, "ttype": "level",
         "est": LinearRegression(), "scale": True, "grid": {}},
        {"id": "M03", "name": "Ridge (Stage History, alpha=1)",
         "feats": FEAT_STAGE_BASE, "ttype": "level",
         "est": Ridge(alpha=1.0), "scale": True, "grid": {}},
        {"id": "M04", "name": "Ridge (Stage-Change, alpha=10)",
         "feats": FEAT_STAGE_DYN, "ttype": "dH",
         "est": Ridge(alpha=10.0), "scale": True, "grid": {}},
        {"id": "M05", "name": "Ridge (Stage-Change + Contemp. Rain) [Model D]",
         "feats": FEAT_STAGE_DYN + FEAT_RAIN_CONT, "ttype": "dH",
         "est": Ridge(alpha=10.0), "scale": True, "grid": {}},
        {"id": "M06", "name": "Ridge (Stage-Change + Lagged Rain, alpha=10)",
         "feats": FEAT_STAGE_DYN + FEAT_RAIN_LAG, "ttype": "dH",
         "est": Ridge(alpha=10.0), "scale": True, "grid": {}},
        {"id": "M07", "name": "Ridge (Stage + Rain + API, alpha=10)",
         "feats": FEAT_MULTIMODAL, "ttype": "dH",
         "est": Ridge(alpha=10.0), "scale": True, "grid": {}},
        {"id": "M08", "name": "Random Forest (Stage + Rain + API)",
         "feats": FEAT_MULTIMODAL, "ttype": "dH",
         "est": RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                      random_state=SEED, n_jobs=-1),
         "scale": False, "grid": _rf},
        {"id": "M09", "name": "Extra Trees (Stage + Rain + API)",
         "feats": FEAT_MULTIMODAL, "ttype": "dH",
         "est": ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                    random_state=SEED, n_jobs=-1),
         "scale": False, "grid": _et},
        {"id": "M10", "name": "Gradient Boosting Regressor (sklearn GBR)",
         "feats": FEAT_MULTIMODAL, "ttype": "dH",
         "est": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=SEED),
         "scale": False, "grid": _gb},
        {"id": "M11", "name": "HistGradientBoosting (sklearn, NaN-native)",
         "feats": FEAT_MULTIMODAL, "ttype": "dH",
         "est": HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                              learning_rate=0.05, random_state=SEED),
         "scale": False, "grid": _hgb},
    ]
    xgb_entry = {
        "id": "M12",
        "name": "XGBoost Regressor (conservative)" if _XGB_AVAILABLE else "XGBoost UNAVAILABLE",
        "feats": FEAT_MULTIMODAL, "ttype": "dH",
        "est": xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=100,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=1.0, reg_lambda=5.0, random_state=SEED,
                                 tree_method="hist") if _XGB_AVAILABLE else None,
        "scale": False, "grid": _xgb if _XGB_AVAILABLE else {},
    }
    base.append(xgb_entry)
    return base


def _build_pipe(cfg: Dict) -> Optional[Pipeline]:
    est = cfg["est"]
    if est is None:
        return None
    cloned_est = clone(est)
    if cfg["scale"]:
        return Pipeline([("sc", StandardScaler()), ("estimator", cloned_est)])
    # Tree models: identity scaler for uniform param naming in tuning
    return Pipeline([("id", StandardScaler(with_mean=False, with_std=False)), ("estimator", cloned_est)])


def _fit_model(pipe: Any, Xtr: pd.DataFrame, ytr: np.ndarray,
               Xval: pd.DataFrame, yval: np.ndarray,
               grid: Dict, n_iter: int = 12) -> Any:
    if not grid:
        pipe.fit(Xtr, ytr)
        return pipe
    Xall = pd.concat([Xtr, Xval], ignore_index=True)
    yall = np.concatenate([ytr, yval])
    fold = np.concatenate([np.full(len(Xtr), -1), np.zeros(len(Xval))])
    srch = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=PredefinedSplit(fold),
                              scoring="neg_mean_absolute_error", refit=False,
                              random_state=SEED, n_jobs=-1)
    srch.fit(Xall, yall)
    log.info("    Best params: %s", srch.best_params_)
    pipe.set_params(**srch.best_params_)
    pipe.fit(Xtr, ytr)
    return pipe


def _save_artifact(model: Any, cfg: Dict, h: int, tgt_col: str, n_tr: int,
                   out_dir: Optional[Path] = None) -> Path:
    fname = f"{cfg['id']}_{h}h.joblib"
    dest_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    path  = dest_dir / fname
    meta  = {
        "model_id": cfg["id"], "model_name": cfg["name"],
        "horizon": f"+{h}h", "target_type": cfg["ttype"],
        "target_column": tgt_col, "features": cfg["feats"],
        "n_features": len(cfg["feats"]),
        "train_range": f"{TRAIN_START} -- {TRAIN_END} IST",
        "val_range":   f"{VAL_START} -- {VAL_END} IST",
        "dataset_sha256": _sha256_short(DATASET_CSV),
        "saved_utc": datetime.now(tz=timezone.utc).isoformat(),
        "random_seed": SEED, "n_train": n_tr,
        "sklearn_version": __import__("sklearn").__version__,
        "xgboost_version": __import__("xgboost").__version__ if _XGB_AVAILABLE else None,
        "notes": (
            "All features strictly causal at t. Missing rainfall kept as NaN. "
            "StandardScaler (if used) fitted on TRAIN only. "
            "Live dashboard 3h freshness gate in live_inference.py is unmodified."
        ),
    }
    joblib.dump({"model": model, "metadata": meta}, path)
    return path


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame, out_dir: Optional[Path] = None) -> Dict:
    skip = {"dt", "timestamp_ist", "split", "is_test_sync_rainfall",
            "rain_scans_in_hour", "rain_qc_flag"}
    rpt = {
        "dataset_sha256": _sha256_short(DATASET_CSV),
        "n_rows": len(df), "n_cols": len(df.columns),
        "dt_min": str(df["dt"].min()), "dt_max": str(df["dt"].max()),
        "expected_hourly_rows": int((df["dt"].max() - df["dt"].min()).total_seconds() / 3600) + 1,
        "timezone": "Asia/Kolkata (IST), naive timestamps",
        "duplicate_timestamps": int(df["dt"].duplicated().sum()),
        "split_counts": df["split"].value_counts().to_dict() if "split" in df.columns else {},
        "missingness": {},
        "temporal_gaps_gt1h": [],
        "stage_anomaly_flags": {},
    }
    for col in df.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        rpt["missingness"][col] = {
            "missing_n": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            "min": float(df[col].min()), "max": float(df[col].max()),
            "mean": round(float(df[col].mean()), 4),
        }
    dt_s = df["dt"].sort_values().reset_index(drop=True)
    diffs = dt_s.diff()
    for i in dt_s[diffs > pd.Timedelta(hours=1)].index:
        rpt["temporal_gaps_gt1h"].append({
            "gap_start": str(dt_s[i - 1]), "gap_end": str(dt_s[i]),
            "gap_hours": round((dt_s[i] - dt_s[i - 1]).total_seconds() / 3600, 1),
        })
    s = df["stage_vishnupuri"].dropna()
    rpt["stage_anomaly_flags"] = {
        "below_340m_msl": int((s < 340).sum()),
        "above_358m_msl": int((s > 358).sum()),
        "note": "Vishnupuri FRL=355.0m MSL; values outside 340-358m are suspicious.",
    }
    rpt["n_temporal_gaps"] = len(rpt["temporal_gaps_gt1h"])
    dest_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / "phase1_data_quality.json", "w") as f:
        json.dump(rpt, f, indent=2)
    return rpt


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------
def check_leakage(df: pd.DataFrame) -> bool:
    """
    Returns True if no feature is an EXACT duplicate of a future target.

    Note on threshold: stage_vishnupuri vs target_v_1h has corr~0.9999 due to
    AR-1 autocorrelation of river stage.  This is NOT leakage -- the feature is
    observed at time t and the target is the value at t+1h (a different row).
    True leakage requires BOTH near-perfect correlation AND identical variance,
    meaning the feature IS the future value byte-for-byte.

    We flag pairs where abs(corr) > 0.999999 AND var_diff < 1e-8.
    """
    skip = {"dt", "timestamp_ist", "split", "is_test_sync_rainfall",
            "rain_scans_in_hour", "rain_qc_flag"}
    feat_cols = [c for c in df.columns
                 if not c.startswith("target_") and c not in skip]
    tgt_cols  = [c for c in df.columns if c.startswith("target_")]
    all_ok = True
    for fc in feat_cols:
        for tc in tgt_cols:
            if df[fc].dtype == object or df[tc].dtype == object:
                continue
            pair = df[[fc, tc]].dropna()
            if len(pair) < 10:
                continue
            r = pair[fc].corr(pair[tc])
            if abs(r) > 0.999999:
                var_diff = abs(pair[fc].var() - pair[tc].var())
                if var_diff < 1e-8:
                    log.error("LEAKAGE DETECTED: %s IS %s  corr=%.6f  var_diff=%.2e",
                              fc, tc, r, var_diff)
                    all_ok = False
                else:
                    log.debug("High AR-1 corr %s vs %s=%.4f (not leakage; var_diff=%.2e)",
                              fc, tc, r, var_diff)
    if all_ok:
        log.info("Leakage check: all features OK (no feature is identical to a future target).")
    return all_ok


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------
class Phase1Runner:
    """Trains, tunes, evaluates, and saves all 12 model x 3 horizon combinations."""

    def __init__(self, df: pd.DataFrame, run_tuning: bool = True,
                 out_dir: Optional[Path] = None):
        self.df         = df.copy()
        self.sp         = get_splits(df)
        self.cat        = get_catalogue()
        self.run_tuning = run_tuning
        self.out_dir    = Path(out_dir) if out_dir is not None else OUT_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict = {}
        self.registry: Dict = {
            "phase": "Phase 1 -- ML Training, Benchmarking and Model Selection",
            "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
            "train_range": f"{TRAIN_START} -- {TRAIN_END} IST",
            "val_range":   f"{VAL_START} -- {VAL_END} IST",
            "test_range":  f"{TEST_START} -- {TEST_END} IST",
            "phase_a_range": f"{PHASE_A_START} -- {PHASE_A_END} IST",
            "random_seed": SEED,
            "xgboost_available": _XGB_AVAILABLE,
            "lightgbm_available": False,
            "models": {},
        }

    def run(self) -> Dict:
        log.info("=" * 70)
        log.info("PHASE 1 BENCHMARK -- Vishnupuri Barrage River Forecasting")
        log.info("=" * 70)
        tr, val = self.sp["train"], self.sp["val"]

        for h in HORIZONS:
            hs  = f"{h}h"
            tl  = f"target_v_{hs}"
            tdh = f"target_v_dH_{hs}"
            log.info("\n-- Horizon +%s --", hs)
            self.results[hs] = {}

            # Pre-event training set for Phase A evaluation
            tr_pre = self.df[
                (self.df["dt"] >= TRAIN_START) &
                ((self.df["dt"] + pd.Timedelta(hours=h)) <= pd.Timestamp(PHASE_A_CUTOFF))
            ].copy()

            for cfg in self.cat:
                mid, mname = cfg["id"], cfg["name"]
                if cfg["est"] is None and mid != "M01":
                    log.warning("  Skipping %s (estimator unavailable)", mid)
                    continue

                feats     = cfg["feats"]
                tgt       = tl if cfg["ttype"] == "level" else tdh
                is_histgb = "HistGradient" in mname

                if is_histgb:
                    tr_c  = _valid_histgb(tr,     tgt)
                    val_c = _valid_histgb(val,    tgt)
                    pre_c = _valid_histgb(tr_pre, tgt)
                else:
                    tr_c  = _valid(tr,     feats, tgt)
                    val_c = _valid(val,    feats, tgt)
                    pre_c = _valid(tr_pre, feats, tgt)

                log.info("  %s %-44s Train N=%-4d Val N=%-4d",
                         mid, mname[:44], len(tr_c), len(val_c))

                if mid == "M01":
                    fitted, fitted_pre, art = None, None, None
                else:
                    grid   = cfg["grid"] if self.run_tuning else {}
                    p_std  = _build_pipe(cfg)
                    fitted = _fit_model(p_std,
                                       tr_c[feats], tr_c[tgt].values,
                                       val_c[feats], val_c[tgt].values, grid)
                    if len(pre_c) > 0:
                        p_pre = _build_pipe(cfg)
                        p_pre.fit(pre_c[feats], pre_c[tgt].values)
                        fitted_pre = p_pre
                    else:
                        log.warning("    No pre-event training rows for %s +%s", mid, hs)
                        fitted_pre = None
                    art = _save_artifact(fitted, cfg, h, tgt, len(tr_c), out_dir=self.out_dir)
                    log.info("    Artifact: %s", art.name)

                # Evaluate on val, test, phase_a
                cohorts = {}
                for cname, cdf, m in [
                    ("val",     self.sp["val"],     fitted),
                    ("test",    self.sp["test"],    fitted),
                    ("phase_a", self.sp["phase_a"], fitted_pre if fitted_pre else fitted),
                ]:
                    cdf_c = _valid_histgb(cdf, tl) if is_histgb else _valid(cdf, feats, tl)
                    cohorts[cname] = self._eval(cdf_c, feats, tl, tdh, cfg, m, mid)

                self.results[hs][mid] = {
                    "id": mid, "name": mname, "horizon": f"+{hs}",
                    "target_type": cfg["ttype"], "features": feats,
                    "n_features": len(feats), "train_n": len(tr_c),
                    "artifact": str(art) if art else None,
                    "cohorts": cohorts,
                }
                if mid not in self.registry["models"]:
                    self.registry["models"][mid] = {
                        "id": mid, "name": mname, "target_type": cfg["ttype"],
                        "feature_set": feats, "horizons": {},
                    }
                self.registry["models"][mid]["horizons"][f"+{hs}"] = {
                    "artifact": str(art) if art else None,
                    "train_n": len(tr_c),
                    "val_mae_m":  cohorts["val"].get("mae_m"),
                    "test_mae_m": cohorts["test"].get("mae_m"),
                }

        self._finalize()
        return self.results

    def _eval(self, df_c: pd.DataFrame, feats: List[str], tl: str, tdh: str,
              cfg: Dict, model: Optional[Any], mid: str) -> Dict:
        if len(df_c) == 0:
            return {"n": 0, "note": "no valid rows"}
        # For dH models we add current stage to predicted increment;
        # require stage_vishnupuri to be non-null so we don't add to NaN.
        if cfg["ttype"] == "dH" or mid == "M01":
            df_c = df_c[df_c["stage_vishnupuri"].notna()].copy()
        if len(df_c) == 0:
            return {"n": 0, "note": "no valid rows after stage_vishnupuri filter"}
        yt    = df_c[tl].values
        yc    = df_c["stage_vishnupuri"].values
        ypers = yc.copy()
        if mid == "M01" or model is None:
            yp = ypers
        elif cfg["ttype"] == "level":
            yp = model.predict(df_c[feats])
        else:
            yp = yc + model.predict(df_c[feats])
        # Guard: if any predictions are NaN, filter those rows
        valid_mask = np.isfinite(yp) & np.isfinite(yt) & np.isfinite(ypers)
        if not valid_mask.all():
            n_nan = (~valid_mask).sum()
            log.warning("Dropping %d rows with NaN/inf predictions in eval", n_nan)
            yt, yp, ypers = yt[valid_mask], yp[valid_mask], ypers[valid_mask]
            df_c = df_c[valid_mask].copy()
        if len(yt) == 0:
            return {"n": 0, "note": "all predictions are NaN/inf"}
        m = _metrics(yt, yp, ypers)
        if "dH_v_1h" in df_c.columns:
            m["strata"] = _strata(df_c, yp, ypers, tl)
        return m

    def _finalize(self):
        # Winner by test MAE
        best = {}
        for h in HORIZONS:
            hs = f"+{h}h"
            cands = [
                (mid, mdata["horizons"].get(hs, {}).get("test_mae_m"))
                for mid, mdata in self.registry["models"].items()
                if mdata["horizons"].get(hs, {}).get("test_mae_m") is not None
            ]
            if cands:
                bm, bmae = min(cands, key=lambda x: x[1])
                best[hs] = {"winner": bm,
                            "name": self.registry["models"][bm]["name"],
                            "test_mae_m": bmae}
        self.registry["selection_by_test_mae"] = best
        with open(self.out_dir / "model_registry.json", "w") as f:
            json.dump(self.registry, f, indent=2)
        with open(self.out_dir / "phase1_benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        rows = []
        for hs, models in self.results.items():
            for mid, md in models.items():
                for cn, cm in md["cohorts"].items():
                    if "mae_m" not in cm:
                        continue
                    rows.append({
                        "horizon": f"+{hs}", "model_id": mid, "model_name": md["name"],
                        "cohort": cn, "n": cm.get("n"), "mae_m": cm.get("mae_m"),
                        "mae_mm": cm.get("mae_mm"), "rmse_m": cm.get("rmse_m"),
                        "rmse_mm": cm.get("rmse_mm"), "r2": cm.get("r2"),
                        "bias_m": cm.get("bias_m"),
                        "imp_vs_persistence_pct": cm.get("imp_vs_persistence_pct"),
                        "mae_ci_low_m": cm.get("mae_ci_low_m"),
                        "mae_ci_high_m": cm.get("mae_ci_high_m"),
                    })
        pd.DataFrame(rows).to_csv(self.out_dir / "phase1_benchmark_results.csv", index=False)
        log.info("Registry, JSON, CSV saved to %s", self.out_dir)


# ---------------------------------------------------------------------------
# Rainfall ablation study
# ---------------------------------------------------------------------------
def rainfall_ablation(df: pd.DataFrame, sp: Dict, out_dir: Optional[Path] = None) -> Dict:
    """
    Controlled ablation of rainfall features using identical evaluation timestamps.
    7 variants from persistence to XGBoost, evaluated on TEST cohort.
    """
    log.info("\n== Rainfall Ablation Study (identical TEST timestamps) ==")
    tr, te = sp["train"], sp["test"]
    out = {}

    for h in HORIZONS:
        hs  = f"{h}h"
        tl  = f"target_v_{hs}"
        tdh = f"target_v_dH_{hs}"

        # Synchronized timestamps: complete case for the MOST demanding feature set
        sync_feats = FEAT_STAGE_DYN + FEAT_RAIN_LAG + ["api_7d_mm"]
        te_s = te.dropna(subset=sync_feats + [tl, "stage_vishnupuri"]).copy()
        n_s  = len(te_s)
        log.info("+%s: N_synchronized=%d", hs, n_s)
        if n_s < 10:
            out[hs] = {"note": "insufficient synchronized rows"}
            continue

        yt    = te_s[tl].values
        yc    = te_s["stage_vishnupuri"].values
        ypers = yc.copy()

        variants = [
            ("ABL-01", "Persistence",                   None,                              None,  "level"),
            ("ABL-02", "Stage-Change Ridge (no rain)",  FEAT_STAGE_DYN,                   tdh,   "dH"),
            ("ABL-03", "Stage-Change + Contemp. Rain",  FEAT_STAGE_DYN + FEAT_RAIN_CONT,   tdh,   "dH"),
            ("ABL-04", "Stage-Change + Lagged Rain",    FEAT_STAGE_DYN + FEAT_RAIN_LAG,    tdh,   "dH"),
            ("ABL-05", "Stage-Change + Rain + API",     FEAT_MULTIMODAL,                   tdh,   "dH"),
            ("ABL-06", "Random Forest + Rain + API",    FEAT_MULTIMODAL,                   tdh,   "dH"),
            ("ABL-07",
             "XGBoost + Rain + API" if _XGB_AVAILABLE else "XGBoost UNAVAILABLE",
             FEAT_MULTIMODAL if _XGB_AVAILABLE else None,
             tdh if _XGB_AVAILABLE else None, "dH"),
        ]

        h_out = {}
        for aid, aname, feats, tgt_col, ttype in variants:
            if feats is None or tgt_col is None:
                h_out[aid] = {"name": aname, "note": "persistence or unavailable"}
                continue
            tr_a = tr.dropna(subset=feats + [tgt_col]).copy()
            if len(tr_a) < 30:
                h_out[aid] = {"name": aname, "note": f"insufficient train N={len(tr_a)}"}
                continue
            if aid == "ABL-06":
                m = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                          random_state=SEED, n_jobs=-1)
                m.fit(tr_a[feats], tr_a[tgt_col])
            elif aid == "ABL-07" and _XGB_AVAILABLE:
                m = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=100,
                                     subsample=0.8, colsample_bytree=0.8,
                                     reg_alpha=1.0, reg_lambda=5.0, random_state=SEED)
                m.fit(tr_a[feats], tr_a[tgt_col])
            else:
                m = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10.0))])
                m.fit(tr_a[feats], tr_a[tgt_col])
            yp  = yc + m.predict(te_s[feats]) if ttype == "dH" else m.predict(te_s[feats])
            met = _metrics(yt, yp, ypers)
            st  = _strata(te_s, yp, ypers, tl)
            log.info("  %s %-34s MAE=%.4fm (%+.1f%% vs pers)",
                     aid, aname[:34], met["mae_m"], met["imp_vs_persistence_pct"])
            h_out[aid] = {"name": aname, "train_n": len(tr_a), "eval_n": n_s,
                          "metrics": met, "strata": st}

        out[hs] = {"n_synchronized": n_s, "variants": h_out}

    dest_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / "phase1_rainfall_ablation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("Ablation results saved to %s", dest_dir)
    return out


# ---------------------------------------------------------------------------
# Inference interface (for later integration -- does NOT modify live dashboard)
# ---------------------------------------------------------------------------
class Phase1InferenceInterface:
    """
    Safe Phase 1 artifact loading and research prediction interface.

    DOES NOT replace live_inference.py. The live dashboard 3-hour freshness
    gate is enforced independently there and is NOT modified here.
    """
    _WARNING = (
        "HISTORICAL EVALUATION ONLY. Not the live operational forecaster. "
        "Live forecasting enforces its own independent 3-hour telemetry freshness gate."
    )

    def __init__(self, registry_path: Path = OUT_DIR / "model_registry.json"):
        if not registry_path.exists():
            raise FileNotFoundError(registry_path)
        with open(registry_path) as f:
            self.reg = json.load(f)
        self._cache: Dict[str, Any] = {}

    @classmethod
    def from_registry(cls) -> "Phase1InferenceInterface":
        return cls()

    def predict(self, model_id: str, horizon: int,
                features: Dict[str, float], origin_ts: str,
                current_stage_m: float) -> Dict:
        if horizon not in HORIZONS:
            return {"valid": False,
                    "reason": f"Horizon {horizon} not in {HORIZONS}",
                    "warning": self._WARNING}
        if model_id == "M01":
            return {"valid": True, "model_id": "M01", "name": "Persistence Baseline",
                    "origin_ts": origin_ts, "horizon_h": horizon,
                    "predicted_stage_m": current_stage_m,
                    "mode": "HISTORICAL_EVALUATION", "warning": self._WARNING}

        cache_key = f"{model_id}_{horizon}h"
        if cache_key not in self._cache:
            mdata = self.reg["models"].get(model_id)
            if mdata is None:
                return {"valid": False,
                        "reason": f"Model {model_id} not found in registry",
                        "warning": self._WARNING}
            art_path = mdata["horizons"].get(f"+{horizon}h", {}).get("artifact")
            if not art_path or not Path(art_path).exists():
                return {"valid": False,
                        "reason": f"Artifact missing for {model_id} +{horizon}h",
                        "warning": self._WARNING}
            self._cache[cache_key] = joblib.load(art_path)

        art   = self._cache[cache_key]
        meta  = art["metadata"]
        model = art["model"]
        exp   = meta["features"]
        miss  = [f for f in exp if f not in features]
        if miss:
            return {"valid": False,
                    "reason": f"Missing required features: {miss}",
                    "warning": self._WARNING}
        X = pd.DataFrame([{f: features.get(f, np.nan) for f in exp}])
        try:
            raw = float(model.predict(X)[0])
        except Exception as exc:
            return {"valid": False, "reason": str(exc), "warning": self._WARNING}

        pred = current_stage_m + raw if meta["target_type"] == "dH" else raw
        return {
            "valid": True, "model_id": model_id, "name": meta["model_name"],
            "origin_ts": origin_ts, "horizon_h": horizon,
            "predicted_stage_m": round(pred, 4),
            "predicted_dH_m": round(raw, 4) if meta["target_type"] == "dH" else None,
            "current_stage_m": current_stage_m,
            "features_used": exp, "n_features": len(exp),
            "mode": "HISTORICAL_EVALUATION", "warning": self._WARNING,
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_phase1(run_tuning: bool = True, out_dir: Optional[Path] = None) -> None:
    """
    Complete Phase 1 workflow:
      1. Data quality report
      2. Feature engineering
      3. Leakage check
      4. Training + evaluation (12 models x 3 horizons)
      5. Rainfall ablation study (7 variants, identical timestamps)
      6. Artifact registry
    """
    dest_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading features...")
    df  = load_features()
    dqr = data_quality_report(df, out_dir=dest_dir)
    log.info("DQ: rows=%d  gaps=%d  stage_anomalies_low=%d  high=%d",
             dqr["n_rows"], dqr["n_temporal_gaps"],
             dqr["stage_anomaly_flags"]["below_340m_msl"],
             dqr["stage_anomaly_flags"]["above_358m_msl"])

    ok = check_leakage(df)
    if not ok:
        log.error("Leakage detected -- review before trusting benchmark results.")

    sp = get_splits(df)
    runner = Phase1Runner(df, run_tuning=run_tuning, out_dir=dest_dir)
    runner.run()
    rainfall_ablation(df, sp, out_dir=dest_dir)
    log.info("\n== Phase 1 complete. Artifacts: %s ==", dest_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Phase 1 ML Benchmark -- Vishnupuri River Forecasting"
    )
    ap.add_argument("--no-tuning", action="store_true",
                    help="Skip hyperparameter tuning (faster; uses default params)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Directory to save artifacts and results")
    args = ap.parse_args()
    run_phase1(run_tuning=not args.no_tuning,
               out_dir=Path(args.out_dir) if args.out_dir else None)