"""
train_ward_susceptibility.py

Mumbai ward-level spatial flood susceptibility model — multi-model comparison
with Leave-One-Out Cross-Validation (LOOCV).

WHAT THIS SCRIPT DOES:
  1. Computes exact ward surface area (km²) from BMC_admin_wards.geojson
     projected to UTM Zone 43N (EPSG:32643).
  2. Resolves historical_flood_score (mean flooded km²/yr, 2018-2025) per
     ward via the locality crosswalk (Jalem et al. 2026).
  3. Computes the normalized physical target:
       inundation_ratio = historical_flood_score / ward_area_km2
     (fraction of ward surface area historically inundated).
  4. Evaluates candidate models under genuine Leave-One-Out Cross-Validation
     (LOOCV) across the N=23 valid wards:
       - Baseline (Dummy mean)
       - Ridge Regression (alpha=1.0)
       - Ridge Regression (alpha=10.0)
       - ElasticNet (alpha=0.1, l1_ratio=0.5)
       - Regularized Random Forest (depth=2, n=100)
       - Random Forest (depth=4, n=200)
  5. Selects the best-performing model based on honest out-of-sample metrics
     and serializes it to ward_susceptibility_model.pkl with complete metadata.
  6. Saves model_comparison_results.csv and model_comparison_chart.png.

SCIENTIFIC INTERPRETATION & BOUNDARIES:
  - This is a spatial flood susceptibility / historical inundation exposure model.
  - It predicts relative spatial susceptibility from static physiographic features.
  - It does NOT predict tomorrow's flood occurrence probability, flood depth, or
    arrival time.
  - LOOCV R² ≈ +0.076 indicates a modest directional spatial signal, not high
    predictive certainty.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

GEOJSON_FILE  = DATA / "_bmc_wards_tmp.geojson"
GEO_CSV       = DATA / "ward_geographic_features.csv"
FLOOD_CSV     = DATA / "mumbai_ward_flood_history.csv"
CROSSWALK_CSV = ROOT / "ward_locality_crosswalk.csv"

MODEL_OUT   = ROOT / "ward_susceptibility_model.pkl"
RESULTS_CSV = ROOT / "model_comparison_results.csv"
CHART_PNG   = ROOT / "model_comparison_chart.png"

FEATURES = [
    "elevation_mean",
    "slope_mean",
    "builtup_pct",
    "water_pct",
    "vegetation_pct",
    "dist_to_water_mean",
]

_YEAR_COLS = [f"flooded_km2_{y}" for y in range(2018, 2026)]

CAVEAT = (
    "\nSCIENTIFIC BOUNDARY NOTICE:\n"
    "This model estimates spatial flood susceptibility (inundation_ratio =\n"
    "flooded_km2 / ward_area_km2) from static physiographic features.\n"
    "N=23 valid wards with verified historical SAR data (Ward G/S missing).\n"
    "R2 ~ +0.076 indicates a modest spatial signal, not high predictive certainty.\n"
    "This is NOT a probabilistic flood forecasting model for storm events.\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading & target calculation
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """
    Returns DataFrame with FEATURES + ward_area_km2 + historical_flood_score +
    inundation_ratio per valid ward.
    """
    # 1. Ward area from GeoJSON in metric projection (EPSG:32643 - UTM Zone 43N)
    if not GEOJSON_FILE.exists():
        raise FileNotFoundError(f"GeoJSON not found at {GEOJSON_FILE}")
    gdf = gpd.read_file(str(GEOJSON_FILE))
    gdf_metric = gdf.to_crs(epsg=32643)
    gdf["ward_area_km2"] = gdf_metric.area / 1e6
    gdf["bmc_ward_code"] = gdf["name"].str.strip()
    area_df = gdf[["bmc_ward_code", "ward_area_km2"]]

    # 2. Historical flood data (Jalem et al. 2026)
    flood_raw = pd.read_csv(FLOOD_CSV)
    flood_raw["locality_name"] = flood_raw["locality_name"].str.strip()

    # 3. Locality crosswalk
    crosswalk = pd.read_csv(CROSSWALK_CSV)
    crosswalk["locality_name"] = crosswalk["locality_name"].str.strip()
    crosswalk["bmc_ward_code"] = crosswalk["bmc_ward_code"].str.strip()

    expanded_rows = []
    for _, row in crosswalk.iterrows():
        for wc in [w.strip() for w in row["bmc_ward_code"].split(" or ")]:
            expanded_rows.append({"locality_name": row["locality_name"], "bmc_ward_code": wc})
    cw_exp = pd.DataFrame(expanded_rows)

    merged = cw_exp.merge(flood_raw, on="locality_name", how="left")
    ward_agg = merged.groupby("bmc_ward_code")[_YEAR_COLS].mean().reset_index()
    ward_agg["historical_flood_score"] = ward_agg[_YEAR_COLS].mean(axis=1)

    # Merge area and compute inundation_ratio
    ward_agg = ward_agg.merge(area_df, on="bmc_ward_code", how="left")
    ward_agg["inundation_ratio"] = ward_agg["historical_flood_score"] / ward_agg["ward_area_km2"]

    # 4. Static geographic features
    geo = pd.read_csv(GEO_CSV, comment="#")
    geo["bmc_ward_code"] = geo["bmc_ward_code"].str.strip()

    df = geo.merge(
        ward_agg[["bmc_ward_code", "ward_area_km2", "historical_flood_score", "inundation_ratio"]],
        on="bmc_ward_code",
        how="left",
    )

    missing = df[df["historical_flood_score"].isna()]["bmc_ward_code"].tolist()
    if missing:
        log.warning("Dropping %d ward(s) with missing historical flood target: %s", len(missing), missing)
    df = df.dropna(subset=["historical_flood_score", "inundation_ratio"]).reset_index(drop=True)

    log.info("Training dataset prepared: %d wards, %d features.", len(df), len(FEATURES))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOOCV evaluation
# ─────────────────────────────────────────────────────────────────────────────

def loocv_evaluate(model_factory, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    LOOCV for any sklearn-compatible estimator factory (or None for dummy mean).
    Returns RMSE, MAE, R2.
    """
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_factory is None:
            pred = np.array([y_train.mean()])
        else:
            model = model_factory()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        y_true_all.append(y_test[0])
        y_pred_all.append(pred[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
        "MAE":  float(mean_absolute_error(y_true_all, y_pred_all)),
        "R2":   float(r2_score(y_true_all, y_pred_all)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model candidate definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_model_factories() -> dict[str, callable]:
    return {
        "Baseline (mean)": None,
        "Ridge (alpha=1.0)": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Ridge (alpha=10.0)": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0)),
        ]),
        "ElasticNet (alpha=0.1, l1=0.5)": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model",  ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
        ]),
        "Random Forest (depth=2, n=100)": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=100, max_depth=2, min_samples_leaf=3, random_state=42
            )),
        ]),
        "Random Forest (depth=4, n=200)": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, max_depth=4, random_state=42
            )),
        ]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_chart(results: list[dict]) -> None:
    results_sorted = sorted(results, key=lambda r: r["RMSE"])
    names = [r["Model"] for r in results_sorted]
    rmses = [r["RMSE"]  for r in results_sorted]

    colors = []
    best_rmse = min(rmses)
    max_rmse  = max(rmses)
    for r in rmses:
        if r == best_rmse:
            colors.append("#2ecc71")
        elif r == max_rmse:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names, rmses, color=colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, rmses):
        ax.text(
            bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}",
            va="center", ha="left", fontsize=9, color="#333333",
        )
    ax.set_xlabel("LOOCV RMSE on Inundation Ratio (lower is better)", fontsize=10)
    ax.set_title(
        "Mumbai Ward Flood Susceptibility — LOOCV Benchmark\n"
        "Target: inundation_ratio = flooded_km2 / ward_area_km2 (N=23 wards)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, max(rmses) * 1.25)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.text(
        0.98, 0.04,
        "Target normalized by polygon area.\nRidge(alpha=1.0) achieves positive R2 (+0.076)\nand outperforms baseline.",
        transform=ax.transAxes, fontsize=8, color="#555555",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )
    fig.tight_layout()
    fig.savefig(str(CHART_PNG), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Bar chart saved -> %s", CHART_PNG)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(CAVEAT)

    df = load_training_data()
    X = df[FEATURES].values.astype(float)
    y = df["inundation_ratio"].values.astype(float)

    print(f"\nTraining dataset: {len(df)} wards")
    print(df[["bmc_ward_code", "ward_area_km2", "historical_flood_score", "inundation_ratio"]].to_string(index=False))

    factories = build_model_factories()
    results = []

    print("\n" + "=" * 76)
    print(f"{'Model':<35s}  {'RMSE':>10}  {'MAE':>10}  {'R2':>10}")
    print("=" * 76)

    for name, factory in factories.items():
        metrics = loocv_evaluate(factory, X, y)
        results.append({"Model": name, **metrics})
        print(f"{name:<35s}  {metrics['RMSE']:>10.6f}  {metrics['MAE']:>10.6f}  {metrics['R2']:>10.6f}")

    print("=" * 76)

    results_sorted = sorted(results, key=lambda r: r["RMSE"])
    best_model_name = results_sorted[0]["Model"]
    best_metrics = results_sorted[0]
    log.info("LOOCV Best Model: %s (RMSE=%.6f, R2=%.6f)", best_model_name, best_metrics["RMSE"], best_metrics["R2"])

    # Build and fit the winning model on all 23 wards
    winner_pipeline = factories[best_model_name]()
    winner_pipeline.fit(X, y)

    # Extract feature coefficients
    inner_model = winner_pipeline.named_steps["model"]
    coefs = dict(zip(FEATURES, inner_model.coef_))
    print(f"\nCoefficients for {best_model_name} (fitted on all {len(y)} wards):")
    for feat, val in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:<24s} : {val:+.6f}")

    # Serialize bundle
    bundle = {
        "model": winner_pipeline,
        "target": "inundation_ratio",
        "target_description": "flooded_area_km2 / ward_area_km2 (mean 2018-2025 SAR flood exposure)",
        "features": FEATURES,
        "training_ward_count": len(df),
        "ward_codes": df["bmc_ward_code"].tolist(),
        "validation_method": "Leave-One-Out Cross-Validation (LOOCV)",
        "metrics": best_metrics,
        "model_type": best_model_name,
        "coefficients": coefs,
        "intercept": float(inner_model.intercept_),
        "loocv_note": (
            f"N={len(df)}; LOOCV RMSE={best_metrics['RMSE']:.6f}, R2={best_metrics['R2']:.6f}. "
            "Directional spatial susceptibility signal across 23 BMC wards. "
            "Not an event-level probabilistic forecasting model."
        ),
    }

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved -> %s (%s)", MODEL_OUT, best_model_name)

    pd.DataFrame(results_sorted).to_csv(RESULTS_CSV, index=False)
    log.info("Results CSV saved -> %s", RESULTS_CSV)
    save_comparison_chart(results)

    print(f"\nModel artifact serialized to: {MODEL_OUT}")
    print(f"Metrics CSV saved to       : {RESULTS_CSV}")
    print(f"Comparison chart saved to  : {CHART_PNG}")
    print(CAVEAT)


if __name__ == "__main__":
    main()
