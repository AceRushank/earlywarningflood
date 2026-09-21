"""
compare_models.py

Rigorous LOOCV comparison across model families for ward-level spatial susceptibility:
1. Dummy Mean Baseline
2. Ridge (alpha=1.0)
3. Ridge with small alpha grid (RidgeCV: 0.01, 0.1, 1.0, 10.0, 100.0)
4. ElasticNet (fixed and ElasticNetCV)
5. Lasso (fixed and LassoCV)
6. Random Forest Regressor
7. Gradient Boosting Regressor
8. SVR with RBF kernel
9. KNN Regressor (k=3, k=5)
10. XGBoost Regressor (checked for availability)

Strict rules:
- StandardScaler inside Pipeline (no leakage)
- Tuning strictly inside each training fold (N=22)
- Genuine Leave-One-Out Cross-Validation (N=23)
- Training R² recorded for overfitting diagnostics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from train_ward_susceptibility import load_training_data, FEATURES, RESULTS_CSV, CHART_PNG

def evaluate_loocv(model_name: str, model_factory, X: np.ndarray, y: np.ndarray) -> dict:
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

    loocv_rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    loocv_mae = float(mean_absolute_error(y_true_all, y_pred_all))
    loocv_r2 = float(r2_score(y_true_all, y_pred_all))

    # Full training R2 for diagnostic/overfitting check
    if model_factory is None:
        train_r2 = 0.0
    else:
        full_model = model_factory()
        full_model.fit(X, y)
        train_preds = full_model.predict(X)
        train_r2 = float(r2_score(y, train_preds))

    return {
        "Model": model_name,
        "LOOCV_RMSE": round(loocv_rmse, 6),
        "LOOCV_MAE": round(loocv_mae, 6),
        "LOOCV_R2": round(loocv_r2, 6),
        "Train_R2": round(train_r2, 6),
    }

def main():
    df = load_training_data()
    X = df[FEATURES].values.astype(float)
    y = df["inundation_ratio"].values.astype(float)

    print(f"Loaded {len(df)} wards. Target: inundation_ratio, Features: {len(FEATURES)}")

    models = [
        ("1. Dummy Mean Baseline", None),
        ("2. Ridge (alpha=1.0)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])),
        ("3. Ridge (GridCV: 0.01, 0.1, 1, 10, 100)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
        ])),
        ("4a. ElasticNet (alpha=0.01, l1=0.5)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)),
        ])),
        ("4b. ElasticNet (GridCV: alpha in [1e-3..1], l1 in [0.1..0.9])", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1.0], l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=42)),
        ])),
        ("5a. Lasso (alpha=0.01)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, random_state=42)),
        ])),
        ("5b. Lasso (GridCV: alpha in [1e-4..0.1])", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1], cv=3, random_state=42)),
        ])),
        ("6. Random Forest (depth=2, n=100)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_leaf=2, random_state=42)),
        ])),
        ("7. Gradient Boosting (depth=2, lr=0.05, n=50)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=50, max_depth=2, min_samples_leaf=2, learning_rate=0.05, random_state=42)),
        ])),
        ("8. SVR (RBF, C=1.0, eps=0.01)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, epsilon=0.01)),
        ])),
        ("9a. KNN Regressor (k=3)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=3)),
        ])),
        ("9b. KNN Regressor (k=5)", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5)),
        ])),
    ]

    # Check for XGBoost
    try:
        import xgboost as xgb
        models.append(("10. XGBoost Regressor", lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)),
        ])))
    except ImportError:
        print("\nNote: XGBoost is NOT installed in the environment -> reported as unavailable.")

    results = []
    for name, factory in models:
        res = evaluate_loocv(name, factory, X, y)
        results.append(res)

    res_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print(f"{'Model':<45} | {'LOOCV RMSE':<10} | {'LOOCV MAE':<10} | {'LOOCV R2':<10} | {'Train R2'}")
    print("-" * 80)
    for _, row in res_df.iterrows():
        print(f"{row['Model']:<45} | {row['LOOCV_RMSE']:<10.6f} | {row['LOOCV_MAE']:<10.6f} | {row['LOOCV_R2']:<10.6f} | {row['Train_R2']:<10.6f}")
    print("=" * 80)

    # Save to model_comparison_results.csv
    res_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved comparison results to: {RESULTS_CSV}")

    # Plot comparison chart
    plt.figure(figsize=(12, 6))
    colors = ["#2b5c8f" if r > 0 else "#c0392b" for r in res_df["LOOCV_R2"]]
    bars = plt.barh(res_df["Model"], res_df["LOOCV_R2"], color=colors, alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("LOOCV R² Score (Leave-One-Out Cross-Validation)")
    plt.title("Model Comparison on Ward Susceptibility (inundation_ratio, N=23)", fontsize=13, pad=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(CHART_PNG, dpi=200)
    plt.close()
    print(f"Saved updated comparison chart to: {CHART_PNG}")

if __name__ == "__main__":
    main()
