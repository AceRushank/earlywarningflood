# -*- coding: utf-8 -*-
"""
forecast_mosdac_rainfall.py (v2: gap-corrected)
================================================
MOSDAC half-hourly rainfall forecasting prototype.

TEMPORAL-GAP VALIDATION (v2 -- corrected):
  A supervised row at index i is valid ONLY if every timestamp required by
  lag features (up to lag_12), rolling windows (up to 12-step), and
  target horizons (+2, +4, +6 steps = +1h, +2h, +3h) exists at EXACTLY
  30-minute intervals in the real MOSDAC series.
  Validation uses actual timestamp differences, NOT array position.
  pandas .shift() is position-based and silently bridges gaps.
  v1 used .shift() naively: 289 interior gap-contaminated rows leaked through.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    USE_XGB = True
    MODEL_NAME = "XGBoost"
except ImportError:
    USE_XGB = False
    MODEL_NAME = "RandomForest"

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "mosdac")
OUT_DIR  = BASE_DIR
os.makedirs(os.path.join(OUT_DIR, "outputs"), exist_ok=True)

PIXEL_CSV    = os.path.join(DATA_DIR, "mumbai_rainfall_30min.csv")
FORECAST_CSV = os.path.join(DATA_DIR, "mosdac_forecast_dataset.csv")
RESULTS_TXT  = os.path.join(OUT_DIR, "outputs", "mosdac_forecast_results.txt")
PRED_CSV     = os.path.join(OUT_DIR, "outputs", "mosdac_forecast_predictions.csv")
EVAL_PNG     = os.path.join(OUT_DIR, "outputs", "mosdac_forecast_evaluation.png")
QC_TXT       = os.path.join(OUT_DIR, "outputs", "mosdac_forecast_data_qc.txt")

HORIZONS      = {"+1h": 2, "+2h": 4, "+3h": 6}
LAGS          = [1, 2, 3, 4, 6, 8, 12]
ROLL_WINS     = [2, 4, 6, 12]
ROLL_MAX_WINS = [4, 12]
ACCUM_WINS    = [4, 8]
STEP_MIN      = 30
STEP          = np.timedelta64(STEP_MIN, "m")

C_ACTUAL  = "#1F497D"
C_PERSIST = "#E67E22"
C_ML      = "#27AE60"

# v1 results for comparison table
V1 = {
    "+1h": {"pm":0.6166,"pr":2.6078,"pr2":-0.3364,"mm":0.3964,"mr":1.8974,"mr2":0.2926},
    "+2h": {"pm":0.7159,"pr":2.7337,"pr2":-0.4692,"mm":0.4774,"mr":2.1179,"mr2":0.1181},
    "+3h": {"pm":0.7575,"pr":2.9050,"pr2":-0.6576,"mm":0.5507,"mr":2.2855,"mr2":-0.0260},
}

# =============================================================================
# STEP 1 -- Load pixel data and build regional mean series
# =============================================================================
print("Loading MOSDAC pixel data...")
raw = pd.read_csv(PIXEL_CSV)
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
rain_col = "rainfall_mm_hr"

ts_sorted_all = np.sort(raw["timestamp"].unique())
full_range    = pd.date_range(ts_sorted_all[0], ts_sorted_all[-1], freq="30min")
missing_ts    = sorted(set(full_range) - set(pd.DatetimeIndex(ts_sorted_all)))
ppt           = raw.groupby("timestamp").size()
dupes         = raw.duplicated(subset=["timestamp","latitude","longitude"]).sum()
zero_frac     = (raw[rain_col] == 0).mean()
ts_means_all  = raw.groupby("timestamp")[rain_col].mean()

ts_series = (raw.groupby("timestamp")[rain_col]
             .mean().rename("rainfall_mumbai_mean_mmhr")
             .sort_index().reset_index())
ts_series.columns = ["timestamp", "rainfall_mumbai_mean_mmhr"]
r = "rainfall_mumbai_mean_mmhr"

ts_arr = ts_series["timestamp"].values
vals   = ts_series[r].values
n      = len(ts_arr)
print("  Regional mean series: %d timesteps" % n)

# =============================================================================
# STEP 2 -- Strict temporal-gap validation
# =============================================================================
print("Computing strict gap-validity mask...")

valid_mask = np.zeros(n, dtype=bool)
drop_lag = drop_tgt = drop_oob = 0

for i in range(n):
    ok = True
    for k in LAGS:
        if i - k < 0:
            ok = False; drop_oob += 1; break
        diff = int((ts_arr[i] - ts_arr[i-k]) / np.timedelta64(1, "m"))
        if diff != k * STEP_MIN:
            ok = False; drop_lag += 1; break
    if ok:
        for s in HORIZONS.values():
            if i + s >= n:
                ok = False; drop_oob += 1; break
            diff = int((ts_arr[i+s] - ts_arr[i]) / np.timedelta64(1, "m"))
            if diff != s * STEP_MIN:
                ok = False; drop_tgt += 1; break
    valid_mask[i] = ok

n_valid   = int(valid_mask.sum())
n_dropped = n - n_valid
n_old_valid = 651
n_leaked    = n_dropped - (n - n_old_valid)

print("  Total rows     : %d" % n)
print("  Valid rows     : %d" % n_valid)
print("  Dropped total  : %d" % n_dropped)
print("    out-of-bounds: %d" % drop_oob)
print("    lag gap      : %d" % drop_lag)
print("    target gap   : %d" % drop_tgt)
print("  v1 valid rows (leaked): %d" % n_old_valid)
print("  Interior rows leaked in v1: %d" % n_leaked)

# =============================================================================
# STEP 3 -- Feature engineering (gap-safe, direct index access)
# =============================================================================
print("\nEngineering features on valid rows...")
valid_idx = np.where(valid_mask)[0]
rows = []
for i in valid_idx:
    row = {"timestamp": ts_arr[i], r: vals[i]}
    for k in LAGS:
        row["lag_%d" % k] = vals[i - k]
    for w in ROLL_WINS:
        row["roll_mean_%d" % w] = float(np.mean(vals[i-w:i]))
    for w in ROLL_MAX_WINS:
        row["roll_max_%d" % w] = float(np.max(vals[i-w:i]))
    row["delta_1"] = vals[i] - vals[i-1]
    row["delta_2"] = vals[i] - vals[i-2]
    for w in ACCUM_WINS:
        row["accum_%dstep" % w] = float(np.sum(vals[i-w:i]) * 0.5)
    ts_pd = pd.Timestamp(ts_arr[i])
    row["hour_sin"] = np.sin(2 * np.pi * ts_pd.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * ts_pd.hour / 24)
    for label, s in HORIZONS.items():
        row["target_%s" % label] = vals[i + s]
    rows.append(row)

feat_clean   = pd.DataFrame(rows).reset_index(drop=True)
TARGET_COLS  = ["target_%s" % h for h in HORIZONS]
FEATURE_COLS = [c for c in feat_clean.columns if c != "timestamp" and c not in TARGET_COLS]

assert feat_clean[FEATURE_COLS + TARGET_COLS].isna().sum().sum() == 0, \
    "NaN detected after gap-validation -- logic error!"
print("  Clean rows : %d" % len(feat_clean))
print("  Features   : %d" % len(FEATURE_COLS))
print("  NaN check  : PASSED")

feat_clean[["timestamp"] + FEATURE_COLS + TARGET_COLS].to_csv(FORECAST_CSV, index=False)
print("  Forecast dataset -> %s" % FORECAST_CSV)

# =============================================================================
# STEP 4 -- Chronological 60/20/20 split
# =============================================================================
nf = len(feat_clean)
train_end = int(nf * 0.60)
val_end   = int(nf * 0.80)
df_train = feat_clean.iloc[:train_end].copy()
df_val   = feat_clean.iloc[train_end:val_end].copy()
df_test  = feat_clean.iloc[val_end:].copy()

print("\nChronological split:")
print("  Train : %4d rows  %s -> %s" % (len(df_train),
      str(df_train.timestamp.min())[:19], str(df_train.timestamp.max())[:19]))
print("  Val   : %4d rows  %s -> %s" % (len(df_val),
      str(df_val.timestamp.min())[:19], str(df_val.timestamp.max())[:19]))
print("  Test  : %4d rows  %s -> %s" % (len(df_test),
      str(df_test.timestamp.min())[:19], str(df_test.timestamp.max())[:19]))

X_train = df_train[FEATURE_COLS].values
X_val   = df_val[FEATURE_COLS].values
X_test  = df_test[FEATURE_COLS].values

# =============================================================================
# STEP 5 -- Persistence baseline and XGBoost model
# =============================================================================
def metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = float(np.sum((y_true - y_true.mean())**2))
    r2 = 1.0 - float(np.sum((y_true - y_pred)**2)) / denom if denom > 0 else float("nan")
    return {"label":label, "MAE":mae, "RMSE":rmse, "R2":r2, "n":len(y_true)}

results   = {}
preds_all = {}

for horizon, steps in HORIZONS.items():
    tc = "target_%s" % horizon
    y_train = df_train[tc].values
    y_val   = df_val[tc].values
    y_test  = df_test[tc].values
    persist_val  = np.clip(df_val[r].values,  0, None)
    persist_test = np.clip(df_test[r].values, 0, None)

    if USE_XGB:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval_m = xgb.DMatrix(X_val,   label=y_val)
        dtest  = xgb.DMatrix(X_test,  label=y_test)
        params = {"objective":"reg:squarederror","max_depth":4,"learning_rate":0.05,
                  "subsample":0.8,"colsample_bytree":0.8,"min_child_weight":5,
                  "eval_metric":"rmse","verbosity":0,"seed":42}
        model = xgb.train(params, dtrain, num_boost_round=500,
                          evals=[(dtrain,"train"),(dval_m,"val")],
                          early_stopping_rounds=30, verbose_eval=False)
        ml_val_pred  = np.clip(model.predict(dval_m), 0, None)
        ml_test_pred = np.clip(model.predict(dtest),  0, None)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=6,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        ml_val_pred  = np.clip(model.predict(X_val),  0, None)
        ml_test_pred = np.clip(model.predict(X_test), 0, None)

    results[horizon] = {
        "persistence_val":  metrics(y_val,  persist_val,  "Persistence %s [val]" % horizon),
        "persistence_test": metrics(y_test, persist_test, "Persistence %s [test]" % horizon),
        "ml_val":           metrics(y_val,  ml_val_pred,  "%s %s [val]" % (MODEL_NAME, horizon)),
        "ml_test":          metrics(y_test, ml_test_pred, "%s %s [test]" % (MODEL_NAME, horizon)),
    }
    preds_all[horizon] = {
        "timestamps":    df_test["timestamp"].values,
        "y_actual":      y_test,
        "y_persistence": persist_test,
        "y_ml":          ml_test_pred,
    }
    print("\n%s:" % horizon)
    for v in results[horizon].values():
        print("  %-45s  MAE=%.4f  RMSE=%.4f  R2=%.4f" % (
              v["label"], v["MAE"], v["RMSE"], v["R2"]))

# =============================================================================
# STEP 6 -- Evaluation plot
# =============================================================================
print("\nGenerating evaluation plot...")
fig, axes = plt.subplots(3, 1, figsize=(15, 12), facecolor="white", sharex=False)
title_line1 = "MOSDAC Half-Hourly Rainfall Forecast Evaluation -- Test Set (v2: gap-corrected)"
title_line2 = "(INSAT-3DS HEM, 8-22 Sep 2026, %s vs Persistence)" % MODEL_NAME
title_line3 = "WARNING: 15-day prototype only. Results NOT generalisable to operational use."
fig.suptitle(title_line1 + "\n" + title_line2 + "\n" + title_line3,
             fontsize=11, fontweight="bold", color="#1B2A41")

for ax, (horizon, pd_) in zip(axes, preds_all.items()):
    ts = pd.DatetimeIndex(pd_["timestamps"])
    ax.set_facecolor("#F7F9FC")
    ax.grid(color="#C8D4E3", linewidth=0.5, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.plot(ts, pd_["y_actual"],      color=C_ACTUAL,  linewidth=1.2, label="Actual", zorder=3)
    ax.plot(ts, pd_["y_persistence"], color=C_PERSIST, linewidth=1.0,
            alpha=0.8, linestyle="--", label="Persistence", zorder=2)
    ax.plot(ts, pd_["y_ml"],          color=C_ML,      linewidth=1.0,
            alpha=0.8, linestyle="-.", label=MODEL_NAME, zorder=2)
    r_p = results[horizon]["persistence_test"]
    r_m = results[horizon]["ml_test"]
    improved = r_m["MAE"] < r_p["MAE"]
    verdict  = "ML < Persistence" if improved else "ML >= Persistence"
    info = ("Horizon %s  |  n=%d (gap-corrected)\n"
            "Persistence: MAE=%.3f  RMSE=%.3f  R2=%.3f\n"
            "%s:  MAE=%.3f  RMSE=%.3f  R2=%.3f\n"
            "Verdict: %s") % (horizon, r_p["n"], r_p["MAE"], r_p["RMSE"], r_p["R2"],
                               MODEL_NAME, r_m["MAE"], r_m["RMSE"], r_m["R2"], verdict)
    ax.text(0.01, 0.97, info, transform=ax.transAxes, fontsize=8, va="top", ha="left",
            color="#1B2A41", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#C8D4E3", alpha=0.92))
    ax.set_ylabel("Rainfall rate (mm/hr)", fontsize=9)
    ax.set_title("Horizon %s  (v2: gap-corrected)" % horizon,
                 fontsize=10, fontweight="bold", color="#1B2A41")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7)

fig.tight_layout(h_pad=2.5)
fig.savefig(EVAL_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  Saved -> %s" % EVAL_PNG)

# =============================================================================
# STEP 7 -- Predictions CSV
# =============================================================================
pred_rows = []
for horizon, pd_ in preds_all.items():
    for i, ts in enumerate(pd_["timestamps"]):
        pred_rows.append({"timestamp":str(ts),"horizon":horizon,
                          "actual_mmhr":pd_["y_actual"][i],
                          "persistence_mmhr":pd_["y_persistence"][i],
                          "ml_mmhr":pd_["y_ml"][i]})
pd.DataFrame(pred_rows).to_csv(PRED_CSV, index=False)
print("  Predictions -> %s" % PRED_CSV)

# =============================================================================
# Results report
# =============================================================================
rl = [
    "=" * 68,
    "MOSDAC RAINFALL FORECAST -- FINAL RESULTS REPORT (v2: gap-corrected)",
    "Model: %s" % MODEL_NAME,
    "=" * 68,
    "",
    "1. MOSDAC OBSERVATION COUNT",
    "   Total pixel observations  : %d" % len(raw),
    "   Unique timestamps         : %d" % len(ts_sorted_all),
    "   Spatial pixels/timestamp  : 153 (constant)",
    "   Archive start             : %s" % raw.timestamp.min(),
    "   Archive end               : %s" % raw.timestamp.max(),
    "",
    "2. DATE/TIME COVERAGE",
    "   Nominal: 2026-09-08 00:00 to 2026-09-22 13:30 UTC",
    "   Actual: %d of %d expected 30-min slots" % (len(ts_sorted_all), len(full_range)),
    "",
    "3. MISSING TIMESTAMPS",
    "   Count   : %d" % len(missing_ts),
    "   Pattern : 17:30 and 19:00 UTC systematically absent from 09-Sep onwards.",
    "   Action  : Rows whose feature/target window crosses a gap are excluded (not imputed).",
    "",
    "4. SPATIAL PIXELS",
    "   Per timestamp: 153 (constant)",
    "   Lat: %.3f - %.3f N" % (raw.latitude.min(), raw.latitude.max()),
    "   Lon: %.3f - %.3f E" % (raw.longitude.min(), raw.longitude.max()),
    "",
    "5. FEATURES",
    "   Count: %d" % len(FEATURE_COLS),
    "   List : %s" % str(FEATURE_COLS),
    "",
    "6. TEMPORAL-GAP VALIDATION (v2 correction)",
    "   Validation: actual timestamp differences, NOT array position.",
    "   Row valid only if ts[i]-ts[i-k]==k*30min for all lags k,",
    "   AND ts[i+s]-ts[i]==s*30min for all target steps s.",
    "   Total rows in series      : %d" % n,
    "   Valid rows (gap-free)     : %d" % n_valid,
    "   Dropped (total)           : %d" % n_dropped,
    "     out-of-bounds           : %d" % drop_oob,
    "     lag gap                 : %d" % drop_lag,
    "     target gap              : %d" % drop_tgt,
    "   v1 valid rows (used shift): %d" % n_old_valid,
    "   Interior rows leaked in v1: %d" % n_leaked,
    "",
    "7. DATASET SPLIT (chronological, no shuffling)",
    "   Total clean rows  : %d" % nf,
    "   Train (60%%)      : %d rows  %s - %s" % (len(df_train),
        str(df_train.timestamp.min())[:19], str(df_train.timestamp.max())[:19]),
    "   Validation (20%%): %d rows  %s - %s" % (len(df_val),
        str(df_val.timestamp.min())[:19], str(df_val.timestamp.max())[:19]),
    "   Test (20%%)       : %d rows  %s - %s" % (len(df_test),
        str(df_test.timestamp.min())[:19], str(df_test.timestamp.max())[:19]),
    "",
    "8. PERSISTENCE BASELINE -- TEST SET",
]
for h in HORIZONS:
    p = results[h]["persistence_test"]
    rl.append("   %s:  MAE=%.4f  RMSE=%.4f  R2=%.4f  (n=%d)" % (h,p["MAE"],p["RMSE"],p["R2"],p["n"]))

rl += ["", "9. %s -- TEST SET" % MODEL_NAME]
for h in HORIZONS:
    m = results[h]["ml_test"]
    rl.append("   %s:  MAE=%.4f  RMSE=%.4f  R2=%.4f  (n=%d)" % (h,m["MAE"],m["RMSE"],m["R2"],m["n"]))

rl += ["", "10. ML vs PERSISTENCE VERDICT (TEST SET, by MAE)"]
for h in HORIZONS:
    p = results[h]["persistence_test"]
    m = results[h]["ml_test"]
    improved = m["MAE"] < p["MAE"]
    diff_pct = (p["MAE"] - m["MAE"]) / p["MAE"] * 100 if p["MAE"] > 0 else 0.0
    verdict = ("ML IMPROVES over persistence by %.1f%% MAE reduction" % diff_pct
               if improved else
               "ML does NOT improve over persistence (MAE diff: %.1f%%)" % diff_pct)
    rl.append("   %s: %s" % (h, verdict))

rl += ["",
    "11. COMPARISON: v1 (leaked) vs v2 (gap-corrected) -- TEST SET, MAE",
    "    Horizon   Persist-v1  Persist-v2  Delta     ML-v1   ML-v2   Delta",
]
for h in HORIZONS:
    p1 = V1[h]["pm"]; p2 = results[h]["persistence_test"]["MAE"]
    m1 = V1[h]["mm"]; m2 = results[h]["ml_test"]["MAE"]
    rl.append("    %s        %6.4f      %6.4f  %+.4f    %6.4f  %6.4f  %+.4f" % (
        h, p1, p2, p2-p1, m1, m2, m2-m1))

rl += [
    "",
    "12. IMPORTANT LIMITATIONS",
    "    * PROTOTYPE: ~15 days only (669 timestamps). NOT operationally generalisable.",
    "    * High zero-rainfall fraction (94.3% of pixels).",
    "    * Rainfall RATE (mm/hr) is forecast, not accumulated rainfall.",
    "    * MOSDAC HEM is satellite-retrieved, not gauge-corrected.",
    "    * 31 missing timestamps. 289 rows that leaked into v1 now correctly excluded.",
    "    * Spatial pixels aggregated to regional mean; sub-pixel variation not modelled.",
    "    * Historical IMD daily rainfall NOT used as training data.",
    "    * This model predicts RAINFALL only. It does NOT predict floods.",
    "",
    "13. FILES CREATED",
    "    data/mosdac/mosdac_forecast_dataset.csv",
    "    outputs/mosdac_forecast_data_qc.txt",
    "    outputs/mosdac_forecast_results.txt",
    "    outputs/mosdac_forecast_predictions.csv",
    "    outputs/mosdac_forecast_evaluation.png",
    "",
    "=" * 68,
    "Generated by: scripts/forecast_mosdac_rainfall.py (v2: gap-corrected)",
    "=" * 68,
]
with open(RESULTS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(rl))
print("  Results -> %s" % RESULTS_TXT)

# QC report
qcl = [
    "=" * 68,
    "MOSDAC FORECAST DATA QC REPORT (v2: gap-corrected)",
    "Source: data/mosdac/mumbai_rainfall_30min.csv",
    "=" * 68,
    "",
    "RAW PIXEL DATA",
    "  Total rows             : %d" % len(raw),
    "  Earliest timestamp     : %s" % raw.timestamp.min(),
    "  Latest  timestamp      : %s" % raw.timestamp.max(),
    "  Unique timestamps      : %d" % len(ts_sorted_all),
    "  Expected 30-min slots  : %d" % len(full_range),
    "  Missing timestamps     : %d" % len(missing_ts),
    "  Duplicate pixel rows   : %d" % int(dupes),
    "  Pixels per timestamp   : %d (constant)" % ppt.min(),
    "",
    "MISSING TIMESTAMP LIST",
]
for mt in missing_ts:
    qcl.append("  %s" % mt)
qcl += [
    "",
    "  17:30 and 19:00 UTC systematically absent from 09-Sep-2026.",
    "  Not imputed. Rows needing these are excluded from supervised dataset.",
    "",
    "RAINFALL_MM_HR STATISTICS (pixel level)",
    "  Mean %.4f  Median %.4f  Std %.4f" % (
        raw[rain_col].mean(), raw[rain_col].median(), raw[rain_col].std()),
    "  Min  %.4f  Max   %.4f" % (raw[rain_col].min(), raw[rain_col].max()),
    "  Zero%%: %.1f%%  NaN: %d  Negative: %d" % (
        zero_frac*100, raw[rain_col].isna().sum(), (raw[rain_col]<0).sum()),
    "",
    "TEMPORAL-GAP VALIDATION SUMMARY",
    "  Total rows          : %d" % n,
    "  Valid (gap-free)    : %d" % n_valid,
    "  Dropped             : %d" % n_dropped,
    "    out-of-bounds     : %d" % drop_oob,
    "    lag gap           : %d" % drop_lag,
    "    target gap        : %d" % drop_tgt,
    "  v1 leaked rows      : %d" % n_leaked,
    "",
    "UNIT: rainfall_mm_hr is INSAT-3DS HEM rainfall RATE in mm/hr.",
    "Targets are future RATES, NOT accumulated rainfall.",
    "",
    "=" * 68,
]
with open(QC_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(qcl))
print("  QC -> %s" % QC_TXT)
print("\nDone.")
