# -*- coding: utf-8 -*-
"""
gap_audit.py -- diagnoses temporal-gap leakage in the existing forecast dataset.
"""
import pandas as pd
import numpy as np

raw = pd.read_csv('data/mosdac/mumbai_rainfall_30min.csv')
raw['timestamp'] = pd.to_datetime(raw['timestamp'])
ts_series = (raw.groupby('timestamp')['rainfall_mm_hr']
             .mean()
             .rename('rainfall_mumbai_mean_mmhr')
             .sort_index()
             .reset_index())
ts_series.columns = ['timestamp', 'rainfall_mumbai_mean_mmhr']

ts_arr = ts_series['timestamp'].values   # numpy datetime64
n = len(ts_arr)
STEP = np.timedelta64(30, 'm')

# Parameters mirroring the existing forecast script
LAGS         = [1, 2, 3, 4, 6, 8, 12]
TARGET_STEPS = [2, 4, 6]          # +1h, +2h, +3h in 30-min steps
# Rolling windows: roll_mean/max use shift(1).rolling(w) -- they look at
# positions [i-w .. i-1], so the farthest lookback is w=12 steps.
ROLL_MAX_LOOKBACK = 12            # largest rolling window used

# A row at index i is valid IFF:
#   For each lag k in LAGS:          ts[i] - ts[i-k] == k * 30min  exactly
#   For rolling up to ROLL_MAX_LOOKBACK: same as lag 12 (already covered)
#   For each target step s in TARGET_STEPS: ts[i+s] - ts[i] == s * 30min exactly

valid_mask   = np.zeros(n, dtype=bool)
drop_because_lag    = 0
drop_because_target = 0
drop_out_of_bounds  = 0

drop_log = []

for i in range(n):
    ok = True
    reason = ""

    # check lags
    for k in LAGS:
        if i - k < 0:
            ok = False
            reason = "out-of-bounds lag_%d" % k
            drop_out_of_bounds += 1
            break
        delta = int((ts_arr[i] - ts_arr[i - k]) / np.timedelta64(1, 'm'))
        expected = k * 30
        if delta != expected:
            ok = False
            reason = "lag_%d: expected %dmin got %dmin" % (k, expected, delta)
            drop_because_lag += 1
            break

    if ok:
        # check targets
        for s in TARGET_STEPS:
            if i + s >= n:
                ok = False
                reason = "out-of-bounds target+%d" % s
                drop_out_of_bounds += 1
                break
            delta = int((ts_arr[i + s] - ts_arr[i]) / np.timedelta64(1, 'm'))
            expected = s * 30
            if delta != expected:
                ok = False
                reason = "target+%d: expected %dmin got %dmin" % (s, expected, delta)
                drop_because_target += 1
                break

    valid_mask[i] = ok
    if not ok:
        drop_log.append((i, str(ts_arr[i])[:19], reason))

n_valid = int(valid_mask.sum())
n_dropped = n - n_valid

print("=" * 60)
print("TEMPORAL-GAP LEAKAGE AUDIT")
print("=" * 60)
print("Total timesteps in regional series : %d" % n)
print("Strictly valid (gap-free) rows     : %d" % n_valid)
print("Total rows to drop                 : %d" % n_dropped)
print("  - out-of-bounds (head/tail)      : %d" % drop_out_of_bounds)
print("  - dropped because lag gap        : %d" % drop_because_lag)
print("  - dropped because target gap     : %d" % drop_because_target)
print()

# Compare with old script: old script dropped via dropna(features+targets)
# dropna catches NaN from shift() only at head (first 12) and tail (last 6)
# = 18 rows max.  Any interior gap row slipped through.
n_old_dropped = 669 - 651   # = 18 from the previous run
n_interior_leakers = n_dropped - n_old_dropped
print("Old script rows dropped (via dropna) : ~%d" % n_old_dropped)
print("Interior gap rows that leaked in     : %d" % n_interior_leakers)
print()
print("First 20 dropped rows:")
for i, ts, reason in drop_log[:20]:
    print("  row %3d  %s  %s" % (i, ts, reason))
if len(drop_log) > 20:
    print("  ... (%d more)" % (len(drop_log) - 20))
print()
print("Last 5 dropped rows:")
for i, ts, reason in drop_log[-5:]:
    print("  row %3d  %s  %s" % (i, ts, reason))
