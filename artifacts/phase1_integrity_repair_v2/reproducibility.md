# Benchmark Reproducibility Guide
**Benchmark:** Phase 1 ML Benchmark (Vishnupuri Barrage River Forecasting) — Repaired Version 2  
**Target Output Directory:** `artifacts/phase1_integrity_repair_v2/`  
**Execution Timestamp:** 2026-10-18  

---

## 1. Computational Environment

- **Operating System:** Windows 11 (build x86_64)
- **Python Version:** 3.13.5 (64-bit)
- **Key Python Packages:**
  - `scikit-learn`: `1.6.1`
  - `xgboost`: `3.4.1`
  - `joblib`: `1.4.2`
  - `numpy`: `2.2.3`
  - `pandas`: `2.2.3`
  - `scipy`: `1.15.2`
  - `pytest`: `8.2.0`

---

## 2. Deterministic Seeds & Parameters

- **Random Seed:** `42` (`SEED = 42`)
- **Block-Bootstrap Block Length:** `6` hours (`block_len = 6`)
- **Bootstrap Replicates:** `2,000` (`B = 2000`)
- **Antecedent Precipitation Index (API):** Daily decay $k=0.9$ applied hourly ($0.9^{1/24} \approx 0.995635$)
- **Temporal Splits (Asia/Kolkata IST):**
  - TRAIN: `2026-06-15 00:00:00` to `2026-08-09 23:00:00`
  - VAL: `2026-08-10 00:00:00` to `2026-08-31 23:00:00`
  - TEST: `2026-09-01 00:00:00` to `2026-09-30 23:00:00`
  - PHASE A (Cutoff): `2026-07-29 23:00:00`

---

## 3. Step-by-Step Reproduction Instructions

From the repository root (`c:\Users\aceru\floodsystem`):

### Step 1: Run the Complete Automated Test Suite
Verify that all unit and regression tests pass, including the new estimator cloning suite:
```bash
python -m pytest river_forecasting/tests/
```
Expected output: `158 passed in ~60-75s`.

### Step 2: Rebuild the Phase 1 Benchmark into the Versioned Directory
Run the pipeline to train all 12 model configurations across +1h, +2h, and +3h horizons, execute the synchronized ablation study, and save all artifacts:
```bash
python -m river_forecasting.src.phase1_ml_pipeline --no-tuning --out-dir artifacts/phase1_integrity_repair_v2
```
This generates:
- 33 `.joblib` model artifacts (`M02_1h.joblib` ... `M12_3h.joblib`)
- `model_registry.json`
- `phase1_benchmark_results.json`
- `phase1_benchmark_results.csv`
- `phase1_data_quality.json`
- `phase1_rainfall_ablation.json`

### Step 3: Independently Validate Artifacts and Generate Manifest
Run the independent validation script to reload all 33 artifacts, recreate test features, verify exact prediction parity, and build the metadata manifest:
```bash
python scratch/validate_and_analyze.py
```
This generates:
- `artifact_manifest.json`
- `benchmark_results.csv`
- `ablation_results.csv`

### Step 4: Run Paired Moving-Block Bootstrap Comparisons
Run the inferential statistics script to calculate paired differences on identical timestamps:
```bash
python scratch/run_paired_comparisons.py
```
This generates:
- `paired_comparisons.csv`

---

## 4. Input Data Checksums
- `data/historical/dataset_2026_full_season_aligned.csv`  
  SHA256: `ea58df0cff7ee54bfa0d941ab5e2dfe4cc8b848fa6836c6cecb414d7162bd213`
- `data/historical/insat3ds_hem_catchment_hourly_2026.csv`  
  SHA256: `36b12a2db0a0fbbfdc9eeb7004626061487f879c0d30a6d790577e8a779ca784`
