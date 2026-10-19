"""
river_forecasting/src/run_event1_pipeline.py
============================================
Master Execution Script for the Complete Event 1 Pipeline.

Sequentially executes:
1. Phase 3: Catchment Rainfall Extraction & Alignment Rebuild
2. Phase 4: Comprehensive Lag Correlation Reassessment (0-72h)
3. Phase 5: Leakage-Free Model Evaluation & Benchmark Comparison
4. Test Suite: Verification of all 20 river_forecasting unit tests
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from river_forecasting.src.phase3_rebuild_rainfall_alignment import run_phase3
from river_forecasting.src.phase4_lag_correlation_analysis import run_phase4_lag_analysis
from river_forecasting.src.phase5_model_evaluation import run_phase5_evaluation


def run_pipeline():
    start_time = datetime.now(timezone.utc)
    print("=" * 80)
    print("MASTER PIPELINE: EVENT 1 RAINFALL & RIVER FORECASTING")
    print(f"Started UTC: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    # Step 1: Run Phase 3
    print("\n>>> [STEP 1/4] EXECUTING PHASE 3: REBUILD RAINFALL & ALIGNMENT...")
    run_phase3()

    # Step 2: Run Phase 4
    print("\n>>> [STEP 2/4] EXECUTING PHASE 4: LAG CORRELATION ANALYSIS (0-72h)...")
    run_phase4_lag_analysis()

    # Step 3: Run Phase 5
    print("\n>>> [STEP 3/4] EXECUTING PHASE 5: HONEST LEAKAGE-FREE MODEL EVALUATION...")
    run_phase5_evaluation()

    # Step 4: Run Unit Tests
    print("\n>>> [STEP 4/4] EXECUTING UNIT TEST SUITE...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(REPO_ROOT / "river_forecasting" / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Elapsed Time: {duration:.2f} seconds")
    print(f"Unit Tests Run: {test_result.testsRun}, Errors: {len(test_result.errors)}, Failures: {len(test_result.failures)}")
    print("=" * 80)

    if not test_result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
