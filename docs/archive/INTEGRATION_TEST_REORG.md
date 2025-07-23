# Integration Test Reorganization Plan

## Current Root-Level Integration Tests Analysis

### Should Move:
1. `test_api_integration.py` → **api/**
   - Tests API endpoints, belongs with other API tests

2. `test_baseline_persistence_e2e.py` → **storage/**
   - Tests baseline repository persistence

3. `test_baseline_persistence_pipeline.py` → **pipeline/**
   - We just moved this here, but it's about pipeline + baselines

4. `test_ensemble_pipeline_activity.py` → **pipeline/**
   - Tests ensemble pipeline with activity data

5. `test_ensemble_predictions.py` → **ml/**
   - Tests machine learning ensemble functionality

### Should Stay at Root:
1. `test_di_smoke.py`
   - Cross-cutting dependency injection tests

2. `test_memory_bounds.py`
   - System-wide memory constraints

3. `test_openapi_contract.py`
   - API contract validation (cross-cutting)

4. `test_hr_hrv_e2e.py`
   - Full end-to-end test across multiple components

5. `test_progress_indication_integration.py`
   - Progress tracking across multiple components

6. `test_real_data_sleep_math.py`
   - Real data validation (cross-cutting)

## Rationale

Keep at root when test:
- Spans multiple architectural layers
- Tests system-wide constraints
- Validates contracts between components
- Is truly end-to-end

Move to subdirectory when test:
- Focuses on one architectural area
- Has clear ownership by a component
- Would benefit from grouping with similar tests