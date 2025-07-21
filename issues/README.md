# Tech Debt Issues

This directory contains GitHub issue templates for tracked technical debt in the Big Mood Detector project.

## Creating Issues

Run the script to create all issues at once:
```bash
./scripts/create-tech-debt-issues.sh
```

## Current Issues

1. **streaming-parser-date-bug.md** - Streaming parser date filtering comparison bug
   - Affects: Large file processing with date filters
   - Test: `tests/integration/test_memory_bounds.py`
   - Marker: `Issue #38`

2. **baseline-persistence-legacy-api.md** - Outdated domain entity APIs in tests
   - Affects: Personal baseline calibration tests
   - Tests: `tests/integration/test_baseline_persistence_pipeline.py`
   - Marker: `Issue #39`

3. **xgboost-booster-predict-proba.md** - Missing predict_proba in JSON-loaded models
   - Affects: Ensemble predictions with JSON models
   - Test: `tests/integration/pipeline/test_full_pipeline.py::test_pipeline_with_ensemble`
   - Marker: `Issue #40`

## Updating Issue Numbers

✅ Issues created and markers updated:
- Issue #38: Streaming parser date filtering bug
- Issue #39: Baseline persistence legacy API
- Issue #40: XGBoost predict_proba missing

## @CLAUDE Integration

All these issues were identified and documented by @CLAUDE during the integration work.