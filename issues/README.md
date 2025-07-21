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
   - Marker: `Issue #TBD-1`

2. **baseline-persistence-legacy-api.md** - Outdated domain entity APIs in tests
   - Affects: Personal baseline calibration tests
   - Tests: `tests/integration/test_baseline_persistence_pipeline.py`
   - Marker: `Issue #TBD-2`

3. **xgboost-booster-predict-proba.md** - Missing predict_proba in JSON-loaded models
   - Affects: Ensemble predictions with JSON models
   - Test: `tests/integration/pipeline/test_full_pipeline.py::test_pipeline_with_ensemble`
   - Marker: `Issue #TBD-3`

## Updating Issue Numbers

After creating the issues, update the test markers:
1. Note the issue numbers created by the script
2. Replace `#TBD-1`, `#TBD-2`, `#TBD-3` with actual issue numbers
3. Commit the updated markers

## @CLAUDE Integration

All these issues were identified and documented by @CLAUDE during the integration work.