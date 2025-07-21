# Test Suite Analysis Report

## Current Issues

### 1. Organizational Problems

#### A. Misplaced Tests
- **Integration tests in unit/**: 
  - `test_pipeline_ensemble_integration.py` in unit/application/
  - `test_baseline_repository_integration.py` in unit/domain/
  - `test_repository_factory_integration.py` in unit/infrastructure/
  
#### B. Duplicate/Similar Tests
- Multiple sleep aggregator tests split across files:
  - `test_sleep_aggregator.py`
  - `test_sleep_aggregator_apple_3pm.py`
  - `test_sleep_aggregator_midnight.py`
  - `test_sleep_aggregator_regression.py`
  
- Multiple clinical interpreter variations:
  - `test_clinical_interpreter.py`
  - `test_clinical_interpreter_refactored.py`
  - `test_clinical_interpreter_with_config.py`
  - `test_clinical_interpreter_migration.py`

#### C. Orphaned Tests at Root Level
- `test_date_range_filtering.py` (should be in unit/application/)
- `test_progress_indication.py` (should be in unit/application/)
- `test_no_orphaned_code.py` (should be in unit/quality/)

### 2. Skipped/XFailed Tests Summary

#### A. Model-Related Skips (9 occurrences)
- PAT model tests skip when weights not available
- XGBoost tests skip when models not found
- **Action**: These are appropriate - models may not be available in all environments

#### B. Database-Related Skips (8 occurrences)
- TimescaleDB tests skip when container not available
- HR/HRV tests skip when database not available
- **Action**: These are appropriate for CI environments

#### C. Privacy-Related Skips (3 occurrences)
- User ID hashing tests we just added
- **Action**: Need redesign to test functionality without comparing hashed IDs

#### D. Data-Related Skips (5 occurrences)
- Real data export tests skip when export.xml not found
- **Action**: These are appropriate - require real user data

#### E. XFailed Tests (5 tests)
- `test_baseline_persistence_improves_predictions`
- `test_baseline_persistence_after_pipeline_restart`
- `test_memory_bounds` 
- `test_ensemble_without_pat`
- `test_pipeline_with_pat_available`
- **Action**: Review if these should be fixed or remain as known limitations

## Proposed Reorganization

### Phase 1: Move Misplaced Tests
```bash
# Move integration tests out of unit/
mv tests/unit/application/test_pipeline_ensemble_integration.py tests/integration/pipeline/
mv tests/unit/domain/test_baseline_repository_integration.py tests/integration/storage/
mv tests/unit/infrastructure/test_repository_factory_integration.py tests/integration/storage/

# Move orphaned tests
mkdir -p tests/unit/quality
mv tests/unit/test_no_orphaned_code.py tests/unit/quality/
mv tests/unit/test_date_range_filtering.py tests/unit/application/
mv tests/unit/test_progress_indication.py tests/unit/application/
```

### Phase 2: Consolidate Similar Tests
1. Merge sleep aggregator tests into one comprehensive test file
2. Consolidate clinical interpreter tests (keep latest refactored version)
3. Group aggregation pipeline tests by concern

### Phase 3: Fix Import Structure
- Update all imports after moving files
- Run tests to ensure nothing breaks

### Phase 4: Document Test Structure
Create `tests/README.md` with:
- Test organization guidelines
- When to skip vs xfail
- How to add new tests
- Test naming conventions

## Skip/XFail Recommendations

### Keep as Skip:
- Model availability checks (PAT, XGBoost)
- Database availability checks (TimescaleDB, PostgreSQL)
- Real data file checks (export.xml)

### Fix Soon:
- User ID hashing tests (redesign to test functionality)
- Memory bounds test (set realistic limits)

### Investigate XFails:
- Baseline persistence improvements (may be flaky)
- Ensemble predictions (check if PAT integration works)

## Next Steps Priority

1. **High**: Reorganize test structure (Phase 1-3)
2. **Medium**: Fix user ID hashing tests
3. **Medium**: Document test conventions
4. **Low**: Investigate xfailed tests
5. **Low**: Consolidate duplicate tests