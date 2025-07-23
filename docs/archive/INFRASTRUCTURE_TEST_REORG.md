# Infrastructure Test Reorganization Plan

## Issues Identified

### 1. Factories Folder
- **Current**: `tests/factories/health_data_factory.py`
- **Issue**: This is a test utility, not a test file
- **Solution**: Move to `tests/fixtures/` or `tests/utils/`

### 2. Unit/Infrastructure Root Level Tests
19 test files at root level that should be in subdirectories:

## Reorganization Plan

### Move Parsers (5 files) → `parsers/`
```
test_activity_parser.py
test_heart_rate_parser.py
test_sleep_parser.py
test_json_parsers.py
test_streaming_adapter.py
```

### Move Repositories (8 files) → `repositories/`
```
test_baseline_repository_factory.py
test_baseline_repository_hr_hrv.py
test_file_activity_repository.py
test_file_baseline_repository.py
test_file_heart_rate_repository.py
test_file_sleep_repository.py
test_timescale_baseline_repository.py
test_repository_dependency_injection.py
```

### Move ML Models (4 files) → `ml_models/`
```
test_booster_wrapper.py
test_pat_equivalence.py
test_pat_model.py
test_xgboost_models.py
```

### Create DI Folder (2 files) → `di/`
```
test_di_container_baseline_repository.py
test_di_container_phase2.py
```

## Benefits

1. **Clear organization**: Each subdirectory contains related tests
2. **No orphaned tests**: All infrastructure tests properly categorized
3. **Easier navigation**: Developers can find tests by component type
4. **Better scalability**: Clear where to add new infrastructure tests

## Implementation Steps

1. Create missing directories (`di/`, `fixtures/`)
2. Move test files to appropriate subdirectories
3. Update any imports that reference moved files
4. Verify all tests still pass