# Domain Test Reorganization Plan

## Current State
- 44 test files at `tests/unit/domain/` root
- Only 1 subdirectory: `services/` (appears empty)
- Hard to navigate and find specific tests

## Target Structure (Mirror Source)
```
tests/unit/domain/
├── entities/
├── repositories/
├── services/
├── utils/
└── value_objects/
```

## File Mapping

### → entities/ (5 files)
```
test_activity_record.py
test_heart_rate_record.py
test_sleep_record.py
test_sleep_math_debug.py  # Related to sleep entity math
test_user_baseline_hr_hrv.py  # Related to HR entity baseline
```

### → repositories/ (1 file)
```
test_baseline_repository_interface.py
```

### → services/ (35 files)
```
test_activity_aggregator.py
test_activity_feature_calculator.py
test_activity_sequence_extractor.py
test_advanced_feature_engineering.py
test_advanced_feature_engineering_with_persistence.py
test_biomarker_interpreter.py
test_circadian_feature_calculator.py
test_circadian_rhythm_analyzer.py
test_clinical_feature_extractor.py
test_clinical_feature_extractor_with_calibration.py
test_clinical_interpreter.py
test_clinical_interpreter_migration.py
test_clinical_interpreter_refactored.py
test_clinical_interpreter_with_config.py
test_clinical_thresholds.py
test_dlmo_calculator.py
test_dsm5_criteria_evaluator.py
test_early_warning_detector.py
test_episode_interpreter.py
test_feature_engineering_orchestrator.py
test_feature_extraction_service.py
test_feature_orchestrator_interface.py
test_heart_rate_aggregator.py
test_interpolation_strategies.py
test_mood_predictor.py
test_pat_sequence_builder.py
test_risk_level_assessor.py
test_sleep_aggregator.py
test_sleep_feature_calculator.py
test_sleep_window_analyzer.py
test_sparse_data_handler.py
test_temporal_feature_calculator.py
test_treatment_recommender.py
```

### → utils/ (2 files)
```
test_episode_mapper.py
```

### → value_objects/ (1 file)
```
test_time_period.py
```

## Benefits
1. **1:1 mapping** with source files
2. **Easy navigation** - max 35 files per directory (services)
3. **Logical grouping** - entities, services, repositories clearly separated
4. **Future-proof** - clear where to add new tests

## Notes
- Multiple clinical interpreter test variants suggest potential consolidation opportunity
- Services directory has the most tests (35) but matches source structure