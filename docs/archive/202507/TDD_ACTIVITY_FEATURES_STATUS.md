# Test-Driven Development Status: Activity Features in API

## Phase 1: Red (Tests Written) ✅

### Tests Created:

1. **Unit Tests** (`tests/unit/application/test_clinical_feature_extraction.py`)
   - ✅ `test_extract_activity_features` - Verifies activity features are extracted
   - ✅ `test_activity_features_calculation` - Tests correct calculation of daily steps
   - ✅ `test_missing_activity_data_handling` - Tests sensible defaults when no activity
   - ✅ `test_activity_features_in_xgboost_vector` - Ensures features in correct positions
   - ✅ `test_circadian_features_with_activity` - Tests PAT and circadian calculations

2. **Integration Tests** (`tests/integration/api/test_features_endpoint_activity.py`)
   - ✅ `test_extract_features_includes_activity` - API returns activity features
   - ✅ `test_extract_features_activity_from_xml` - XML parsing includes activity
   - ✅ `test_feature_schema_validation` - Schema has all expected fields
   - ✅ `test_missing_activity_data_defaults` - API handles missing data gracefully
   - ✅ `test_backwards_compatibility` - Original features unchanged

3. **E2E Tests** (`tests/integration/test_ensemble_pipeline_activity.py`)
   - ✅ `test_direct_ensemble_with_activity` - Direct orchestrator call with activity
   - ✅ `test_pipeline_process_with_activity` - Full pipeline includes activity
   - ✅ `test_api_vs_direct_consistency` - API and direct calls match
   - ✅ `test_activity_improves_prediction_confidence` - Activity affects predictions

### Current Test Status:
- **Total Tests**: 14
- **Status**: All skipped with `pytest.skip(reason="awaiting implementation")`
- **Ready for**: Green phase implementation

## Phase 2: Green (Implementation Needed) 🔴

### Required Changes:

1. **Extend API Response Schema**
   - [ ] Add activity fields to `FeatureExtractionResponse`
   - [ ] Fields needed:
     - `daily_steps`
     - `activity_variance`
     - `sedentary_hours`
     - `activity_fragmentation`
     - `sedentary_bout_mean`
     - `activity_intensity_ratio`

2. **Update Feature Extraction**
   - [ ] Option A: Extend `DailyFeatures` to include activity fields
   - [ ] Option B: Replace `DailyFeatures` with `ClinicalFeatureSet`
   - [ ] Update `routes/features.py` to return full feature set

3. **Update Feature Mapping**
   - [ ] Ensure proper mapping between API response and XGBoost features
   - [ ] Maintain backwards compatibility

## Next Steps:

1. Remove `pytest.skip` decorators one by one as features are implemented
2. Run tests to verify they fail for the right reason
3. Implement minimal code to make each test pass
4. Refactor for cleanliness once all tests pass

## Success Metrics:

- [ ] All 14 tests passing
- [ ] API returns 42+ features (36 Seoul + 6 activity)
- [ ] No performance regression (P99 < 250ms)
- [ ] Backwards compatible with existing clients