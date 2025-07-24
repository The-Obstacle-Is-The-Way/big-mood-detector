# Feature Fix Action Plan - TDD Approach
Generated: 2025-07-23

## Problem Summary

The XGBoost models fail because they receive single-day clinical features instead of 30-day statistical summaries. The models were trained on Seoul paper features (12 base metrics × 3 statistics = 36 features), but we're passing different features from our ClinicalFeatureExtractor.

## Solution: Use the Existing AggregationPipeline

The good news: **The correct implementation already exists!** The `AggregationPipeline` with `DailyFeatures` generates exactly what XGBoost needs.

## TDD Implementation Plan

### Phase 1: Create Failing Tests
1. **Test Seoul Feature Generation**
   ```python
   def test_aggregation_pipeline_generates_36_seoul_features():
       # Given: 30 days of health data
       # When: Process through AggregationPipeline
       # Then: Should generate all 36 features with correct names
   ```

2. **Test XGBoost Prediction Integration**
   ```python
   def test_xgboost_prediction_with_aggregation_features():
       # Given: DailyFeatures from AggregationPipeline
       # When: Pass to XGBoostModels.predict()
       # Then: Should return valid predictions without errors
   ```

3. **Test Feature Name Mapping**
   ```python
   def test_daily_features_to_dict_maps_correctly():
       # Given: DailyFeatures object
       # When: Call to_dict()
       # Then: Should have all XGBoost expected names (MN/SD/Z suffixes)
   ```

### Phase 2: Fix the Pipeline

The fix is straightforward - update the prediction use case to use `AggregationPipeline` for XGBoost features:

1. **Current (broken) flow**:
   ```
   ClinicalFeatureExtractor → SeoulXGBoostFeatures → XGBoost (FAILS)
   ```

2. **Fixed flow**:
   ```
   AggregationPipeline → DailyFeatures → to_dict() → XGBoost (SUCCESS)
   ```

### Phase 3: Implementation Steps

1. **Update `process_health_data_use_case.py`**:
   - For XGBoost predictions: Use `AggregationPipeline`
   - For clinical insights: Continue using `ClinicalFeatureExtractor`

2. **Create a dual-pipeline approach**:
   ```python
   # For predictions
   aggregation_features = self.aggregation_pipeline.aggregate_daily_features(...)
   xgboost_dict = aggregation_features.to_dict()
   prediction = self.xgboost_models.predict(xgboost_dict)
   
   # For clinical insights (optional)
   clinical_features = self.clinical_extractor.extract_clinical_features(...)
   ```

3. **Ensure proper 30-day window**:
   - XGBoost needs statistics over 30 days
   - Make sure we have enough historical data

### Phase 4: Validation Tests

1. **End-to-end test with real data**:
   ```bash
   python src/big_mood_detector/main.py predict data/input/health_auto_export/ --report
   ```

2. **Verify all features are present**:
   - Check for all 36 features in the correct order
   - Verify feature values are reasonable

3. **Test ensemble mode**:
   - XGBoost should work
   - PAT should work (if available)
   - Ensemble should combine both

## Key Code Locations

1. **AggregationPipeline**: `src/big_mood_detector/application/services/aggregation_pipeline.py`
   - Already generates correct features
   - `DailyFeatures.to_dict()` has proper mapping

2. **XGBoostModels**: `src/big_mood_detector/infrastructure/ml_models/xgboost_models.py`
   - Expects features from `FEATURE_NAMES` list
   - Can accept dict or array

3. **Use Case**: `src/big_mood_detector/application/use_cases/process_health_data_use_case.py`
   - Need to update prediction logic here

## Benefits of This Approach

1. **Minimal changes**: Reuse existing, tested code
2. **Low risk**: AggregationPipeline is already validated
3. **Maintains architecture**: Clean separation of concerns
4. **Future flexibility**: Can still use ClinicalFeatureExtractor for other purposes

## Success Criteria

1. XGBoost predictions work without "missing fields" errors
2. All 36 features are correctly generated and named
3. Predictions produce reasonable risk scores (not all 50%)
4. Tests pass with real Apple Health data
5. Personal calibration continues to work

## Next Immediate Step

Create the failing test for Seoul feature generation to confirm our understanding is correct.