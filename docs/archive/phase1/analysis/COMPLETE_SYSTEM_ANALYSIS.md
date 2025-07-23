# Complete System Analysis - Big Mood Detector
Generated: 2025-07-23

## System Overview

Big Mood Detector uses two complementary approaches for mood prediction:
1. **XGBoost**: Clinical features based on Seoul paper (30-day statistics)
2. **PAT**: Transformer-based patterns from raw activity (7-day sequences)

## The Core Issue

### What's Broken
- **XGBoost predictions fail** with "missing fields" error
- Features generated don't match what models expect
- Using `ClinicalFeatureExtractor` instead of `AggregationPipeline`

### What's Working
- Data ingestion (XML/JSON parsing) ✓
- Domain entities creation ✓
- PAT pipeline and predictions ✓
- Personal calibration baselines ✓
- API and CLI interfaces ✓

## Feature Generation Paths

### Path 1: Seoul/XGBoost Features (BROKEN)
```
Current (Wrong):
HealthData → ClinicalFeatureExtractor → SeoulXGBoostFeatures → [36 single-day features] → FAIL

Should Be:
HealthData → AggregationPipeline → DailyFeatures → [36 statistical features] → SUCCESS
```

### Path 2: PAT Features (WORKING)
```
ActivityRecords → ActivitySequenceExtractor → PATSequenceBuilder → PATModel → [96 embeddings] → SUCCESS
```

## The Two Feature Systems Explained

### System 1: Seoul Statistical Features (for XGBoost)
**Purpose**: Capture mood-predictive patterns over 30 days

**12 Base Metrics**:
1. `sleep_percentage` - Daily sleep ÷ 1440 minutes
2. `sleep_amplitude` - Sleep depth variation
3. `long_num` - Count of long sleep windows (≥3.75h)
4. `long_len` - Total hours of long windows
5. `long_ST` - Sleep time in long windows
6. `long_WT` - Wake time in long windows
7. `short_num` - Count of short windows (<3.75h)
8. `short_len` - Total hours of short windows
9. `short_ST` - Sleep time in short windows
10. `short_WT` - Wake time in short windows
11. `circadian_amplitude` - Rhythm strength
12. `circadian_phase` - DLMO timing

**Statistics** (×3 for each metric):
- `_MN`: Mean over 30 days
- `_SD`: Standard deviation
- `_Z`: Z-score of current day

**Total**: 12 × 3 = 36 features

### System 2: Clinical Interpretation Features
**Purpose**: Single-day clinical assessment (not for XGBoost)

**36 Direct Measurements**:
- Sleep duration, efficiency, timing
- Activity levels, fragmentation
- Heart rate patterns
- Circadian markers
- Clinical flags (insomnia, hypersomnia)

**Use Cases**:
- Clinical reports
- Feature visualization
- Threshold detection
- NOT for mood prediction models

## Why The Mismatch Happened

1. **Refactoring**: System evolved to support richer features
2. **Dual purpose**: Wanted both statistical and clinical features
3. **Pipeline modification**: Prediction started using clinical extractor
4. **Model expectations**: XGBoost still expects original Seoul features

## The Solution

### Use Existing AggregationPipeline
```python
# In process_health_data_use_case.py
if predicting:
    # For XGBoost predictions
    features = self.aggregation_pipeline.aggregate_daily_features(...)
    feature_dict = features.to_dict()  # Correct Seoul names
    prediction = xgboost.predict(feature_dict)
else:
    # For clinical insights
    clinical = self.clinical_extractor.extract_clinical_features(...)
    # Use for reports, API, etc.
```

### Why This Works
1. `AggregationPipeline` already generates correct features
2. `DailyFeatures.to_dict()` has proper name mapping
3. Minimal code changes required
4. Preserves all existing functionality

## Architecture Strengths

1. **Clean separation**: Models use appropriate features
2. **Modularity**: Each pipeline is independent
3. **Flexibility**: Can enhance without breaking
4. **Testability**: Each component testable in isolation

## Testing Strategy

### Phase 1: Unit Tests
```python
def test_aggregation_generates_seoul_features():
    # Verify 36 features with correct names
    
def test_daily_features_name_mapping():
    # Verify to_dict() produces XGBoost names
    
def test_xgboost_accepts_aggregation_features():
    # Verify prediction works
```

### Phase 2: Integration Tests
```python
def test_full_prediction_pipeline():
    # XML → Aggregation → XGBoost → Prediction
    
def test_ensemble_prediction():
    # Both XGBoost and PAT working together
```

### Phase 3: Real Data Validation
```bash
# Should work after fix
python src/big_mood_detector/main.py predict data/health_auto_export/ --report
```

## Implementation Priority

1. **Immediate**: Fix XGBoost pipeline (use AggregationPipeline)
2. **Next**: Add tests to prevent regression
3. **Future**: Document both feature systems clearly
4. **Optional**: Add feature system selector in config

## Expected Outcomes

After fix:
- XGBoost predictions return valid risk scores
- No more "missing fields" errors  
- Ensemble predictions improve (both models working)
- Clinical features still available for insights
- All tests pass with real data

## Key Takeaway

We have two valid feature systems serving different purposes:
1. **Statistical features** (AggregationPipeline) → For ML models
2. **Clinical features** (ClinicalFeatureExtractor) → For human interpretation

The fix is simply routing the right features to the right consumers.