# Simple Fix Plan for XGBoost Feature Bug
Generated: 2025-07-23

## The Simplest Fix

Instead of creating a whole new flow, we can fix this by:

1. Keep using `ClinicalFeatureExtractor` for clinical insights
2. When making XGBoost predictions, use `AggregationPipeline` to get Seoul features
3. This requires minimal changes to existing code

## Implementation Steps

### Step 1: Add AggregationPipeline to MoodPredictionPipeline
```python
# In __init__
self.aggregation_pipeline = AggregationPipeline()
```

### Step 2: Update prediction logic
```python
if self.config.use_seoul_features and not self.ensemble_orchestrator:
    # Get Seoul features for XGBoost
    seoul_features = self.aggregation_pipeline.aggregate_seoul_features(...)
    if seoul_features:
        feature_dict = seoul_features[-1].to_xgboost_dict()
        # Use XGBoost predictor with proper features
else:
    # Current flow (will fail)
    feature_vector = feature_set.seoul_features.to_xgboost_features()
```

### Step 3: Update XGBoost predictor to accept dict
The XGBoost predictor already has `dict_to_array()` method, so we can pass the feature dict directly.

## Benefits
- Minimal code changes
- Preserves existing architecture
- Easy to toggle with config flag
- Low risk of breaking other features