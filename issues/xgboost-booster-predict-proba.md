## XGBoost Booster objects loaded from JSON lack predict_proba method

### Description
When loading XGBoost models from JSON format (to avoid pickle warnings), the resulting Booster objects don't have the scikit-learn compatible `predict_proba` method that the ensemble orchestrator expects.

### Current Behavior
- Models load successfully from JSON files
- Ensemble predictions fail with: `'Booster' object has no attribute 'predict_proba'`
- Test `test_pipeline_with_ensemble` is marked as `xfail`

### Expected Behavior
- XGBoost models should provide probability predictions regardless of load format
- Ensemble orchestrator should work with both pickle and JSON model formats

### Root Cause
The code loads models as native XGBoost Booster objects from JSON:
```python
model = xgb.Booster()
model.load_model(str(model_path))  # Returns Booster, not XGBClassifier
```

But then tries to use scikit-learn API:
```python
depression_proba = self.models["depression"].predict_proba(features_2d)[0]
```

### Impact
- Ensemble predictions don't work when models are loaded from JSON
- Forces use of pickle format which triggers deprecation warnings
- Blocks migration to more portable JSON model format

### Proposed Solutions

#### Option 1: Wrap Booster in XGBClassifier
```python
booster = xgb.Booster()
booster.load_model(str(model_path))
# Wrap in classifier
model = xgb.XGBClassifier()
model._Booster = booster
model.n_classes_ = 2  # Set for binary classification
```

#### Option 2: Use Booster.predict() with custom probability logic
```python
# Use native Booster API
raw_predictions = booster.predict(dmatrix)
# Apply sigmoid for probabilities
probabilities = 1 / (1 + np.exp(-raw_predictions))
```

#### Option 3: Save models in sklearn format
Retrain and save models using XGBClassifier.save_model() which preserves sklearn interface.

### Test Case
```python
# Should work after fix:
pipeline = MoodPredictionPipeline(config=PipelineConfig(include_pat_sequences=True))
result = pipeline.process_health_data(...)
assert "xgboost" in result.daily_predictions[date]["models_used"]
```

### Labels
- bug
- tech-debt
- ml-models
- architecture
- priority: high

### References
- Test: `tests/integration/pipeline/test_full_pipeline.py::test_pipeline_with_ensemble`
- @CLAUDE identified during ensemble testing
- Blocks JSON model migration