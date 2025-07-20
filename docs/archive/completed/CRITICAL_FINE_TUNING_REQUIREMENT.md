# CRITICAL: Fine-Tuning Requirement for Accurate Predictions

**Date**: July 16, 2025  
**Priority**: CRITICAL  
**Status**: Not Yet Implemented

## The Problem

Our current implementation processes raw Apple Health data through pre-trained XGBoost and PAT models to generate mood predictions. However, **this approach is fundamentally flawed** for real-world use:

1. **The models require individual baseline calibration** - Every person has unique circadian rhythms, sleep patterns, and activity levels
2. **Episode labels are essential** - Without knowing when actual mood episodes occurred, the models cannot learn individual patterns
3. **Pre-trained models give poor results** - The paper's impressive AUC scores (0.80-0.98) are AFTER fine-tuning, not out-of-the-box

## Evidence from Literature

From the PAT paper analysis:
- Pre-trained PAT only achieves AUC ~0.61 for depression
- XGBoost models need 30-60 labeled episode days to reach AUC 0.90+
- "Population models" without personalization perform poorly

## Current Implementation Status

### ❌ What We DON'T Have:
1. **Episode labeling interface** - No way for users to input when they had mood episodes
2. **Baseline calculation** - No storage/calculation of individual baseline metrics
3. **Fine-tuning pipeline** - No code to retrain models on individual data
4. **Model persistence** - No way to save/load personalized model weights

### ✅ What We DO Have:
1. Data processing pipeline that extracts features
2. Pre-trained models that can make (poor) predictions
3. Infrastructure for background processing
4. API and CLI interfaces

## Proposed Solution Architecture

### Phase 1: Data Collection
```python
# User provides:
# 1. Apple Health export
# 2. Episode dates CSV or manual input
episodes = [
    {"start": "2024-01-15", "end": "2024-01-25", "type": "depressive", "severity": 7},
    {"start": "2024-03-10", "end": "2024-03-20", "type": "hypomanic", "severity": 5},
]
```

### Phase 2: Baseline Calculation
```python
# Calculate personal baseline during stable periods
baseline = {
    "sleep_duration_mean": 7.5,
    "sleep_duration_std": 0.8,
    "activity_level_mean": 250.5,
    "hrv_mean": 45.2,
    # ... all 36 features
}
```

### Phase 3: Fine-Tuning
```python
# Quick local fine-tuning (< 1 minute)
personalized_xgb = fine_tune_xgboost(
    base_model=pretrained_xgb,
    features=user_features,
    labels=episode_labels,
    epochs=10
)

personalized_pat = fine_tune_pat(
    base_model=pretrained_pat,
    activity_data=user_activity,
    labels=episode_labels,
    epochs=5
)
```

### Phase 4: Personalized Predictions
```python
# Use personalized models for all future predictions
risk_scores = ensemble_predict(
    personalized_xgb,
    personalized_pat,
    new_data
)
```

## Implementation Priority

### High Priority (Required for Accuracy):
1. **Episode input interface** - CLI command or API endpoint to accept episode dates
2. **Baseline calculator** - Service to compute individual baseline statistics
3. **Fine-tuning wrapper** - Simple interface to retrain models
4. **Model storage** - Save/load personalized weights (SQLite or files)

### Medium Priority (Better UX):
1. **Automatic episode detection** - Infer episodes from hospitalization records
2. **Progressive learning** - Update models as new labels arrive
3. **Confidence scoring** - Lower confidence until enough labels collected

### Low Priority (Nice to Have):
1. **Transfer learning** - Use similar users' patterns as starting point
2. **Federated learning** - Privacy-preserving collaborative training
3. **Multi-user support** - Handle multiple profiles

## Temporary Workaround

Until fine-tuning is implemented, we should:

1. **Add a clear warning** to all outputs:
   ```
   ⚠️ WARNING: These predictions use population averages, not your personal baseline.
   Accuracy will be low until the system learns your patterns.
   ```

2. **Collect but don't rely on predictions**:
   - Still process and store all data
   - Show trends but not clinical recommendations
   - Focus on data quality metrics

3. **Prepare for fine-tuning**:
   - Store all raw features in a database
   - Create episode labeling interface
   - Build baseline calculation logic

## Action Items

1. [ ] Create `Episode` domain entity
2. [ ] Add `baseline_calculator.py` service  
3. [ ] Implement `model_personalizer.py` with fine-tuning logic
4. [ ] Add CLI command: `big-mood calibrate --episodes episodes.csv`
5. [ ] Create model storage with user-specific paths
6. [ ] Update predictions to check for personalized models first
7. [ ] Add confidence scoring based on available training data

## The Bottom Line

**The current system is a prototype that demonstrates the pipeline, but it cannot provide accurate clinical predictions without personalization.** This is not a bug—it's a fundamental requirement of the approach that we need to implement.

For the paper's authors, this was assumed (they always fine-tuned). For our production system, we need to make this seamless and invisible to users while maintaining clinical accuracy.


  Current Reality: The system makes predictions but they're not personalized
  What's Needed: User provides episode dates → System learns their baseline → Accurate
  predictions
  Priority: CRITICAL - This is the #1 thing to implement next

  You're absolutely right that this "throws a wrench" but we can work around it. The
  architecture is there, we just need to add:
  1. Episode labeling interface
  2. Baseline calculation
  3. Model fine-tuning logic
  4. Personalized model storage
