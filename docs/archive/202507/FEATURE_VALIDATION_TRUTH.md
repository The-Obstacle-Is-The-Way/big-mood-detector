# Feature Validation Source of Truth
Generated: 2025-07-23

## What We're Trying to Validate

### 1. The Two Models Are Separate:
- **XGBoost Model**: Predicts tomorrow's mood risk using 36 features from today
- **PAT Transformer**: Analyzes 7 days of minute-level activity (1440 values/day = 10,080 total)

### 2. The Problem We Found:
The basic CLI `process` command only outputs 10 features in the CSV:
- sleep_duration_hours
- sleep_efficiency  
- sleep_onset_hour
- wake_time_hour
- daily_steps
- activity_variance
- sedentary_hours
- activity_fragmentation
- sedentary_bout_mean
- activity_intensity_ratio

But the XGBoost model expects 36 features (as documented in Seoul paper).

### 3. Why This Happened:
Looking at the code, the `to_dict()` method in ClinicalFeatureSet (line 196) only exports 10 features, not all 36. This is likely because:
- The CLI was originally built for basic feature extraction
- The full 36-feature extraction happens internally during prediction
- The CSV export was never updated to include all features

### 4. Your Concern About Ensemble:
YES, your concern makes sense! The ensemble combines:
- XGBoost predictions (based on 36 features from today)
- PAT embeddings (based on 7 days of activity)

These operate on different time windows and should remain separate. The confusion comes from:
- The pipeline trying to do too many things at once
- Unclear separation between feature extraction vs prediction

## What We Should NOT Change:
1. The core prediction logic (it's working correctly)
2. The model weights or loading logic
3. The ensemble combination logic

## What We Need to Test:
1. **Verify XGBoost gets all 36 features during prediction** (internal pipeline)
2. **Verify PAT gets 7-day sequences during prediction** (when ensemble is used)
3. **Create a proper validation script** that tests the prediction pipeline, not just CSV export

## The Real Issue:
The CSV export is just for debugging/analysis. The actual prediction pipeline uses all features correctly internally. We need to validate the prediction pipeline, not the CSV export.

## Next Steps:
1. Create a validation script that uses the PREDICTION pipeline (not just process)
2. Log/verify all 36 features are computed during prediction
3. Test with your real data through the predict command
4. Don't modify the core feature extraction - it's working

## Safe Test Command:
```bash
# This uses the full pipeline with all features:
python src/big_mood_detector/main.py predict data/input/health_auto_export/ --report

# NOT this (only exports subset):
python src/big_mood_detector/main.py process data/input/health_auto_export/
```