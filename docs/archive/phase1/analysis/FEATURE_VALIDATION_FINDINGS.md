# Feature Validation Findings
Generated: 2025-07-23

## CRITICAL FINDING: Feature Name Mismatch

### The Real Problem:
The XGBoost models are failing because of a **feature name mismatch**:

1. **Model expects** (from error message):
   - ST_long_MN, ST_long_SD, ST_long_Zscore
   - WT_long_MN, WT_long_SD, WT_long_Zscore
   - LongSleepWindow_length_MN, LongSleepWindow_length_SD
   - Sleep_percentage_MN, Sleep_percentage_SD
   - Circadian_phase_MN, Circadian_phase_SD
   - etc. (36 features with MN/SD/Zscore suffixes)

2. **Our code generates** (from clinical_feature_extractor.py):
   - sleep_duration_hours
   - sleep_efficiency
   - sleep_onset_hour
   - circadian_phase_advance
   - etc. (36 features with descriptive names)

### Why This Happened:
- The XGBoost models were trained on the original Seoul paper's feature names
- Our implementation translated these to more readable names
- But we never mapped back to the original names when feeding to the model

### Evidence:
```
xgboost prediction failed: training data did not have the following fields: 
ST_long_MN, ST_long_SD, ST_long_Zscore, ST_short_MN, ST_short_SD...
```

### Test Results:
1. **XGBoost Only**: Got 39.1% depression risk (only 1 day processed)
2. **Ensemble**: 50% for all risks (failed predictions = default values)
3. **With Calibration**: Same 50% (failed predictions)

### The Good News:
- The feature extraction is working correctly
- The models are loading correctly
- Personal calibration is saving baselines
- Only the feature name mapping is broken

## Next Steps:
1. Find where the feature mapping should happen
2. Create proper mapping from our names to Seoul names
3. Test again to verify predictions work

## Important:
- Do NOT change the feature extraction logic
- Do NOT change the model files
- Only fix the feature name mapping