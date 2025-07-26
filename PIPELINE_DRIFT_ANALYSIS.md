# Pipeline Drift Analysis - Big Mood Detector
**Date**: July 26, 2025
**Author**: Analysis of current implementation vs. intended design

## Executive Summary

We have drifted from the original two parallel pipelines design. The system is currently trying to enforce common data requirements across both pipelines, causing failures when sparse data prevents XGBoost feature calculation.

## Original Design Intent

### Two Independent Parallel Pipelines

1. **PAT Pipeline** (Current State Assessment)
   - Input: 7 consecutive days of minute-level activity data (10,080 data points)
   - Output: Current depression probability based on past 7 days
   - Paper: "pretrained on week-long actigraphy data"
   - No circadian features needed - transformer learns patterns directly

2. **XGBoost Pipeline** (Future Risk Prediction)
   - Input: 30-60 days of aggregated daily data
   - Output: Tomorrow's depression/mania/hypomania risk
   - Paper: "60-day ranges" for training, mentions 30-day minimum
   - Requires 36 Seoul features including circadian calculations

## Current Implementation Problems

### 1. Shared Minimum Requirements
The code currently enforces 7 consecutive days minimum for BOTH pipelines:
```python
# In feature_extraction_service.py
if (feature_date - min_date).days < 7:  # Need at least 7 days history
    continue
```

This is wrong because:
- PAT only needs 7 consecutive days
- XGBoost needs 30-60 days for circadian rhythm calculations

### 2. Feature Extraction Coupling
Both pipelines go through the same feature extraction:
```python
# Current flow
XML → Parse → Aggregate → Extract 36 Features → Predict
```

But should be:
```python
# PAT flow
XML → Parse → Extract 7-day sequences → PAT → Depression probability

# XGBoost flow  
XML → Parse → Aggregate 30-60 days → Calculate 36 features → XGBoost → Tomorrow's risk
```

### 3. Consecutive Days Requirement
The system requires consecutive days for both pipelines, but:
- PAT: MUST have 7 consecutive days (one week of continuous data)
- XGBoost: Can work with sparse data over 30-60 days (calculates averages/patterns)

## Why Your Data Failed

Your XML has only 7 non-consecutive days over 30 days:
- June 30, July 2, 7, 8, 9, 11, 15 (gaps between days)
- PAT fails: Needs 7 CONSECUTIVE days
- XGBoost fails: Needs more days to calculate circadian patterns

## Recommendations

### 1. Separate Pipeline Requirements
```python
class PATRequirements:
    MIN_CONSECUTIVE_DAYS = 7
    REQUIRED_MINUTES = 10_080  # 7 * 24 * 60
    
class XGBoostRequirements:
    MIN_DAYS = 30  # Can be non-consecutive
    OPTIMAL_DAYS = 60
    REQUIRED_FEATURES = 36
```

### 2. Independent Data Validation
```python
def validate_for_pat(health_data) -> bool:
    # Find any 7 consecutive days with complete data
    return has_consecutive_days(health_data, days=7)

def validate_for_xgboost(health_data) -> bool:
    # Need enough days for circadian calculations
    days_with_data = count_days_with_data(health_data)
    return days_with_data >= 30
```

### 3. Parallel Processing
```python
# Process in parallel, not sequential
pat_result = None
xgboost_result = None

if validate_for_pat(data):
    pat_result = pat_pipeline.process(data)
    
if validate_for_xgboost(data):
    xgboost_result = xgboost_pipeline.process(data)

# Combine results if both available
if pat_result and xgboost_result:
    return TemporalEnsemble(pat_result, xgboost_result)
elif pat_result:
    return CurrentStateOnly(pat_result)
elif xgboost_result:
    return FutureRiskOnly(xgboost_result)
else:
    return InsufficientData()
```

## Implementation Changes Needed

### 1. Update ProcessHealthDataUseCase
- Remove shared minimum days requirement
- Process pipelines independently
- Handle partial results gracefully

### 2. Create Separate Validators
- `PATDataValidator`: Checks for 7 consecutive days
- `XGBoostDataValidator`: Checks for 30+ days of data

### 3. Fix Feature Extraction
- PAT: Direct sequence extraction, no feature engineering
- XGBoost: Full 36-feature calculation with circadian

### 4. Update CLI Output
- Show which models could run
- Explain why others couldn't
- Provide specific data requirements not met

## Testing with Current Data

With your sparse 7-day data:
- PAT: Would fail (non-consecutive)
- XGBoost: Would fail (too few days)

To get results, you need either:
1. Find 7 consecutive days in your export
2. Use a date range with 30+ days of data
3. Get a more complete Apple Health export

## Conclusion

We built a unified pipeline when we needed two parallel ones. The fix is to:
1. Separate validation logic
2. Process independently
3. Combine results when both available
4. Handle partial results gracefully

This matches the original research papers and the clinical use case where:
- PAT tells you "how are you NOW"
- XGBoost tells you "what's your risk TOMORROW"

They serve different purposes and have different data requirements.