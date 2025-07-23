# Feature Mapping Analysis - Big Mood Detector
Generated: 2025-07-23

## Executive Summary

The XGBoost prediction failures are caused by a **feature generation mismatch**, not a naming issue. Our clinical feature extractor generates features using a different algorithm than what the XGBoost models were trained on.

## The Real Problem

### What We Found:
1. **XGBoost expects**: Features calculated using the Seoul paper's specific algorithms
   - Example: `sleep_percentage_MN` = mean of (daily sleep minutes ÷ 1440)
   - These are statistical summaries over multiple days (MN=mean, SD=std dev, Z=z-score)

2. **Our code generates**: Single-day clinical features
   - Example: `sleep_duration_hours` = hours slept on a specific day
   - These are direct measurements, not statistical summaries

### The Critical Mismatch:
- XGBoost models were trained on **30-day statistical summaries** (12 base features × 3 statistics = 36)
- Our `ClinicalFeatureExtractor` generates **single-day measurements** (36 different features)

## Data Flow Analysis

### Current Pipeline:
```
Apple Health XML/JSON
    ↓
Domain Entities (SleepRecord, ActivityRecord, etc.)
    ↓
ClinicalFeatureExtractor.extract_seoul_features()
    ↓
SeoulXGBoostFeatures (single day features)
    ↓
.to_xgboost_features() → [36 floats]
    ↓
XGBoostModels.predict() → FAILS (wrong feature algorithm)
```

### What Should Happen:
```
Apple Health XML/JSON
    ↓
Domain Entities (30 days of records)
    ↓
AggregationPipeline (calculate 12 base Seoul features per day)
    ↓
Statistical summaries over 30 days (mean, std, z-score)
    ↓
DailyFeatures.to_dict() → {feature_name: value}
    ↓
XGBoostModels.predict() → SUCCESS
```

## The Two Feature Systems

### System 1: Seoul Paper Features (for XGBoost)
- **Purpose**: Statistical patterns over 30 days
- **Implementation**: `AggregationPipeline` + `DailyFeatures`
- **Features**: 12 base × 3 statistics = 36 total
- **Examples**:
  - `sleep_percentage_MN` = mean(sleep_minutes/1440) over 30 days
  - `long_num_SD` = std dev of daily long sleep window counts
  - `circadian_phase_Z` = z-score of DLMO hours

### System 2: Clinical Features (for interpretability)
- **Purpose**: Single-day clinical assessment
- **Implementation**: `ClinicalFeatureExtractor`
- **Features**: 36 direct measurements
- **Examples**:
  - `sleep_duration_hours` = hours slept today
  - `sleep_fragmentation` = fragmentation index today
  - `circadian_phase_advance` = phase shift today

## Why This Happened

1. **Feature Engineering Orchestrator** was added to enhance features
2. **ClinicalFeatureExtractor** was created for richer clinical insights
3. The prediction pipeline was modified to use the new extractor
4. But the XGBoost models still expect the original Seoul features

## The Solution Path

### Option 1: Use the Existing Aggregation Pipeline
- The `AggregationPipeline` already generates correct Seoul features
- Just need to route predictions through it instead of `ClinicalFeatureExtractor`
- Minimal code changes required

### Option 2: Add Feature Mapping
- Calculate Seoul features from clinical features
- Requires implementing the statistical calculations
- More complex but preserves new architecture

### Option 3: Dual Pipeline
- Use `AggregationPipeline` for XGBoost predictions
- Use `ClinicalFeatureExtractor` for clinical insights
- Best of both worlds but more complex

## Test Results Explained

```
xgboost prediction failed: training data did not have the following fields: 
ST_long_MN, ST_long_SD, ST_long_Zscore...
```

This error occurs because:
1. The model expects `long_ST_MN` (mean sleep time in long windows over 30 days)
2. We're providing `sleep_duration_hours` (today's total sleep)
3. These are fundamentally different features

## Recommendation

Use **Option 1** - Route predictions through the existing `AggregationPipeline`:
1. It's already implemented and tested
2. It generates the exact features the models expect
3. Minimal risk of breaking existing functionality
4. Can still use `ClinicalFeatureExtractor` for reports/insights

## Next Steps

1. Verify `AggregationPipeline` generates all 36 Seoul features correctly
2. Update prediction pipeline to use `AggregationPipeline` for XGBoost
3. Keep `ClinicalFeatureExtractor` for clinical reports and insights
4. Test with real data to ensure predictions work
5. Document the two feature systems clearly