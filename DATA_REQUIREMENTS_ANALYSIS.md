# Data Requirements Analysis - XGBoost vs PAT

## Literature Review Findings

### XGBoost Paper (Lim et al., 2024)
**"Accurately predicting mood episodes in mood disorder patients using wearable sleep and circadian rhythm features"**

Key Data Requirements:
- **Training**: "60-day ranges" where half represented episodic days
- **Minimum**: 30 days mentioned as reduced training set
- **Average data**: 429 days total, 267 days excluding missing
- **Features**: 36 sleep and circadian features
- **Data density**: Can handle missing data (267/429 = 62% density)
- **Prediction window**: Next-day predictions

Quote: "Using the training data consisting of the 60-day ranges, we achieved AUCs of 0.80, 0.98, and 0.95"

### PAT Paper (Ruan et al., 2024) 
**"AI Foundation Models for Wearable Movement Data in Mental Health Research"**

Key Data Requirements:
- **Input**: "week-long actigraphy data" = exactly 7 days
- **Format**: 10,080 minutes (7 × 24 × 60) continuous
- **Pretraining**: NHANES datasets with full weeks
- **Architecture**: Patches of 18 minutes → 560 tokens
- **Requirement**: MUST be consecutive (transformer expects continuous sequence)

Quote: "one week of actigraphy data produces over 10,000 tokens with minute-level sampling"

## Critical Differences

### 1. Time Windows
- **XGBoost**: 30-60 days (longer historical context)
- **PAT**: Exactly 7 consecutive days (fixed window)

### 2. Data Continuity
- **XGBoost**: Can work with sparse data (calculates daily aggregates)
- **PAT**: Requires continuous 7-day sequence (no gaps)

### 3. Feature Engineering
- **XGBoost**: Extensive - 36 hand-crafted features including circadian calculations
- **PAT**: None - raw minute-level activity fed directly to transformer

### 4. Temporal Focus
- **XGBoost**: Predicts TOMORROW based on patterns over past month+
- **PAT**: Assesses NOW based on past week

## Current Implementation Issues

### 1. Unified Minimum Days Check
```python
# WRONG - applies same requirement to both
if days_of_data < 7:
    raise InsufficientDataError("Need at least 7 days")
```

Should be:
```python
# PAT check
consecutive_weeks = find_consecutive_weeks(data)
if not consecutive_weeks:
    pat_available = False

# XGBoost check  
total_days = count_days_with_data(data)
if total_days < 30:
    xgboost_available = False
```

### 2. Feature Extraction Pipeline
Current: Both models go through same feature extraction
```
Data → Aggregate → Extract 36 Features → Models
```

Should be:
```
Data ─┬→ [PAT] Extract 7-day sequences → PAT Model
      └→ [XGB] Aggregate 30-60 days → Calculate Features → XGBoost
```

### 3. Seoul Features Calculation

The 36 Seoul features require extended time periods:
- **Circadian phase shifts**: Need multiple days to detect patterns
- **Sleep regularity index**: Requires variance calculation over weeks
- **Interdaily stability**: Needs 7+ days minimum
- **Intradaily variability**: Best with 30+ days

These CANNOT be calculated from just 7 days of sparse data.

## Validation Logic Needed

### For PAT
```python
def find_valid_pat_windows(health_data):
    """Find all 7-day consecutive windows in data."""
    windows = []
    for start_date in all_dates:
        end_date = start_date + timedelta(days=7)
        if has_continuous_data(start_date, end_date):
            windows.append((start_date, end_date))
    return windows
```

### For XGBoost
```python
def validate_xgboost_data(health_data):
    """Check if enough data for circadian features."""
    days_with_sleep = count_days_with_metric(health_data, 'sleep')
    days_with_activity = count_days_with_metric(health_data, 'activity')
    
    # Need substantial data for each metric
    return (days_with_sleep >= 30 and 
            days_with_activity >= 30 and
            data_density > 0.5)  # 50% minimum density
```

## Why User's Data Failed

User has 7 non-consecutive days over 30 days:
- Days: June 30, July 2, 7, 8, 9, 11, 15
- Longest consecutive run: 3 days (July 7-9)
- Total density: 7/30 = 23%

Results:
- ❌ PAT: Needs 7 consecutive days (max found: 3)
- ❌ XGBoost: Needs 30+ days with >50% density (has: 7 days, 23%)

## Recommended Pipeline Redesign

### 1. Independent Validators
```python
@dataclass
class DataValidation:
    pat_windows: List[Tuple[date, date]]  # Valid 7-day windows
    xgboost_ready: bool  # Has 30+ days
    xgboost_days: int  # Actual days available
    data_density: float  # Percentage of days with data
    recommendations: List[str]  # What user needs
```

### 2. Parallel Processing
```python
def process_health_data(xml_path):
    data = parse_xml(xml_path)
    validation = validate_data(data)
    
    results = {}
    
    # Try PAT for each valid window
    if validation.pat_windows:
        pat_results = []
        for start, end in validation.pat_windows:
            window_data = extract_window(data, start, end)
            pat_results.append(run_pat(window_data))
        results['pat'] = best_result(pat_results)
    
    # Try XGBoost if enough data
    if validation.xgboost_ready:
        features = calculate_36_features(data)
        results['xgboost'] = run_xgboost(features)
    
    return results
```

### 3. Clear User Feedback
```
Analysis complete:
- Found 7 days of data over 30 days (23% density)
- Longest consecutive period: 3 days

❌ PAT Depression Assessment: Insufficient data
   - Requires: 7 consecutive days
   - Found: Maximum 3 consecutive days
   - Recommendation: Wear device continuously for a week

❌ XGBoost Mood Prediction: Insufficient data  
   - Requires: 30+ days with >50% density
   - Found: 7 days with 23% density
   - Recommendation: Need 23 more days of data

Try using a date range with more complete data, or ensure
consistent device usage going forward.
```

## Conclusion

The fundamental issue is that we built a unified pipeline requiring both models to work together, when they should operate independently with different data requirements. The solution is to:

1. Separate validation for each model
2. Process in parallel, not serial
3. Return partial results when available
4. Provide clear feedback on what's missing

This aligns with the clinical use case where a patient might have enough data for current state assessment (PAT) but not future prediction (XGBoost), or vice versa.