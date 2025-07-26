# Implementation Plan: Fixing Big Mood Detector Architecture

**Date**: July 26, 2025  
**Priority**: CRITICAL - User cannot get predictions with current implementation

## Phase 1: Emergency Sleep Duration Fix (TODAY)

### Task 1.1: Fix SleepAggregator Overlap Bug
**File**: `domain/services/sleep_aggregator.py`

```python
# Add new method to SleepAggregator class:
def _merge_overlapping_records(self, records: list[SleepRecord]) -> list[tuple[datetime, datetime]]:
    """Merge overlapping sleep records from multiple devices."""
    if not records:
        return []
    
    # Convert to intervals and sort
    intervals = [(r.start_date, r.end_date) for r in records]
    intervals.sort()
    
    # Merge overlapping intervals
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_end = merged[-1][1]
        if start <= last_end:
            # Overlapping - extend the last interval
            merged[-1] = (merged[-1][0], max(last_end, end))
        else:
            # Non-overlapping - add new interval
            merged.append((start, end))
    
    return merged

# Update _create_daily_summary to use merging:
def _create_daily_summary(self, day: date, records: list[SleepRecord]) -> DailySleepSummary:
    # Get actual sleep records only
    sleep_records = [r for r in records if r.is_actual_sleep]
    bed_records = records  # All records for time in bed
    
    # Merge overlapping records
    merged_sleep = self._merge_overlapping_records(sleep_records)
    merged_bed = self._merge_overlapping_records(bed_records)
    
    # Calculate actual durations from merged intervals
    total_sleep_time = sum((end - start).total_seconds() / 3600 for start, end in merged_sleep)
    total_bed_time = sum((end - start).total_seconds() / 3600 for start, end in merged_bed)
    
    # Cap at 24 hours with warning
    if total_bed_time > 24.0:
        logger.warning(f"Capping bed time from {total_bed_time:.1f}h to 24h for {day}")
        total_bed_time = 24.0
        total_sleep_time = min(total_sleep_time, 24.0)
```

### Task 1.2: Add Device Overlap Detection
**File**: `infrastructure/parsers/xml/sleep_parser.py`

Add method to detect overlapping records during parsing:
```python
def detect_overlaps(self, sleep_records: list[SleepRecord]) -> list[tuple[SleepRecord, SleepRecord]]:
    """Detect overlapping sleep records from different devices."""
    overlaps = []
    sorted_records = sorted(sleep_records, key=lambda r: r.start_date)
    
    for i in range(len(sorted_records) - 1):
        for j in range(i + 1, len(sorted_records)):
            r1, r2 = sorted_records[i], sorted_records[j]
            # Check if different devices and overlapping
            if r1.source_name != r2.source_name:
                if r1.start_date < r2.end_date and r2.start_date < r1.end_date:
                    overlaps.append((r1, r2))
    
    return overlaps
```

## Phase 2: Pipeline Architecture Split (URGENT)

### Task 2.1: Create Independent Pipeline Validators
**New File**: `application/validators/pipeline_validators.py`

```python
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

@dataclass
class ValidationResult:
    is_valid: bool
    days_available: int
    consecutive_days: int
    missing_data: list[str]
    can_run: bool
    message: str

class PATValidator:
    """Validates data sufficiency for PAT model (7 consecutive days)."""
    
    REQUIRED_CONSECUTIVE_DAYS = 7
    
    def validate(self, activity_records, start_date: date, end_date: date) -> ValidationResult:
        # Check for 7 consecutive days with activity data
        dates_with_data = {r.start_date.date() for r in activity_records}
        
        # Find longest consecutive sequence
        consecutive_days = self._find_max_consecutive_days(dates_with_data)
        
        return ValidationResult(
            is_valid=consecutive_days >= self.REQUIRED_CONSECUTIVE_DAYS,
            days_available=len(dates_with_data),
            consecutive_days=consecutive_days,
            missing_data=[] if consecutive_days >= 7 else ["Need 7 consecutive days"],
            can_run=consecutive_days >= self.REQUIRED_CONSECUTIVE_DAYS,
            message=f"PAT requires 7 consecutive days, found {consecutive_days}"
        )

class XGBoostValidator:
    """Validates data sufficiency for XGBoost model (30-60 days, can be sparse)."""
    
    MINIMUM_DAYS = 30
    OPTIMAL_DAYS = 60
    
    def validate(self, sleep_records, activity_records, start_date, end_date) -> ValidationResult:
        # XGBoost can work with sparse data
        dates_with_sleep = {r.start_date.date() for r in sleep_records}
        dates_with_activity = {r.start_date.date() for r in activity_records}
        
        # Union of all dates with any data
        all_dates = dates_with_sleep | dates_with_activity
        days_available = len(all_dates)
        
        return ValidationResult(
            is_valid=days_available >= self.MINIMUM_DAYS,
            days_available=days_available,
            consecutive_days=0,  # Not required for XGBoost
            missing_data=[] if days_available >= 30 else [f"Need {30 - days_available} more days"],
            can_run=days_available >= self.MINIMUM_DAYS,
            message=f"XGBoost needs 30+ days (any distribution), found {days_available}"
        )
```

### Task 2.2: Split MoodPredictionPipeline
**New Files**: Create separate pipeline classes

1. **`application/pipelines/pat_pipeline.py`**
```python
class PATPipeline:
    """Independent pipeline for PAT depression assessment."""
    
    def __init__(self, pat_loader, validator: PATValidator):
        self.pat_loader = pat_loader
        self.validator = validator
    
    def can_run(self, activity_records, start_date, end_date) -> ValidationResult:
        return self.validator.validate(activity_records, start_date, end_date)
    
    def process(self, activity_records, target_date) -> Optional[PATResult]:
        # Extract 7-day window
        # Create 10,080-minute sequence
        # Run PAT model
        # Return current depression assessment
```

2. **`application/pipelines/xgboost_pipeline.py`**
```python
class XGBoostPipeline:
    """Independent pipeline for XGBoost mood prediction."""
    
    def __init__(self, feature_extractor, xgboost_predictor, validator: XGBoostValidator):
        self.feature_extractor = feature_extractor
        self.predictor = xgboost_predictor
        self.validator = validator
    
    def can_run(self, records, start_date, end_date) -> ValidationResult:
        return self.validator.validate(records, start_date, end_date)
    
    def process(self, records, target_date) -> Optional[XGBoostResult]:
        # Extract Seoul features for available days
        # Calculate statistics over time window
        # Run XGBoost prediction
        # Return tomorrow's mood risk
```

### Task 2.3: Update Process Use Case
**File**: `application/use_cases/process_health_data_use_case.py`

```python
def execute(self, command: ProcessHealthDataCommand) -> ProcessingResult:
    # ... existing data loading ...
    
    # Independent validation for each pipeline
    pat_validation = self.pat_pipeline.can_run(parsed_data.activity_records, start_date, end_date)
    xgboost_validation = self.xgboost_pipeline.can_run(all_records, start_date, end_date)
    
    # Run available pipelines
    predictions = []
    insights = []
    
    if pat_validation.can_run:
        pat_result = self.pat_pipeline.process(parsed_data.activity_records, end_date)
        if pat_result:
            predictions.append(pat_result)
            insights.append(f"PAT: Current depression risk = {pat_result.risk_score:.1%}")
    else:
        insights.append(f"PAT unavailable: {pat_validation.message}")
    
    if xgboost_validation.can_run:
        xgboost_result = self.xgboost_pipeline.process(all_records, end_date)
        if xgboost_result:
            predictions.append(xgboost_result)
            insights.append(f"XGBoost: Tomorrow's mood risk = {xgboost_result.risk_level}")
    else:
        insights.append(f"XGBoost unavailable: {xgboost_validation.message}")
    
    # Return partial results
    return ProcessingResult(
        feature_count=len(features),
        prediction_count=len(predictions),
        predictions=predictions,
        clinical_insights=insights,
        validation_errors=[]  # No longer block on validation
    )
```

## Phase 3: Data Window Selection Intelligence

### Task 3.1: Smart Window Finder
**New File**: `application/services/data_window_selector.py`

```python
class DataWindowSelector:
    """Intelligently selects optimal data windows for each model."""
    
    def find_best_pat_window(self, activity_records, target_date: date) -> Optional[DateRange]:
        """Find the best 7-consecutive-day window nearest to target date."""
        dates = sorted({r.start_date.date() for r in activity_records})
        
        # Find all 7-day consecutive windows
        windows = []
        for i in range(len(dates) - 6):
            if (dates[i+6] - dates[i]).days == 6:  # Consecutive
                windows.append((dates[i], dates[i+6]))
        
        if not windows:
            return None
        
        # Pick window closest to target date
        best_window = min(windows, key=lambda w: abs((w[1] - target_date).days))
        return DateRange(start=best_window[0], end=best_window[1])
    
    def find_best_xgboost_window(self, all_records, target_date: date) -> DateRange:
        """Find optimal 30-60 day window with most data."""
        # XGBoost is flexible - just need 30+ days total
        all_dates = {r.start_date.date() for r in all_records}
        
        if len(all_dates) >= 60:
            # Use most recent 60 days
            return DateRange(
                start=target_date - timedelta(days=59),
                end=target_date
            )
        elif len(all_dates) >= 30:
            # Use all available data
            return DateRange(
                start=min(all_dates),
                end=max(all_dates)
            )
        else:
            # Not enough data
            return None
```

## Phase 4: Testing & Validation

### Task 4.1: Test Sleep Duration Fix
```bash
# Create test for overlapping sleep records
pytest tests/unit/domain/services/test_sleep_aggregator.py::test_overlapping_records -xvs
```

### Task 4.2: Test Independent Pipelines
```bash
# Test PAT can run alone
pytest tests/integration/test_pat_pipeline_independent.py -xvs

# Test XGBoost can run alone  
pytest tests/integration/test_xgboost_pipeline_independent.py -xvs
```

### Task 4.3: Test with User's Data
```bash
# Process with new architecture
python src/big_mood_detector/main.py process data/health_auto_export/export.xml --report

# Should show:
# - PAT: Cannot run (need consecutive days)
# - XGBoost: Cannot run (only 7 days available, need 30+)
# - Sleep duration: ~7-8 hours (not 12+)
```

## Phase 5: Documentation Updates

### Task 5.1: Update CLAUDE.md
- Document the two-pipeline architecture
- Explain data requirements clearly
- Add troubleshooting for overlap issues

### Task 5.2: Create User Guide
- How to handle multiple device exports
- Data requirements for each model
- Understanding partial predictions

## Implementation Priority

1. **TODAY**: Fix sleep duration bug (Phase 1) - User's data is wrong
2. **THIS WEEK**: Split pipelines (Phase 2) - Enable partial predictions  
3. **NEXT SPRINT**: Smart data windows (Phase 3) - Better UX
4. **ONGOING**: Testing and documentation (Phases 4-5)

## Success Criteria

1. Sleep duration shows realistic values (6-9 hours average)
2. PAT can run independently with 7 consecutive days
3. XGBoost can run independently with 30+ sparse days
4. Clear user feedback about what's available
5. All tests passing
6. No mypy errors

## Risk Mitigation

1. **Backward Compatibility**: Keep old pipeline available via feature flag
2. **Data Migration**: Ensure cached features remain valid
3. **Model Compatibility**: Verify model weights work with new pipeline
4. **Performance**: Profile to ensure no regression

This plan addresses the critical architectural issues while maintaining system stability and improving user experience.