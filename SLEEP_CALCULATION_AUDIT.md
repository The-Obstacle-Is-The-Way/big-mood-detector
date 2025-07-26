# Sleep Duration Calculation Audit

## The Problem

Average sleep duration showing as **12.29 hours** is unrealistically high for most people. Normal adult sleep is 7-9 hours.

## Raw Data Analysis

From `features.csv`:
- June 30: 12.75 hours
- July 2: 12.0 hours  
- July 7: 14.78 hours (!)
- July 8: 12.03 hours
- July 9: 9.21 hours
- July 11: 11.58 hours
- July 15: 13.73 hours

Average: 12.29 hours (as reported)

## Root Cause: Overlapping Sleep Records

The warnings tell the story:
```
Total bed time 27.6h exceeds 24h for 2024-08-30, capping at 24h. This may indicate overlapping sleep records.
Total bed time 27.0h exceeds 24h for 2025-01-12, capping at 24h. This may indicate overlapping sleep records.
...
```

Your Apple Health export contains **overlapping sleep records** - likely from:
1. Multiple devices recording simultaneously (iPhone + Apple Watch)
2. Manual entries overlapping with automatic tracking
3. Third-party apps adding duplicate data
4. Naps being counted as part of main sleep

## How the Calculation Works

1. **SleepAggregator** sums ALL sleep records for each day
2. If total > 24 hours, it caps at 24 hours
3. But even capped values (12-14 hours) are too high

The code is correctly summing durations:
```python
total_bed_time = sum(r.duration_hours for r in records)
total_sleep_time = sum(r.duration_hours for r in records if r.is_actual_sleep)
```

But it's not handling overlaps BEFORE summing.

## The Real Issue

Apple Health can have multiple overlapping records:
```
Device 1: 22:00 - 06:00 (8 hours)
Device 2: 22:30 - 06:30 (8 hours)  
Manual:   23:00 - 07:00 (8 hours)
Total:    24 hours (but actual sleep was only ~8 hours)
```

## Missing Overlap Detection

The current code doesn't merge overlapping periods:
```python
# Current (WRONG)
def aggregate_sleep(records):
    return sum(r.duration for r in records)  # Double counts!

# Needed (RIGHT)  
def aggregate_sleep(records):
    merged = merge_overlapping_periods(records)
    return sum(r.duration for r in merged)
```

## Impact on Models

1. **XGBoost**: Will think you're sleeping 12+ hours (hypersomnia signal)
2. **PAT**: Activity patterns distorted by incorrect sleep boundaries
3. **Clinical flags**: May trigger false "hypersomnia pattern" warnings

## Why This Relates to Data Selection

This overlapping issue is even more critical when selecting data windows:
- User might have good data for some periods
- But overlapping records make it unusable
- Need to identify "clean" periods without overlaps

## Recommendations

### 1. Implement Overlap Merging
```python
def merge_overlapping_sleep(records: List[SleepRecord]) -> List[SleepRecord]:
    """Merge overlapping sleep periods to avoid double counting."""
    if not records:
        return []
    
    # Sort by start time
    sorted_records = sorted(records, key=lambda r: r.start_date)
    merged = [sorted_records[0]]
    
    for current in sorted_records[1:]:
        last = merged[-1]
        
        # Check for overlap
        if current.start_date <= last.end_date:
            # Merge by extending end time
            last.end_date = max(last.end_date, current.end_date)
        else:
            # No overlap, add as new record
            merged.append(current)
    
    return merged
```

### 2. Add Data Quality Metrics
```python
@dataclass
class DataQualityMetrics:
    overlap_ratio: float  # How much data is overlapped
    coverage: float  # Percentage of day with data
    device_count: int  # Number of recording devices
    confidence: float  # Overall quality score
```

### 3. Provide Clear Warnings
```
WARNING: Detected overlapping sleep records on 7 days
- July 7: 27.6h of records for 24h day (multiple devices?)
- Merged to estimated 8.2h actual sleep
- Confidence: MEDIUM
```

## The Bigger Picture

This sleep calculation issue reveals a fundamental challenge:
- Apple Health aggregates data from multiple sources
- No built-in deduplication
- Raw summation gives incorrect results
- Need intelligent merging algorithms

For both PAT and XGBoost to work correctly, we need:
1. Clean, non-overlapping data
2. Accurate sleep duration calculations  
3. Quality metrics to identify usable periods
4. User guidance on data hygiene