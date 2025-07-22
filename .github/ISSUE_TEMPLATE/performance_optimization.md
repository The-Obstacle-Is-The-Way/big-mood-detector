# Performance Optimization: Fix O(n*m) Scaling in Aggregation Pipeline

## Summary
The aggregation pipeline has O(n*m) complexity when processing large datasets, causing 365-day analyses to take ~170 seconds instead of the <60 second target. This only affects bulk imports or yearly reports - typical daily usage (1-30 days) performs acceptably.

## Current State
- **Impact**: Year-long data processing takes 3 minutes instead of 1 minute
- **Tests**: 3 performance tests marked as `xfail` to unblock CI
- **Production**: Working correctly, just slower than ideal for large datasets

## Root Cause
The issue is in `AggregationPipeline` where it scans ALL records for EACH day:

```python
# Current O(n*m) pattern in multiple methods:
day_activity = [a for a in activity_records if a.start_date.date() == target_date]
```

With 365 days × 500k records = 182 million comparisons.

## Existing Solution
`OptimizedAggregationPipeline` already implements pre-indexing:
- ✅ Indexes records by date for O(1) lookups
- ✅ Already optimizes `_calculate_activity_metrics()`
- ❌ But still calls parent's O(n*m) methods for circadian and DLMO

## Required Work

### 1. Complete the optimization (partially done)
- [x] `_calculate_circadian_metrics_optimized()` - implemented, needs testing
- [x] `_calculate_dlmo_optimized()` - implemented, needs testing
- [ ] Verify performance meets targets
- [ ] Remove xfail markers from tests

### 2. Performance Targets
- 365-day processing: <60s (currently 170s)
- Linear O(n) scaling (currently O(n*m))
- Maintain <100MB memory usage

### 3. Tests to Re-enable
- `test_optimized_vs_original_performance`
- `test_aggregation_performance_target` 
- `test_aggregation_should_be_linear_not_quadratic`

## Recommendation
This is **not urgent** - only affects edge cases. Implement when:
- Users need yearly reports
- Bulk import performance becomes important
- You have a spare dev day

## References
- Tracking: `docs/performance/OPTIMIZATION_TRACKING.md`
- Analysis: `PERFORMANCE_INVESTIGATION.md`
- Tests: `tests/integration/test_aggregation_performance_issue.py`