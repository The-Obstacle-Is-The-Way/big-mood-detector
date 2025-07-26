# Data Selection and Validation Analysis

## The Core Challenge

We need to intelligently select usable data windows from Apple Health exports that may contain:
- Sparse data (missing days)
- Overlapping records (multiple devices)
- Different quality levels across time periods
- Varying requirements for PAT vs XGBoost

## Current Data Selection Approach

### What We Have
```python
# CLI options for date filtering
--days-back 30  # Last N days
--date-range 2025-01-01:2025-03-31  # Specific range
--start-date / --end-date  # Boundaries
```

### What's Missing
1. **No quality assessment** before processing
2. **No intelligent window finding** for PAT
3. **No overlap detection** before aggregation
4. **No user guidance** on best periods to use

## Proposed Data Selection Framework

### 1. Data Quality Assessment Phase
```python
@dataclass
class DataQualityReport:
    """Comprehensive quality metrics for a date range."""
    date_range: Tuple[date, date]
    total_days: int
    days_with_data: int
    data_density: float  # 0.0 - 1.0
    
    # Overlap metrics
    days_with_overlaps: int
    max_overlap_ratio: float  # e.g., 27.6h/24h = 1.15
    
    # Continuity metrics
    consecutive_runs: List[Tuple[date, date]]
    longest_consecutive_days: int
    gaps: List[Tuple[date, date]]
    
    # Device metrics
    device_sources: Set[str]
    multi_device_days: int
    
    # Completeness by type
    sleep_coverage: float
    activity_coverage: float
    heart_rate_coverage: float
    
    # Pipeline eligibility
    pat_eligible_windows: List[Tuple[date, date]]
    xgboost_eligible: bool
    xgboost_confidence: float
```

### 2. Intelligent Window Selection

#### For PAT (7 consecutive days)
```python
def find_pat_windows(data: HealthData) -> List[PATWindow]:
    """Find all valid 7-day windows for PAT analysis."""
    windows = []
    
    # Scan for consecutive runs
    consecutive_runs = find_consecutive_days(data)
    
    for run_start, run_end in consecutive_runs:
        run_length = (run_end - run_start).days + 1
        
        # Slide 7-day window through each run
        if run_length >= 7:
            for offset in range(run_length - 6):
                window_start = run_start + timedelta(days=offset)
                window_end = window_start + timedelta(days=6)
                
                # Assess window quality
                quality = assess_window_quality(
                    data, window_start, window_end
                )
                
                if quality.is_acceptable():
                    windows.append(PATWindow(
                        start=window_start,
                        end=window_end,
                        quality_score=quality.score,
                        warnings=quality.warnings
                    ))
    
    # Sort by quality score
    return sorted(windows, key=lambda w: w.quality_score, reverse=True)
```

#### For XGBoost (30-60 days)
```python
def find_xgboost_range(data: HealthData) -> Optional[XGBoostRange]:
    """Find optimal date range for XGBoost analysis."""
    
    # Start with most recent data
    end_date = max(data.dates)
    
    # Try different window sizes
    for days in [60, 45, 30]:  # Prefer larger windows
        start_date = end_date - timedelta(days=days-1)
        
        # Count available data
        available = count_days_in_range(data, start_date, end_date)
        density = available / days
        
        if density >= 0.5:  # 50% minimum density
            # Assess quality
            quality = assess_range_quality(
                data, start_date, end_date
            )
            
            if quality.has_sufficient_circadian_data():
                return XGBoostRange(
                    start=start_date,
                    end=end_date,
                    days_available=available,
                    density=density,
                    confidence=quality.circadian_confidence
                )
    
    return None  # No suitable range found
```

### 3. Overlap Resolution

#### Before Aggregation
```python
def resolve_overlaps(records: List[SleepRecord]) -> List[SleepRecord]:
    """Merge overlapping records before aggregation."""
    if not records:
        return []
    
    # Group by source
    by_source = defaultdict(list)
    for record in records:
        by_source[record.source_name].append(record)
    
    # Prioritize sources (e.g., Apple Watch > iPhone > Manual)
    source_priority = {
        "Apple Watch": 1,
        "iPhone": 2,
        "Manual Entry": 3,
        "Third Party": 4
    }
    
    # For each time period, keep highest priority source
    merged = []
    for time_slot in generate_time_slots(records):
        candidates = [
            r for r in records 
            if overlaps(r, time_slot)
        ]
        
        if candidates:
            # Sort by priority and duration
            best = max(
                candidates,
                key=lambda r: (
                    -source_priority.get(r.source_name, 999),
                    r.duration_hours
                )
            )
            merged.append(best)
    
    return merge_adjacent(merged)
```

### 4. User Guidance System

#### Pre-Processing Report
```python
def generate_data_assessment(file_path: Path) -> str:
    """Generate user-friendly data assessment."""
    
    # Quick scan of data
    report = scan_health_export(file_path)
    
    message = f"""
DATA QUALITY ASSESSMENT
======================
File: {file_path.name}
Total days span: {report.total_days}
Days with data: {report.days_with_data} ({report.data_density:.0%})

OVERLAP ISSUES
--------------
{'✅ No significant overlaps detected' if report.days_with_overlaps == 0 else
f'⚠️  {report.days_with_overlaps} days have overlapping records
   Max overlap: {report.max_overlap_ratio:.1f}x normal
   Likely cause: Multiple devices or manual entries'}

PAT ANALYSIS (Current State)
---------------------------
{'✅ Found ' + str(len(report.pat_eligible_windows)) + ' eligible 7-day windows:' 
if report.pat_eligible_windows else '❌ No 7-day consecutive periods found'}
{format_pat_windows(report.pat_eligible_windows)}

XGBOOST ANALYSIS (Future Risk)
-----------------------------
{'✅ Eligible for mood prediction' if report.xgboost_eligible else 
'❌ Insufficient data for circadian analysis'}
{f'   Confidence: {report.xgboost_confidence:.0%}' if report.xgboost_eligible else
f'   Need {30 - report.days_with_data} more days of data'}

RECOMMENDATIONS
---------------
{generate_recommendations(report)}
"""
    return message
```

#### Recommendation Engine
```python
def generate_recommendations(report: DataQualityReport) -> List[str]:
    """Generate actionable recommendations."""
    recs = []
    
    # For PAT
    if not report.pat_eligible_windows:
        gap_days = 7 - report.longest_consecutive_days
        recs.append(
            f"• For depression screening: Wear device for {gap_days} "
            f"more consecutive days"
        )
    elif report.pat_eligible_windows:
        best_window = report.pat_eligible_windows[0]
        recs.append(
            f"• Best PAT window: {best_window[0]} to {best_window[1]}"
        )
    
    # For XGBoost  
    if not report.xgboost_eligible:
        if report.days_with_data < 30:
            recs.append(
                f"• For mood prediction: Need {30 - report.days_with_data} "
                f"more days total (gaps OK)"
            )
        else:
            recs.append(
                "• For mood prediction: Data too sparse, "
                "increase daily device usage"
            )
    
    # For overlaps
    if report.days_with_overlaps > 5:
        recs.append(
            "• Multiple devices detected: Consider using "
            "only primary device for cleaner data"
        )
    
    # Suggest date ranges
    if report.xgboost_eligible or report.pat_eligible_windows:
        recs.append("\nSUGGESTED COMMANDS:")
        
        if report.pat_eligible_windows:
            w = report.pat_eligible_windows[0]
            recs.append(
                f"  big-mood predict export.xml "
                f"--date-range {w[0]}:{w[1]}"
            )
        
        if report.xgboost_eligible:
            recs.append(
                f"  big-mood predict export.xml "
                f"--days-back {report.xgboost_range.days}"
            )
    
    return recs
```

### 5. Enhanced CLI Integration

#### New Commands
```bash
# Assess data quality without processing
big-mood assess export.xml

# Find best windows automatically
big-mood predict export.xml --auto-select

# Show all valid PAT windows
big-mood find-windows export.xml

# Process with quality thresholds
big-mood predict export.xml --min-quality 0.8
```

#### Modified Predict Flow
```python
def predict_command_enhanced(file_path, **options):
    """Enhanced predict with intelligent data selection."""
    
    # 1. Quick assessment
    if options.get('assess_first', True):
        report = generate_data_assessment(file_path)
        click.echo(report)
        
        if not report.has_usable_data():
            if not click.confirm("No ideal data found. Continue anyway?"):
                return
    
    # 2. Auto-select best windows
    if options.get('auto_select'):
        pat_window = select_best_pat_window(file_path)
        xgb_range = select_best_xgb_range(file_path)
        
        click.echo(f"Auto-selected:")
        if pat_window:
            click.echo(f"  PAT: {pat_window}")
        if xgb_range:
            click.echo(f"  XGBoost: {xgb_range}")
    
    # 3. Process with quality awareness
    results = process_with_quality_checks(
        file_path,
        pat_window=pat_window,
        xgb_range=xgb_range,
        min_quality=options.get('min_quality', 0.7)
    )
    
    # 4. Report with confidence levels
    generate_report_with_confidence(results)
```

## Implementation Priority

### Phase 1: Core Quality Metrics (1-2 days)
1. Implement `DataQualityReport` class
2. Add overlap detection to `SleepAggregator`
3. Create `find_consecutive_days` utility

### Phase 2: Window Selection (2-3 days)
1. Implement PAT window finder
2. Implement XGBoost range optimizer
3. Add quality scoring algorithms

### Phase 3: User Experience (2-3 days)
1. Add `assess` command
2. Enhance `predict` with auto-select
3. Improve error messages with recommendations

### Phase 4: Advanced Features (1 week)
1. Multi-device handling strategies
2. Timezone correction
3. Data cleaning pipelines
4. Export quality reports

## Benefits

1. **User Confidence**: Know data quality before processing
2. **Better Results**: Use optimal data windows
3. **Clear Guidance**: Actionable recommendations
4. **Reduced Errors**: Catch issues early
5. **Research Value**: Quality metrics for validation

## Summary

The current system processes whatever data is provided without intelligence. We need:
1. Pre-processing quality assessment
2. Intelligent window selection for each model
3. Overlap resolution before aggregation
4. Clear user guidance throughout

This would transform the user experience from "error: insufficient data" to "here's exactly what you need to do to get predictions."