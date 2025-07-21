# XML Processing Fix Action Plan

**Date:** 2025-07-20  
**Priority:** CRITICAL - Main use case broken  
**Effort:** 3 approaches (Quick, Medium, Proper)  

## Problem Summary

520MB XML file times out because `DataParsingService` loads ALL records into memory despite having a streaming parser. The parser streams, but the service layer collects everything into lists.

## Approach 1: Quick Fix (2 hours)

### Add Date Range Filtering to CLI

**File:** `src/big_mood_detector/interfaces/cli/commands.py`

```python
@click.option(
    "--days-back",
    type=int,
    help="Process only the last N days of data (reduces memory usage)"
)
def process(file_path: Path, output: Path, days_back: Optional[int] = None):
    """Process health data."""
    if days_back:
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        # Pass dates to pipeline
```

**Benefits:**
- Users can process last 90 days: `--days-back 90`
- Reduces 500k records to ~10k
- No architecture changes needed
- Immediate relief for users

## Approach 2: Batch Processing (1 day)

### Process in Time Windows

**New file:** `src/big_mood_detector/application/services/batch_processor.py`

```python
class BatchProcessor:
    """Process large files in manageable chunks."""
    
    def process_in_batches(
        self,
        file_path: Path,
        batch_days: int = 30,
        output_path: Path = None
    ):
        # Get date range from file
        date_range = self._scan_date_range(file_path)
        
        # Process each batch
        for batch_start, batch_end in self._get_batches(date_range, batch_days):
            # Parse only this batch
            records = self.parsing_service.parse_xml_export(
                file_path,
                start_date=batch_start,
                end_date=batch_end
            )
            
            # Extract features
            features = self.pipeline.extract_features_batch(
                records,
                batch_start,
                batch_end
            )
            
            # Save incrementally
            self._save_batch_features(features, output_path)
            
            # Clear memory
            del records
            gc.collect()
            
            # Update progress
            self._report_progress(batch_end, date_range.end)
```

**Benefits:**
- Handles any file size
- Memory usage capped at ~30 days of data
- Progress visible to users
- Can be interrupted and resumed

## Approach 3: True Streaming (3-5 days)

### Refactor Pipeline for Streaming

**Core idea:** Never hold more than a few days in memory

```python
class StreamingPipeline:
    """Process data as it streams from parser."""
    
    def stream_process(self, file_path: Path, output_path: Path):
        # Daily accumulator
        daily_records = defaultdict(lambda: {
            'sleep': [], 'activity': [], 'heart': []
        })
        
        # Feature writer
        with FeatureWriter(output_path) as writer:
            # Stream records
            for record in self.parser.parse_file(file_path):
                date = record.start_date.date()
                record_type = self._get_record_type(record)
                
                # Accumulate by day
                daily_records[date][record_type].append(record)
                
                # Process complete days
                for complete_date in self._get_complete_days(daily_records):
                    if self._has_enough_history(daily_records, complete_date):
                        # Extract features for this day
                        features = self._extract_day_features(
                            daily_records,
                            complete_date
                        )
                        
                        # Write immediately
                        writer.write_day(complete_date, features)
                        
                        # Clean old data (keep 30-day window)
                        self._cleanup_old_days(daily_records, complete_date)
```

**Benefits:**
- True streaming - constant memory usage
- Scales to any file size
- Most elegant solution
- Follows streaming best practices

## Implementation Priority

### Phase 1: Quick Relief (This Week)
1. ✅ Add `--days-back` parameter
2. ✅ Add `--date-range` parameter  
3. ✅ Update documentation
4. ✅ Test with 500MB file

### Phase 2: Robust Solution (Next Week)
1. 📝 Implement BatchProcessor
2. 📝 Add progress bars with tqdm
3. 📝 Add checkpoint/resume capability
4. 📝 Memory usage monitoring

### Phase 3: Proper Architecture (v0.3.0)
1. 🔮 Design streaming pipeline
2. 🔮 Refactor service layer
3. 🔮 Add SQLite caching option
4. 🔮 Benchmark performance

## Testing Plan

### Test Files Needed
```python
# Generate test XML files
def generate_test_xml(size_mb: int, days: int):
    """Generate realistic test data."""
    records_per_day = (size_mb * 1024 * 1024) // (days * 200)  # ~200 bytes/record
    # Generate sleep, activity, heart records
```

### Test Scenarios
1. **Small file:** 10MB, 30 days
2. **Medium file:** 100MB, 180 days  
3. **Large file:** 500MB, 2 years
4. **Huge file:** 2GB, 5 years

### Performance Targets
- 100MB in < 30 seconds
- 500MB in < 2 minutes
- 2GB in < 10 minutes
- Memory < 1GB for any size

## Migration Safety

### Backward Compatibility
- Keep existing API unchanged
- Add new parameters as optional
- Default behavior = current behavior

### Feature Flags
```python
FEATURES = {
    'batch_processing': False,
    'streaming_pipeline': False,
    'date_filtering': True,  # Safe to enable
}
```

### Rollback Plan
- All changes behind feature flags
- Old code paths preserved
- Can disable per user if issues

## Success Metrics

### User Experience
- ✅ 500MB files process successfully
- ✅ Progress visible during processing
- ✅ Memory usage stays reasonable
- ✅ Can process last 90 days quickly

### Technical Metrics
- 📊 Processing speed > 50k records/second
- 📊 Memory usage < 1GB peak
- 📊 Linear scaling with file size
- 📊 No timeouts up to 5GB

## Next Steps

1. **Today:** Implement `--days-back` parameter
2. **Tomorrow:** Test with real 500MB file
3. **This Week:** Release v0.2.1 with quick fix
4. **Next Week:** Start batch processor
5. **v0.3.0:** Full streaming pipeline

---

*"Start with quick relief, build toward elegance."*