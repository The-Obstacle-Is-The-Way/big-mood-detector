## Streaming parser date filtering fails with string/datetime comparison

### Description
The FastStreamingXMLParser's date filtering logic has a bug where it tries to compare string dates from XML attributes with datetime objects, causing the filtering to fail and potentially over-filter or under-filter records.

### Current Behavior
- When using `--days-back` or `--date-range` with large XML files, the parser may not correctly filter records
- The `test_memory_bounds.py` test is currently marked as `xfail` due to this issue

### Expected Behavior
- Date strings from XML attributes should be properly parsed to datetime objects before comparison
- Date filtering should work correctly for all record types

### Root Cause
In `fast_streaming_parser.py`, the date comparison logic needs to handle the conversion:
```python
date_str = elem.get("startDate")  # This is a string
if start_date and record_date < start_date:  # start_date is a datetime
```

### Impact
- Users with large XML files (500MB+) cannot effectively use date filtering to reduce processing time
- Memory usage optimization via date filtering is not working as intended

### Proposed Solution
1. Parse the date string to datetime before comparison
2. Add proper error handling for malformed dates
3. Update the test and remove xfail marker once fixed

### Test Case
```bash
python src/big_mood_detector/main.py process large_export.xml --days-back 90
```

### Labels
- bug
- tech-debt
- parser
- priority: high

### References
- Related test: `tests/integration/test_memory_bounds.py`
- @CLAUDE identified this during code review