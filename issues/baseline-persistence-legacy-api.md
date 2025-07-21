## Baseline persistence tests use outdated domain entity APIs

### Description
The integration tests in `test_baseline_persistence_pipeline.py` are using an outdated API for domain entities (SleepRecord, ActivityRecord, HeartRateRecord) that no longer matches the current domain model.

### Current Behavior
- Tests are marked as `xfail` because they use old constructor signatures
- Tests try to pass parameters like `date`, `sleep_duration_hours`, `intensity` that don't exist in current entities

### Expected Behavior
- Tests should use the current domain entity APIs
- Tests should validate that personal baselines persist and improve predictions over time

### Examples of Outdated API Usage
```python
# OLD (test is using):
SleepRecord(
    date=current_date,
    sleep_start=...,
    sleep_end=...,
    sleep_duration_hours=...,
    sleep_efficiency=...
)

# CURRENT (should be):
SleepRecord(
    source_name="test",
    start_date=...,
    end_date=...,
    state=SleepState.ASLEEP
)
```

### Impact
- Critical feature (personal baseline calibration) lacks proper test coverage
- Cannot verify that baselines actually improve prediction accuracy
- Risk of regression in personalization features

### Proposed Solution
1. Rewrite tests to use current domain entity constructors
2. Ensure tests validate the core value prop: predictions improve with personal data
3. Add proper test data generation that matches real-world patterns

### Affected Tests
- `test_baseline_persistence_improves_predictions`
- `test_baseline_persistence_after_pipeline_restart`

### Labels
- tech-debt
- testing
- domain
- priority: high

### References
- @CLAUDE identified during test failure investigation
- Critical for v0.3.0 personalization features