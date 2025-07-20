# CHANGELOG - Refactoring Progress

## [2025-07-20] - Critical Sleep Duration Bug Fix & Quality Improvements

### Fixed
- **CRITICAL**: Fixed sleep duration calculation bug where `sleep_percentage * 24` was producing 2-5 hour results instead of correct 7.5 hours
  - Root cause: Aggregation pipeline was using window-based percentage instead of SleepAggregator
  - Solution: Added `_get_actual_sleep_duration()` method using proper SleepAggregator
  - Added `sleep_duration_hours` field to daily_metrics for accurate tracking

### Added
- **Apple Health 3pm Rule**: Implemented Apple's sleep date assignment convention
  - Sleep ending before 3pm stays on wake date
  - Sleep ending after 3pm assigned to next day
  - Comprehensive tests for midnight boundaries and edge cases

- **Wire-tap Logging**: Added debug logging to SleepAggregator for observability
  - Logs sleep record date assignments
  - Logs daily aggregation summaries
  - Helps debug date boundary issues

- **Property-based Testing**: Added hypothesis for incremental statistics validation
  - Tests mathematical correctness against numpy
  - Validates Welford's algorithm implementation
  - Ensures numerical stability with large values

- **Regression Guards**: Multiple layers of protection against sleep bug returning
  - CI script `check_no_sleep_percentage.sh` to catch pattern in code
  - Unit test ensuring sleep stays in [6,10] hour bounds
  - Application-level regression test with multi-day validation

### Improved
- **Baseline Persistence**: Fixed triple-write issue
  - Moved `persist_baselines()` outside date loop
  - Added file locking for thread safety (pytest-xdist compatible)
  - Now saves only once per pipeline run

- **HR/HRV Baselines**: Skip default values (70 bpm / 50 ms)
  - Only update baselines with real sensor data
  - Prevents skewing statistics with placeholder values

### Technical Debt Addressed
- Added hypothesis to dev dependencies for property-based testing
- Updated documentation organization (moved to docs/architecture/)
- Fixed E2E test date assignment to work with Apple 3pm rule
- Improved test coverage for edge cases (midnight, 3pm boundary, fragmented sleep)

### Metrics
- Sleep duration accuracy: Now correctly reports ~7.5 hours (was 2-5 hours)
- Test coverage: Added 17 new sleep aggregator tests
- Performance: No regression in processing speed

### Next Steps
- Write integration test for baseline persistence in pipeline
- Remove magic HR/HRV defaults (use explicit values)
- Add configuration for TimescaleDB vs File repository selection