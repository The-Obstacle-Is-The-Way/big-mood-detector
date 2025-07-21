# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2025-07-20

### Added
- Date range filtering for XML processing with `--days-back` and `--date-range` CLI options
- Integration tests for date filtering and memory bounds
- Wire-tap logging in SleepAggregator for debugging sleep date assignment
- Property-based testing with hypothesis for incremental statistics
- Heart rate aggregation in AggregationPipeline
- HR/HRV field support in TimescaleDB baseline repository
- Comprehensive test suite for TimescaleDB HR/HRV functionality
- Test organization: created repositories/ subfolder for repository tests

### Changed
- Implemented Apple Health 3pm cutoff rule for sleep date assignment
- Made HR/HRV fields optional in UserBaseline (no more magic defaults)
- Updated FileBaselineRepository to preserve None values for HR/HRV
- Fixed sleep duration calculation bug (was using sleep_percentage * 24)
- Updated all datetime.utcnow() calls to datetime.now(timezone.utc)
- Improved AdvancedFeatureEngineer to only update baselines with real data

### Fixed
- 848 linting issues resolved with ruff
- 55 type checking errors fixed
- SQLAlchemy 2.0 import compatibility
- Structlog logger initialization order
- Application regression test now uses 3 days of data for statistics
- TimescaleDB repository now handles baseline updates properly
- Sleep duration calculation now caps at 24 hours to handle overlapping records
- Episode deletion in SQLite repository now works correctly
- XGBoost model loading now looks in correct directory (converted/ instead of pretrained/)
- Fixed all deprecated datetime.utcnow() usage to use timezone-aware datetime.now(UTC)
- Fixed deprecated datetime.utcfromtimestamp() to datetime.fromtimestamp(..., UTC)

### Removed
- Magic HR/HRV defaults (70 bpm / 50 ms) that would skew personal baselines
- Deprecated datetime.utcnow() usage throughout codebase

### Technical Debt (Tracked)
- Issue #38: Streaming parser date filtering bug (test: test_memory_bounds.py)
- Issue #39: Baseline persistence tests use legacy entity APIs (test: test_baseline_persistence_pipeline.py)
- Issue #40: XGBoost JSON models lack predict_proba method (test: test_pipeline_with_ensemble)
- All xfail tests have strict=True to alert when fixed
- Nightly CI job added to monitor slow/xfail tests
- Repository pattern redundancy needs review
- SQLite repository now uses unique constraints to prevent duplicate episodes

### Developer Notes
- Run `./scripts/create-tech-debt-issues.sh` to create GitHub issues
- Update issue numbers in xfail markers after creation
- See `issues/` directory for detailed issue descriptions

## [0.1.0] - 2024-01-01

### Added
- Initial release with core functionality
- XGBoost + PAT ensemble models
- CLI with 6 commands
- FastAPI server
- Docker deployment
- Comprehensive test suite (695 tests)