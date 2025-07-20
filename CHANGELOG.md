# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Date range filtering for XML processing with `--days-back` and `--date-range` CLI options
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

### Removed
- Magic HR/HRV defaults (70 bpm / 50 ms) that would skew personal baselines
- Deprecated datetime.utcnow() usage throughout codebase

## [0.1.0] - 2024-01-01

### Added
- Initial release with core functionality
- XGBoost + PAT ensemble models
- CLI with 6 commands
- FastAPI server
- Docker deployment
- Comprehensive test suite (695 tests)