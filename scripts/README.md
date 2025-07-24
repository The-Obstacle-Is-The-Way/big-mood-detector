# Scripts Directory Organization

This directory contains various utility scripts for development, testing, and maintenance of the Big Mood Detector project.

## Directory Structure

### `/pat_training/`
PAT (Pretrained Actigraphy Transformer) model training and analysis scripts:
- **Active Training Scripts:**
  - `train_pat_l_run_now.py` - Current PAT-L training with normalization fix
  - `train_pat_l_advanced.py` - Advanced training with progressive unfreezing
  - `train_pat_l_fixed.py` - Fixed training script with proper normalization
- **Analysis & Debugging:**
  - `analyze_pat_training.py` - Analyze training checkpoints and suggest improvements
  - `debug_pat_training.py` - Debug data quality and model initialization
  - `monitor_training.py` - Monitor ongoing training progress
  - `debug_pat_architecture.py` - Debug PAT architecture issues
  - `debug_nhanes_columns.py` - Debug NHANES data column issues
- **Testing:**
  - `test_pat_pytorch_smoke.py` - Smoke tests for PyTorch PAT implementation
  - `test_pat_weight_parity.py` - Test weight conversion parity
  - `test_depression_load.py` - Test depression data loading
- **Utilities:**
  - `train_all_pat_models.sh` - Train all PAT model sizes

### `/experiments/`
Experimental scripts for testing new features:
- `test_dlmo.py`, `test_dlmo_comparison.py`, `test_dlmo_validation.py` - DLMO feature experiments
- `test_data_processing.py` - Data processing experiments
- `test_xml_complete_flow.py`, `test_xml_end_to_end.py` - XML processing tests

### `/validation/`
Validation and verification scripts:
- `validate_full_pipeline.py` - Full pipeline validation
- `test_full_36_features.py` - Validate all 36 Seoul features
- `analyze_data_coverage.py` - Analyze data completeness
- `golden_run_june_2025.sh` - Golden run for regression testing

### `/maintenance/`
System maintenance and setup scripts:
- `convert_xgboost_models.py` - Convert XGBoost model formats
- `download_model_weights.py` - Download pretrained weights
- `setup_reference_repos.sh` - Setup reference repositories
- `test-in-docker.sh` - Test in Docker environment

### `/utilities/`
Import and code fixing utilities:
- `fix_*.py` - Various import fixing scripts
- `restore_test_imports.py` - Restore test imports

### `/github/`
GitHub integration scripts:
- `create_github_issues_for_todos.sh` - Create GitHub issues from TODOs
- `extract_todos_for_github.py` - Extract TODOs for issue creation
- `check_todo_format.py` - Validate TODO format
- `create-tech-debt-issues.sh` - Create tech debt issues

### `/archive/`
Archived scripts no longer actively used:
- `/deprecated/` - Deprecated scripts from earlier versions
- `/needs_fixing/` - Scripts that need updates to work with current codebase
- `/pat_training_old/` - Older PAT training scripts superseded by current versions

## Key Scripts

### Data Processing
- `process_large_xml.py` - Process large XML files efficiently
- `benchmark_xml_parser.py` - Benchmark XML parsing performance

### Testing & Validation
- `check_sleep_features.sh` - Validate sleep feature calculations
- `assert_feature_schema.py` - Assert feature schema compliance
- `benchmark_ensemble.py` - Benchmark ensemble model performance
- `test_api_ensemble.py` - Test API ensemble endpoints

### Documentation
- `generate_licenses.py` - Generate license documentation
- `generate_requirements.py` - Generate requirements files
- `inventory_docs.py` - Inventory documentation files

### Utilities
- `migrate_user_ids_to_hashed.py` - Migrate to hashed user IDs
- `trace_sleep_math.py` - Trace sleep calculation logic
- `archive_docs.sh`, `clean_remaining_docs.sh` - Documentation cleanup

## Usage

Most Python scripts can be run directly:
```bash
python scripts/pat_training/monitor_training.py
```

Shell scripts should be executed with appropriate permissions:
```bash
./scripts/validation/golden_run_june_2025.sh
```

## Recent Updates (July 2025)

- Reorganized PAT training scripts into dedicated folder
- Fixed PAT-L training normalization issue (see `pat_training/train_pat_l_run_now.py`)
- Archived old training scripts that have been superseded
- Created clear directory structure for better organization