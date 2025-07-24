# Scripts Directory

Utility scripts for development, maintenance, and experimentation.

## Directory Structure

```
scripts/
├── experiments/       # Test scripts and experimental code
│   ├── test_data_processing.py
│   ├── test_dlmo*.py          # DLMO validation experiments
│   └── test_xml_*.py          # XML processing tests
│
├── maintenance/       # Setup and maintenance utilities
│   ├── setup_reference_repos.sh
│   ├── download_model_weights.py
│   ├── convert_xgboost_models.py
│   ├── fix_type_errors.py
│   └── test-in-docker.sh
│
├── validation/        # Data analysis and validation
│   ├── analyze_data_coverage.py
│   ├── analyze_xml_record_types.py
│   └── validate_full_pipeline.py
│
└── archive/          # Deprecated/old scripts
    ├── deprecated/
    └── needs_fixing/
```

## Key Scripts

### Maintenance
- `setup_reference_repos.sh` - Clone reference repositories for development
- `download_model_weights.py` - Download pre-trained model weights
- `convert_xgboost_models.py` - Convert XGBoost models from pickle to JSON

### Validation
- `validate_full_pipeline.py` - End-to-end pipeline validation
- `analyze_data_coverage.py` - Analyze health data coverage and gaps

### Experiments
- `test_data_processing.py` - Test data processing pipeline
- `test_xml_*.py` - Various XML processing experiments