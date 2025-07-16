# Scripts Directory

This directory contains utility scripts for data analysis, model setup, and testing.

## Working Scripts

### Data Analysis
- **`analyze_data_coverage.py`** - Analyzes data coverage and quality in JSON files
- **`analyze_xml_record_types.py`** - Discovers and counts record types in Apple Health XML exports

### Model Setup
- **`download_model_weights.py`** - Downloads pretrained PAT and XGBoost model weights

### DLMO Testing
- **`test_dlmo.py`** - Tests DLMO calculation with different sleep patterns
- **`test_dlmo_comparison.py`** - Comprehensive DLMO testing with visualization
- **`test_dlmo_validation.py`** - Validates DLMO calculations against known test cases

### Pipeline Validation
- **`validate_full_pipeline.py`** - Comprehensive end-to-end pipeline validation

## Archive

### Deprecated Scripts (`archive/deprecated/`)
Scripts that are no longer needed:
- Synthetic data generators (real data is available)
- Incomplete experimental scripts
- Redundant test scripts

### Scripts Needing Fixes (`archive/needs_fixing/`)
Scripts that could be useful but need import fixes or updates:
- XML parser debugging scripts
- Pipeline comparison scripts
- Full ML pipeline tests

## Usage Examples

### Analyze your Apple Health export:
```bash
python analyze_xml_record_types.py
```

### Download model weights:
```bash
python download_model_weights.py
```

### Test DLMO calculations:
```bash
python test_dlmo.py
```

### Validate the complete pipeline:
```bash
python validate_full_pipeline.py
```