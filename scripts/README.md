# Scripts Directory

Organized utility scripts for the Big Mood Detector project.

## Directory Structure

```
scripts/
├── pat_training/          # Canonical PAT model training scripts
├── validation/            # Pipeline validation and testing
├── maintenance/           # System maintenance utilities
├── github/               # GitHub automation scripts
└── archive/              # Deprecated/old scripts for reference
```

## Main Scripts

### Performance & Benchmarking
- `benchmark_ensemble.py` - Benchmarks ensemble model prediction performance
- `benchmark_xml_parser.py` - Tests XML parsing speed for large Apple Health exports
- `process_large_xml.py` - Processes large XML files with progress tracking

### Data Quality & Validation
- `check_no_sleep_percentage.sh` - CI guard against sleep calculation regression
- `check_sleep_features.sh` - Validates sleep feature extraction accuracy
- `migrate_user_ids_to_hashed.py` - Migrates plaintext user IDs to hashed versions

### Build & Maintenance
- `generate_licenses.py` - Generates third-party license file from dependencies
- `generate_requirements.py` - Generates requirements.txt from pyproject.toml

## PAT Training Scripts (`/pat_training/`)

### 🎯 Canonical Training Scripts
- **`train_pat_canonical.py`** - Unified launcher for all model sizes
- **`train_pat_s_canonical.py`** - PAT-Small training (0.560 AUC target)
- **`train_pat_m_canonical.py`** - PAT-Medium training (0.559 AUC target)
- **`train_pat_l_run_now.py`** - PAT-Large training (0.610 AUC target)

### Advanced Training
- `train_pat_l_advanced.py` - PAT-L with advanced options (resume, schedulers)
- `train_pat_l_fixed.py` - PAT-L with normalization fixes

### Training Utilities
- `debug_pat_training.py` - Debug training issues
- `analyze_pat_training.py` - Analyze training results
- `monitor_training.py` - Real-time training monitoring
- `test_pat_pytorch_smoke.py` - Smoke tests for PyTorch implementation
- `test_pat_weight_parity.py` - Verify TF→PyTorch weight conversion

### Shell Scripts (Being Phased Out)
- `run_pat_*.sh` - Legacy launchers (use Python scripts instead)
- `start_pat_l_training.sh` - Legacy tmux launcher

## Validation Scripts (`/validation/`)

- `validate_full_pipeline.py` - Full pipeline validation
- `test_full_36_features.py` - Validate all 36 Seoul features
- `analyze_data_coverage.py` - Analyze data completeness
- `golden_run_june_2025.sh` - Golden run for regression testing
- `test_prediction_pipeline.py` - Test prediction accuracy

## Maintenance Scripts (`/maintenance/`)

- `convert_xgboost_models.py` - Convert XGBoost model formats
- `download_model_weights.py` - Download pretrained weights
- `setup_reference_repos.sh` - Setup reference repositories
- `test-in-docker.sh` - Test in Docker environment
- `fix_type_errors.py` - Fix type annotation issues

## GitHub Integration (`/github/`)

- `create_github_issues_for_todos.sh` - Create GitHub issues from TODOs
- `extract_todos_for_github.py` - Extract TODOs for issue creation
- `check_todo_format.py` - Validate TODO format
- `create-tech-debt-issues.sh` - Create tech debt issues

## Usage Examples

### Train PAT Models
```bash
# Train all models
python scripts/pat_training/train_pat_canonical.py --model-size all

# Train specific model
python scripts/pat_training/train_pat_canonical.py --model-size small

# Resume PAT-L training with advanced settings (in tmux)
python scripts/pat_training/train_pat_l_run_now.py \
  --resume model_weights/pat/pytorch/pat_l_training/best_stage1_auc_0.5788.pt \
  --unfreeze-last-n 4 \
  --head-lr 3e-4 \
  --encoder-lr 3e-5 \
  --epochs 60 \
  --scheduler cosine \
  --patience 10 \
  --output-dir model_weights/pat/pytorch/pat_l_retry
```

### Benchmark Performance
```bash
# Test XML parsing speed
python scripts/benchmark_xml_parser.py data/export.xml

# Benchmark ensemble predictions
python scripts/benchmark_ensemble.py
```

### Validation
```bash
# Run golden validation
cd scripts/validation && ./golden_run_june_2025.sh

# Validate full pipeline
python scripts/validation/validate_full_pipeline.py
```

## Archive Structure

```
archive/
├── pat_training_old/     # Previous PAT training implementations
├── experiments/          # One-off experimental scripts
├── analysis_tools/       # Old analysis utilities
├── one_off_fixes/       # Import fixes and temporary patches
├── deprecated/          # Deprecated implementations
├── needs_fixing/        # Scripts requiring updates
└── old_backups/         # Old backup files
```

## Important Notes

### PAT Training
- All canonical PAT training scripts include the normalization fix from v0.4.0
- Training scripts automatically handle MPS (Apple Silicon) acceleration
- Default hyperparameters are tuned for depression detection on NHANES data
- Models are saved to `model_weights/pat/pytorch/pat_[s|m|l]_training/`

### Data Requirements
- PAT training requires cached NHANES data at `data/cache/nhanes_pat_data_subsetNone.npz`
- Run data preparation scripts first if cache doesn't exist
- Validation scripts expect test data in standard locations

### Best Practices
- Always use canonical scripts for production training
- Check training logs for normalization warnings
- Monitor AUC progression - expect 0.50+ in Stage 1
- Use tmux for long-running training sessions

## Recent Updates (v0.4.0 - July 24, 2025)

✅ **Major Improvements:**
- Created canonical training scripts for all PAT model sizes
- Fixed critical normalization bug in PAT-L training
- Achieved paper parity: PAT-S (0.56), PAT-M (0.54), PAT-L (0.58+)
- Cleaned and organized scripts directory structure
- Archived 50+ old/one-off scripts

🔧 **Training Status:**
- PAT-S: Ready with canonical script
- PAT-M: Ready with canonical script  
- PAT-L: Actively training with improved hyperparameters