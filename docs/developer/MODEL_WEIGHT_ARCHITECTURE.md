# Model Weight Architecture and Management

This document provides the definitive guide to model weight management in the Big Mood Detector project.

## Model Weight Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Reference Repository                          │
│  reference_repos/mood_ml/                                       │
│  - XGBoost_DE.pkl  (Depression Episode)                        │
│  - XGBoost_HME.pkl (Hypomanic Episode)                         │
│  - XGBoost_ME.pkl  (Manic Episode)                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼ Convert (optional)
┌─────────────────────────────────────────────────────────────────┐
│                    Converted Models                              │
│  model_weights/xgboost/converted/                               │
│  - XGBoost_DE.pkl  + XGBoost_DE.json                          │
│  - XGBoost_HME.pkl + XGBoost_HME.json                         │
│  - XGBoost_ME.pkl  + XGBoost_ME.json                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼ Rename
┌─────────────────────────────────────────────────────────────────┐
│                    Production Models                             │
│  model_weights/xgboost/pretrained/                             │
│  - depression_model.pkl                                         │
│  - hypomanic_model.pkl                                          │
│  - manic_model.pkl                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Model Naming Convention

### Original Research Names
From Seoul National University Study:
- **DE** = Depression Episode
- **HME** = Hypomanic Episode  
- **ME** = Manic Episode

### Production Names
Used in infrastructure layer:
- **depression_model.pkl** (from XGBoost_DE.pkl)
- **hypomanic_model.pkl** (from XGBoost_HME.pkl)
- **manic_model.pkl** (from XGBoost_ME.pkl)

## Model Loading Pipeline

### 1. Infrastructure Layer (`XGBoostMoodPredictor`)
```python
# Location: infrastructure/ml_models/xgboost_models.py
# Default path: model_weights/xgboost/pretrained/
# Expected files:
#   - depression_model.pkl
#   - hypomanic_model.pkl
#   - manic_model.pkl
```

### 2. Domain Layer (`MoodPredictor`)
```python
# Location: domain/services/mood_predictor.py
# Default path: model_weights/xgboost/pretrained/
# Uses same renamed files as infrastructure layer
```

### 3. Application Layer (`MoodPredictionPipeline`)
```python
# Location: application/use_cases/process_health_data_use_case.py
# Uses infrastructure XGBoostMoodPredictor
# Default: model_weights/xgboost/pretrained/
```

## PAT Model Weights

### Naming Convention
- **PAT-S_29k_weights.h5** = Small model (285K parameters)
- **PAT-M_29k_weights.h5** = Medium model (1M parameters)
- **PAT-L_29k_weights.h5** = Large model (1.99M parameters)

### Loading
- Default path: `model_weights/pat/pretrained/`
- Environment override: `BIG_MOOD_PAT_WEIGHTS_DIR`

## Model Conversion Process

### Why Convert?
1. Eliminate XGBoost version warnings
2. Create JSON format for cross-language compatibility
3. Ensure reproducible predictions

### Conversion Script
```bash
python scripts/convert_xgboost_models.py
```

Creates both `.pkl` and `.json` formats in `converted/` directory.

### Full Model Journey
1. **Original**: `reference_repos/mood_ml/XGBoost_DE.pkl` (from research)
2. **Converted**: `model_weights/xgboost/converted/XGBoost_DE.pkl` (version updated)
3. **Production**: `model_weights/xgboost/pretrained/depression_model.pkl` (renamed for clarity)

The `converted/` directory is an intermediate step and is NOT used by the application.

## Source of Truth

### Model Architecture
- **36 features** as defined in Seoul National University paper
- **XGBoost hyperparameters** from original training
- **PAT architecture** from Harvard/Google paper

### Model Weights
1. **Original weights**: `reference_repos/mood_ml/*.pkl`
2. **Production weights**: `model_weights/xgboost/pretrained/*.pkl`
3. **PAT weights**: `model_weights/pat/pretrained/PAT-*_29k_weights.h5`

### Directory Structure
```
model_weights/
├── xgboost/
│   ├── pretrained/      # Production models (renamed)
│   ├── converted/       # Version-updated models (original names)
│   └── finetuned/       # User-specific fine-tuned models
└── pat/
    ├── pretrained/      # Harvard pretrained weights
    └── finetuned/       # User-specific fine-tuned weights
```

## Best Practices

### Adding New Models
1. Place original weights in `reference_repos/`
2. Run conversion script if needed
3. Copy to `pretrained/` with descriptive names
4. Update this documentation

### Model Loading
1. Always use infrastructure layer for loading
2. Specify explicit paths in configs
3. Log model loading success/failure
4. Validate loaded models before use

### Version Control
1. Track model metadata (not weights) in git
2. Use git-lfs or external storage for weights
3. Document model provenance
4. Tag model versions with releases

## Troubleshooting

### Common Issues
1. **FileNotFoundError**: Check naming convention and path
2. **Version warnings**: Run conversion script
3. **Wrong predictions**: Verify feature order matches training

### Debugging Commands
```bash
# List all model files
find model_weights -name "*.pkl" -o -name "*.h5"

# Check model file sizes
ls -la model_weights/xgboost/pretrained/

# Verify PAT weights location
echo $BIG_MOOD_PAT_WEIGHTS_DIR
```

## Future Improvements

1. **Model Registry**: Implement centralized model registry
2. **Versioning**: Add semantic versioning to models
3. **Validation**: Add model checksum verification
4. **Monitoring**: Track model performance over time