# Model Weights Directory

## Structure

```
model_weights/
├── production/      # ✅ USE THESE FOR THE APP
│   ├── pat_conv_l_v0.5929.pth    # Best PAT model (24.3MB)
│   └── pat_conv_l_v0.5929.json   # Model metadata
│
├── pretrained/      # Original pretrained weights (DO NOT MODIFY)
│   ├── PAT-L_29k_weights.h5      # Large model (7.7MB)
│   ├── PAT-M_29k_weights.h5      # Medium model (3.9MB)
│   └── PAT-S_29k_weights.h5      # Small model (1.1MB)
│
└── xgboost/         # XGBoost models for mood prediction
```

## Usage

### Loading Production Models

```python
# PAT-Conv-L for depression detection
MODEL_PATH = "model_weights/production/pat_conv_l_v0.5929.pth"

# Load with PyTorch
import torch
model = torch.load(MODEL_PATH, map_location='cpu')
```

### Model Information

#### PAT-Conv-L v0.5929
- **Trained**: July 25, 2025
- **Task**: Depression classification (PHQ-9 ≥ 10)
- **Performance**: 0.5929 AUC
- **Architecture**: PAT-L with Conv1d patch embedding
- **Parameters**: 1,984,289

## Important Notes

1. **Always use `/production/` models** for the application
2. **Never modify `/pretrained/`** - these are the original PAT weights
3. **Old `/pat/` directory** should be removed after migration

## Model Versions

| Model | Version | AUC | Status |
|-------|---------|-----|--------|
| PAT-Conv-L | v0.5929 | 0.5929 | Production ✅ |
| PAT-L | v0.5888 | 0.5888 | Superseded |
| XGBoost | v1.0 | 0.80-0.98* | Production ✅ |

*XGBoost AUC varies by mood state (depression/mania/hypomania)
