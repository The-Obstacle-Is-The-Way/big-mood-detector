# Training Output Structure Guide

## Current State (MESSY)
- Logs scattered in: `/logs/`, `/docs/archive/pat_experiments/`
- Models in: `/model_weights/pat/pytorch/` with inconsistent names
- No clear separation between experiments and production models

## Proposed Clean Structure

```
big-mood-detector/
├── training/                      # All training-related files
│   ├── experiments/              # Experimental runs (not for production)
│   │   ├── archived/            # Old experiments (compressed)
│   │   └── active/              # Current experiments
│   │       └── 2025-07-25_pat_conv_l/
│   │           ├── config.yaml
│   │           ├── training.log
│   │           └── checkpoints/
│   │
│   ├── logs/                     # Canonical training logs
│   │   └── pat_conv_l_v0.5929_20250725.log
│   │
│   └── results/                  # Training results and analysis
│       └── pat_conv_l_v0.5929_summary.md
│
├── model_weights/                # Production-ready models only
│   ├── production/              # Models ready for use
│   │   └── pat_conv_l_v0.5929.pth  # Our best model
│   │
│   └── pretrained/              # Original pretrained weights
│       └── PAT-L_29k_weights.h5
│
└── docs/
    └── training/                # Training documentation
        ├── PAT_CONV_L_TRAINING_GUIDE.md
        └── REPRODUCIBILITY.md
```

## Naming Convention

**Model files**: `{model_type}_v{auc}_{date}.{ext}`
- Example: `pat_conv_l_v0.5929_20250725.pth`

**Log files**: `{model_type}_training_{date}_{time}.log`
- Example: `pat_conv_l_training_20250725_104438.log`

**Experiment folders**: `{date}_{model_type}_{experiment_name}/`
- Example: `2025-07-25_pat_conv_l_simplified/`

## Clean Up Plan

1. **Keep Production Model**:
   - `pat_conv_l_simple_best.pth` → `model_weights/production/pat_conv_l_v0.5929.pth`

2. **Archive Everything Else**:
   - Move all other `.pth` files to `training/experiments/archived/`
   - Compress old logs into `training/experiments/archived/logs_before_20250725.tar.gz`

3. **Document Key Result**:
   - Create `training/results/pat_conv_l_v0.5929_summary.md`
   - Include all training parameters, results, and reproduction steps

4. **Remove Duplicates**:
   - Delete redundant model files with lower AUC
   - Clean up scattered logs

## Environment Variables

Add to `.env`:
```bash
TRAINING_DIR="./training"
MODEL_WEIGHTS_DIR="./model_weights/production"
EXPERIMENT_DIR="./training/experiments/active"
```

## Logging Configuration

Update logging to use structured paths:
```python
# In training scripts
import os
from datetime import datetime

def get_log_path(model_type: str, experiment: bool = False):
    if experiment:
        base = os.getenv("EXPERIMENT_DIR", "./training/experiments/active")
        return f"{base}/{datetime.now():%Y-%m-%d}_{model_type}/"
    else:
        base = os.getenv("TRAINING_DIR", "./training/logs")
        return f"{base}/{model_type}_training_{datetime.now():%Y%m%d_%H%M%S}.log"
```

## Next Steps

1. Execute cleanup script to reorganize files
2. Update all training scripts to use new structure
3. Create comprehensive documentation
4. Set up automated archival for old experiments