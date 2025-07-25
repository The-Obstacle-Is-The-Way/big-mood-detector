#!/usr/bin/env python3
"""Fix the model weights directory structure to be clean and consistent."""

import os
import shutil
from pathlib import Path

def fix_structure():
    """Fix the model weights structure."""
    
    print("🔧 Fixing model weights structure...")
    
    # 1. Move pretrained PAT weights to the root pretrained folder
    old_pretrained = Path("model_weights/pat/pretrained")
    new_pretrained = Path("model_weights/pretrained")
    
    if old_pretrained.exists():
        for file in old_pretrained.glob("*.h5"):
            dest = new_pretrained / file.name
            shutil.copy2(file, dest)
            print(f"  Moved: {file.name} → model_weights/pretrained/")
    
    # 2. The structure should be:
    # model_weights/
    # ├── production/      # Production-ready models
    # │   ├── pat_conv_l_v0.5929.pth
    # │   └── pat_conv_l_v0.5929.json
    # ├── pretrained/      # Original pretrained weights
    # │   ├── PAT-L_29k_weights.h5
    # │   ├── PAT-M_29k_weights.h5
    # │   └── PAT-S_29k_weights.h5
    # └── xgboost/         # XGBoost models
    
    print("\n✅ Correct structure:")
    print("model_weights/")
    print("├── production/      # Production models (USE THESE)")
    print("│   ├── pat_conv_l_v0.5929.pth")
    print("│   └── pat_conv_l_v0.5929.json")
    print("├── pretrained/      # Original PAT weights") 
    print("│   ├── PAT-L_29k_weights.h5")
    print("│   ├── PAT-M_29k_weights.h5")
    print("│   └── PAT-S_29k_weights.h5")
    print("└── xgboost/         # XGBoost models")
    
    print("\n⚠️  Old structure to remove (after verification):")
    print("model_weights/pat/  # This whole directory can be archived")

def create_model_weights_readme():
    """Create a clear README for model weights."""
    readme = """# Model Weights Directory

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
"""
    
    with open("model_weights/README.md", "w") as f:
        f.write(readme)
    print("✅ Created model_weights/README.md")

if __name__ == "__main__":
    fix_structure()
    create_model_weights_readme()