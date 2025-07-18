# Model Weights Directory

This directory contains pretrained and fine-tuned model weights for the Big Mood Detector system.

## ⚠️ Important Notice

Model weight files (`.h5`, `.pkl`, `.joblib`, etc.) are **NOT** included in the repository due to their large size. You must download or obtain them separately.

## Directory Structure

```
model_weights/
├── pat/                    # Pretrained Actigraphy Transformer models
│   ├── pretrained/        # Original PAT weights from Dartmouth
│   │   ├── PAT-S_29k_weights.h5  (285 KB)
│   │   ├── PAT-M_29k_weights.h5  (1.0 MB)
│   │   └── PAT-L_29k_weights.h5  (1.99 MB)
│   └── finetuned/         # Fine-tuned PAT models for specific tasks
│       └── (your fine-tuned models here)
│
└── xgboost/               # XGBoost models for mood prediction
    ├── pretrained/        # CANONICAL: Production models (descriptive names)
    │   ├── depression_model.pkl    # From XGBoost_DE.pkl
    │   ├── hypomanic_model.pkl     # From XGBoost_HME.pkl
    │   └── manic_model.pkl         # From XGBoost_ME.pkl
    ├── converted/         # Version-updated models (original research names)
    │   ├── XGBoost_DE.pkl + .json
    │   ├── XGBoost_HME.pkl + .json
    │   └── XGBoost_ME.pkl + .json
    └── finetuned/         # Fine-tuned or retrained models
        └── (your custom models here)
```

## Obtaining Model Weights

### PAT Models

Download the pretrained PAT weights from the official repository:

1. **Option 1: Direct Download Links** (from PAT paper repository)
   - [PAT-S (Small)](https://www.dropbox.com/scl/fi/12ip8owx1psc4o7b2uqff/PAT-S_29k_weights.h5?rlkey=ffaf1z45a74cbxrl7c9i2b32h&st=mfk6f0y5&dl=1)
   - [PAT-M (Medium)](https://www.dropbox.com/scl/fi/hlfbni5bzsfq0pynarjcn/PAT-M_29k_weights.h5?rlkey=frbkjtbgliy9vq2kvzkquruvg&st=mxc4uet9&dl=1)
   - [PAT-L (Large)](https://www.dropbox.com/scl/fi/exk40hu1nxc1zr1prqrtp/PAT-L_29k_weights.h5?rlkey=t1e5h54oob0e1k4frqzjt1kmz&st=7a20pcox&dl=1)

2. **Option 2: Copy from Reference Repository**
   ```bash
   # If you have the reference repos cloned
   cp reference_repos/Pretrained-Actigraphy-Transformer/model_weights/PAT-*_29k_weights.h5 model_weights/pat/pretrained/
   ```

3. **Option 3: Download Script**
   ```bash
   # Run the download script (if available)
   python scripts/download_model_weights.py --model pat
   ```

### XGBoost Models

The XGBoost models are from the Seoul National University study (Yun et al., 2022).

**Option 1: Copy from Reference Repository** (Recommended)
```bash
# If you have the reference repos cloned
cp reference_repos/mood_ml/XGBoost_DE.pkl model_weights/xgboost/pretrained/depression_model.pkl
cp reference_repos/mood_ml/XGBoost_HME.pkl model_weights/xgboost/pretrained/hypomanic_model.pkl
cp reference_repos/mood_ml/XGBoost_ME.pkl model_weights/xgboost/pretrained/manic_model.pkl
```

**Option 2: Download from Original Repository**
- Clone from: https://github.com/mcqeen1207/mood_ml
- Files needed: XGBoost_DE.pkl, XGBoost_HME.pkl, XGBoost_ME.pkl

**Option 3: Train Your Own Models**
```bash
# Train XGBoost models on your data
python scripts/train_xgboost_models.py
```

## Usage in Code

### Loading PAT Models

```python
from pathlib import Path
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

# Initialize model
model = PATModel(model_size="medium")

# Load pretrained weights
weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")
if model.load_pretrained_weights(weights_path):
    print("Model loaded successfully!")
else:
    print("Failed to load model weights. Please download them first.")
```

### Loading XGBoost Models

```python
import joblib
from pathlib import Path

# Load depression model
model_path = Path("model_weights/xgboost/pretrained/depression_model.pkl")
if model_path.exists():
    depression_model = joblib.load(model_path)
else:
    print("Model not found. Please train or download it first.")
```

## Model Information

### PAT Models
- **PAT-S (Small)**: 285K parameters, 18-min patches, 1 encoder layer
- **PAT-M (Medium)**: 1.0M parameters, 18-min patches, 2 encoder layers  
- **PAT-L (Large)**: 1.99M parameters, 9-min patches, 4 encoder layers

All PAT models are pretrained on 29,307 participants from NHANES using masked autoencoding.

#### Important Note on PAT Weights
The PAT H5 files contain encoder-only weights extracted after masked autoencoder pretraining. These files:
- Do NOT include model configuration (architecture definition)
- Use custom attention layer naming (separate Q/K/V projections)
- Were saved using TensorFlow's custom layer implementations
- Require exact architecture reconstruction to load properly

**Single Source of Truth (SSOT)**: The PAT paper and original Jupyter notebooks define the authoritative architecture. The H5 files are legitimate pretrained weights from Dartmouth's foundation model research.

### XGBoost Models

**SOURCE OF TRUTH**: The `pretrained/` directory contains the canonical production models with descriptive names.

Model Mapping:
| Production Name | Original Research Name | Abbreviation Meaning |
|-----------------|------------------------|---------------------|
| depression_model.pkl | XGBoost_DE.pkl | DE = Depression Episode |
| hypomanic_model.pkl | XGBoost_HME.pkl | HME = Hypomanic Episode |
| manic_model.pkl | XGBoost_ME.pkl | ME = Manic Episode |

Models use 36 engineered features from sleep, activity, and circadian patterns.

## Security Notes

1. **Never commit model weights to Git** - They are in `.gitignore`
2. **Verify checksums** when downloading models from external sources
3. **Keep personal fine-tuned models** in the `finetuned/` subdirectories
4. **Document model versions** and training dates for reproducibility

## Troubleshooting

If you get "Model not found" errors:
1. Check that you've downloaded the weights to the correct directory
2. Verify file permissions (weights should be readable)
3. Ensure the file extension matches (`.h5` for PAT, `.pkl` for XGBoost)
4. Check the exact filename - it's case-sensitive

### PAT Loading Issues
If PAT models fail to load with "No model config found" or layer mismatch errors:
- This is expected behavior with the current H5 files
- The system will gracefully fall back to XGBoost-only predictions
- Full ensemble mode requires reconstructing the exact PAT architecture
- XGBoost-only mode still provides excellent performance (AUC 0.80-0.98)

**Current Status**: PAT integration is a work in progress. The foundation weights are valid but require custom loading logic that matches the original TensorFlow implementation exactly.

## Contributing

When adding new models:
1. Place them in the appropriate subdirectory
2. Update this README with download/training instructions
3. Add the model info to the relevant section
4. Ensure the model files are in `.gitignore`