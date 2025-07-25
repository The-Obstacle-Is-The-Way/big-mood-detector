# PAT Depression Training Guide

## Overview

This repository implements PAT (Pretrained Actigraphy Transformer) models for depression prediction using NHANES 2013-2014 data. We achieve competitive results compared to the paper's benchmarks.

## Current Results

| Model | Our Best AUC | Paper's AUC | Gap |
|-------|--------------|-------------|-----|
| PAT-L | 0.580 | 0.610 | 0.030 |
| PAT-Conv-L | 0.592 | 0.625 | 0.033 |

## Quick Start

### 1. Environment Setup
```bash
# Python 3.12+ required
python3.12 -m venv .venv-wsl
source .venv-wsl/bin/activate

# Install dependencies (order matters!)
pip install 'numpy<2.0'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers h5py scikit-learn pandas matplotlib tqdm
pip install -e ".[dev,ml,monitoring]"
```

### 2. Data Setup
- NHANES data is auto-downloaded on first run
- Pretrained weights required in `model_weights/pat/pretrained/`
  - PAT-L_29k_weights.h5 (7.7MB)
  - PAT-M_29k_weights.h5 (3.9MB)  
  - PAT-S_29k_weights.h5 (1.1MB)

### 3. Training

**Stable baseline (recommended):**
```bash
python scripts/pat_training/train_pat_conv_l_simple.py
```

**Multiple runs with seeds:**
```bash
python scripts/pat_training/train_pat_stable.py --model pat-l --conv --runs 3
```

## Key Scripts

### Core Training Scripts
- `train_pat_conv_l_simple.py` - Simple training that achieves 0.592 AUC
- `train_pat_stable.py` - SSOT for reproducible multi-run experiments
- `train_pat_l_corrected.py` - PAT-L with corrected normalization

### Model Implementations
- `src/big_mood_detector/infrastructure/ml_models/pat_pytorch.py` - PyTorch PAT encoder
- Conv variant implemented in training scripts

## Technical Details

### What Works
1. **Data normalization**: Global StandardScaler (mean=0, std=1)
2. **Simple optimizer**: AdamW with uniform LR=1e-4
3. **Direct fine-tuning**: Skip linear probing phase
4. **Conv architecture**: 1D Conv with kernel_size=patch_size=9

### Known Issues
- Complex LP→FT training with scheduler issues
- Gap to paper results (~0.03 AUC) - likely needs hyperparameter tuning

### Training Configuration
```python
# Proven configuration
optimizer = AdamW(lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(T_max=15_epochs)
batch_size = 32
gradient_clip = 1.0
```

## Repository Structure
```
big-mood-detector/
├── scripts/
│   └── pat_training/          # Active training scripts
├── src/
│   └── big_mood_detector/
│       └── infrastructure/
│           └── ml_models/     # PAT PyTorch implementation
├── model_weights/
│   └── pat/
│       └── pretrained/        # Place weights here
├── data/
│   └── cache/                 # Auto-generated NHANES cache
└── logs/                      # Training logs
```

## Contact

For questions about PAT implementation or to collaborate on reaching paper parity, please open an issue or contact the maintainers.

## References

- Original PAT paper: [Self-Supervised Learning of Accelerometer Data Provides New Insights for Sleep and Its Association with Mortality](https://arxiv.org/abs/2305.09930)
- NHANES dataset: [2013-2014 cycle](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013)