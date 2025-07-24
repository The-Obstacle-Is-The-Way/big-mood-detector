# 📦 PC Migration Checklist - Big Mood Detector

## 🚨 CRITICAL: Save Training State First!

Before ANY migration, save your current training checkpoint:

```bash
# In your tmux session, add this to your training script:
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_auc': val_auc,
    'train_loss': train_loss,
    'config': {
        'batch_size': batch_size,
        'learning_rate': lr_stage2,
        'unfrozen_blocks': 2,
        'stage': 2,
        'epoch_in_stage': 5
    }
}, 'model_weights/pat/pytorch/pat_l_training/checkpoint_before_migration.pt')
```

## 📁 Files to Transfer to PC

### 1. **Training Data & Cache** (~4-8GB)
```
data/
├── cache/
│   └── nhanes_pat_data_subsetNone.npz  # CRITICAL - cached training data
├── nhanes/2013-2014/
│   ├── DEMO_H.xpt                      # Demographics
│   ├── DPQ_H.xpt                       # Depression questionnaire  
│   ├── PAXDAY_H.xpt                    # Day-level activity
│   ├── PAXHD_H.xpt                     # Header data
│   ├── PAXMIN_H.xpt                    # Minute-level activity
│   ├── RXQ_DRUG.xpt                    # Drug info
│   ├── RXQ_RX_H.xpt                    # Prescriptions
│   └── SMQRTU_H.xpt                    # Smoking data
└── NCHS/
    └── NHANES_2013_2014_MORT_2019_PUBLIC.dat  # Mortality data
```

### 2. **Model Weights** (~500MB)
```
model_weights/
├── xgboost/converted/
│   ├── XGBoost_DE.json                 # Depression model
│   ├── XGBoost_HME.json                # Hypomania model
│   └── XGBoost_ME.json                 # Mania model
├── pat/
│   ├── pretrained/
│   │   ├── PAT-S_29k_weights.h5       # Small (285KB)
│   │   ├── PAT-M_29k_weights.h5       # Medium (1MB)
│   │   └── PAT-L_29k_weights.h5       # Large (2MB)
│   └── pytorch/
│       └── pat_l_training/
│           ├── best_stage1_auc_0.5788.pt
│           ├── checkpoint_before_migration.pt  # NEW - save this!
│           └── training_summary.json
```

### 3. **Reference Data** (optional but helpful)
```
reference_repos/
└── mood_ml-main/                       # If you have the reference implementation
```

### 4. **Your Health Data** (if testing)
```
data/input/
├── apple_export/export.xml             # Your Apple Health export
└── health_auto_export/*.json           # Health Auto Export files
```

## 🗂️ Creating the Transfer Archive

```bash
# Create a transfer directory
mkdir -p ~/Desktop/big_mood_transfer
cd ~/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector

# 1. Save git state
git bundle create ~/Desktop/big_mood_transfer/big-mood-detector.bundle --all
echo "Current branch: $(git branch --show-current)" > ~/Desktop/big_mood_transfer/git_info.txt
git log --oneline -20 >> ~/Desktop/big_mood_transfer/git_info.txt

# 2. Copy data files (excluding git repo)
rsync -av --progress \
  --include="data/cache/nhanes_pat_data_*.npz" \
  --include="data/nhanes/2013-2014/*.xpt" \
  --include="data/NCHS/*.dat" \
  --include="model_weights/**/*.json" \
  --include="model_weights/**/*.h5" \
  --include="model_weights/**/*.pt" \
  --include="model_weights/**/training_summary.json" \
  --exclude=".git/" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  data/ model_weights/ \
  ~/Desktop/big_mood_transfer/

# 3. Save environment info
pip freeze > ~/Desktop/big_mood_transfer/requirements_mac.txt
conda env export > ~/Desktop/big_mood_transfer/environment_mac.yml
python --version > ~/Desktop/big_mood_transfer/python_version.txt
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')" >> ~/Desktop/big_mood_transfer/python_version.txt

# 4. Create manifest
cat > ~/Desktop/big_mood_transfer/MANIFEST.txt << EOF
Big Mood Detector PC Migration
Created: $(date)
Mac PyTorch device: mps
Training interrupted at: Stage 2, Epoch 5, AUC ~0.55

Files included:
- big-mood-detector.bundle (git repository)
- data/ (NHANES training data, cached arrays)
- model_weights/ (all model files and checkpoints)
- requirements_mac.txt (pip packages)
- environment_mac.yml (conda environment)
- git_info.txt (current branch and commits)

CRITICAL PATHS TO PRESERVE:
- data/cache/nhanes_pat_data_subsetNone.npz
- model_weights/pat/pytorch/pat_l_training/best_stage1_auc_0.5788.pt
- model_weights/xgboost/converted/*.json

EOF

# 5. Create the zip
cd ~/Desktop
zip -r big_mood_transfer_$(date +%Y%m%d).zip big_mood_transfer/
```

## 🖥️ PC Setup Instructions

### 1. **Extract and Clone**
```powershell
# Extract the zip
# Clone from bundle
git clone big-mood-detector.bundle big-mood-detector
cd big-mood-detector

# Or fresh clone from GitHub
git clone https://github.com/Clarity-Digital-Twin/big-mood-detector.git
cd big-mood-detector
```

### 2. **Setup Python Environment**
```powershell
# Option A: Conda (recommended)
conda create -n bigmood python=3.12
conda activate bigmood

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project
pip install -e ".[dev,ml,monitoring]"

# Option B: Docker (easiest)
docker compose up --build
```

### 3. **Restore Data Files**
```powershell
# Copy from transfer archive
xcopy /E /I transfer\data data
xcopy /E /I transfer\model_weights model_weights
```

### 4. **Verify Setup**
```powershell
python scripts/verify_setup.py
```

### 5. **Resume Training**
```python
# Modify train_pat_l_run_now.py to load checkpoint:
checkpoint = torch.load('model_weights/pat/pytorch/pat_l_training/checkpoint_before_migration.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## ⚠️ Path Compatibility Issues

### Windows Path Fixes
```python
# In your code, ensure using pathlib:
from pathlib import Path

# Bad (Mac-specific)
cache_path = "data/cache/nhanes_pat_data_subsetNone.npz"

# Good (cross-platform)
cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
```

### Common Windows Issues
1. **Long paths**: Enable long path support or use shorter paths
2. **Case sensitivity**: Windows is case-insensitive, Mac isn't
3. **Line endings**: Set `git config core.autocrlf false`
4. **Backslashes**: Always use forward slashes or pathlib

## 📊 Training State Summary

Current training state to document:
- Model: PAT-L
- Stage: 2 (unfrozen last 2 blocks)
- Epoch: 5/30 in Stage 2 (35/60 total)
- Best AUC: 0.5788 (from Stage 1)
- Current AUC: ~0.55 (stagnant)
- Learning rate: 1e-4
- Batch size: 32
- Device: MPS (Mac) → CUDA (PC)

## 🚀 Performance Expectations on PC

With RTX 4090:
- Mac M1/M2: ~110s/epoch
- RTX 4090: ~15-20s/epoch (5-7x faster)
- Larger batch sizes possible (64-128)
- Mixed precision training available

## 📝 Pre-Migration Checklist

- [ ] Save current training checkpoint
- [ ] Commit all code changes
- [ ] Document current hyperparameters
- [ ] Export conda/pip environment
- [ ] Create data archive (~5-10GB)
- [ ] Test verify_setup.py locally
- [ ] Note any running experiments
- [ ] Save training logs

---

**Remember**: The most critical file is `data/cache/nhanes_pat_data_subsetNone.npz`. 
Without it, you'll need to regenerate from raw NHANES files, which takes time!