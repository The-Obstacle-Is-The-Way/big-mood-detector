# 📁 Complete Data Files Manifest - Big Mood Detector

This document lists ALL gitignored data files, model weights, and their exact locations for replication.

## 🌳 Complete Directory Structure

```
big-mood-detector/
├── data/
│   ├── cache/                                    # Preprocessed data cache
│   │   └── nhanes_pat_data_subsetNone.npz      # [CRITICAL] 200MB - Cached training data
│   │
│   ├── nhanes/                                   # Raw NHANES data files
│   │   └── 2013-2014/
│   │       ├── DEMO_H.xpt                       # Demographics (3.5MB)
│   │       ├── DPQ_H.xpt                        # Depression questionnaire (1.2MB)
│   │       ├── PAXDAY_H.xpt                     # Day-level activity (0.8MB)
│   │       ├── PAXHD_H.xpt                      # Header data (1.1MB)
│   │       ├── PAXMIN_H.xpt                     # [LARGE] Minute activity (1.5GB)
│   │       ├── RXQ_DRUG.xpt                     # Drug database (45MB)
│   │       ├── RXQ_RX_H.xpt                     # Prescriptions (4.2MB)
│   │       └── SMQRTU_H.xpt                     # Smoking data (1.8MB)
│   │
│   ├── NCHS/                                     # Mortality linkage files
│   │   └── NHANES_2013_2014_MORT_2019_PUBLIC.dat # Mortality data (2.1MB)
│   │
│   ├── baselines/                                # User baseline calibration
│   │   └── [user_id]/                          # Per-user baseline files (created at runtime)
│   │       └── baseline_history.json
│   │
│   └── input/                                    # User health data (optional for testing)
│       ├── apple_export/
│       │   └── export.xml                       # Apple Health export (varies, 100MB-2GB)
│       └── health_auto_export/
│           └── *.json                           # JSON exports (varies)
│
├── model_weights/
│   ├── xgboost/
│   │   └── converted/                           # [REQUIRED] XGBoost models
│   │       ├── XGBoost_DE.json                  # Depression model (15MB)
│   │       ├── XGBoost_HME.json                 # Hypomania model (15MB)
│   │       └── XGBoost_ME.json                  # Mania model (15MB)
│   │
│   └── pat/
│       ├── pretrained/                          # [REQUIRED] Pretrained PAT weights
│       │   ├── PAT-S_29k_weights.h5             # Small model (285KB)
│       │   ├── PAT-M_29k_weights.h5             # Medium model (1MB)
│       │   └── PAT-L_29k_weights.h5             # Large model (1.99MB)
│       │
│       ├── heads/                               # Fine-tuned depression heads
│       │   └── pat_depression_head.pt           # (if trained, ~10MB)
│       │
│       └── pytorch/                             # PyTorch training outputs
│           ├── pat_s_training/
│           │   ├── best_stage1_auc_*.pt
│           │   ├── best_overall_auc_*.pt
│           │   └── training_summary.json
│           ├── pat_m_training/
│           │   └── (similar structure)
│           └── pat_l_training/                  # [IMPORTANT] Current training
│               ├── best_stage1_auc_0.5788.pt    # Best checkpoint (8MB)
│               ├── checkpoint_migration.pt       # Migration checkpoint
│               ├── final_mac_checkpoint.pt       # Final save before migration
│               ├── training_summary.json         # Training metadata
│               └── migration_checkpoint_info.json
│
└── logs/                                        # Application logs
    └── *.log                                    # Runtime logs (created automatically)
```

## 📦 File Categories by Priority

### 🔴 CRITICAL (App won't run without these)
```
model_weights/xgboost/converted/XGBoost_DE.json
model_weights/xgboost/converted/XGBoost_HME.json  
model_weights/xgboost/converted/XGBoost_ME.json
```

### 🟡 IMPORTANT (For PAT functionality)
```
model_weights/pat/pretrained/PAT-S_29k_weights.h5
model_weights/pat/pretrained/PAT-M_29k_weights.h5
model_weights/pat/pretrained/PAT-L_29k_weights.h5
```

### 🟢 TRAINING (For continuing PAT training)
```
data/cache/nhanes_pat_data_subsetNone.npz         # Preprocessed data
model_weights/pat/pytorch/pat_l_training/*.pt     # Checkpoints
```

### 🔵 OPTIONAL (For retraining from scratch)
```
data/nhanes/2013-2014/*.xpt                       # Raw NHANES files
data/NCHS/*.dat                                    # Mortality data
```

## 📊 File Sizes Summary

- **Minimal setup** (XGBoost only): ~45MB
- **With PAT models**: ~48MB  
- **With training cache**: ~250MB
- **Full NHANES raw data**: ~2GB
- **Everything**: ~2.5GB

## 🗂️ Creating Transfer Archives

```bash
cd ~/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector

# 1. Essential models only (45MB)
tar -czf essential_models.tar.gz \
  model_weights/xgboost/converted/*.json

# 2. All models including PAT (48MB)
tar -czf all_models.tar.gz \
  model_weights/xgboost/converted/*.json \
  model_weights/pat/pretrained/*.h5

# 3. Training continuation pack (260MB)
tar -czf training_pack.tar.gz \
  data/cache/nhanes_pat_data_subsetNone.npz \
  model_weights/pat/pytorch/pat_l_training/ \
  model_weights/pat/pretrained/*.h5 \
  model_weights/xgboost/converted/*.json

# 4. Complete data archive (2.5GB)
tar -czf complete_data.tar.gz \
  data/cache/ \
  data/nhanes/ \
  data/NCHS/ \
  model_weights/
```

## 🔧 Directory Creation Script

Save this as `create_data_dirs.sh` on your PC:

```bash
#!/bin/bash
# Create all required directories for Big Mood Detector

mkdir -p data/cache
mkdir -p data/nhanes/2013-2014
mkdir -p data/NCHS
mkdir -p data/baselines
mkdir -p data/input/apple_export
mkdir -p data/input/health_auto_export

mkdir -p model_weights/xgboost/converted
mkdir -p model_weights/pat/pretrained
mkdir -p model_weights/pat/heads
mkdir -p model_weights/pat/pytorch/pat_s_training
mkdir -p model_weights/pat/pytorch/pat_m_training
mkdir -p model_weights/pat/pytorch/pat_l_training

mkdir -p logs

echo "✅ All directories created!"
```

## 🌐 Download URLs (if available)

### NHANES Data Files
Base URL: `https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/`
- DEMO_H.XPT
- DPQ_H.XPT
- PAXDAY_H.XPT
- PAXHD_H.XPT
- PAXMIN_H.XPT
- RXQ_RX_H.XPT
- SMQRTU_H.XPT

### NCHS Mortality
URL: `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/`
- NHANES_2013_2014_MORT_2019_PUBLIC.dat

### Model Weights
- Check GitHub releases or contact maintainers
- XGBoost models are from Seoul National University study
- PAT weights are from Dartmouth's release

## ✅ Verification Commands

After extracting on PC:

```bash
# Check file existence
ls -la model_weights/xgboost/converted/*.json
ls -la model_weights/pat/pretrained/*.h5
ls -la data/cache/*.npz

# Check file sizes
du -sh data/cache/nhanes_pat_data_subsetNone.npz
du -sh model_weights/pat/pytorch/pat_l_training/

# Run setup verification
python scripts/verify_setup.py
```

## 📝 Notes

1. **Paths are case-sensitive** on Mac but not Windows - maintain exact case
2. **Use forward slashes** even on Windows (Python handles conversion)
3. **Create directories first** before extracting archives
4. The `.npz` cache file is **crucial** for training - without it, you need to regenerate from raw XPT files
5. **Git LFS** not used currently - all weights are gitignored

---

This manifest provides complete replication instructions for all data files needed by the Big Mood Detector.