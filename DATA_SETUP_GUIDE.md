# 📁 Data Setup Guide - Big Mood Detector

This guide helps you set up all required data files after cloning the repository.

## 🚨 Quick Start Checklist

After cloning, you need to download/create these files:

```
✅ Required for basic functionality:
□ model_weights/xgboost/converted/*.json   # XGBoost models (3 files)
□ Your health data (JSON or XML format)    # From Apple Health or Health Auto Export

🔧 Optional for enhanced features:
□ model_weights/pat/pretrained/*.h5        # PAT transformer weights (3 files)
□ data/cache/nhanes_pat_data_*.npz         # For PAT training only
□ data/nhanes/2013-2014/*.xpt              # For training new models only
```

## 📋 Detailed Setup Instructions

### 1. Download Model Weights (Required)

The application needs pre-trained model weights that are too large for Git.

#### Option A: Use the download script
```bash
python scripts/maintenance/download_model_weights.py
```

#### Option B: Manual download
Download from the releases page and place in correct directories:

**XGBoost Models** (Required - 3 files):
- `XGBoost_DE.json` → `model_weights/xgboost/converted/`
- `XGBoost_HME.json` → `model_weights/xgboost/converted/`
- `XGBoost_ME.json` → `model_weights/xgboost/converted/`

**PAT Models** (Optional - 3 files):
- `PAT-S_29k_weights.h5` (285KB) → `model_weights/pat/pretrained/`
- `PAT-M_29k_weights.h5` (1MB) → `model_weights/pat/pretrained/`
- `PAT-L_29k_weights.h5` (2MB) → `model_weights/pat/pretrained/`

### 2. Prepare Your Health Data

#### Apple Health Export (XML)
1. On iPhone: Health app → Profile → Export All Health Data
2. Extract the zip file
3. Place `export.xml` in: `data/input/apple_export/`

#### Health Auto Export (JSON)
1. Use the Health Auto Export app
2. Export as JSON files
3. Place all JSON files in: `data/input/health_auto_export/`

### 3. Directory Structure

Create these directories if they don't exist:

```bash
mkdir -p model_weights/xgboost/converted
mkdir -p model_weights/pat/pretrained
mkdir -p model_weights/pat/heads
mkdir -p model_weights/pat/pytorch
mkdir -p data/input/apple_export
mkdir -p data/input/health_auto_export
mkdir -p data/cache
mkdir -p data/baselines
mkdir -p data/nhanes/2013-2014
mkdir -p data/nhanes/processed
```

### 4. For Developers: Training Data

If you want to train/fine-tune models, you'll need NHANES data:

#### Download NHANES Data
```bash
# Physical activity data
wget https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/PAXMIN_H.XPT -P data/nhanes/2013-2014/

# Depression questionnaire
wget https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DPQ_H.XPT -P data/nhanes/2013-2014/

# Demographics
wget https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT -P data/nhanes/2013-2014/
```

## 🔍 Verify Your Setup

Run this verification script:

```python
# save as verify_setup.py
from pathlib import Path

def check_file(path, required=True):
    exists = Path(path).exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = " (REQUIRED)" if required else " (optional)"
    print(f"{status} {path}{req_text}")
    return exists

print("🔍 Checking Big Mood Detector Setup...\n")

# Required files
print("Required Model Files:")
all_required = True
all_required &= check_file("model_weights/xgboost/converted/XGBoost_DE.json")
all_required &= check_file("model_weights/xgboost/converted/XGBoost_HME.json")
all_required &= check_file("model_weights/xgboost/converted/XGBoost_ME.json")

print("\nOptional PAT Models:")
check_file("model_weights/pat/pretrained/PAT-S_29k_weights.h5", False)
check_file("model_weights/pat/pretrained/PAT-M_29k_weights.h5", False)
check_file("model_weights/pat/pretrained/PAT-L_29k_weights.h5", False)

print("\nHealth Data (need at least one):")
has_json = check_file("data/input/health_auto_export/", False) and \
           any(Path("data/input/health_auto_export/").glob("*.json"))
has_xml = check_file("data/input/apple_export/export.xml", False)

if all_required and (has_json or has_xml):
    print("\n✅ Setup complete! You can run predictions.")
else:
    print("\n❌ Setup incomplete. Download required files.")
```

## 🐳 Docker Setup

For consistent cross-platform setup, use Docker:

```dockerfile
# Dockerfile snippet for data volumes
VOLUME ["/app/data", "/app/model_weights"]

# docker-compose.yml
volumes:
  - ./data:/app/data
  - ./model_weights:/app/model_weights
```

## 🌐 Environment Variables

Optional environment variables for custom paths:

```bash
# Custom model weights directory
export BIG_MOOD_PAT_WEIGHTS_DIR=/path/to/weights

# Custom data directory
export BIGMOOD_DATA_DIR=/path/to/data
```

## ⚠️ Common Issues

### "Model file not found"
- Ensure XGBoost JSON files are in `model_weights/xgboost/converted/`
- File names must match exactly (case-sensitive)

### "No health data found"
- Place files in the exact directories specified
- XML file must be named `export.xml`
- JSON files must have `.json` extension

### "PAT weights not found" (warning only)
- This is optional - app will work without PAT
- If you need PAT, download the H5 files

## 📊 What Happens with Missing Files?

| Missing File | Impact | Fallback |
|--------------|--------|----------|
| XGBoost models | ❌ App won't run | None - required |
| PAT weights | ⚠️ No PAT predictions | Uses XGBoost only |
| Baseline data | ⚠️ No personalization | Uses population defaults |
| NHANES data | ⚠️ Can't train models | N/A - training only |

## 🔐 Security Note

**NEVER commit these files to Git:**
- Personal health data (XML/JSON)
- Model predictions
- Clinical reports
- Any files containing health information

These are all listed in `.gitignore` for your protection.

---

For more help, see the main [README.md](README.md) or open an issue on GitHub.