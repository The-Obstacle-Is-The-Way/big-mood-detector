"""
Central path configuration for the project.

Provides absolute paths that work regardless of where the code is executed from.
"""

from pathlib import Path

# Project root is 4 levels up from this file
# src/big_mood_detector/core/paths.py -> src/big_mood_detector -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Key directories
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "model_weights"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model weight paths
XGBOOST_PRETRAINED_DIR = MODEL_WEIGHTS_DIR / "xgboost" / "pretrained"
XGBOOST_FINETUNED_DIR = MODEL_WEIGHTS_DIR / "xgboost" / "finetuned"
PAT_PRETRAINED_DIR = MODEL_WEIGHTS_DIR / "pat" / "pretrained"
PAT_FINETUNED_DIR = MODEL_WEIGHTS_DIR / "pat" / "finetuned"

# Data paths
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
UPLOADS_DIR = DATA_DIR / "uploads"
TEMP_DIR = DATA_DIR / "temp"

# Ensure critical directories exist
for directory in [DATA_DIR, LOGS_DIR, OUTPUT_DIR, UPLOADS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
