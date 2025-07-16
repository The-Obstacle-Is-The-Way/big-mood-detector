"""
ML Models Infrastructure

This module provides the machine learning model implementations for the
Big Mood Detector system, including PAT and XGBoost models.
"""

from .pat_model import PATFeatureExtractor, PATModel
from .xgboost_models import XGBoostMoodPredictor

# Single source of truth for PAT implementation
# The DirectPATModel is used internally by PATModel
# Other prototypes (pat_architecture.py, pat_custom_layers.py) are deprecated

__all__ = [
    "PATModel",
    "PATFeatureExtractor",
    "XGBoostMoodPredictor",
]
