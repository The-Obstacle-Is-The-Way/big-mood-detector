"""
ML Models Infrastructure

This module provides the machine learning model implementations for the
Big Mood Detector system, including PAT and XGBoost models.
"""

from .xgboost_models import XGBoostMoodPredictor

# Conditional import for PAT model (requires TensorFlow)
try:
    from .pat_model import PATFeatureExtractor, PATModel
    PAT_AVAILABLE = True
except ImportError:
    from typing import Any
    PAT_AVAILABLE = False
    PATModel = Any  # type: ignore
    PATFeatureExtractor = Any  # type: ignore

# Single source of truth for PAT implementation
# The DirectPATModel is used internally by PATModel
# Other prototypes (pat_architecture.py, pat_custom_layers.py) are deprecated

__all__ = [
    "XGBoostMoodPredictor",
    "PAT_AVAILABLE",
]

# Only export PAT if available
if PAT_AVAILABLE:
    __all__.extend(["PATModel", "PATFeatureExtractor"])
