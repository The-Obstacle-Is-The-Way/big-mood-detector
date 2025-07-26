"""
ML Models Infrastructure

This module provides the machine learning model implementations for the
Big Mood Detector system, including PAT and XGBoost models.
"""

from .pat_conv_depression_model import SimplePATConvLModel

# We now use pure PyTorch implementation - no TensorFlow dependency!
from .pat_production_loader import ProductionPATLoader
from .pat_pytorch import PATDepressionNet
from .xgboost_models import XGBoostMoodPredictor

# For backward compatibility with tests
PAT_AVAILABLE = True  # Always available with PyTorch

# Alias for tests expecting old names
PATModel = ProductionPATLoader
PATFeatureExtractor = ProductionPATLoader

__all__ = [
    "XGBoostMoodPredictor",
    "PAT_AVAILABLE",
    "ProductionPATLoader",
    "PATDepressionNet",
    "SimplePATConvLModel",
    "PATModel",  # Backward compatibility
    "PATFeatureExtractor",  # Backward compatibility
]
