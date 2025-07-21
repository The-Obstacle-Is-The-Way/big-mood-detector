"""
Booster Wrapper for XGBoost Compatibility

Wraps raw XGBoost Booster objects to provide scikit-learn compatible predict_proba method.
This is needed because Booster objects loaded from JSON don't have predict_proba.
"""

import numpy as np


class BoosterPredictProbaWrapper:
    """Wrapper to add predict_proba to raw XGBoost Booster objects."""

    def __init__(self, booster):
        """Initialize with a raw XGBoost Booster.

        Args:
            booster: Raw xgboost.Booster object
        """
        self.booster = booster

    def predict(self, X):
        """Raw prediction using the booster.

        Args:
            X: Feature matrix

        Returns:
            Raw prediction scores
        """
        import xgboost as xgb

        # Convert to DMatrix if needed
        if not isinstance(X, xgb.DMatrix):
            X = xgb.DMatrix(X)

        return self.booster.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities.

        For binary classification, returns probabilities for both classes.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        # Get raw predictions (these are already probabilities for binary classification)
        raw_predictions = self.predict(X)

        # For binary classification, create 2-class probability array
        n_samples = len(raw_predictions)
        proba = np.zeros((n_samples, 2))

        # Class 0 probability = 1 - positive class probability
        proba[:, 0] = 1 - raw_predictions
        # Class 1 probability = positive class probability
        proba[:, 1] = raw_predictions

        return proba

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped booster."""
        return getattr(self.booster, name)
