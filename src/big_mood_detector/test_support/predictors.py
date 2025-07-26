"""Test predictors for clean integration testing."""

from typing import Any

import numpy as np

from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class ConstantMoodPredictor:
    """A predictor that returns constant values for testing."""
    
    def __init__(
        self, 
        depression: float = 0.3, 
        hypomanic: float = 0.2, 
        manic: float = 0.1, 
        confidence: float = 0.85
    ) -> None:
        self.depression = depression
        self.hypomanic = hypomanic
        self.manic = manic
        self.confidence = confidence
        self.is_loaded = True
    
    def predict(self, features: np.ndarray | list[float]) -> MoodPrediction:
        """Return constant prediction regardless of input."""
        return MoodPrediction(
            depression_risk=self.depression,
            hypomanic_risk=self.hypomanic,
            manic_risk=self.manic,
            confidence=self.confidence
        )