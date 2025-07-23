"""
PAT Predictor Interface

Domain interface for PAT-based mood prediction.
This is a pure domain concept with no infrastructure dependencies.
"""

from abc import ABC, abstractmethod

import numpy as np

from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class PATPredictorInterface(ABC):
    """
    Interface for PAT-based mood predictors.
    
    This interface defines the contract that any PAT predictor implementation
    must follow. It takes PAT embeddings and returns mood predictions.
    """
    
    @abstractmethod
    def predict_from_embeddings(self, embeddings: np.ndarray) -> MoodPrediction:
        """
        Predict mood state from PAT embeddings.
        
        Args:
            embeddings: 96-dimensional PAT embedding vector
            
        Returns:
            MoodPrediction with depression, hypomanic, and manic risks
            
        Raises:
            ValueError: If embeddings have wrong dimensions
        """
        pass