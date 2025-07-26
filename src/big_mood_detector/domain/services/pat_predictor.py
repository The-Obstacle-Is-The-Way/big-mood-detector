"""
PAT Predictor Interface

Domain interface for PAT-based mood prediction.
Based on PAT paper capabilities: binary depression and medication proxy predictions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PATBinaryPredictions:
    """
    Binary predictions from PAT model.

    Based on PAT paper's actual training tasks:
    - Depression: PHQ-9 >= 10 (AUC 0.589)
    - Benzodiazepine usage: Proxy for mood stabilization (AUC 0.767)

    Note: PAT cannot distinguish hypomania from mania.
    """
    depression_probability: float  # P(PHQ-9 >= 10)
    benzodiazepine_probability: float  # P(taking benzodiazepines)
    confidence: float  # Overall prediction confidence


class PATPredictorInterface(ABC):
    """
    Interface for PAT-based binary predictors.

    PAT was trained on binary classification tasks, NOT 3-class mood prediction.
    This interface reflects PAT's actual capabilities from the literature.
    """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for predictions."""
        pass

    @abstractmethod
    def predict_from_embeddings(self, embeddings: NDArray[np.float32]) -> PATBinaryPredictions:
        """
        Get binary predictions from PAT embeddings.

        Args:
            embeddings: 96-dimensional PAT embedding vector

        Returns:
            PATBinaryPredictions with depression and medication probabilities

        Raises:
            ValueError: If embeddings have wrong dimensions
        """
        pass

    @abstractmethod
    def predict_depression(self, embeddings: NDArray[np.float32]) -> float:
        """
        Predict probability of depression (PHQ-9 >= 10).

        Args:
            embeddings: 96-dimensional PAT embedding vector

        Returns:
            Probability between 0 and 1
        """
        pass

    @abstractmethod
    def predict_medication_proxy(self, embeddings: NDArray[np.float32]) -> float:
        """
        Predict probability of benzodiazepine usage.

        This is a proxy for mood stabilization, NOT direct mania detection.

        Args:
            embeddings: 96-dimensional PAT embedding vector

        Returns:
            Probability between 0 and 1
        """
        pass
