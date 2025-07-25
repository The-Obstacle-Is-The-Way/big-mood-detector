"""
PAT Encoder Interface

Domain interface for encoding activity sequences into embeddings.
This abstraction allows the domain to work with PAT without depending
on specific ML framework implementations.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class PATEncoderInterface(ABC):
    """
    Interface for encoding activity sequences into embeddings.

    The PAT (Pretrained Actigraphy Transformer) encoder converts
    7-day activity sequences into 96-dimensional embeddings that
    capture behavioral patterns.
    """

    @abstractmethod
    def encode(self, activity_sequence: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Encode activity sequence into embeddings.

        Args:
            activity_sequence: 7-day activity data as (7, 1440) array
                              where each day has 1440 minute-level readings

        Returns:
            96-dimensional embedding vector

        Raises:
            ValueError: If input shape is incorrect
        """
        pass

    @abstractmethod
    def validate_sequence(self, activity_sequence: NDArray[np.float32]) -> bool:
        """
        Validate that activity sequence has correct format.

        Args:
            activity_sequence: Activity data to validate

        Returns:
            True if valid, False otherwise
        """
        pass
