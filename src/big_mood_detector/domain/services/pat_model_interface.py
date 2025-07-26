"""
PAT Model Interface

Defines the contract that all PAT model implementations must follow.
This ensures clean separation between TensorFlow and PyTorch implementations.
"""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class PATModelInterface(Protocol):
    """
    Interface for all PAT model implementations.
    
    Both TensorFlow and PyTorch implementations must satisfy this contract.
    """
    
    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded."""
        ...
    
    def extract_features(self, sequence: Any) -> NDArray[np.float32]:
        """
        Extract 96-dimensional embeddings from activity sequence.
        
        Args:
            sequence: Either PATSequence object or raw activity array (10080,)
            
        Returns:
            96-dimensional embedding vector
        """
        ...
    
    def predict(self, features: NDArray[np.float32]) -> Any:
        """
        Make predictions from features.
        
        Args:
            features: Input features (shape depends on model)
            
        Returns:
            Model-specific prediction object
        """
        ...