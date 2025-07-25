"""
PAT Depression Head

Binary classification head for current depression state assessment.
Based on PAT paper: predicts PHQ-9 >= 10 from past 7 days of activity.

TEMPORAL NOTE: This predicts CURRENT depression state, not future risk.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from big_mood_detector.domain.services.pat_predictor import (
    PATBinaryPredictions,
    PATPredictorInterface,
)
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)


class PATDepressionHead(nn.Module):
    """
    Binary classification head for depression detection.

    Maps 96-dimensional PAT embeddings to probability of current depression.
    Based on PAT paper: PHQ-9 >= 10 (AUC 0.589 in paper).
    """

    def __init__(self, input_dim: int = 96, hidden_dim: int = 256, dropout: float = 0.5):
        """Initialize depression head.

        Args:
            input_dim: Dimension of PAT embeddings (default 96)
            hidden_dim: Hidden layer dimension (default 256)
            dropout: Dropout rate for regularization (default 0.5)
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Binary output
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through depression head.

        Args:
            embeddings: PAT embeddings of shape (batch_size, 96)

        Returns:
            Logits of shape (batch_size, 1)
        """
        logits: torch.Tensor = self.mlp(embeddings)
        return logits


class PATDepressionPredictor(PATPredictorInterface):
    """
    Current depression state predictor using PAT embeddings.

    IMPORTANT: This assesses CURRENT depression (based on past 7 days),
    NOT future depression risk. For future risk, use XGBoost.
    """

    def __init__(self, model_path: Path | None = None):
        """Initialize depression predictor.

        Args:
            model_path: Path to saved model weights (optional for testing)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depression_head = PATDepressionHead().to(self.device)

        if model_path and model_path.exists():
            logger.info(f"Loading PAT depression head from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.depression_head.load_state_dict(state_dict)
        else:
            logger.info("Initialized PAT depression head with random weights")

        self.depression_head.eval()

    def predict_from_embeddings(self, embeddings: NDArray[np.float32]) -> PATBinaryPredictions:
        """Get current state predictions from PAT embeddings.

        Args:
            embeddings: 96-dimensional PAT embedding vector

        Returns:
            PATBinaryPredictions with current depression probability

        Raises:
            ValueError: If embeddings have wrong dimensions
        """
        if embeddings.shape[0] != 96:
            raise ValueError(f"Expected 96-dim embeddings, got {embeddings.shape[0]}")

        # Get depression probability
        depression_prob = self.predict_depression(embeddings)

        # For now, only depression head is implemented
        # Benzodiazepine set to 0.5 (unknown) until implemented
        return PATBinaryPredictions(
            depression_probability=depression_prob,
            benzodiazepine_probability=0.5,  # Not implemented yet
            confidence=self._calculate_confidence(depression_prob)
        )

    def predict_depression(self, embeddings: NDArray[np.float32]) -> float:
        """Predict probability of current depression (PHQ-9 >= 10).

        This is CURRENT state based on past 7 days, not future prediction.

        Args:
            embeddings: 96-dimensional PAT embedding vector

        Returns:
            Probability between 0 and 1
        """
        # Convert to tensor and add batch dimension
        embeddings_tensor = torch.from_numpy(embeddings).float().unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.depression_head(embeddings_tensor)
            probability = torch.sigmoid(logits).squeeze().item()

        return probability

    def predict_medication_proxy(self, embeddings: NDArray[np.float32]) -> float:
        """Predict probability of benzodiazepine usage.

        NOT IMPLEMENTED YET - returns 0.5 (unknown).

        Args:
            embeddings: 96-dimensional PAT embedding vector

        Returns:
            0.5 (placeholder until implemented)
        """
        logger.warning("Benzodiazepine head not implemented yet")
        return 0.5

    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence based on how far from 0.5 the prediction is.

        Args:
            probability: Binary prediction probability

        Returns:
            Confidence score between 0 and 1
        """
        # Confidence is high when prediction is far from 0.5
        distance_from_uncertain = abs(probability - 0.5) * 2
        return distance_from_uncertain

    def save_model(self, save_path: Path) -> None:
        """Save model weights to disk.

        Args:
            save_path: Path to save model weights
        """
        torch.save(self.depression_head.state_dict(), save_path)
        logger.info(f"Saved PAT depression head to {save_path}")
