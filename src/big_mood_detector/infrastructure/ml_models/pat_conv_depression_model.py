"""
PAT-Conv Depression Model

Specialized model for loading trained PAT-Conv-L depression classifier.
This handles the architectural differences between standard PAT and PAT-Conv models.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

from big_mood_detector.infrastructure.ml_models.pat_pytorch import (
    ConvPatchEmbedding,
    PATPyTorchEncoder,
    PATBlock,
    SinusoidalPositionalEmbedding,
)

logger = logging.getLogger(__name__)


class SimplePATConvLModel(nn.Module):
    """
    PAT-Conv-L model matching the training script architecture.
    
    This exactly replicates the model structure used in training
    to ensure proper weight loading.
    """
    
    def __init__(self, model_size: str = "large"):
        super().__init__()
        
        # Create encoder with Conv embedding
        self.encoder = PATConvLEncoder(model_size=model_size)
        
        # Simple linear head for binary classification
        self.head = nn.Linear(96, 1)  # 96 -> 1 for binary classification
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        embeddings = self.encoder(x)  # (B, 96)
        logits = self.head(embeddings)  # (B, 1)
        return logits.squeeze()  # (B,)


class PATConvLEncoder(PATPyTorchEncoder):
    """
    PAT-Conv-L encoder with convolutional patch embedding.
    
    Inherits from PATPyTorchEncoder but overrides patch embedding
    to use Conv1d instead of Linear projection.
    """
    
    def __init__(self, model_size: str = "large", dropout: float = 0.1):
        # Initialize parent WITHOUT conv_embedding first
        super().__init__(model_size=model_size, dropout=dropout, conv_embedding=False)
        
        # Replace linear patch embedding with conv variant
        self.patch_embed = ConvPatchEmbedding(
            patch_size=self.config["patch_size"],  # 9 for PAT-L
            embed_dim=self.config["embed_dim"],    # 96 for all PAT models
        )
        
        # Mark as conv model
        self.conv_embedding = True
        
        logger.info(f"Created PAT-Conv-{model_size.upper()} encoder")
        logger.info(f"Patch size: {self.config['patch_size']}, Embed dim: {self.config['embed_dim']}")
        
    def load_tf_weights(self, h5_path: Path) -> bool:
        """
        Load transformer weights from TF file, skipping patch embedding.
        
        Conv models have different patch embedding architecture,
        so we only load the transformer block weights.
        """
        # Call parent but it will skip patch embed for conv models
        return super().load_tf_weights(h5_path)