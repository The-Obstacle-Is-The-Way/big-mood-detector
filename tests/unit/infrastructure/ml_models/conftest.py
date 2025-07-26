"""
Fixtures for PAT model tests.

Provides clean test fixtures without private attributes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
    NHANESNormalizer,
)
from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
    ProductionPATLoader,
)


@pytest.fixture
def mock_normalizer():
    """Create a mock normalizer for testing."""
    normalizer = MagicMock(spec=NHANESNormalizer)
    # Make transform return the input unchanged by default
    normalizer.transform.side_effect = lambda x: x
    return normalizer


@pytest.fixture
def pat_loader_no_normalization(mock_normalizer):
    """
    Create a PAT loader that bypasses normalization.
    
    This is the clean replacement for setting _test_skip_normalization.
    """
    loader = ProductionPATLoader(
        skip_loading=True,
        normalizer=mock_normalizer
    )
    return loader


@pytest.fixture
def pat_loader_with_weights(tmp_path):
    """Create a PAT loader with mock weight file."""
    # Create mock weights file
    weights_path = tmp_path / "model_weights" / "pat" / "conv_depression"
    weights_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = weights_path / "pat_conv_l_depression_epoch10_auc0.5929.pt"
    
    # Create minimal checkpoint
    checkpoint = {
        'model_state_dict': {
            'encoder.patch_embed.conv.weight': torch.randn(96, 1, 15),
            'encoder.patch_embed.conv.bias': torch.randn(96),
            'head.weight': torch.randn(1, 96),
            'head.bias': torch.randn(1),
        },
        'epoch': 10,
        'best_auc': 0.5929,
    }
    torch.save(checkpoint, checkpoint_path)
    
    # Create loader pointing to this path
    loader = ProductionPATLoader(model_path=checkpoint_path)
    return loader


@pytest.fixture
def activity_sequence_7days():
    """Generate a valid 7-day activity sequence."""
    # 10080 minutes = 7 days * 24 hours * 60 minutes
    return np.random.rand(10080).astype(np.float32)


@pytest.fixture
def pat_embeddings_96d():
    """Generate valid 96-dimensional PAT embeddings."""
    return np.random.randn(96).astype(np.float32)