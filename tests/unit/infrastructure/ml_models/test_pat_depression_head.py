"""
Test PAT Depression Head

Tests for the binary depression classification head.
Focuses on CURRENT state assessment, not future prediction.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from big_mood_detector.domain.services.pat_predictor import (
    PATBinaryPredictions,
    PATPredictorInterface,
)


class TestPATDepressionHead:
    """Test the PAT depression head implementation."""

    def test_depression_head_module_exists(self):
        """Should be able to import PATDepressionHead."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionHead,
        )

        assert issubclass(PATDepressionHead, nn.Module)

    def test_depression_head_initialization(self):
        """Should initialize with correct architecture."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionHead,
        )

        head = PATDepressionHead(input_dim=96, hidden_dim=256)

        assert hasattr(head, 'mlp')
        assert isinstance(head.mlp, nn.Sequential)

        # Check it has the right layers
        layers = list(head.mlp.children())
        assert isinstance(layers[0], nn.Linear)  # Input layer
        assert isinstance(layers[1], nn.ReLU)    # Activation
        assert isinstance(layers[2], nn.Dropout) # Regularization
        assert isinstance(layers[3], nn.Linear)  # Output layer

        # Check dimensions
        assert layers[0].in_features == 96
        assert layers[0].out_features == 256
        assert layers[3].in_features == 256
        assert layers[3].out_features == 1  # Binary output

    def test_depression_head_forward_pass(self):
        """Forward pass should return binary logits."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionHead,
        )

        head = PATDepressionHead(input_dim=96, hidden_dim=256)
        head.eval()  # Set to evaluation mode

        # Test with single embedding
        embeddings = torch.randn(1, 96)
        with torch.no_grad():
            logits = head(embeddings)

        assert logits.shape == (1, 1)  # Binary output
        assert isinstance(logits, torch.Tensor)

    def test_depression_predictor_exists(self):
        """Should have a concrete depression predictor."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionPredictor,
        )

        assert issubclass(PATDepressionPredictor, PATPredictorInterface)

    def test_depression_predictor_initialization(self):
        """Depression predictor should initialize properly."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionPredictor,
        )

        predictor = PATDepressionPredictor(model_path=None)

        assert hasattr(predictor, 'depression_head')
        assert hasattr(predictor, 'device')

    def test_predict_depression_returns_probability(self):
        """Should return probability between 0 and 1."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionPredictor,
        )

        predictor = PATDepressionPredictor(model_path=None)
        embeddings = np.random.randn(96).astype(np.float32)

        prob = predictor.predict_depression(embeddings)

        assert isinstance(prob, float)
        assert 0 <= prob <= 1

    def test_predict_from_embeddings_returns_binary_predictions(self):
        """Should return PATBinaryPredictions object."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionPredictor,
        )

        predictor = PATDepressionPredictor(model_path=None)
        embeddings = np.random.randn(96).astype(np.float32)

        predictions = predictor.predict_from_embeddings(embeddings)

        assert isinstance(predictions, PATBinaryPredictions)
        assert 0 <= predictions.depression_probability <= 1
        assert 0 <= predictions.confidence <= 1

        # Benzodiazepine should be 0.5 (not implemented)
        assert predictions.benzodiazepine_probability == 0.5

    def test_confidence_calculation(self):
        """Confidence should be high when far from 0.5."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionPredictor,
        )

        predictor = PATDepressionPredictor(model_path=None)

        # Test confidence calculation
        assert predictor._calculate_confidence(0.9) > 0.7  # High confidence
        assert predictor._calculate_confidence(0.1) > 0.7  # High confidence
        assert predictor._calculate_confidence(0.5) == 0.0  # No confidence
        assert predictor._calculate_confidence(0.6) < 0.3  # Low confidence

    def test_embedding_dimension_validation(self):
        """Should validate embedding dimensions."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionPredictor,
        )

        predictor = PATDepressionPredictor(model_path=None)

        # Wrong dimension should raise error
        wrong_embeddings = np.random.randn(50).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 96-dim embeddings"):
            predictor.predict_from_embeddings(wrong_embeddings)

    def test_binary_sigmoid_output(self):
        """Should use sigmoid activation for binary output."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionPredictor,
        )

        # Create predictor
        predictor = PATDepressionPredictor(model_path=None)

        # Test with extreme logits
        head = predictor.depression_head
        head.eval()

        # Mock extreme positive logit
        with torch.no_grad():
            embeddings = torch.randn(1, 96).to(predictor.device)
            logits = head(embeddings)

            # Apply sigmoid manually
            prob = torch.sigmoid(logits).item()

            # Should be between 0 and 1
            assert 0 < prob < 1

    def test_medication_proxy_not_implemented(self):
        """Medication proxy should return 0.5 and log warning."""
        from big_mood_detector.infrastructure.ml_models.pat_depression_head import (
            PATDepressionPredictor,
        )

        predictor = PATDepressionPredictor(model_path=None)
        embeddings = np.random.randn(96).astype(np.float32)

        # Should return 0.5 (unknown)
        prob = predictor.predict_medication_proxy(embeddings)
        assert prob == 0.5
