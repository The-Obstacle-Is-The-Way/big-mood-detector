"""
Test cases for PAT Model Integration

Tests the integration of Pretrained Actigraphy Transformer models
including loading pretrained weights and making predictions.
Following TDD principles - tests written before implementation.
"""

# Check if TensorFlow is available
import importlib.util
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

# Skip all tests in this module if TensorFlow is not available
pytestmark = pytest.mark.skipif(
    not HAS_TENSORFLOW,
    reason="TensorFlow not installed - PAT model tests skipped"
)

class TestPATModel:
    """Test the PAT model wrapper for inference."""

    def test_model_initialization(self):
        """Test basic model initialization."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        # Import will fail initially (TDD)

        model = PATModel(model_size="medium")

        assert model.model_size == "medium"
        assert model.patch_size == 18  # Medium uses 18-minute patches
        assert model.embed_dim == 96
        assert model.is_loaded is False

    def test_model_size_configurations(self):
        """Test different model size configurations match paper specs."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        # Test Small model
        model_s = PATModel(model_size="small")
        assert model_s.patch_size == 18
        assert model_s.encoder_num_heads == 6
        assert model_s.encoder_num_layers == 1

        # Test Medium model
        model_m = PATModel(model_size="medium")
        assert model_m.patch_size == 18
        assert model_m.encoder_num_heads == 12
        assert model_m.encoder_num_layers == 2

        # Test Large model
        model_l = PATModel(model_size="large")
        assert model_l.patch_size == 9
        assert model_l.encoder_num_heads == 12
        assert model_l.encoder_num_layers == 4

    def test_invalid_model_size(self):
        """Test that invalid model size raises error."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        with pytest.raises(ValueError, match="Invalid model size"):
            PATModel(model_size="extra-large")

    @pytest.mark.skip(
        reason="Requires TensorFlow mocking which conflicts with PAT stub"
    )
    @patch("pathlib.Path.exists")
    @patch("tensorflow.keras.models.load_model")
    def test_load_pretrained_weights(self, mock_load_model, mock_exists):
        """Test loading pretrained weights from file."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        # Mock file exists
        mock_exists.return_value = True

        # Mock the loaded model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        model = PATModel(model_size="medium")
        weights_path = Path("model_weights/PAT-M_29k_weights.h5")

        success = model.load_pretrained_weights(weights_path)

        assert success is True
        assert model.is_loaded is True
        mock_load_model.assert_called_once()

    def test_load_pretrained_weights_file_not_found(self):
        """Test handling of missing weights file."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        model = PATModel(model_size="medium")
        fake_path = Path("nonexistent/weights.h5")

        success = model.load_pretrained_weights(fake_path)

        assert success is False
        assert model.is_loaded is False

    @pytest.mark.skip(
        reason="Requires TensorFlow mocking which conflicts with PAT stub"
    )
    @patch("pathlib.Path.exists")
    @patch("tensorflow.keras.models.load_model")
    def test_extract_features_from_sequence(self, mock_load_model, mock_exists):
        """Test extracting features from a PAT sequence."""
        from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        # Mock file exists
        mock_exists.return_value = True

        # Mock model that returns features
        mock_model = MagicMock()
        # Model should return (batch, num_patches, embed_dim)
        num_patches = 560  # 10080 / 18 for medium model
        mock_features = np.random.randn(1, num_patches, 96)
        mock_model.predict.return_value = mock_features
        mock_load_model.return_value = mock_model

        # Create model and load weights
        model = PATModel(model_size="medium")
        model.load_pretrained_weights(Path("fake_weights.h5"))

        # Create test sequence
        sequence = PATSequence(
            end_date=date(2025, 5, 15),
            activity_values=np.random.randn(10080),
            missing_days=[],
            data_quality_score=1.0,
        )

        # Extract features
        features = model.extract_features(sequence)

        assert features is not None
        assert features.shape == (96,)  # Should return 1D feature vector
        mock_model.predict.assert_called_once()

    def test_extract_features_without_loaded_model(self):
        """Test that feature extraction fails gracefully without loaded model."""
        from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        model = PATModel(model_size="medium")

        sequence = PATSequence(
            end_date=date(2025, 5, 15),
            activity_values=np.random.randn(10080),
            missing_days=[],
            data_quality_score=1.0,
        )

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.extract_features(sequence)

    @pytest.mark.skip(
        reason="Requires TensorFlow mocking which conflicts with PAT stub"
    )
    @patch("pathlib.Path.exists")
    @patch("tensorflow.keras.models.load_model")
    def test_batch_feature_extraction(self, mock_load_model, mock_exists):
        """Test extracting features from multiple sequences."""
        from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        # Mock file exists
        mock_exists.return_value = True

        # Mock model
        batch_size = 3
        mock_model = MagicMock()
        # Model should return (batch, num_patches, embed_dim)
        num_patches = 560  # 10080 / 18 for medium model
        mock_features = np.random.randn(batch_size, num_patches, 96)
        mock_model.predict.return_value = mock_features
        mock_load_model.return_value = mock_model

        # Create model
        model = PATModel(model_size="medium")
        model.load_pretrained_weights(Path("fake_weights.h5"))

        # Create test sequences
        sequences = []
        for _ in range(batch_size):
            sequences.append(
                PATSequence(
                    end_date=date(2025, 5, 15),
                    activity_values=np.random.randn(10080),
                    missing_days=[],
                    data_quality_score=1.0,
                )
            )

        # Extract features
        features = model.extract_features_batch(sequences)

        assert features.shape == (batch_size, 96)
        mock_model.predict.assert_called_once()

    def test_prepare_input_for_model(self):
        """Test input preparation matches PAT paper requirements."""
        from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        model = PATModel(model_size="medium")

        sequence = PATSequence(
            end_date=date(2025, 5, 15),
            activity_values=np.random.randn(10080),
            missing_days=[],
            data_quality_score=1.0,
        )

        # Prepare input
        model_input = model._prepare_input(sequence)

        # Should be normalized and shaped correctly
        assert model_input.shape == (1, 10080)  # Batch dimension added
        assert np.abs(np.mean(model_input)) < 0.1  # Should be normalized
        assert np.abs(np.std(model_input) - 1.0) < 0.1

    def test_get_attention_weights(self):
        """Test extracting attention weights for explainability."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        # This would require a more complex mock of the model
        # For now, we'll test the interface exists
        model = PATModel(model_size="medium")

        # Method should exist
        assert hasattr(model, "get_attention_weights")

class TestPATModelIntegration:
    """Test integration with the mood prediction pipeline."""

    @pytest.mark.skip(
        reason="Requires TensorFlow mocking which conflicts with PAT stub"
    )
    @patch("pathlib.Path.exists")
    @patch("tensorflow.keras.models.load_model")
    def test_pat_features_for_mood_prediction(self, mock_load_model, mock_exists):
        """Test that PAT features can be used for mood prediction."""
        from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        # Mock file exists
        mock_exists.return_value = True

        # Mock model
        mock_model = MagicMock()
        # Model should return (batch, num_patches, embed_dim)
        num_patches = 560  # 10080 / 18 for medium model
        mock_features = np.random.randn(1, num_patches, 96)
        mock_model.predict.return_value = mock_features
        mock_load_model.return_value = mock_model

        # Create model
        model = PATModel(model_size="medium")
        model.load_pretrained_weights(Path("fake_weights.h5"))

        # Create sequence
        sequence = PATSequence(
            end_date=date(2025, 5, 15),
            activity_values=np.random.randn(10080),
            missing_days=[],
            data_quality_score=1.0,
        )

        # Extract features
        features = model.extract_features(sequence)

        # Features should be suitable for downstream tasks
        assert len(features) == 96
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32 or features.dtype == np.float64

    def test_model_info(self):
        """Test model information retrieval."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        model = PATModel(model_size="large")
        info = model.get_model_info()

        assert "model_size" in info
        assert "patch_size" in info
        assert "num_patches" in info
        assert "parameters" in info
        assert info["model_size"] == "large"
        assert info["patch_size"] == 9
        assert info["num_patches"] == 1120  # 10080 / 9
