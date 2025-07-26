"""
Test Production PAT Loader - Clean TDD Implementation

Following Uncle Bob's principles:
- Tests should be F.I.R.S.T (Fast, Independent, Repeatable, Self-validating, Timely)
- One assert per test when possible
- Test behavior, not implementation
- Minimal mocking - only at boundaries
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from big_mood_detector.domain.services.pat_predictor import PATPredictorInterface


class TestProductionPATLoaderContract:
    """Test that ProductionPATLoader fulfills its contract."""
    
    def test_implements_pat_predictor_interface(self):
        """ProductionPATLoader must implement PATPredictorInterface."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        assert issubclass(ProductionPATLoader, PATPredictorInterface)
    
    def test_can_create_loader_without_weights(self):
        """Should be able to create loader for testing without loading weights."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        # This allows unit testing without file I/O
        loader = ProductionPATLoader(skip_loading=True)
        assert loader is not None
    
    def test_default_model_path_points_to_production_weights(self):
        """Default path should point to our 0.5929 AUC model."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        assert loader.model_path.name == "pat_conv_l_v0.5929.pth"
        assert "production" in str(loader.model_path)


class TestProductionPATLoaderArchitecture:
    """Test the model architecture is correct."""
    
    def test_creates_conv_model_not_linear(self):
        """Must create Conv variant, not standard linear PAT."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        
        # Verify Conv architecture
        encoder = loader.model.encoder
        assert hasattr(encoder.patch_embed, 'conv'), "Must use Conv1d patch embedding"
        assert encoder.conv_embedding is True
    
    def test_model_is_in_eval_mode(self):
        """Model must be in eval mode to disable dropout during inference."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        assert loader.model.training is False
    
    def test_uses_correct_device(self):
        """Should use CUDA if available, CPU otherwise."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        
        if torch.cuda.is_available():
            assert loader.device.type == 'cuda'
        else:
            assert loader.device.type == 'cpu'


class TestProductionPATLoaderPredictions:
    """Test prediction functionality."""
    
    def test_predict_depression_validates_input_shape(self):
        """Should only accept 7 days (10,080 minutes) of data."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        
        # Wrong shape should raise error
        wrong_shape = np.zeros(5000, dtype=np.float32)
        
        with pytest.raises(ValueError, match="10080"):
            loader.predict_depression(wrong_shape)
    
    def test_predict_depression_returns_probability(self):
        """Depression prediction must return value between 0 and 1."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        
        # Mock just the model forward pass
        with patch.object(loader.model, 'forward', return_value=torch.tensor([0.0])):
            activity = np.zeros(10080, dtype=np.float32)
            probability = loader.predict_depression(activity)
            
            assert isinstance(probability, float)
            assert 0 <= probability <= 1
    
    def test_applies_sigmoid_to_model_output(self):
        """Must apply sigmoid since model outputs logits."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        
        # Test with known logit -> probability mapping
        test_cases = [
            (-10.0, 0.0, 0.01),  # Very negative logit -> ~0
            (0.0, 0.45, 0.55),   # Zero logit -> ~0.5
            (10.0, 0.99, 1.0),   # Very positive logit -> ~1
        ]
        
        for logit, min_prob, max_prob in test_cases:
            with patch.object(loader.model, 'forward', return_value=torch.tensor([logit])):
                activity = np.zeros(10080, dtype=np.float32)
                probability = loader.predict_depression(activity)
                assert min_prob <= probability <= max_prob
    
    def test_uses_no_grad_for_inference(self):
        """Inference should not compute gradients for efficiency."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        
        # Track if we're in no_grad context
        grad_enabled_during_forward = None
        
        def mock_forward(x):
            nonlocal grad_enabled_during_forward
            grad_enabled_during_forward = torch.is_grad_enabled()
            return torch.tensor([0.0])
        
        with patch.object(loader.model, 'forward', side_effect=mock_forward):
            activity = np.zeros(10080, dtype=np.float32)
            loader.predict_depression(activity)
            
        assert grad_enabled_during_forward is False


class TestProductionPATLoaderIntegration:
    """Integration tests with real PyTorch operations."""
    
    @pytest.fixture
    def temp_checkpoint(self, tmp_path):
        """Create a minimal valid checkpoint file."""
        from big_mood_detector.infrastructure.ml_models.pat_conv_depression_model import (
            SimplePATConvLModel,
        )
        
        # Create model and save checkpoint
        model = SimplePATConvLModel(model_size="large")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 2,
            'val_auc': 0.5929
        }
        
        checkpoint_path = tmp_path / "test_model.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Also create normalizer stats
        stats_dir = tmp_path / "model_weights" / "production"
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_file = stats_dir / "nhanes_scaler_stats.json"
        stats_file.write_text(json.dumps({
            "mean": [0.0] * 10080,
            "std": [1.0] * 10080,
            "dataset": "test"
        }))
        
        return checkpoint_path
    
    def test_can_load_real_checkpoint(self, temp_checkpoint):
        """Should successfully load a real PyTorch checkpoint."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
            NHANESNormalizer,
        )
        
        # Create a fitted normalizer for testing
        normalizer = NHANESNormalizer()
        normalizer.mean = np.zeros(10080, dtype=np.float32)
        normalizer.std = np.ones(10080, dtype=np.float32)
        normalizer.fitted = True
        
        loader = ProductionPATLoader(
            model_path=temp_checkpoint,
            normalizer=normalizer
        )
        
        # Verify model loaded and is functional
        assert loader.model is not None
        
        # Can run forward pass
        activity = np.random.randn(10080).astype(np.float32)
        probability = loader.predict_depression(activity)
        
        assert 0 <= probability <= 1
        assert not np.isnan(probability)
    
    def test_raises_clear_error_for_missing_weights(self):
        """Should provide helpful error when weights file is missing."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        fake_path = Path("/definitely/does/not/exist.pth")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            ProductionPATLoader(model_path=fake_path)
            
        assert "pat_conv_l_v0.5929.pth" in str(exc_info.value) or "not found" in str(exc_info.value)


class TestProductionPATLoaderNormalization:
    """Test NHANES normalization integration."""
    
    def test_uses_nhanes_normalizer(self):
        """Should normalize input data using NHANES statistics."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        # Create a mock normalizer
        mock_normalizer = MagicMock()
        mock_normalizer.transform.return_value = np.zeros(10080, dtype=np.float32)
        
        # Create loader with the mock normalizer
        loader = ProductionPATLoader(
            skip_loading=True,
            normalizer=mock_normalizer
        )
        
        # Disable bypass normalization for this test
        loader._bypass_normalization = False
        
        # Run prediction
        with patch.object(loader.model, 'forward', return_value=torch.tensor([0.0])):
            activity = np.ones(10080, dtype=np.float32)  # Non-zero input
            loader.predict_depression(activity)
        
        # Verify normalizer was used
        mock_normalizer.transform.assert_called_once()
        np.testing.assert_array_equal(
            mock_normalizer.transform.call_args[0][0],
            activity
        )


class TestPATPredictorInterfaceCompliance:
    """Test compliance with domain interface."""
    
    def test_predict_from_embeddings_returns_correct_type(self):
        """predict_from_embeddings must return PATBinaryPredictions."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        from big_mood_detector.domain.services.pat_predictor import PATBinaryPredictions
        
        loader = ProductionPATLoader(skip_loading=True)
        
        # Mock the head forward pass
        with patch.object(loader.model.head, 'forward', return_value=torch.tensor([[0.0]])):
            embeddings = np.zeros(96, dtype=np.float32)
            result = loader.predict_from_embeddings(embeddings)
        
        assert isinstance(result, PATBinaryPredictions)
        assert hasattr(result, 'depression_probability')
        assert hasattr(result, 'benzodiazepine_probability')
        assert hasattr(result, 'confidence')
    
    def test_implements_all_interface_methods(self):
        """Must implement all methods from PATPredictorInterface."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        loader = ProductionPATLoader(skip_loading=True)
        
        # Check all interface methods exist
        assert hasattr(loader, 'predict_from_embeddings')
        assert hasattr(loader, 'predict_depression')
        assert hasattr(loader, 'predict_medication_proxy')
        
        # All should be callable
        assert callable(loader.predict_from_embeddings)
        assert callable(loader.predict_depression)
        assert callable(loader.predict_medication_proxy)