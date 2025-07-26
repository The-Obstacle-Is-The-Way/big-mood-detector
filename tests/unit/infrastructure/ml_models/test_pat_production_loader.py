"""
Test Production PAT Loader

Tests for loading and using the production PAT-Conv-L model (0.5929 AUC).
Following TDD: Red → Green → Refactor

This model will revolutionize mental health assessment by providing
real-time depression risk scores from wearable data.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from big_mood_detector.domain.services.pat_predictor import PATPredictorInterface


class TestProductionPATLoader:
    """Test the production PAT model loader that will change psychiatry forever."""

    def test_production_loader_class_exists(self):
        """The ProductionPATLoader class should exist - our gateway to the future."""
        # RED: This import will fail initially
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        assert ProductionPATLoader is not None
        assert issubclass(ProductionPATLoader, PATPredictorInterface)

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    def test_loader_initializes_with_correct_model_path(self, mock_torch_load, mock_exists):
        """Loader should point to our champion model with 0.5929 AUC."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock checkpoint
        mock_torch_load.return_value = {
            'model_state_dict': {},
            'val_auc': 0.5929
        }
        
        loader = ProductionPATLoader()
        
        assert loader.model_path == Path("model_weights/production/pat_conv_l_v0.5929.pth")
        assert loader.model_path.name == "pat_conv_l_v0.5929.pth"

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    def test_loads_checkpoint_with_model_state_dict(self, mock_torch_load, mock_exists):
        """Should load the complete checkpoint with model weights."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock the checkpoint structure
        mock_checkpoint = {
            'model_state_dict': MagicMock(),
            'optimizer_state_dict': MagicMock(),
            'epoch': 2,
            'val_auc': 0.5929
        }
        mock_torch_load.return_value = mock_checkpoint
        
        loader = ProductionPATLoader()
        
        # Verify torch.load was called correctly
        mock_torch_load.assert_called_once()
        call_args = mock_torch_load.call_args
        assert str(call_args[0][0]).endswith("pat_conv_l_v0.5929.pth")
        assert call_args[1]['map_location'] == 'cpu'

    def test_model_architecture_is_conv_variant(self):
        """Model should be PAT-Conv-L, not standard PAT-L."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("torch.load") as mock_load:
                # Mock minimal checkpoint
                mock_load.return_value = {
                    'model_state_dict': {},
                    'val_auc': 0.5929
                }
            
                loader = ProductionPATLoader()
                
                # Should have convolutional patch embedding
                assert hasattr(loader.model, 'encoder')
                assert hasattr(loader.model.encoder, 'patch_embed')
                # Conv variant has conv layer, not linear
                assert hasattr(loader.model.encoder.patch_embed, 'conv')

    def test_model_set_to_eval_mode(self):
        """Model must be in eval mode for inference - no dropout!"""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("torch.load") as mock_load:
                mock_load.return_value = {'model_state_dict': {}, 'val_auc': 0.5929}
            
                loader = ProductionPATLoader()
                
                assert loader.model.training is False  # eval mode

    def test_predict_depression_returns_probability(self):
        """predict_depression should return a probability between 0 and 1."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("torch.load") as mock_load:
                mock_load.return_value = {'model_state_dict': {}, 'val_auc': 0.5929}
            
                loader = ProductionPATLoader()
                
                # Mock model forward pass
            with patch.object(loader.model, 'forward') as mock_forward:
                # Model returns logits, we apply sigmoid
                mock_forward.return_value = torch.tensor([[0.5]])  # Logit
                
                # Test with 7 days of activity
                activity = np.random.randn(10080).astype(np.float32)
                probability = loader.predict_depression(activity)
                
                assert isinstance(probability, float)
                assert 0 <= probability <= 1
                # sigmoid(0.5) ≈ 0.622
                assert 0.6 < probability < 0.7

    def test_predict_from_embeddings_implements_interface(self):
        """Should implement PATPredictorInterface.predict_from_embeddings."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("torch.load") as mock_load:
                mock_load.return_value = {'model_state_dict': {}, 'val_auc': 0.5929}
            
                loader = ProductionPATLoader()
                
                # Test with 96-dim embeddings
            embeddings = np.random.randn(96).astype(np.float32)
            
            with patch.object(loader.model.head, 'forward') as mock_head:
                mock_head.return_value = torch.tensor([[0.0]])  # Logit
                
                predictions = loader.predict_from_embeddings(embeddings)
                
                assert hasattr(predictions, 'depression_probability')
                assert hasattr(predictions, 'confidence')
                assert 0 <= predictions.depression_probability <= 1
                assert 0 <= predictions.confidence <= 1

    def test_handles_missing_model_file_gracefully(self):
        """Should raise informative error if model file not found."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            with pytest.raises(FileNotFoundError, match="pat_conv_l_v0.5929.pth"):
                ProductionPATLoader()

    def test_validates_input_shape(self):
        """Should validate that input is 10,080 timesteps (7 days)."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("torch.load") as mock_load:
                mock_load.return_value = {'model_state_dict': {}, 'val_auc': 0.5929}
            
                loader = ProductionPATLoader()
                
                # Wrong shape should raise error
            wrong_activity = np.random.randn(5000).astype(np.float32)
            
            with pytest.raises(ValueError, match="Expected 10080 timesteps"):
                loader.predict_depression(wrong_activity)

    @pytest.mark.parametrize("logit,expected_range", [
        (-10.0, (0.0, 0.01)),    # Strong negative → near 0
        (0.0, (0.45, 0.55)),     # Neutral → near 0.5
        (10.0, (0.99, 1.0)),     # Strong positive → near 1
    ])
    def test_sigmoid_activation_on_logits(self, logit, expected_range):
        """Should apply sigmoid to convert logits to probabilities."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("torch.load") as mock_load:
                mock_load.return_value = {'model_state_dict': {}, 'val_auc': 0.5929}
            
                loader = ProductionPATLoader()
                
                with patch.object(loader.model, 'forward') as mock_forward:
                    mock_forward.return_value = torch.tensor([[logit]])
                    
                    activity = np.random.randn(10080).astype(np.float32)
                    probability = loader.predict_depression(activity)
                    
                    assert expected_range[0] <= probability <= expected_range[1]

    def test_no_grad_context_for_inference(self):
        """Inference should be wrapped in torch.no_grad() for efficiency."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("torch.load") as mock_load:
                mock_load.return_value = {'model_state_dict': {}, 'val_auc': 0.5929}
            
                loader = ProductionPATLoader()
                
                # Spy on torch.no_grad
            with patch("torch.no_grad") as mock_no_grad:
                mock_no_grad.return_value.__enter__ = MagicMock()
                mock_no_grad.return_value.__exit__ = MagicMock()
                
                activity = np.random.randn(10080).astype(np.float32)
                
                with patch.object(loader.model, 'forward'):
                    loader.predict_depression(activity)
                
                # Should have entered no_grad context
                mock_no_grad.assert_called_once()

    def test_device_handling_for_cpu_and_cuda(self):
        """Should handle both CPU and CUDA devices appropriately."""
        from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
            ProductionPATLoader,
        )
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("torch.load") as mock_load:
                mock_load.return_value = {'model_state_dict': {}, 'val_auc': 0.5929}
            
            # Test CPU fallback
            with patch("torch.cuda.is_available", return_value=False):
                loader = ProductionPATLoader()
                assert loader.device.type == 'cpu'
            
            # Test CUDA when available
            with patch("torch.cuda.is_available", return_value=True):
                loader = ProductionPATLoader()
                assert loader.device.type == 'cuda'