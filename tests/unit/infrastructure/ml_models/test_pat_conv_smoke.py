"""
Smoke test for PAT Conv checkpoint loading.

Ensures the convolutional architecture matches the trained weights.
"""

import pytest
import torch

from big_mood_detector.infrastructure.ml_models.pat_conv_depression_model import (
    SimplePATConvLModel,
)


class TestPATConvSmoke:
    """Smoke tests to guard against architectural changes."""

    def test_conv_layer_shape_matches_checkpoint(self):
        """Verify conv layer has expected shape for checkpoint compatibility."""
        # Create model
        model = SimplePATConvLModel(model_size="large")

        # Check patch embedding is Conv1d
        assert hasattr(model.encoder, 'patch_embed'), "Encoder missing patch_embed"
        patch_embed = model.encoder.patch_embed

        # Verify it's a Conv patch embedding
        assert hasattr(patch_embed, 'conv'), "Patch embed missing conv layer"
        conv_layer = patch_embed.conv

        # Check conv layer parameters match checkpoint expectations
        # PAT models use embed_dim=96
        assert isinstance(conv_layer, torch.nn.Conv1d), "Expected Conv1d layer"
        assert conv_layer.in_channels == 1, f"Expected 1 input channel, got {conv_layer.in_channels}"
        assert conv_layer.out_channels == 96, f"Expected 96 output channels, got {conv_layer.out_channels}"

        # Patch size could be 9 (base PAT-L) or 15 (Conv variant)
        # Both are valid depending on training configuration
        kernel_size = conv_layer.kernel_size[0]
        assert kernel_size in [9, 15], f"Expected kernel size 9 or 15, got {kernel_size}"
        assert conv_layer.stride == conv_layer.kernel_size, "Expected stride to match kernel size"

    def test_head_layer_shape_for_binary_classification(self):
        """Verify head layer is configured for binary classification."""
        # Create model
        model = SimplePATConvLModel(model_size="large")

        # Check head layer
        assert hasattr(model, 'head'), "Model missing classification head"
        head = model.head

        # Verify it's a linear layer for binary classification
        assert isinstance(head, torch.nn.Linear), "Expected Linear layer for head"
        assert head.in_features == 96, f"Expected 96 input features, got {head.in_features}"
        assert head.out_features == 1, f"Expected 1 output for binary classification, got {head.out_features}"

    def test_model_forward_pass_shape(self):
        """Verify model produces correct output shape."""
        # Create model
        model = SimplePATConvLModel(model_size="large")
        model.eval()

        # Create input: batch of 7-day sequences
        batch_size = 2
        sequence_length = 10080  # 7 days * 24 hours * 60 minutes
        x = torch.randn(batch_size, sequence_length)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Check output shape for binary classification
        assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"

    def test_checkpoint_weight_names_match_model(self):
        """Verify checkpoint weight names align with model structure."""
        # Create model
        model = SimplePATConvLModel(model_size="large")

        # Expected weight names from checkpoint
        expected_weights = {
            'encoder.patch_embed.conv.weight',
            'encoder.patch_embed.conv.bias',
            'head.weight',
            'head.bias',
        }

        # Get actual model state dict keys
        state_dict = model.state_dict()

        # Check critical weights exist
        for weight_name in expected_weights:
            assert weight_name in state_dict, f"Missing expected weight: {weight_name}"

            # Verify shapes match checkpoint expectations
            if weight_name == 'encoder.patch_embed.conv.weight':
                # Conv1d weight shape: (out_channels, in_channels, kernel_size)
                actual_shape = tuple(state_dict[weight_name].shape)
                assert actual_shape[0] == 96, f"Expected 96 out_channels, got {actual_shape[0]}"
                assert actual_shape[1] == 1, f"Expected 1 in_channel, got {actual_shape[1]}"
                assert actual_shape[2] in [9, 15], f"Expected kernel size 9 or 15, got {actual_shape[2]}"
            elif weight_name == 'encoder.patch_embed.conv.bias':
                expected_shape = (96,)
                actual_shape = tuple(state_dict[weight_name].shape)
                assert actual_shape == expected_shape, \
                    f"Conv bias shape mismatch: expected {expected_shape}, got {actual_shape}"
            elif weight_name == 'head.weight':
                expected_shape = (1, 96)
                actual_shape = tuple(state_dict[weight_name].shape)
                assert actual_shape == expected_shape, \
                    f"Head weight shape mismatch: expected {expected_shape}, got {actual_shape}"
            elif weight_name == 'head.bias':
                expected_shape = (1,)
                actual_shape = tuple(state_dict[weight_name].shape)
                assert actual_shape == expected_shape, \
                    f"Head bias shape mismatch: expected {expected_shape}, got {actual_shape}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_model_runs_on_cuda(self):
        """Verify model can be moved to CUDA and run."""
        # Create model and move to CUDA
        model = SimplePATConvLModel(model_size="large")
        model = model.cuda()
        model.eval()

        # Create input on CUDA
        x = torch.randn(1, 10080).cuda()

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Check output is on CUDA
        assert output.is_cuda, "Output should be on CUDA"
        assert output.device == x.device, "Output device should match input device"
