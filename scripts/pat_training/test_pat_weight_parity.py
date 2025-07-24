#!/usr/bin/env python3
"""
Unit test to verify TensorFlow and PyTorch PAT models produce identical outputs.
This ensures weight conversion accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
import torch

from big_mood_detector.infrastructure.ml_models.pat_loader_direct import DirectPATModel
from big_mood_detector.infrastructure.ml_models.pat_pytorch import PATPyTorchEncoder


def test_weight_parity():
    """Test that TF and PyTorch models produce identical outputs."""
    print("Testing TensorFlow vs PyTorch weight parity...")

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)

    # Load weights path
    weights_path = Path("model_weights/pat/pretrained/PAT-S_29k_weights.h5")
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        return False

    # Create random test data (z-scored)
    batch_size = 4
    test_data = np.random.randn(batch_size, 10080).astype(np.float32)

    # Test 1: TensorFlow model
    print("\n1. Loading TensorFlow model...")
    tf_model = DirectPATModel(model_size="small")
    tf_model.load_weights(weights_path)

    # Get TF predictions
    tf_features = tf_model.extract_features(test_data).numpy()
    print(f"   TF output shape: {tf_features.shape}")
    print(f"   TF output sample: {tf_features[0, :5]}")

    # Test 2: PyTorch model
    print("\n2. Loading PyTorch model...")
    pt_model = PATPyTorchEncoder(model_size="small")
    pt_model.load_tf_weights(weights_path)
    pt_model.eval()

    # Get PyTorch predictions
    with torch.no_grad():
        pt_tensor = torch.from_numpy(test_data)
        pt_features = pt_model(pt_tensor).numpy()

    print(f"   PT output shape: {pt_features.shape}")
    print(f"   PT output sample: {pt_features[0, :5]}")

    # Test 3: Compare outputs
    print("\n3. Comparing outputs...")

    # Calculate differences
    abs_diff = np.abs(tf_features - pt_features)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"   Max absolute difference: {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")

    # Check if differences are within tolerance
    tolerance = 1e-3  # Allow small numerical differences
    if max_diff < tolerance:
        print("   ✅ PASSED: Models produce identical outputs (within tolerance)")
        return True
    else:
        print(f"   ❌ FAILED: Outputs differ by more than {tolerance}")

        # Show where differences occur
        problematic_indices = np.where(abs_diff > tolerance)
        if len(problematic_indices[0]) > 0:
            print(f"   First mismatch at index {problematic_indices[0][0]}, {problematic_indices[1][0]}")
            print(f"   TF value: {tf_features[problematic_indices[0][0], problematic_indices[1][0]]}")
            print(f"   PT value: {pt_features[problematic_indices[0][0], problematic_indices[1][0]]}")

        return False


def test_gradient_flow_real_weights():
    """Test gradient flow with real pretrained weights."""
    print("\n\nTesting gradient flow with real weights...")

    # Load model with real weights
    from big_mood_detector.infrastructure.ml_models.pat_pytorch import PATDepressionNet

    model = PATDepressionNet(model_size="small", unfreeze_last_n=1)
    weights_path = Path("model_weights/pat/pretrained/PAT-S_29k_weights.h5")

    if model.load_pretrained_encoder(weights_path):
        print("✅ Loaded real weights successfully")
    else:
        print("❌ Failed to load weights")
        return

    # Test forward/backward
    x = torch.randn(4, 10080, requires_grad=True)
    y = torch.randint(0, 2, (4,)).float()

    logits = model(x)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()

    # Check gradients
    has_grad = 0
    no_grad = 0

    for _name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad += 1
            else:
                no_grad += 1

    print("\nGradient check:")
    print(f"  Parameters with gradients: {has_grad}")
    print(f"  Parameters without gradients: {no_grad}")

    if no_grad == 0:
        print("✅ All trainable parameters have gradients")
    else:
        print("❌ Some trainable parameters missing gradients")


if __name__ == "__main__":
    print("PAT Weight Parity Test")
    print("="*50)

    # Run parity test
    parity_ok = test_weight_parity()

    # Run gradient test
    test_gradient_flow_real_weights()

    if parity_ok:
        print("\n✅ Weight conversion is accurate!")
        print("You can now run the full training with confidence.")
    else:
        print("\n❌ Weight conversion has issues. Fix before training.")

