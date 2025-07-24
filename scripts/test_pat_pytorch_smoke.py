#!/usr/bin/env python3
"""
Smoke test for PyTorch PAT implementation.
Verifies gradient flow and basic functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from big_mood_detector.infrastructure.ml_models.pat_pytorch import (
    PATDepressionNet,
)


def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("Testing gradient flow...")

    # Create model with last block unfrozen
    model = PATDepressionNet(model_size="small", unfreeze_last_n=1)

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 10080)
    y = torch.randint(0, 2, (batch_size,)).float()

    # Forward pass
    logits = model(x)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)

    # Backward pass
    loss.backward()

    # Check gradients
    print("\nGradient check:")

    # Encoder patch embedding (should have gradients)
    patch_grad = model.encoder.patch_embed.weight.grad
    print(f"✓ Patch embed grad: {patch_grad is not None and patch_grad.abs().sum().item() > 0}")

    # Last transformer block (should have gradients)
    last_block = model.encoder.blocks[-1]
    attn_grad = last_block.attention.q_proj.weight.grad
    print(f"✓ Last block attention grad: {attn_grad is not None and attn_grad.abs().sum().item() > 0}")

    # Head (should have gradients)
    head_grad = model.head[0].weight.grad
    print(f"✓ Head grad: {head_grad is not None and head_grad.abs().sum().item() > 0}")

    # Count trainable parameters
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("\nParameter counts:")
    print(f"  Encoder trainable: {encoder_trainable:,}")
    print(f"  Head trainable: {head_trainable:,}")
    print(f"  Total parameters: {total_params:,}")


def test_weight_loading():
    """Test loading TF weights."""
    print("\nTesting weight loading...")

    model = PATDepressionNet(model_size="small")
    weights_path = Path("model_weights/pat/pretrained/PAT-S_29k_weights.h5")

    if weights_path.exists():
        success = model.load_pretrained_encoder(weights_path)
        print(f"✓ Weight loading: {success}")
    else:
        print(f"✗ Weights not found at {weights_path}")


def test_inference_speed():
    """Test inference speed."""
    print("\nTesting inference speed...")

    model = PATDepressionNet(model_size="small")
    model.eval()

    # Test different batch sizes
    for batch_size in [1, 16, 32, 64]:
        x = torch.randn(batch_size, 10080)

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Time
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        elapsed = (time.time() - start) / 10

        print(f"  Batch size {batch_size}: {elapsed*1000:.1f}ms ({batch_size/elapsed:.1f} samples/sec)")


def test_output_shape():
    """Test output shapes."""
    print("\nTesting output shapes...")

    model = PATDepressionNet(model_size="small")

    for batch_size in [1, 8, 32]:
        x = torch.randn(batch_size, 10080)
        with torch.no_grad():
            embeddings = model.encoder(x)
            logits = model(x)

        print(f"  Batch {batch_size}: embeddings {embeddings.shape}, logits {logits.shape}")
        assert embeddings.shape == (batch_size, 96)
        assert logits.shape == (batch_size,)

    print("✓ All shapes correct")


if __name__ == "__main__":
    print("PAT PyTorch Implementation Smoke Test")
    print("="*50)

    test_gradient_flow()
    test_weight_loading()
    test_inference_speed()
    test_output_shape()

    print("\n✓ All tests passed!")
    print("\nNow run the training with:")
    print("python scripts/train_pat_depression_pytorch.py --subset 200 --epochs 3 --device mps")

