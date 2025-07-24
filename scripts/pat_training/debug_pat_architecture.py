#!/usr/bin/env python3
"""Debug PAT architecture differences between TF and PyTorch."""

import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.ml_models.pat_loader_direct import DirectPATModel

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# Create simple test input
test_input = np.random.randn(1, 10080).astype(np.float32)

# Load TF model and trace execution
print("Loading TensorFlow PAT-S model...")
tf_model = DirectPATModel(model_size="small")
weights_path = Path("model_weights/pat/pretrained/PAT-S_29k_weights.h5")
tf_model.load_weights(weights_path)

# Manually step through TF computation
print("\nManual TF computation:")
# 1. Reshape to patches
patches = test_input.reshape(1, 560, 18)
print(f"1. Patches shape: {patches.shape}")

# 2. Linear projection (patch embedding)
with h5py.File(weights_path, 'r') as f:
    dense_kernel = np.array(f['dense/dense/kernel:0'])
    dense_bias = np.array(f['dense/dense/bias:0'])

x = np.matmul(patches, dense_kernel) + dense_bias
print(f"2. After patch embed: {x.shape}, sample: {x[0, 0, :5]}")

# 3. Add positional embeddings
position = np.arange(560, dtype=np.float32)[:, np.newaxis]
div_term = np.exp(np.arange(0, 96, 2).astype(np.float32) * (-np.log(10000.0) / 96))
sin_embeddings = np.sin(position * div_term)
cos_embeddings = np.cos(position * div_term)
pos_embeddings = np.zeros((560, 96), dtype=np.float32)
pos_embeddings[:, 0::2] = sin_embeddings
pos_embeddings[:, 1::2] = cos_embeddings
x = x + pos_embeddings[np.newaxis, :, :]
print(f"3. After pos embed: {x.shape}, sample: {x[0, 0, :5]}")

# 4. Through transformer block
# For now, just check if the model runs
print("\n4. Running full TF model...")
tf_output = tf_model.extract_features(test_input)
print(f"   TF final output: {tf_output.shape}, sample: {tf_output[0, :5]}")

# Now test our PyTorch implementation step by step
print("\n\nPyTorch implementation:")
from big_mood_detector.infrastructure.ml_models.pat_pytorch import (
    PATPyTorchEncoder,  # noqa: E402
)

pt_model = PATPyTorchEncoder(model_size="small")
pt_model.load_tf_weights(weights_path)
pt_model.eval()

with torch.no_grad():
    # Convert input
    pt_input = torch.from_numpy(test_input)

    # Step through forward pass
    # 1. Reshape to patches
    pt_patches = pt_input.view(1, 560, 18)
    print(f"1. PT Patches shape: {pt_patches.shape}")

    # 2. Patch embedding
    pt_x = pt_model.patch_embed(pt_patches)
    print(f"2. PT After patch embed: {pt_x.shape}, sample: {pt_x[0, 0, :5]}")

    # 3. Add positional embeddings
    pt_x = pt_model.pos_embed(pt_x)
    print(f"3. PT After pos embed: {pt_x.shape}, sample: {pt_x[0, 0, :5]}")

    # 4. Full forward
    pt_output = pt_model(pt_input)
    print(f"4. PT final output: {pt_output.shape}, sample: {pt_output[0, :5]}")

# Compare outputs
print("\n\nComparison:")
tf_np = tf_output.numpy()
pt_np = pt_output.numpy()

diff = np.abs(tf_np - pt_np)
print(f"Max difference: {np.max(diff):.6f}")
print(f"Mean difference: {np.mean(diff):.6f}")

if np.max(diff) < 1e-3:
    print("✅ Models match!")
else:
    print("❌ Models differ significantly")
    # Find where they differ
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Biggest difference at index {max_idx}:")
    print(f"  TF: {tf_np[max_idx]}")
    print(f"  PT: {pt_np[max_idx]}")

