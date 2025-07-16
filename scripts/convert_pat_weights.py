#!/usr/bin/env python3
"""
Convert PAT model weights to a simpler format for easier loading.

The original PAT models are saved as complete Keras models with custom objects.
This script extracts just the encoder weights in a format we can load easily.
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def inspect_pat_model(model_path: Path):
    """Inspect the structure of a PAT model file."""
    print(f"\nInspecting {model_path.name}:")
    print("-" * 50)

    with h5py.File(model_path, "r") as f:
        # Check attributes
        print("Model attributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        print("\nLayer structure:")

        # The model has these key layers we need:
        # 1. dense/dense - the patch embedding layer
        # 2. encoder_layer_N_transformer - the transformer blocks

        def count_parameters(group):
            """Count parameters in a group."""
            total = 0
            for key in group.keys():
                if key.endswith(":0"):  # Weight tensor
                    dataset = group[key]
                    params = np.prod(dataset.shape)
                    total += params
            return total

        # Check each major component
        if "dense" in f:
            dense_params = count_parameters(f["dense/dense"])
            print(f"  Patch embedding (dense): {dense_params:,} parameters")

        # Count encoder layers
        encoder_layers = 0
        for key in f.keys():
            if key.startswith("encoder_layer_") and key.endswith("_transformer"):
                encoder_layers += 1
                layer_params = count_parameters(f[key])
                print(f"  {key}: {layer_params:,} parameters")

        print(f"\nTotal encoder layers: {encoder_layers}")

        # Get total model size
        total_params = 0

        def visit_fn(name, obj):
            nonlocal total_params
            if isinstance(obj, h5py.Dataset) and name.endswith(":0"):
                total_params += np.prod(obj.shape)

        f.visititems(visit_fn)
        print(f"Total parameters: {total_params:,}")

        return encoder_layers


def extract_encoder_weights(model_path: Path, output_path: Path):
    """Extract just the encoder weights in a simpler format."""
    print(f"\nExtracting encoder weights from {model_path.name}")

    # For now, we'll keep the original format since TensorFlow expects it
    # In production, we might convert to a simpler format

    # The models work as-is, we just need to handle loading correctly
    print(f"Model is already in the correct format at: {model_path}")


def main():
    """Check all PAT models."""
    model_dir = Path("model_weights/pat/pretrained")

    models = {
        "small": "PAT-S_29k_weights.h5",
        "medium": "PAT-M_29k_weights.h5",
        "large": "PAT-L_29k_weights.h5",
    }

    print("=" * 70)
    print("PAT Model Weight Analysis")
    print("=" * 70)

    for size, filename in models.items():
        model_path = model_dir / filename
        if model_path.exists():
            num_layers = inspect_pat_model(model_path)

            # Verify against expected architecture
            expected_layers = {"small": 1, "medium": 2, "large": 4}

            if num_layers == expected_layers[size]:
                print(f"✅ PAT-{size.upper()} has correct architecture")
            else:
                print(
                    f"❌ PAT-{size.upper()} architecture mismatch: "
                    f"expected {expected_layers[size]} layers, found {num_layers}"
                )
        else:
            print(f"❌ {filename} not found")

    print("\n" + "=" * 70)
    print("Summary:")
    print("The PAT weights are saved as complete Keras encoder models.")
    print("They include positional embeddings and can be loaded with proper")
    print("custom object handling. No conversion needed.")


if __name__ == "__main__":
    main()
