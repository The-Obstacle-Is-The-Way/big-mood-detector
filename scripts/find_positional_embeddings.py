#!/usr/bin/env python3
"""
Find positional embeddings in PAT H5 files.
"""

from pathlib import Path

import h5py


def find_positional_embeddings(h5_path):
    """Search for positional embeddings in H5 file."""
    print(f"\nSearching for positional embeddings in: {h5_path}")
    print("=" * 70)

    with h5py.File(h5_path, "r") as f:

        def search_recursively(name, obj):
            if "position" in name.lower() or "pos" in name.lower():
                print(f"Found: {name}")
                if isinstance(obj, h5py.Dataset):
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")

        f.visititems(search_recursively)

        # Also check for tf.__operators__.add layers which might contain pos embeddings
        print("\nChecking tf.__operators__.add layers:")
        for key in f.keys():
            if "tf.__operators__.add" in key:
                print(f"Found add layer: {key}")
                if key in f and "weight_names" in f[key].attrs:
                    weight_names = f[key].attrs["weight_names"]
                    print(f"  Weight names: {weight_names}")


def main():
    """Check all PAT models."""
    models = [
        "model_weights/pat/pretrained/PAT-S_29k_weights.h5",
        "model_weights/pat/pretrained/PAT-M_29k_weights.h5",
        "model_weights/pat/pretrained/PAT-L_29k_weights.h5",
    ]

    for model_path in models:
        if Path(model_path).exists():
            find_positional_embeddings(model_path)
        else:
            print(f"Model not found: {model_path}")


if __name__ == "__main__":
    main()
