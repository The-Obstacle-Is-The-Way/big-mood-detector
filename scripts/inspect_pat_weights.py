#!/usr/bin/env python3
"""
Inspect PAT H5 Weight Files

This script examines the structure of PAT weight files to understand
the exact layer names and weight shapes.
"""

import h5py
from pathlib import Path
import numpy as np


def inspect_h5_file(file_path):
    """Inspect the structure of an H5 file."""
    print(f"\nInspecting: {file_path}")
    print("=" * 70)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    with h5py.File(file_path, 'r') as f:
        # Print attributes
        print("\nFile attributes:")
        for key, value in f.attrs.items():
            if key == 'layer_names':
                layer_names = [n.decode('utf8') if isinstance(n, bytes) else n for n in value]
                print(f"  {key}: {layer_names}")
            else:
                print(f"  {key}: {value}")
        
        # Print groups and datasets
        print("\nFile structure:")
        
        def print_structure(name, obj):
            """Print the structure recursively."""
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}{name}/")
                # Print group attributes
                if obj.attrs:
                    for attr_key, attr_val in obj.attrs.items():
                        if attr_key == 'weight_names':
                            weight_names = [n.decode('utf8') if isinstance(n, bytes) else n for n in attr_val]
                            print(f"{indent}  @{attr_key}: {weight_names}")
                        else:
                            print(f"{indent}  @{attr_key}: {attr_val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
        
        f.visititems(print_structure)
        
        # Detailed weight inspection
        print("\nDetailed weight shapes:")
        if 'layer_names' in f.attrs:
            layer_names = [n.decode('utf8') if isinstance(n, bytes) else n for n in f.attrs['layer_names']]
            for layer_name in layer_names:
                if layer_name in f:
                    print(f"\n  Layer: {layer_name}")
                    layer_group = f[layer_name]
                    if 'weight_names' in layer_group.attrs:
                        weight_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                                      for n in layer_group.attrs['weight_names']]
                        for weight_name in weight_names:
                            if weight_name in layer_group:
                                weight = layer_group[weight_name]
                                print(f"    {weight_name}: shape={weight.shape}")


def main():
    """Inspect all PAT weight files."""
    print("PAT Weight File Inspector")
    print("=" * 70)
    
    # Paths to inspect
    weight_files = [
        "model_weights/pat/pretrained/PAT-S_29k_weights.h5",
        "model_weights/pat/pretrained/PAT-M_29k_weights.h5", 
        "model_weights/pat/pretrained/PAT-L_29k_weights.h5"
    ]
    
    for file_path in weight_files:
        inspect_h5_file(Path(file_path))
    
    print("\n" + "=" * 70)
    print("Analysis complete!")


if __name__ == "__main__":
    main()