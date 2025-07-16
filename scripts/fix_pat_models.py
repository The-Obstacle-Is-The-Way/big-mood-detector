#!/usr/bin/env python3
"""
Fix PAT models by creating properly loadable versions.

The original PAT H5 files contain only weights without model configuration.
This script creates mock PAT models that maintain compatibility.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_pat_encoder(model_size="medium"):
    """Create a PAT encoder model architecture."""
    
    # Model configurations from the paper
    configs = {
        "small": {
            "patch_size": 18,
            "embed_dim": 96,
            "encoder_num_heads": 6,
            "encoder_ff_dim": 256,
            "encoder_num_layers": 1,
            "encoder_rate": 0.1,
        },
        "medium": {
            "patch_size": 18,
            "embed_dim": 96,
            "encoder_num_heads": 12,
            "encoder_ff_dim": 256,
            "encoder_num_layers": 2,
            "encoder_rate": 0.1,
        },
        "large": {
            "patch_size": 9,
            "embed_dim": 96,
            "encoder_num_heads": 12,
            "encoder_ff_dim": 256,
            "encoder_num_layers": 4,
            "encoder_rate": 0.1,
        },
    }
    
    config = configs[model_size]
    input_size = 10080  # 7 days * 1440 minutes
    num_patches = input_size // config["patch_size"]
    
    # Build model
    inputs = keras.Input(shape=(input_size,), name="inputs")
    
    # Reshape to patches
    reshaped = keras.layers.Reshape((num_patches, config["patch_size"]), name="reshape")(inputs)
    
    # Patch embedding
    x = keras.layers.Dense(config["embed_dim"], name="dense")(reshaped)
    
    # Add positional encoding (simplified for compatibility)
    pos_encoding = keras.layers.Embedding(
        input_dim=num_patches, 
        output_dim=config["embed_dim"],
        name="positional_encoding"
    )(tf.range(num_patches))
    
    x = keras.layers.Add(name="add_pos_encoding")([x, pos_encoding])
    
    # Transformer blocks
    for i in range(config["encoder_num_layers"]):
        # Multi-head attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=config["encoder_num_heads"],
            key_dim=config["embed_dim"],
            dropout=config["encoder_rate"],
            name=f"encoder_{i+1}_mha"
        )(x, x)
        
        # Add & Norm
        x = keras.layers.Add(name=f"encoder_{i+1}_add1")([x, attn_output])
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"encoder_{i+1}_norm1")(x)
        
        # Feed-forward network
        ffn = keras.Sequential([
            keras.layers.Dense(config["encoder_ff_dim"], activation="relu"),
            keras.layers.Dense(config["embed_dim"]),
            keras.layers.Dropout(config["encoder_rate"])
        ], name=f"encoder_{i+1}_ffn")
        
        ffn_output = ffn(x)
        
        # Add & Norm
        x = keras.layers.Add(name=f"encoder_{i+1}_add2")([x, ffn_output])
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"encoder_{i+1}_norm2")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=x, name=f"PAT_{model_size}_encoder")
    
    return model


def create_fixed_models():
    """Create fixed versions of all PAT models."""
    
    models_dir = Path("model_weights/pat/pretrained")
    fixed_dir = Path("model_weights/pat/pretrained_fixed")
    fixed_dir.mkdir(exist_ok=True)
    
    models = {
        "small": "PAT-S_29k_weights.h5",
        "medium": "PAT-M_29k_weights.h5",
        "large": "PAT-L_29k_weights.h5"
    }
    
    print("Creating fixed PAT models...")
    print("=" * 70)
    
    for size, filename in models.items():
        print(f"\nProcessing PAT-{size.upper()}...")
        
        # Create model architecture
        model = create_pat_encoder(size)
        print(f"  Created architecture: {model.count_params():,} parameters")
        
        # For production, we would load the original weights here
        # For now, we'll save with random initialization as a working example
        
        # Save in a format that can be loaded
        fixed_path = fixed_dir / filename
        model.save(fixed_path, save_format="h5")
        print(f"  Saved to: {fixed_path}")
        
        # Verify it can be loaded
        try:
            test_model = keras.models.load_model(fixed_path, compile=False)
            print(f"  ✅ Verified: Model can be loaded successfully")
            print(f"     Input shape: {test_model.input_shape}")
            print(f"     Output shape: {test_model.output_shape}")
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"Fixed models saved to: {fixed_dir}")
    print("\nTo use the fixed models:")
    print("1. Update PAT model loader to use 'pretrained_fixed' directory")
    print("2. Or copy fixed models over the original ones")
    print("\nNote: These are placeholder models with random weights.")
    print("For production, transfer weights from the original H5 files.")


def main():
    """Main function."""
    create_fixed_models()
    
    print("\n" + "=" * 70)
    print("Alternative Solution:")
    print("Since the original PAT weights are research artifacts,")
    print("consider using the ensemble system with XGBoost only")
    print("until proper PAT model files are available.")
    print("\nThe system already gracefully degrades to XGBoost-only mode!")


if __name__ == "__main__":
    main()