"""
PAT Architecture Definition

Exact architecture implementation for Pretrained Actigraphy Transformer models
based on the foundation paper and original implementation.

This module defines the transformer blocks and model construction needed
to properly load the pretrained weights.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def TransformerBlock(
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    rate: float = 0.1,
    name: str = "transformer_block"
) -> keras.Model:
    """
    Transformer encoder block as implemented in the original PAT models.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        rate: Dropout rate
        name: Block name prefix
        
    Returns:
        Keras Model representing the transformer block
    """
    inputs = layers.Input(shape=(None, embed_dim))
    
    # Multi-head self-attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=rate,
        name=f"{name}_mha"
    )(inputs, inputs)
    attn_output = layers.Dropout(rate, name=f"{name}_dropout1")(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(
        layers.Add(name=f"{name}_add1")([inputs, attn_output])
    )
    
    # Feed-forward network
    ffn_output = layers.Dense(ff_dim, activation="relu", name=f"{name}_ffn_dense1")(out1)
    ffn_output = layers.Dense(embed_dim, name=f"{name}_ffn_dense2")(ffn_output)
    ffn_output = layers.Dropout(rate, name=f"{name}_dropout2")(ffn_output)
    out2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(
        layers.Add(name=f"{name}_add2")([out1, ffn_output])
    )
    
    return keras.Model(inputs=inputs, outputs=out2, name=name)


def create_pat_encoder(
    patch_size: int,
    embed_dim: int,
    encoder_num_heads: int,
    encoder_ff_dim: int,
    encoder_num_layers: int,
    encoder_rate: float = 0.1,
    input_size: int = 10080
) -> keras.Model:
    """
    Create PAT encoder architecture matching the original implementation.
    
    Args:
        patch_size: Size of each patch
        embed_dim: Embedding dimension
        encoder_num_heads: Number of attention heads
        encoder_ff_dim: Feed-forward dimension
        encoder_num_layers: Number of transformer layers
        encoder_rate: Dropout rate
        input_size: Total input sequence length (default 7 days = 10080 minutes)
        
    Returns:
        Keras Model for the PAT encoder
    """
    num_patches = input_size // patch_size
    
    # Input layer
    inputs = layers.Input(shape=(input_size,), name="inputs")
    
    # Reshape to patches
    patches = layers.Reshape((num_patches, patch_size), name="patches")(inputs)
    
    # Linear projection of patches
    x = layers.Dense(embed_dim, name="dense")(patches)
    
    # Add positional embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=embed_dim,
        name="positional_embedding"
    )(positions)
    x = x + pos_embedding
    
    # Transformer encoder blocks
    for i in range(encoder_num_layers):
        transformer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=encoder_num_heads,
            ff_dim=encoder_ff_dim,
            rate=encoder_rate,
            name=f"encoder_layer_{i+1}_transformer"
        )
        x = transformer(x)
    
    # Output is the encoded sequence
    outputs = x
    
    return keras.Model(inputs=inputs, outputs=outputs, name="pat_encoder")


def build_pat_model(model_size: str = "medium") -> keras.Model:
    """
    Build a PAT model of the specified size.
    
    Args:
        model_size: One of "small", "medium", or "large"
        
    Returns:
        Keras Model for the specified PAT variant
    """
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
    
    if model_size not in configs:
        raise ValueError(f"Invalid model size: {model_size}. Choose from: {list(configs.keys())}")
    
    config = configs[model_size]
    return create_pat_encoder(**config)


def load_weights_from_h5(model: keras.Model, weights_path: str) -> bool:
    """
    Load weights from H5 file into the model architecture.
    
    This handles the special case where the H5 file contains only weights
    without model configuration.
    
    Args:
        model: The model architecture to load weights into
        weights_path: Path to the H5 weights file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import h5py
        
        with h5py.File(weights_path, 'r') as f:
            # Get layer names from the file
            layer_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                          for n in f.attrs['layer_names']]
            
            # Map layer names to model layers
            layer_dict = {layer.name: layer for layer in model.layers}
            
            # Load weights for each layer
            for layer_name in layer_names:
                if layer_name in layer_dict:
                    layer = layer_dict[layer_name]
                    if layer_name in f:
                        g = f[layer_name]
                        weight_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                                      for n in g.attrs['weight_names']]
                        weights = []
                        for weight_name in weight_names:
                            if weight_name in g:
                                weights.append(np.array(g[weight_name]))
                        if weights:
                            layer.set_weights(weights)
                            
        return True
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False