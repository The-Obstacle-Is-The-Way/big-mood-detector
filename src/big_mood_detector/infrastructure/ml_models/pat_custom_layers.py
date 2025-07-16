"""
Custom PAT Layers

Implements the exact attention mechanism used in the original PAT models
to enable proper weight loading from the H5 files.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PATMultiHeadAttention(layers.Layer):
    """
    Custom multi-head attention layer matching PAT's implementation.
    
    Uses separate Dense layers for Q/K/V projections with specific naming.
    """
    
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        dropout: float = 0.1,
        name: str = "attention",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        # Create separate layers for Q, K, V projections
        # Note: PAT uses full embed_dim output for each, not head_dim
        self.query_dense = layers.Dense(
            num_heads * embed_dim,  # Not head_dim!
            name=f"query"
        )
        self.key_dense = layers.Dense(
            num_heads * embed_dim,
            name=f"key"
        )
        self.value_dense = layers.Dense(
            num_heads * embed_dim,
            name=f"value"
        )
        
        # Output projection
        self.output_dense = layers.Dense(
            embed_dim,
            name=f"attention_output"
        )
        
        self.dropout_layer = layers.Dropout(dropout)
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Project inputs to Q, K, V
        # Shape: (batch, seq_len, num_heads * embed_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Reshape to separate heads
        # From: (batch, seq_len, num_heads * embed_dim)
        # To: (batch, num_heads, seq_len, embed_dim)
        query = tf.reshape(query, (batch_size, seq_len, self.num_heads, self.embed_dim))
        query = tf.transpose(query, [0, 2, 1, 3])
        
        key = tf.reshape(key, (batch_size, seq_len, self.num_heads, self.embed_dim))
        key = tf.transpose(key, [0, 2, 1, 3])
        
        value = tf.reshape(value, (batch_size, seq_len, self.num_heads, self.embed_dim))
        value = tf.transpose(value, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, value)
        
        # Reshape back
        # From: (batch, num_heads, seq_len, embed_dim)  
        # To: (batch, seq_len, num_heads * embed_dim)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(
            attention_output,
            (batch_size, seq_len, self.num_heads * self.embed_dim)
        )
        
        # Final projection
        output = self.output_dense(attention_output)
        
        return output


class PATTransformerBlock(layers.Layer):
    """
    Transformer block matching PAT's implementation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        name: str = "transformer_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        # Use custom attention
        self.attention = PATMultiHeadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout=dropout,
            name=f"{name.replace('_transformer', '_attention')}"
        )
        
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name.replace('_transformer', '_norm1')}"
        )
        
        # Feed-forward network with specific naming
        base_name = name.replace('_transformer', '')
        self.ff1 = layers.Dense(
            ff_dim,
            activation="relu",
            name=f"ff1"
        )
        self.ff2 = layers.Dense(
            embed_dim,
            name=f"ff2"
        )
        
        self.dropout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name.replace('_transformer', '_norm2')}"
        )
        
    def call(self, inputs, training=None):
        # Self-attention
        attn_output = self.attention(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        
        # Feed-forward
        ff_output = self.ff1(out1)
        ff_output = self.ff2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.norm2(out1 + ff_output)
        
        return out2


def create_pat_encoder_custom(
    patch_size: int,
    embed_dim: int,
    encoder_num_heads: int,
    encoder_ff_dim: int,
    encoder_num_layers: int,
    encoder_rate: float = 0.1,
    input_size: int = 10080
) -> keras.Model:
    """
    Create PAT encoder with custom layers matching the original implementation.
    """
    num_patches = input_size // patch_size
    
    # Input
    inputs = layers.Input(shape=(input_size,), name="inputs")
    
    # Reshape to patches
    patches = layers.Reshape(
        (num_patches, patch_size),
        name="reshape"
    )(inputs)
    
    # Linear projection
    encoded = layers.Dense(embed_dim, name="dense")(patches)
    
    # Add positional embeddings (simplified - original uses sinusoidal)
    # This is represented in the H5 as tf.__operators__.add
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=embed_dim,
        name="positional_embedding"
    )(positions)
    
    # Manual add operation to match the saved model structure
    encoded = encoded + position_embedding
    
    # Transformer blocks
    x = encoded
    for i in range(encoder_num_layers):
        x = PATTransformerBlock(
            embed_dim=embed_dim,
            num_heads=encoder_num_heads,
            ff_dim=encoder_ff_dim,
            dropout=encoder_rate,
            name=f"encoder_layer_{i+1}_transformer"
        )(x)
    
    return keras.Model(inputs=inputs, outputs=x, name="pat_encoder")


def build_pat_model_custom(model_size: str = "medium") -> keras.Model:
    """
    Build PAT model with custom layers.
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
        raise ValueError(f"Invalid model size: {model_size}")
    
    config = configs[model_size]
    return create_pat_encoder_custom(**config)


def load_pat_weights_custom(model: keras.Model, weights_path: str) -> bool:
    """
    Load PAT weights with custom weight mapping.
    """
    try:
        import h5py
        
        with h5py.File(weights_path, 'r') as f:
            # Get layer names
            layer_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                          for n in f.attrs['layer_names']]
            
            # Load weights for each layer
            for layer_name in layer_names:
                if layer_name in ['inputs', 'reshape', 'tf.__operators__.add_6',
                                 'tf.__operators__.add_8', 'tf.__operators__.add_12',
                                 'top_level_model_weights']:
                    # Skip non-weight layers
                    continue
                    
                if layer_name == 'dense' and layer_name in f:
                    # Load dense layer weights
                    layer = model.get_layer('dense')
                    kernel = np.array(f[layer_name]['dense']['kernel:0'])
                    bias = np.array(f[layer_name]['dense']['bias:0'])
                    layer.set_weights([kernel, bias])
                    print(f"✅ Loaded weights for layer: {layer_name}")
                    
                elif '_transformer' in layer_name and layer_name in f:
                    # Load transformer block weights
                    block = model.get_layer(layer_name)
                    g = f[layer_name]
                    
                    # Load attention weights
                    attn_name = layer_name.replace('_transformer', '_attention')
                    
                    # Q, K, V weights - need to reshape
                    for proj in ['query', 'key', 'value']:
                        kernel_path = f"{attn_name}/{proj}/kernel:0"
                        bias_path = f"{attn_name}/{proj}/bias:0"
                        
                        if kernel_path in g:
                            kernel = np.array(g[kernel_path])
                            bias = np.array(g[bias_path])
                            
                            # Reshape from (embed_dim, num_heads, embed_dim) to (embed_dim, num_heads * embed_dim)
                            kernel_reshaped = kernel.reshape(kernel.shape[0], -1)
                            bias_reshaped = bias.reshape(-1)
                            
                            # Set weights for the Dense layer
                            proj_layer = getattr(block.attention, f"{proj}_dense")
                            proj_layer.set_weights([kernel_reshaped, bias_reshaped])
                    
                    # Output projection
                    output_kernel = np.array(g[f"{attn_name}/attention_output/kernel:0"])
                    output_bias = np.array(g[f"{attn_name}/attention_output/bias:0"])
                    
                    # Reshape from (num_heads, embed_dim, embed_dim) to (num_heads * embed_dim, embed_dim)
                    output_kernel_reshaped = output_kernel.reshape(-1, output_kernel.shape[-1])
                    
                    block.attention.output_dense.set_weights([output_kernel_reshaped, output_bias])
                    
                    # Layer norms
                    norm1_name = layer_name.replace('_transformer', '_norm1')
                    norm1_gamma = np.array(g[f"{norm1_name}/gamma:0"])
                    norm1_beta = np.array(g[f"{norm1_name}/beta:0"])
                    block.norm1.set_weights([norm1_gamma, norm1_beta])
                    
                    norm2_name = layer_name.replace('_transformer', '_norm2')
                    norm2_gamma = np.array(g[f"{norm2_name}/gamma:0"])
                    norm2_beta = np.array(g[f"{norm2_name}/beta:0"])
                    block.norm2.set_weights([norm2_gamma, norm2_beta])
                    
                    # Feed-forward weights
                    ff1_name = layer_name.replace('_transformer', '_ff1')
                    ff1_kernel = np.array(g[f"{ff1_name}/kernel:0"])
                    ff1_bias = np.array(g[f"{ff1_name}/bias:0"])
                    block.ff1.set_weights([ff1_kernel, ff1_bias])
                    
                    ff2_name = layer_name.replace('_transformer', '_ff2')
                    ff2_kernel = np.array(g[f"{ff2_name}/kernel:0"])
                    ff2_bias = np.array(g[f"{ff2_name}/bias:0"])
                    block.ff2.set_weights([ff2_kernel, ff2_bias])
                    
                    print(f"✅ Loaded weights for transformer block: {layer_name}")
                    
        return True
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False