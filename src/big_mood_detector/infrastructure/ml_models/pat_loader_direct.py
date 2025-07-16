"""
Direct PAT Weight Loader

A simpler approach that builds a minimal model structure and directly
loads the weights without complex layer reconstruction.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DirectPATModel:
    """
    Direct weight loading approach for PAT models.

    This bypasses TensorFlow model construction and directly uses the weights
    for inference, similar to how modern transformers handle weight loading.
    """

    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self.weights = {}
        self.config = self._get_config(model_size)
        self.is_loaded = False

    def _get_config(self, model_size: str) -> dict:
        """Get model configuration."""
        configs = {
            "small": {
                "patch_size": 18,
                "embed_dim": 96,
                "num_heads": 6,
                "ff_dim": 256,
                "num_layers": 1,
                "num_patches": 560,
            },
            "medium": {
                "patch_size": 18,
                "embed_dim": 96,
                "num_heads": 12,
                "ff_dim": 256,
                "num_layers": 2,
                "num_patches": 560,
            },
            "large": {
                "patch_size": 9,
                "embed_dim": 96,
                "num_heads": 12,
                "ff_dim": 256,
                "num_layers": 4,
                "num_patches": 1120,
            },
        }
        return configs[model_size]

    def load_weights(self, weights_path: Path) -> bool:
        """Load weights from H5 file."""
        try:
            with h5py.File(weights_path, "r") as f:
                # Load dense layer
                self.weights["dense_kernel"] = np.array(f["dense"]["dense"]["kernel:0"])
                self.weights["dense_bias"] = np.array(f["dense"]["dense"]["bias:0"])

                # Load transformer layers
                for i in range(1, self.config["num_layers"] + 1):
                    layer_prefix = f"encoder_layer_{i}"

                    # Attention weights
                    attn_prefix = f"{layer_prefix}_transformer/{layer_prefix}_attention"
                    self.weights[f"{layer_prefix}_q_kernel"] = np.array(
                        f[f"{attn_prefix}/query/kernel:0"]
                    )
                    self.weights[f"{layer_prefix}_q_bias"] = np.array(
                        f[f"{attn_prefix}/query/bias:0"]
                    )
                    self.weights[f"{layer_prefix}_k_kernel"] = np.array(
                        f[f"{attn_prefix}/key/kernel:0"]
                    )
                    self.weights[f"{layer_prefix}_k_bias"] = np.array(
                        f[f"{attn_prefix}/key/bias:0"]
                    )
                    self.weights[f"{layer_prefix}_v_kernel"] = np.array(
                        f[f"{attn_prefix}/value/kernel:0"]
                    )
                    self.weights[f"{layer_prefix}_v_bias"] = np.array(
                        f[f"{attn_prefix}/value/bias:0"]
                    )
                    self.weights[f"{layer_prefix}_out_kernel"] = np.array(
                        f[f"{attn_prefix}/attention_output/kernel:0"]
                    )
                    self.weights[f"{layer_prefix}_out_bias"] = np.array(
                        f[f"{attn_prefix}/attention_output/bias:0"]
                    )

                    # Layer norm weights
                    norm_prefix = f"{layer_prefix}_transformer/{layer_prefix}"
                    self.weights[f"{layer_prefix}_norm1_gamma"] = np.array(
                        f[f"{norm_prefix}_norm1/gamma:0"]
                    )
                    self.weights[f"{layer_prefix}_norm1_beta"] = np.array(
                        f[f"{norm_prefix}_norm1/beta:0"]
                    )
                    self.weights[f"{layer_prefix}_norm2_gamma"] = np.array(
                        f[f"{norm_prefix}_norm2/gamma:0"]
                    )
                    self.weights[f"{layer_prefix}_norm2_beta"] = np.array(
                        f[f"{norm_prefix}_norm2/beta:0"]
                    )

                    # FFN weights
                    self.weights[f"{layer_prefix}_ff1_kernel"] = np.array(
                        f[f"{norm_prefix}_ff1/kernel:0"]
                    )
                    self.weights[f"{layer_prefix}_ff1_bias"] = np.array(
                        f[f"{norm_prefix}_ff1/bias:0"]
                    )
                    self.weights[f"{layer_prefix}_ff2_kernel"] = np.array(
                        f[f"{norm_prefix}_ff2/kernel:0"]
                    )
                    self.weights[f"{layer_prefix}_ff2_bias"] = np.array(
                        f[f"{norm_prefix}_ff2/bias:0"]
                    )

            self.is_loaded = True
            logger.info(
                f"Successfully loaded {len(self.weights)} weight tensors for PAT-{self.model_size.upper()}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False

    def _multi_head_attention(self, x, layer_idx):
        """Compute multi-head attention using loaded weights."""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        prefix = f"encoder_layer_{layer_idx}"

        # Reshape weights from (embed_dim, num_heads, embed_dim) to (embed_dim, num_heads * embed_dim)
        q_kernel = self.weights[f"{prefix}_q_kernel"]  # Shape: (96, 12, 96)
        assert q_kernel.shape == (
            self.config["embed_dim"],
            self.config["num_heads"],
            self.config["embed_dim"],
        ), f"Q kernel shape mismatch: {q_kernel.shape}"
        q_kernel = tf.reshape(q_kernel, (q_kernel.shape[0], -1))  # Shape: (96, 1152)
        q_bias = tf.reshape(self.weights[f"{prefix}_q_bias"], (-1,))  # Shape: (1152,)

        k_kernel = self.weights[f"{prefix}_k_kernel"]
        k_kernel = tf.reshape(k_kernel, (k_kernel.shape[0], -1))
        k_bias = tf.reshape(self.weights[f"{prefix}_k_bias"], (-1,))

        v_kernel = self.weights[f"{prefix}_v_kernel"]
        v_kernel = tf.reshape(v_kernel, (v_kernel.shape[0], -1))
        v_bias = tf.reshape(self.weights[f"{prefix}_v_bias"], (-1,))

        # Project to Q, K, V
        q = tf.matmul(x, q_kernel) + q_bias
        k = tf.matmul(x, k_kernel) + k_bias
        v = tf.matmul(x, v_kernel) + v_bias

        # Reshape for multi-head attention
        num_heads = self.config["num_heads"]
        embed_dim = self.config["embed_dim"]

        q = tf.reshape(q, (batch_size, seq_len, num_heads, embed_dim))
        q = tf.transpose(q, [0, 2, 1, 3])

        k = tf.reshape(k, (batch_size, seq_len, num_heads, embed_dim))
        k = tf.transpose(k, [0, 2, 1, 3])

        v = tf.reshape(v, (batch_size, seq_len, num_heads, embed_dim))
        v = tf.transpose(v, [0, 2, 1, 3])

        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(embed_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Apply attention
        attention_output = tf.matmul(attention_weights, v)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(
            attention_output, (batch_size, seq_len, num_heads * embed_dim)
        )

        # Output projection
        # Reshape output kernel from (num_heads, embed_dim, embed_dim) to (num_heads * embed_dim, embed_dim)
        out_kernel = self.weights[f"{prefix}_out_kernel"]  # Shape: (12, 96, 96)
        out_kernel = tf.reshape(
            out_kernel, (-1, out_kernel.shape[-1])
        )  # Shape: (1152, 96)
        out_bias = self.weights[f"{prefix}_out_bias"]

        output = tf.matmul(attention_output, out_kernel) + out_bias

        return output

    def _get_sinusoidal_embeddings(self, num_patches, embed_dim):
        """Generate sinusoidal positional embeddings (from original PAT implementation)."""
        position = tf.range(num_patches, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, embed_dim, 2, dtype=tf.float32)
            * (-tf.math.log(10000.0) / embed_dim)
        )

        # Create sin and cos embeddings
        sin_embeddings = tf.sin(position * div_term)
        cos_embeddings = tf.cos(position * div_term)

        # Concatenate to get full positional embeddings
        pos_embeddings = tf.concat([sin_embeddings, cos_embeddings], axis=-1)

        # Add batch dimension
        return pos_embeddings[tf.newaxis, :, :]

    def _layer_norm(self, x, gamma, beta, epsilon=1e-6):
        """Apply layer normalization."""
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) / tf.sqrt(variance + epsilon)
        return x * gamma + beta

    def _transformer_block(self, x, layer_idx):
        """Process through one transformer block."""
        prefix = f"encoder_layer_{layer_idx}"

        # Self-attention
        attn_output = self._multi_head_attention(x, layer_idx)
        x = self._layer_norm(
            x + attn_output,
            self.weights[f"{prefix}_norm1_gamma"],
            self.weights[f"{prefix}_norm1_beta"],
        )

        # Feed-forward
        ff_output = (
            tf.matmul(x, self.weights[f"{prefix}_ff1_kernel"])
            + self.weights[f"{prefix}_ff1_bias"]
        )
        ff_output = tf.nn.relu(ff_output)
        ff_output = (
            tf.matmul(ff_output, self.weights[f"{prefix}_ff2_kernel"])
            + self.weights[f"{prefix}_ff2_bias"]
        )

        # Add & norm
        x = self._layer_norm(
            x + ff_output,
            self.weights[f"{prefix}_norm2_gamma"],
            self.weights[f"{prefix}_norm2_beta"],
        )

        return x

    @tf.function
    def extract_features(self, inputs):
        """Extract features from input sequence."""
        if not self.is_loaded:
            raise RuntimeError("Model weights not loaded")

        # Reshape to patches
        batch_size = tf.shape(inputs)[0]
        num_patches = self.config["num_patches"]
        patch_size = self.config["patch_size"]

        x = tf.reshape(inputs, (batch_size, num_patches, patch_size))

        # Linear projection
        x = tf.matmul(x, self.weights["dense_kernel"]) + self.weights["dense_bias"]

        # Add sinusoidal positional embeddings (matching original PAT implementation)
        pos_embeddings = self._get_sinusoidal_embeddings(
            num_patches, self.config["embed_dim"]
        )
        x = x + pos_embeddings

        # Process through transformer blocks
        for i in range(1, self.config["num_layers"] + 1):
            x = self._transformer_block(x, i)

        # Average pool over sequence
        features = tf.reduce_mean(x, axis=1)

        return features
