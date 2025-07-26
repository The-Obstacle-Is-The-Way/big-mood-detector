"""
PyTorch Implementation of Pretrained Actigraphy Transformer (PAT)

This provides a pure PyTorch implementation of the PAT encoder,
enabling end-to-end gradient flow for fine-tuning.
"""

import logging
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvPatchEmbedding(nn.Module):
    """
    Convolutional patch embedding for PAT-Conv variants.
    
    Uses 1D convolution to create patch embeddings instead of linear projection.
    This is the key architectural difference between PAT and PAT-Conv models.
    """
    
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 1D Conv layer to create patches
        self.conv = nn.Conv1d(
            in_channels=1,  # Single channel actigraphy data
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,  # Non-overlapping patches
            padding=0  # No padding
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert activity sequence to patch embeddings using convolution.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        # Add channel dimension if needed
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
            
        # Apply convolution
        x = self.conv(x)  # (B, embed_dim, num_patches)
        
        # Transpose to match transformer input format
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings matching the original PAT implementation."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))

        # TF concatenates sin and cos, not interleaves!
        sin_embeddings = torch.sin(position * div_term)
        cos_embeddings = torch.cos(position * div_term)
        pe = torch.cat([sin_embeddings, cos_embeddings], dim=-1)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input tensor."""
        seq_len = x.size(1)
        # Cast to Tensor for mypy - register_buffer ensures this is always a Tensor
        pe_buffer = cast(torch.Tensor, self.pe)
        pe_slice = pe_buffer[:, :seq_len]
        return x + pe_slice


class PATAttention(nn.Module):
    """Multi-head attention module matching PAT's implementation."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim  # PAT uses full embed_dim per head!

        # Q, K, V projections - PAT uses (embed_dim, num_heads, embed_dim) structure
        self.q_proj = nn.Linear(embed_dim, num_heads * embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_heads * embed_dim)
        self.v_proj = nn.Linear(embed_dim, num_heads * embed_dim)
        self.out_proj = nn.Linear(num_heads * embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.embed_dim ** -0.5  # Scale by sqrt(key_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.embed_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.embed_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.embed_dim)

        # Transpose for attention: (B, H, L, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)

        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        output_tensor: torch.Tensor = output
        return output_tensor


class PATBlock(nn.Module):
    """Transformer block matching PAT's architecture."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-6):
        super().__init__()

        # Self-attention
        self.attention = PATAttention(embed_dim, num_heads, dropout)

        # Layer norms (PAT uses eps=1e-6, not 1e-12)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Feed-forward network
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual (post-norm like reference)
        attn_out = self.attention(x)
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)  # Post-norm: normalize AFTER adding residual

        # Feed-forward with residual (post-norm like reference)
        ff_out = self.ff2(F.relu(self.ff1(x)))
        ff_out = self.dropout(ff_out)
        x = self.norm2(x + ff_out)  # Post-norm: normalize AFTER adding residual

        return x


class PATPyTorchEncoder(nn.Module):
    """PyTorch implementation of the PAT encoder."""

    def __init__(self, model_size: str = "small", dropout: float = 0.1):
        super().__init__()

        self.model_size = model_size
        self.config = self._get_config(model_size)

        # Patch embedding layer
        self.patch_embed = nn.Linear(self.config["patch_size"], self.config["embed_dim"])

        # Positional embeddings
        self.pos_embed = SinusoidalPositionalEmbedding(
            self.config["embed_dim"],
            max_len=self.config["num_patches"]
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PATBlock(
                embed_dim=self.config["embed_dim"],
                num_heads=self.config["num_heads"],
                ff_dim=self.config["ff_dim"],
                dropout=dropout,
                layer_norm_eps=1e-6
            )
            for _ in range(self.config["num_layers"])
        ])

        self.dropout = nn.Dropout(dropout)

    def _get_config(self, model_size: str) -> dict[str, Any]:
        """Get model configuration matching the original PAT."""
        configs = {
            "small": {
                "patch_size": 18,
                "embed_dim": 96,
                "num_heads": 6,
                "ff_dim": 256,
                "num_layers": 1,  # PAT-S has only 1 transformer block!
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PAT encoder.

        Args:
            x: Input tensor of shape (batch_size, 10080) for 7 days of minute-level data

        Returns:
            Embeddings of shape (batch_size, embed_dim)
        """
        batch_size = x.shape[0]

        # Reshape to patches: (B, 10080) -> (B, 560, 18)
        x = x.view(batch_size, self.config["num_patches"], self.config["patch_size"])

        # Patch embedding: (B, 560, 18) -> (B, 560, 96)
        x = self.patch_embed(x)

        # Add positional embeddings
        x = self.pos_embed(x)
        x = self.dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling: (B, 560, 96) -> (B, 96)
        x = x.mean(dim=1)

        return x

    def load_tf_weights(self, h5_path: Path) -> bool:
        """
        Load weights from TensorFlow H5 file and convert to PyTorch.

        This carefully maps the TF weight structure to our PyTorch model.
        """
        try:
            with h5py.File(h5_path, 'r') as f:
                # Load patch embedding (dense layer)
                self.patch_embed.weight.data = torch.from_numpy(
                    np.array(f['dense']['dense']['kernel:0']).T  # Transpose for PyTorch
                )
                self.patch_embed.bias.data = torch.from_numpy(
                    np.array(f['dense']['dense']['bias:0'])
                )

                # Load transformer blocks
                for i, block in enumerate(self.blocks):
                    layer_idx = i + 1  # TF uses 1-based indexing
                    layer_prefix = f"encoder_layer_{layer_idx}"

                    # Attention weights
                    attn_prefix = f"{layer_prefix}_transformer/{layer_prefix}_attention"

                    # Q, K, V projections
                    # TF stores as (embed_dim, num_heads, embed_dim) = (96, 6, 96)
                    # We need to reshape to (embed_dim, num_heads * embed_dim) = (96, 576)
                    # Then transpose for PyTorch Linear which expects (out_features, in_features)
                    q_kernel = np.array(f[f"{attn_prefix}/query/kernel:0"])  # (96, 6, 96)
                    q_kernel = q_kernel.reshape(self.config["embed_dim"], -1)  # (96, 576)
                    block.attention.q_proj.weight.data = torch.from_numpy(q_kernel.T)  # (576, 96)

                    # Bias is already (6, 96), just flatten to (576,)
                    q_bias = np.array(f[f"{attn_prefix}/query/bias:0"])  # (6, 96)
                    block.attention.q_proj.bias.data = torch.from_numpy(q_bias.flatten())

                    k_kernel = np.array(f[f"{attn_prefix}/key/kernel:0"])  # (96, 6, 96)
                    k_kernel = k_kernel.reshape(self.config["embed_dim"], -1)  # (96, 576)
                    block.attention.k_proj.weight.data = torch.from_numpy(k_kernel.T)  # (576, 96)
                    k_bias = np.array(f[f"{attn_prefix}/key/bias:0"])  # (6, 96)
                    block.attention.k_proj.bias.data = torch.from_numpy(k_bias.flatten())

                    v_kernel = np.array(f[f"{attn_prefix}/value/kernel:0"])  # (96, 6, 96)
                    v_kernel = v_kernel.reshape(self.config["embed_dim"], -1)  # (96, 576)
                    block.attention.v_proj.weight.data = torch.from_numpy(v_kernel.T)  # (576, 96)
                    v_bias = np.array(f[f"{attn_prefix}/value/bias:0"])  # (6, 96)
                    block.attention.v_proj.bias.data = torch.from_numpy(v_bias.flatten())

                    # Output projection
                    # TF stores as (num_heads, embed_dim, embed_dim) = (6, 96, 96)
                    # We need (num_heads * embed_dim, embed_dim) = (576, 96)
                    out_kernel = np.array(f[f"{attn_prefix}/attention_output/kernel:0"])  # (6, 96, 96)
                    out_kernel = out_kernel.reshape(-1, self.config["embed_dim"])  # (576, 96)
                    block.attention.out_proj.weight.data = torch.from_numpy(out_kernel.T)  # (96, 576)
                    block.attention.out_proj.bias.data = torch.from_numpy(
                        np.array(f[f"{attn_prefix}/attention_output/bias:0"])  # (96,)
                    )

                    # Layer norms
                    norm_prefix = f"{layer_prefix}_transformer/{layer_prefix}"
                    block.norm1.weight.data = torch.from_numpy(
                        np.array(f[f"{norm_prefix}_norm1/gamma:0"])
                    )
                    block.norm1.bias.data = torch.from_numpy(
                        np.array(f[f"{norm_prefix}_norm1/beta:0"])
                    )
                    block.norm2.weight.data = torch.from_numpy(
                        np.array(f[f"{norm_prefix}_norm2/gamma:0"])
                    )
                    block.norm2.bias.data = torch.from_numpy(
                        np.array(f[f"{norm_prefix}_norm2/beta:0"])
                    )

                    # Feed-forward layers
                    block.ff1.weight.data = torch.from_numpy(
                        np.array(f[f"{norm_prefix}_ff1/kernel:0"]).T
                    )
                    block.ff1.bias.data = torch.from_numpy(
                        np.array(f[f"{norm_prefix}_ff1/bias:0"])
                    )
                    block.ff2.weight.data = torch.from_numpy(
                        np.array(f[f"{norm_prefix}_ff2/kernel:0"]).T
                    )
                    block.ff2.bias.data = torch.from_numpy(
                        np.array(f[f"{norm_prefix}_ff2/bias:0"])
                    )

            logger.info(f"Successfully loaded TF weights for PAT-{self.model_size.upper()}")
            return True

        except Exception as e:
            logger.error(f"Failed to load TF weights: {e}")
            return False


class PATDepressionNet(nn.Module):
    """End-to-end PAT model for depression classification with fine-tuning support."""

    def __init__(self, model_size: str = "small", unfreeze_last_n: int = 1,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # PAT encoder
        self.encoder = PATPyTorchEncoder(model_size=model_size)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Freeze encoder except last N blocks
        self._freeze_encoder(unfreeze_last_n)

    def _freeze_encoder(self, unfreeze_last_n: int) -> None:
        """Freeze encoder parameters except last N transformer blocks."""
        # First freeze everything
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze patch embedding (often helpful)
        for param in self.encoder.patch_embed.parameters():
            param.requires_grad = True

        # Unfreeze last N transformer blocks
        if unfreeze_last_n > 0:
            num_blocks = len(self.encoder.blocks)
            start_idx = max(0, num_blocks - unfreeze_last_n)

            for i in range(start_idx, num_blocks):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True

            logger.info(f"Unfroze last {unfreeze_last_n} transformer blocks")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 10080)

        Returns:
            Logits of shape (batch_size,)
        """
        # Encode sequences
        embeddings = self.encoder(x)

        # Classification head
        logits = self.head(embeddings).squeeze(-1)

        logits_tensor: torch.Tensor = logits
        return logits_tensor

    def load_pretrained_encoder(self, h5_path: Path) -> bool:
        """Load pretrained encoder weights from TF checkpoint."""
        return self.encoder.load_tf_weights(h5_path)

