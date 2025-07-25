#!/usr/bin/env python3
"""
PAT-Conv-L Training Script
==========================

Implements PAT-Conv-L with convolutional patch embedding.
This is the variant that achieved 0.625 AUC in the paper vs 0.589 for standard PAT-L.

Conv-L = Same PAT-L transformer + Conv1D patch embedding instead of Linear
"""

import logging
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pat_conv_l_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from big_mood_detector.infrastructure.ml_models.pat_pytorch import PATPyTorchEncoder


class ConvPatchEmbedding(nn.Module):
    """
    Convolutional patch embedding for PAT-Conv-L.
    
    Applies 1D conv on full 24h sequence, then creates patches through stride.
    This is the key difference from standard PAT-L's linear patch embedding.
    """

    def __init__(self, patch_size: int, embed_dim: int, in_channels: int = 1, kernel_size: int = None):
        super().__init__()
        self.patch_size = patch_size

        # Use kernel_size = patch_size by default (sees full patch)
        if kernel_size is None:
            kernel_size = patch_size

        # Conv1D on full sequence with stride=patch_size creates patches
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,  # Direct to embed_dim, no projection needed
            kernel_size=kernel_size,
            stride=patch_size,
            padding=0  # No padding to avoid edge effects
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length) - full 24h sequence (10080 timesteps)
        Returns:
            (batch_size, num_patches, embed_dim) - sequence of patch embeddings
        """
        # Add channel dimension if needed: (batch_size, 1, sequence_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, T) -> (B, 1, T)

        # Apply 1D conv with stride=patch_size: (batch_size, embed_dim, num_patches)
        x = self.conv(x)  # (B, 1, T) -> (B, embed_dim, num_patches)

        # Permute to sequence format: (batch_size, num_patches, embed_dim)
        x = x.permute(0, 2, 1)  # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)

        return x


class PATConvLEncoder(PATPyTorchEncoder):
    """PAT-Conv-L: PAT-L with convolutional patch embedding instead of linear"""

    def __init__(self, model_size: str = "large", dropout: float = 0.1):
        # Initialize parent with dropout
        super().__init__(model_size, dropout=0.1)  # Paper uses 0.1 dropout

        # Replace linear patch embedding with conv variant
        self.patch_embed = ConvPatchEmbedding(
            patch_size=self.config["patch_size"],  # 9 for PAT-L
            embed_dim=self.config["embed_dim"],    # 96 for PAT-L
            in_channels=1  # Single channel actigraphy data
        )

        logger.info("Replaced linear patch embedding with ConvPatchEmbedding")
        logger.info(f"Patch size: {self.config['patch_size']}, Embed dim: {self.config['embed_dim']}")

    def load_tf_weights(self, h5_path: Path) -> bool:
        """
        Load weights from TensorFlow H5 file, but skip patch embedding since Conv is different.
        Only loads transformer block weights.
        """
        try:
            with h5py.File(h5_path, 'r') as f:
                # Skip patch embedding - Conv layer will start from random init
                logger.info("Skipping patch embedding weights (Conv layer will use random init)")

                # Load transformer blocks
                for i, block in enumerate(self.blocks):
                    layer_idx = i + 1  # TF uses 1-based indexing
                    layer_prefix = f"encoder_layer_{layer_idx}"

                    # Attention weights
                    attn_prefix = f"{layer_prefix}_transformer/{layer_prefix}_attention"

                    # Q, K, V projections
                    q_kernel = np.array(f[f"{attn_prefix}/query/kernel:0"])  # (96, 12, 96) for PAT-L
                    q_kernel = q_kernel.reshape(self.config["embed_dim"], -1)
                    block.attention.q_proj.weight.data = torch.from_numpy(q_kernel.T)
                    q_bias = np.array(f[f"{attn_prefix}/query/bias:0"])
                    block.attention.q_proj.bias.data = torch.from_numpy(q_bias.flatten())

                    k_kernel = np.array(f[f"{attn_prefix}/key/kernel:0"])
                    k_kernel = k_kernel.reshape(self.config["embed_dim"], -1)
                    block.attention.k_proj.weight.data = torch.from_numpy(k_kernel.T)
                    k_bias = np.array(f[f"{attn_prefix}/key/bias:0"])
                    block.attention.k_proj.bias.data = torch.from_numpy(k_bias.flatten())

                    v_kernel = np.array(f[f"{attn_prefix}/value/kernel:0"])
                    v_kernel = v_kernel.reshape(self.config["embed_dim"], -1)
                    block.attention.v_proj.weight.data = torch.from_numpy(v_kernel.T)
                    v_bias = np.array(f[f"{attn_prefix}/value/bias:0"])
                    block.attention.v_proj.bias.data = torch.from_numpy(v_bias.flatten())

                    # Output projection
                    out_kernel = np.array(f[f"{attn_prefix}/attention_output/kernel:0"])
                    out_kernel = out_kernel.reshape(-1, self.config["embed_dim"])
                    block.attention.out_proj.weight.data = torch.from_numpy(out_kernel.T)
                    block.attention.out_proj.bias.data = torch.from_numpy(
                        np.array(f[f"{attn_prefix}/attention_output/bias:0"])
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

            logger.info("Successfully loaded transformer weights for PAT-Conv-L")
            logger.info("Conv patch embedding initialized randomly (as intended)")
            return True

        except Exception as e:
            logger.error(f"Failed to load TF weights: {e}")
            return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conv patch embedding.
        
        Args:
            x: Input tensor of shape (batch_size, 10080) for 7 days of minute-level data
        Returns:
            Embeddings of shape (batch_size, embed_dim)
        """
        batch_size = x.shape[0]

        # Apply conv patch embedding to full sequence: (B, 10080) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # ConvPatchEmbedding handles the patching

        # Add positional embeddings
        x = self.pos_embed(x)
        x = self.dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling: (B, num_patches, embed_dim) -> (B, embed_dim)
        x = x.mean(dim=1)

        return x


class SimplePATConvLModel(nn.Module):
    """PAT-Conv-L with simple linear head for depression classification."""

    def __init__(self, model_size: str = "large"):
        super().__init__()

        self.encoder = PATConvLEncoder(model_size=model_size)
        self.head = nn.Linear(96, 1)  # 96 -> 1 for binary classification

        # Initialize head weights
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)  # (B, 96)
        logits = self.head(embeddings)  # (B, 1)
        return logits.squeeze()  # (B,)


def load_data():
    """Load the CORRECTED NHANES depression data."""
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    logger.info(f"Loading corrected data from {cache_path}")

    if not cache_path.exists():
        raise FileNotFoundError(f"Data file not found: {cache_path}")

    data = np.load(cache_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']

    logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Class balance - Train: {(y_train == 1).sum()}/{len(y_train)} positive")

    # Verify normalization is NOT the fixed values problem
    train_mean = X_train.mean()
    train_std = X_train.std()
    logger.info("Data statistics:")
    logger.info(f"  Train - Mean: {train_mean:.6f}, Std: {train_std:.6f}")
    logger.info(f"  Val - Mean: {X_val.mean():.6f}, Std: {X_val.std():.6f}")

    # Check if we have the bad fixed normalization
    if abs(train_mean - (-1.24)) < 0.01:
        logger.warning("❌ BAD NORMALIZATION DETECTED! Fixing automatically...")

        # Reverse bad normalization: X_raw = X_cached * 2.0 + 2.5
        X_train_raw = X_train * 2.0 + 2.5
        X_val_raw = X_val * 2.0 + 2.5

        # Compute proper stats from training data
        train_mean_raw = X_train_raw.mean()
        train_std_raw = X_train_raw.std()

        # Apply proper normalization
        X_train = (X_train_raw - train_mean_raw) / train_std_raw
        X_val = (X_val_raw - train_mean_raw) / train_std_raw

        logger.info("✅ Fixed normalization:")
        logger.info(f"  Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
        logger.info(f"  Val - Mean: {X_val.mean():.6f}, Std: {X_val.std():.6f}")
    else:
        logger.info("✅ Normalization looks good - proceeding")

    return X_train, X_val, y_train, y_val


def main():
    logger.info("="*60)
    logger.info("PAT-Conv-L Training for Depression Classification")
    logger.info("Target: 0.625 AUC (paper's Conv-L result)")
    logger.info("Architecture: PAT-L + Conv1D patch embedding")
    logger.info("="*60)

    # Load data
    X_train, X_val, y_train, y_val = load_data()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create PAT-Conv-L model
    model = SimplePATConvLModel(model_size="large")

    # Try to load PAT-L pretrained weights (transformer layers only, conv layer random)
    weights_path = Path("model_weights/pat/pretrained/PAT-L_29k_weights.h5")
    if weights_path.exists():
        logger.info("Loading PAT-L pretrained transformer weights...")
        try:
            # Only load transformer weights, conv patch embedding starts random
            success = model.encoder.load_tf_weights(weights_path)
            if success:
                logger.info("✅ Loaded transformer weights, conv layer initialized randomly")
            else:
                logger.warning("Failed to load weights, training from scratch")
        except Exception as e:
            logger.warning(f"Weight loading failed: {e}, training from scratch")
    else:
        logger.warning("No pretrained weights found, training from scratch")
        logger.info("Note: For best results, download PAT-L weights from the official repo")

    model = model.to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_params = sum(p.numel() for p in model.encoder.patch_embed.parameters())

    logger.info("Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Conv patch embedding: {conv_params:,}")

    # Create datasets
    batch_size = 32
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # Loss with class weighting for imbalanced data
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f} for class imbalance")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer - Updated based on feedback
    # Separate conv patch embedding params (need higher LR as they start from scratch)
    conv_params = list(model.encoder.patch_embed.parameters())
    # Rest of encoder params (pretrained, need lower LR)
    encoder_params = [p for name, p in model.encoder.named_parameters()
                      if 'patch_embed' not in name]
    head_params = list(model.head.parameters())

    # Paper's learning rates for LP phase
    # Start with Linear Probing (LP) - freeze encoder
    do_linear_probe = True
    lp_epochs = 5

    if do_linear_probe:
        encoder_lr = 0  # Frozen during LP
        head_lr = 5e-4  # Only train head
        logger.info("Starting with Linear Probing (LP) phase - encoder frozen")
    else:
        encoder_lr = 3e-5  # Paper's encoder LR
        head_lr = 5e-4    # Paper's head LR

    logger.info("Optimizer configuration:")
    logger.info(f"  - Encoder LR: {encoder_lr} (higher due to random conv layer)")
    logger.info(f"  - Head LR: {head_lr}")

    # Use AdamW with weight decay as per paper
    # Three param groups: encoder (pretrained), conv patch (random), head
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': 0.01},  # Pretrained encoder
        {'params': conv_params, 'lr': 1e-3, 'weight_decay': 0.01},  # Random conv - higher LR
        {'params': head_params, 'lr': head_lr, 'weight_decay': 0.01}  # Classification head
    ], betas=(0.9, 0.95))

    # Gradient accumulation for stability (optional)
    grad_accumulation_steps = 2
    logger.info(f"Using gradient accumulation: {grad_accumulation_steps} steps")

    # Calculate total steps for scheduler
    steps_per_epoch = len(train_loader) // grad_accumulation_steps
    total_steps = steps_per_epoch * 20  # 20 epochs total
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    # Linear warmup + cosine decay
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    logger.info("\n" + "="*50)
    logger.info("Starting PAT-Conv-L Training")
    logger.info("Expected progression:")
    logger.info("  - Epochs 1-3: Warmup (~0.52-0.55 AUC)")
    logger.info("  - Epochs 4-8: Rapid improvement (0.55-0.58 AUC)")
    logger.info("  - Epochs 8-15: Target range (0.58-0.625 AUC)")
    logger.info("="*50)

    # Log dataset info
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")

    best_auc = 0.0
    patience = 0
    max_patience = 10  # Increased patience for longer plateau periods

    # Track global step for scheduler
    global_step = 0

    # Phase 1: Linear Probing
    if do_linear_probe:
        logger.info("\n" + "="*50)
        logger.info("Phase 1: Linear Probing (5 epochs)")
        logger.info("="*50)

        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

    for epoch in range(20):  # Total 20 epochs (5 LP + 15 FT)
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)}")

            # Extra logging for fine-tuning phase debugging
            if epoch >= 5 and batch_idx % 5 == 0:
                logger.info(f"    FT Debug - Batch {batch_idx}, LR: {optimizer.param_groups[0]['lr']:.2e}")

            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss = loss / grad_accumulation_steps  # Scale loss for accumulation
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                # Step scheduler per batch (not per epoch)
                scheduler.step()
                global_step += 1

            train_loss += loss.item() * grad_accumulation_steps

        # Final optimizer step if needed
        if len(train_loader) % grad_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # Calculate validation loss
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item()

                # Collect predictions
                probs = torch.sigmoid(output.squeeze())
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # Calculate metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Get current learning rates
        encoder_lr_current = optimizer.param_groups[0]['lr']
        conv_lr_current = optimizer.param_groups[1]['lr']
        head_lr_current = optimizer.param_groups[2]['lr']

        # Sanity check for fine-tuning phase (skip on transition epoch)
        if epoch > lp_epochs:  # > instead of >= to skip the transition epoch
            assert encoder_lr_current > 0, f"Encoder LR is {encoder_lr_current} - should be > 0!"

        logger.info(f"Epoch {epoch+1:2d}: "
                   f"Train Loss={avg_train_loss:.4f}, "
                   f"Val Loss={avg_val_loss:.4f}, "
                   f"Val AUC={val_auc:.4f}, "
                   f"Enc LR={encoder_lr_current:.2e}, "
                   f"Conv LR={conv_lr_current:.2e}, "
                   f"Head LR={head_lr_current:.2e}")

        # Check if we need to switch from LP to FT
        if do_linear_probe and epoch == lp_epochs - 1:
            logger.info("\n" + "="*50)
            logger.info("Switching to Fine-Tuning phase - unfreezing encoder")
            logger.info("="*50)

            # Unfreeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = True

            # Update optimizer with new LRs AND initial_lr (for scheduler)
            encoder_lr = 3e-5
            conv_lr = 5e-4  # Higher for random init conv
            optimizer.param_groups[0]['lr'] = encoder_lr  # Encoder
            optimizer.param_groups[1]['lr'] = conv_lr     # Conv patch

            # CRITICAL: Also update initial_lr for scheduler to use
            optimizer.param_groups[0]['initial_lr'] = encoder_lr
            optimizer.param_groups[1]['initial_lr'] = conv_lr

            # CRITICAL: Recreate scheduler to pick up new initial_lr values
            logger.info("Recreating scheduler with new learning rates...")
            # Create new lambda that accounts for current global step
            current_global_step = global_step
            def lr_lambda_ft(current_step):
                # Offset by steps already taken
                effective_step = current_step - current_global_step

                # CRITICAL: Return 1.0 on creation to preserve initial LRs
                if effective_step <= 0:
                    return 1.0

                # Warmup phase
                if effective_step < warmup_steps:
                    return float(effective_step) / float(max(1, warmup_steps))

                # Cosine decay after warmup
                progress = float(effective_step - warmup_steps) / float(max(1, total_steps - warmup_steps - current_global_step))
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Min 10% LR

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_ft)

            # CRITICAL: Force the scheduler to respect our LRs
            # The LambdaLR multiplies by the lambda value, but we need to ensure
            # it doesn't zero out our carefully set LRs
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']

            # Step once to apply lambda(0) = 1.0
            scheduler.step()
            global_step += 1

            # Verify LRs after scheduler recreation and step
            logger.info("LRs after scheduler recreation and initial step:")
            logger.info(f"  Encoder: {optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"  Conv: {optimizer.param_groups[1]['lr']:.2e}")
            logger.info(f"  Head: {optimizer.param_groups[2]['lr']:.2e}")

            # Reset patience for fine-tuning phase
            patience = 0
            logger.info("Reset patience counter for fine-tuning phase")

            do_linear_probe = False

        # Save best model with versioned filename
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0

            # Include LR in filename to avoid overwrites
            lr_str = f"{int(encoder_lr * 1e5)}e5"
            save_path = f"model_weights/pat/pytorch/pat_conv_l_best_{lr_str}.pth"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc,
                'config': {
                    'model_size': 'large',
                    'encoder_lr': encoder_lr,
                    'head_lr': head_lr,
                    'batch_size': batch_size,
                    'grad_accumulation': grad_accumulation_steps
                }
            }, save_path)

            logger.info(f"✅ Saved best PAT-Conv-L with AUC: {val_auc:.4f}")

            # Progress milestones
            if val_auc >= 0.58:
                logger.info("🎯 EXCELLENT: Beat standard PAT-L baseline!")
            if val_auc >= 0.60:
                logger.info("🚀 OUTSTANDING: Strong performance!")
            if val_auc >= 0.625:
                logger.info("🏆 TARGET REACHED! Matched paper's Conv-L result!")
                break

        else:
            patience += 1
            logger.info(f"No improvement. Patience: {patience}/{max_patience}")

            if patience >= max_patience:
                logger.info("Early stopping triggered")
                break

    # Final results
    logger.info("\n" + "="*50)
    logger.info("PAT-Conv-L Training Complete!")
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    logger.info("Baseline comparisons:")
    logger.info("  - Standard PAT-L: ~0.56-0.58")
    logger.info("  - Paper's Conv-L target: 0.625")

    # Performance assessment
    if best_auc > 0.58:
        logger.info("✅ SUCCESS: Outperformed standard PAT-L!")
    if best_auc >= 0.60:
        logger.info("🎯 EXCELLENT: Strong clinical performance!")
    if best_auc >= 0.625:
        logger.info("🏆 PERFECT: Matched paper's benchmark!")
    else:
        logger.info(f"📈 Gap to target: {0.625 - best_auc:.3f} AUC points")

    logger.info("="*50)


if __name__ == "__main__":
    main()
