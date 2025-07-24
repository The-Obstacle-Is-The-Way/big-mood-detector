#!/usr/bin/env python3
"""
PAT-L Training - Ready to Run Version
Simplified from advanced script, focusing on core training
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.ml_models.pat_pytorch import (  # noqa: E402
    PATPyTorchEncoder,
)


class SimplePATDepressionNet(nn.Module):
    """Simple PAT depression model with single layer head for initial training."""

    def __init__(self, model_size: str = "large", dropout: float = 0.1):
        super().__init__()

        # PAT encoder
        self.encoder = PATPyTorchEncoder(model_size=model_size)

        # Simple head to start
        self.head = nn.Sequential(
            nn.Linear(96, 1),
            nn.Dropout(dropout)
        )

        # Freeze encoder initially
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        logits = self.head(embeddings)
        return logits

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze last n transformer blocks."""
        if n > 0:
            num_blocks = len(self.encoder.blocks)
            start_idx = max(0, num_blocks - n)

            for i in range(start_idx, num_blocks):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True

            logger.info(f"Unfroze last {n} transformer blocks")


def load_and_fix_data():
    """Load cached data and fix normalization."""

    # Load cached data
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached data at {cache_path}. Run original training first.")

    logger.info(f"Loading cached data from {cache_path}")
    data = np.load(cache_path)

    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']

    logger.info(f"Loaded {len(X_train)} train, {len(X_val)} val samples")

    # Check if normalization is bad
    train_means = X_train.mean(axis=1)
    if train_means.std() < 0.01:
        logger.warning("Detected bad normalization - fixing...")

        # Reverse the bad normalization
        X_train_raw = X_train * 2.0 + 2.5
        X_val_raw = X_val * 2.0 + 2.5

        # Compute proper stats from training data
        train_mean = X_train_raw.mean()
        train_std = X_train_raw.std()

        logger.info(f"Recomputed stats - Mean: {train_mean:.4f}, Std: {train_std:.4f}")

        # Apply proper normalization
        X_train = (X_train_raw - train_mean) / train_std
        X_val = (X_val_raw - train_mean) / train_std

        # Verify
        logger.info(f"After fix - Train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")

    return X_train, X_val, y_train, y_val


def train():
    """Main training function."""

    # Load and fix data
    X_train, X_val, y_train, y_val = load_and_fix_data()

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = SimplePATDepressionNet(model_size="large", dropout=0.1)

    # Load pretrained weights
    weights_path = Path("model_weights/pat/pretrained/PAT-L_29k_weights.h5")
    if weights_path.exists():
        logger.info(f"Loading pretrained weights from {weights_path}")
        success = model.encoder.load_tf_weights(weights_path)
        if not success:
            logger.warning("Failed to load pretrained weights!")
    else:
        logger.warning(f"No pretrained weights at {weights_path}")

    model = model.to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters - Trainable: {trainable:,}, Total: {total:,}")

    # Data loaders
    batch_size = 32
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Loss with class weighting
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")

    # Stage 1: Train head only with high LR
    epochs_stage1 = 30
    lr_stage1 = 5e-3

    logger.info(f"\n🔵 Stage 1: Training head only for {epochs_stage1} epochs (LR={lr_stage1})")

    optimizer = optim.AdamW(model.parameters(), lr=lr_stage1, weight_decay=0.01)

    best_auc = 0
    output_dir = Path("model_weights/pat/pytorch/pat_l_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs_stage1):
        # Training
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits.squeeze(), y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.squeeze())
                all_labels.extend(y_batch.numpy())

        # Metrics
        val_auc = roc_auc_score(all_labels, all_preds)

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs_stage1}: Loss={avg_train_loss:.4f}, AUC={val_auc:.4f}")

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc
            }, output_dir / f"best_stage1_auc_{val_auc:.4f}.pt")
            logger.info(f"✅ Saved best model with AUC: {val_auc:.4f}")

    # Stage 2: Unfreeze last 2 blocks
    if best_auc > 0.50:  # Only continue if stage 1 worked
        epochs_stage2 = 30
        lr_stage2 = 1e-4

        logger.info(f"\n🟢 Stage 2: Unfreezing last 2 blocks for {epochs_stage2} epochs (LR={lr_stage2})")

        model.unfreeze_last_n_blocks(2)

        # New optimizer with lower LR
        optimizer = optim.AdamW(model.parameters(), lr=lr_stage2, weight_decay=0.01)

        for epoch in range(epochs_stage2):
            # Training
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits.squeeze(), y_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs.squeeze())
                    all_labels.extend(y_batch.numpy())

            val_auc = roc_auc_score(all_labels, all_preds)

            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Stage 2 - Epoch {epoch+1}/{epochs_stage2}: Loss={avg_train_loss:.4f}, AUC={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'epoch': epoch + epochs_stage1,
                    'model_state_dict': model.state_dict(),
                    'val_auc': val_auc
                }, output_dir / f"best_overall_auc_{val_auc:.4f}.pt")
                logger.info(f"✅ New best AUC: {val_auc:.4f}")

    logger.info(f"\n🎉 Training complete! Best AUC: {best_auc:.4f}")
    logger.info(f"Models saved to: {output_dir}")

    # Save summary
    summary = {
        'best_auc': best_auc,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'train_samples': len(X_train),
        'val_samples': len(X_val)
    }

    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    train()
