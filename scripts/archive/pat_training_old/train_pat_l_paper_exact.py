#!/usr/bin/env python3
"""
PAT-L Training - Exact Paper Replication for Depression (PHQ-9)
Target: 0.610 AUC as reported in the paper for PAT Conv-L

Based on careful reading of "AI Foundation Models for Wearable Movement Data in Mental Health Research"
Key findings:
- PAT Conv-L achieved 0.610 AUC averaged across dataset sizes
- Used "small feed forward layer and sigmoid activation" 
- Full fine-tuning (FT) not just linear probing
- Trained in under 6 hours on Colab Pro
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

from big_mood_detector.infrastructure.ml_models.pat_pytorch import (
    PATPyTorchEncoder,
)


class PATPaperDepressionNet(nn.Module):
    """
    Exact replication of paper's architecture:
    - PAT encoder (frozen initially, then full fine-tuning)
    - "Small feed forward layer" (we interpret as 2-layer based on typical usage)
    - Sigmoid activation for binary classification
    """

    def __init__(self, model_size: str = "large", hidden_dim: int = 128):
        super().__init__()

        # PAT encoder
        self.encoder = PATPyTorchEncoder(model_size=model_size)

        # "Small feed forward layer" - interpreting as 2-layer MLP
        # This is common in transformer fine-tuning
        self.head = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
            # Note: Sigmoid is in BCEWithLogitsLoss, not in model
        )

        # Initialize head properly
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)  # Small positive bias for binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        logits = self.head(embeddings)
        return logits

    def freeze_encoder(self):
        """Freeze encoder for initial training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen for full fine-tuning")


def load_and_fix_data():
    """Load cached data and fix normalization."""
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    if not cache_path.exists():
        raise FileNotFoundError(f"No cached data at {cache_path}")

    logger.info(f"Loading cached data from {cache_path}")
    data = np.load(cache_path)

    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']

    logger.info(f"Loaded {len(X_train)} train, {len(X_val)} val samples")

    # Check if normalization is bad (from previous findings)
    train_means = X_train.mean(axis=1)
    if train_means.std() < 0.01:
        logger.warning("Detected bad normalization - fixing...")

        # Reverse and recompute
        X_train_raw = X_train * 2.0 + 2.5
        X_val_raw = X_val * 2.0 + 2.5

        train_mean = X_train_raw.mean()
        train_std = X_train_raw.std()
        logger.info(f"Recomputed stats - Mean: {train_mean:.4f}, Std: {train_std:.4f}")

        X_train = (X_train_raw - train_mean) / train_std
        X_val = (X_val_raw - train_mean) / train_std

        logger.info(f"After fix - Train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")

    return X_train, X_val, y_train, y_val


def train_paper_replication():
    """Train PAT-L exactly as described in the paper."""

    # Load data
    X_train, X_val, y_train, y_val = load_and_fix_data()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create model
    model = PATPaperDepressionNet(model_size="large", hidden_dim=128)

    # Load pretrained weights
    weights_path = Path("model_weights/pat/pretrained/PAT-L_29k_weights.h5")
    if weights_path.exists():
        logger.info(f"Loading pretrained weights from {weights_path}")
        success = model.encoder.load_tf_weights(weights_path)
        if not success:
            logger.warning("Failed to load pretrained weights!")
    else:
        raise FileNotFoundError(f"Pretrained weights required at {weights_path}")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Paper mentions class imbalance handling
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training strategy based on paper:
    # 1. Initial training with frozen encoder (like linear probing)
    # 2. Full fine-tuning with unfrozen encoder

    logger.info("\n" + "="*60)
    logger.info("Stage 1: Training head only (frozen encoder)")
    logger.info("="*60)

    model.freeze_encoder()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    best_val_auc = 0
    patience_counter = 0

    # Stage 1: Train head only
    for epoch in range(20):
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                output = model(data).squeeze()
                val_preds.extend(torch.sigmoid(output).cpu().numpy())
                val_targets.extend(target.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        avg_loss = train_loss / len(train_loader)

        logger.info(f"Stage 1 - Epoch {epoch+1}/20: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'model_weights/pat/pytorch/pat_l_depression_stage1_best.pth')
            logger.info(f"✅ Saved best stage 1 model with AUC: {val_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping Stage 1")
                break

    logger.info(f"\nStage 1 complete. Best AUC: {best_val_auc:.4f}")

    # Load best stage 1 model
    model.load_state_dict(torch.load('model_weights/pat/pytorch/pat_l_depression_stage1_best.pth'))

    logger.info("\n" + "="*60)
    logger.info("Stage 2: Full fine-tuning (unfrozen encoder)")
    logger.info("="*60)

    model.unfreeze_encoder()

    # Different learning rates for encoder and head
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 5e-5},
        {'params': model.head.parameters(), 'lr': 5e-4}
    ])

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_overall_auc = best_val_auc
    patience_counter = 0

    # Stage 2: Full fine-tuning
    for epoch in range(50):
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                output = model(data).squeeze()
                val_preds.extend(torch.sigmoid(output).cpu().numpy())
                val_targets.extend(target.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        avg_loss = train_loss / len(train_loader)

        logger.info(f"Stage 2 - Epoch {epoch+1}/50: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")

        if val_auc > best_overall_auc:
            best_overall_auc = val_auc
            torch.save(model.state_dict(), 'model_weights/pat/pytorch/pat_l_depression_best.pth')
            logger.info(f"✅ Saved best overall model with AUC: {val_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("Early stopping Stage 2")
                break

        scheduler.step()

    logger.info("\n" + "="*60)
    logger.info(f"Training complete! Best AUC: {best_overall_auc:.4f}")
    logger.info("Target from paper: 0.610 for PAT Conv-L")
    logger.info("="*60)

    # Save training info
    training_info = {
        'best_auc': float(best_overall_auc),
        'model_size': 'large',
        'hidden_dim': 128,
        'total_params': total_params,
        'device': str(device),
        'completed_at': datetime.now().isoformat()
    }

    with open('model_weights/pat/pytorch/pat_l_depression_training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)


if __name__ == "__main__":
    # Create output directory
    Path("model_weights/pat/pytorch").mkdir(parents=True, exist_ok=True)

    # Run training
    train_paper_replication()
