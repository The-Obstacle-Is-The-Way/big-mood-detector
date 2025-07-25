#!/usr/bin/env python3
"""
PAT Stable Training Script
==========================

This is our SSOT (Single Source of Truth) training script.
Based on what actually worked: Simple approach that achieved 0.5924 AUC.

Key findings:
- Simple optimizer setup works best (uniform LR)
- Data normalization must be mean=0, std=1
- No complex scheduler gymnastics needed
"""

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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pat_stable_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pat_training.train_pat_conv_l import SimplePATConvLModel, load_data
from scripts.pat_training.train_pat_l_corrected import SimplePATDepressionModel


def train_stable_pat(model_type="pat-l", conv=False):
    """
    Train PAT with stable, proven configuration.
    
    Args:
        model_type: "pat-s", "pat-m", or "pat-l"
        conv: Whether to use Conv variant
    """
    logger.info("="*60)
    logger.info(f"PAT STABLE Training - {model_type.upper()}{' Conv' if conv else ''}")
    logger.info("Using configuration that achieved 0.5924 AUC")
    logger.info("="*60)

    # Load and verify data
    X_train, X_val, y_train, y_val = load_data()

    # Critical data check
    train_mean, train_std = X_train.mean(), X_train.std()
    logger.info(f"Data statistics - Mean: {train_mean:.6f}, Std: {train_std:.6f}")

    if abs(train_std - 1.0) > 0.1:
        logger.error("DATA NORMALIZATION ERROR! Expected std~1.0")
        return None

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    size_to_full = {"s": "small", "m": "medium", "l": "large"}
    full_size = size_to_full[model_type.split('-')[1]]

    if conv:
        model = SimplePATConvLModel(model_size=full_size)
    else:
        model = SimplePATDepressionModel(model_size=full_size)

    # Load pretrained weights
    size_map = {"s": "S", "m": "M", "l": "L"}
    size = size_map[model_type.split('-')[1]]
    weights_path = Path(f"model_weights/pat/pretrained/PAT-{size}_29k_weights.h5")

    if weights_path.exists():
        logger.info(f"Loading pretrained weights from {weights_path}")
        success = model.encoder.load_tf_weights(weights_path)
        if success:
            logger.info("✅ Successfully loaded pretrained weights")
    else:
        logger.warning(f"No pretrained weights found at {weights_path}")

    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Create datasets
    batch_size = 32
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Loss function
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    logger.info(f"Class imbalance - pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # PROVEN OPTIMIZER CONFIGURATION
    base_lr = 1e-4  # This worked!
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),  # Standard AdamW betas
        weight_decay=0.01
    )

    # Simple cosine scheduler
    total_epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * total_epochs,
        eta_min=base_lr * 0.1
    )

    logger.info(f"Optimizer: AdamW with base LR={base_lr}")
    logger.info(f"Training for {total_epochs} epochs")

    # Training loop
    best_auc = 0.0
    patience_counter = 0
    max_patience = 5

    for epoch in range(total_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item()

                probs = torch.sigmoid(output.squeeze())
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # Calculate metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1:2d}/{total_epochs}: "
                   f"Train Loss={avg_train_loss:.4f}, "
                   f"Val Loss={avg_val_loss:.4f}, "
                   f"Val AUC={val_auc:.4f}, "
                   f"LR={current_lr:.2e}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            save_dir = Path("model_weights/pat/stable")
            save_dir.mkdir(parents=True, exist_ok=True)

            model_name = f"{model_type}{'_conv' if conv else ''}"
            save_path = save_dir / f"{model_name}_best.pth"

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc,
                'config': {
                    'model_type': model_type,
                    'conv': conv,
                    'base_lr': base_lr,
                    'batch_size': batch_size,
                    'total_epochs': total_epochs
                }
            }, save_path)

            logger.info(f"✅ Saved best model with AUC: {val_auc:.4f}")

            # Milestones
            if val_auc >= 0.58:
                logger.info("🎯 Excellent: Matched paper's PAT baseline!")
            if val_auc >= 0.60:
                logger.info("🚀 Outstanding: Strong performance!")
            if val_auc >= 0.625:
                logger.info("🏆 Target reached: Matched Conv-L paper result!")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    logger.info("Model saved to: model_weights/pat/stable/")
    logger.info("="*60)

    return best_auc


def run_multiple_seeds(model_type="pat-l", conv=False, n_runs=3):
    """Run training multiple times with different seeds for stability check."""
    results = []

    for i in range(n_runs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {i+1}/{n_runs} - Random seed: {42+i}")
        logger.info(f"{'='*60}")

        # Set seeds for reproducibility
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)

        auc = train_stable_pat(model_type, conv)
        if auc is not None:
            results.append(auc)

    if results:
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary of {len(results)} runs:")
        logger.info(f"AUCs: {results}")
        logger.info(f"Mean AUC: {np.mean(results):.4f} ± {np.std(results):.4f}")
        logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stable PAT Training")
    parser.add_argument("--model", default="pat-l", choices=["pat-s", "pat-m", "pat-l"])
    parser.add_argument("--conv", action="store_true", help="Use Conv variant")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")

    args = parser.parse_args()

    if args.runs > 1:
        run_multiple_seeds(args.model, args.conv, args.runs)
    else:
        train_stable_pat(args.model, args.conv)
