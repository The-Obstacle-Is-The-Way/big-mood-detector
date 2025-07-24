#!/usr/bin/env python3
"""
PAT-L Linear Probe Training (Stage 1)
Following exact hyperparameters from reference implementation
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import (  # noqa: E402
    NHANESProcessor,
)
from big_mood_detector.infrastructure.ml_models.pat_pytorch import (  # noqa: E402
    PATDepressionNet,
)


def prepare_data(
    processor: NHANESProcessor,
    subset: int = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Prepare data with proper preprocessing."""

    # Try to load cached data first
    cache_path = Path(f"data/cache/nhanes_pat_data_subset{subset}.npz")
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        return data['X_train'], data['X_val'], data['y_train'], data['y_val']

    # Load actigraphy and labels
    logger.info("Loading NHANES data...")
    actigraphy = processor.load_actigraphy("PAXMIN_H.xpt")
    depression = processor.load_depression_scores("DPQ_H.xpt")

    # Get subjects with both actigraphy and depression data
    common_subjects = set(actigraphy['participant_id'].unique()) & set(depression.keys())
    logger.info(f"Found {len(common_subjects)} subjects with both actigraphy and depression data")

    if subset:
        common_subjects = list(common_subjects)[:subset]
        logger.info(f"Using subset of {len(common_subjects)} subjects")

    # Prepare sequences and labels
    X, y, valid_indices = processor.prepare_pad_sequences_and_labels(
        actigraphy=actigraphy,
        depression_scores=depression,
        subjects=common_subjects,
        sequence_length=10080  # 7 days of minute-level data
    )

    # Apply valid indices
    X = X[valid_indices]
    y = y[valid_indices]

    logger.info(f"Prepared {len(X)} sequences with {sum(y)} positive samples ({100*sum(y)/len(y):.1f}%)")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # Cache the data
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val
    )
    logger.info(f"Cached data to {cache_path}")

    return X_train, X_val, y_train, y_val


def train_linear_probe(
    X_train, X_val, y_train, y_val,
    model_size: str = "large",
    epochs: int = 150,
    batch_size: int = 32,
    device: str = "cpu",
    learning_rate: float = 1e-4,
    output_dir: Path = None
):
    """Train PAT-L with frozen encoder (linear probe only)."""

    logger.info(f"Using device: {device}")

    # Create model with completely frozen encoder
    logger.info(f"Creating PAT-{model_size.upper()} depression model...")
    model = PATDepressionNet(
        model_size=model_size,
        unfreeze_last_n=0  # Freeze entire encoder
    )
    model = model.to(device)

    # Verify encoder is frozen
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters - Encoder: {encoder_params:,}, Head: {head_params:,}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Calculate positive weight for imbalanced dataset
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], dtype=torch.float32)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")

    # Loss and optimizer - following reference implementation
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=0.01  # Add weight decay as in reference
    )

    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6
    )

    # Training loop
    best_val_auc = 0
    patience_counter = 0
    early_stopping_patience = 50  # Much longer patience

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for _, (X_batch, y_batch) in enumerate(train_loader):
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
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits.squeeze(), y_batch)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.squeeze())
                all_labels.extend(y_batch.cpu().numpy())

        # Calculate metrics
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        val_auc = roc_auc_score(all_labels, all_preds)
        val_pr_auc = average_precision_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, np.array(all_preds) > 0.5)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                   f"val_auc={val_auc:.4f}, val_pr_auc={val_pr_auc:.4f}, val_f1={val_f1:.4f}")

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            # Save best model
            if output_dir:
                checkpoint_path = output_dir / f"pat_{model_size}_lp_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_pr_auc': val_pr_auc,
                    'val_f1': val_f1
                }, checkpoint_path)
                logger.info(f"Saved best model with AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Check if we've plateaued (for stage transition)
        if epoch > 20 and patience_counter > 10:
            recent_improvement = best_val_auc - val_auc
            if recent_improvement < 0.003:  # Less than 0.3% improvement
                logger.info(f"Model has plateaued at AUC: {best_val_auc:.4f}")

    return best_val_auc


def main():
    """Run linear probe training for PAT-L."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, help='Use subset of data for testing')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--model-size', default='large', choices=['small', 'medium', 'large'])
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='model_weights/pat/pytorch/pat_l_linear_probe')
    args = parser.parse_args()

    # Initialize processor
    processor = NHANESProcessor(
        data_dir=Path("data/nhanes/2013-2014"),
        output_dir=Path("data/nhanes/processed")
    )

    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data(
        processor,
        subset=args.subset
    )

    # Train with linear probe
    best_auc = train_linear_probe(
        X_train, X_val, y_train, y_val,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        learning_rate=args.learning_rate,
        output_dir=Path(args.output_dir)
    )

    logger.info(f"Training completed. Best validation AUC: {best_auc:.4f}")

    # Save training summary
    summary = {
        'model_size': args.model_size,
        'training_type': 'linear_probe',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'best_val_auc': best_auc,
        'timestamp': datetime.now().isoformat()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
