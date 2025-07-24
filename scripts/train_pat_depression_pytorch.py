#!/usr/bin/env python3
"""
Production-Ready PyTorch PAT Depression Training

This implements the complete end-to-end training pipeline with:
- Pure PyTorch implementation (no TF/NumPy conversions)
- Proper gradient flow through encoder
- Two-tier learning rates
- Best practices from Clean Code, Karpathy, Hinton, Hassabis
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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
from big_mood_detector.infrastructure.ml_models.pat_pytorch import (
    PATDepressionNet,  # noqa: E402
)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop


def prepare_data(
    processor: NHANESProcessor,
    subset: int = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    cache_dir: Path = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data with caching for efficiency.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Check cache first
    if cache_dir:
        cache_file = cache_dir / f"nhanes_pat_data_subset{subset}.npz"
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            data = np.load(cache_file)
            return (data['X_train'], data['X_val'], data['X_test'],
                   data['y_train'], data['y_val'], data['y_test'])

    # Load data
    logger.info("Loading NHANES data...")
    actigraphy = processor.load_actigraphy("PAXMIN_H.xpt")
    depression = processor.load_depression_scores("DPQ_H.xpt")

    # Get subjects with both actigraphy and depression data
    subjects = set(actigraphy['SEQN'].unique()) & set(depression['SEQN'].unique())
    subjects = sorted(subjects)  # Sort for reproducibility

    if subset:
        logger.info(f"Using subset of {subset} subjects")
        subjects = subjects[:subset]
    else:
        logger.info(f"Using full dataset: {len(subjects)} subjects")

    # Extract sequences
    logger.info("Extracting PAT sequences...")
    sequences = []
    labels = []

    for i, seqn in enumerate(subjects):
        if i % 100 == 0:
            logger.info(f"Processing subject {i}/{len(subjects)}")

        try:
            # Extract full 10,080 sequence with NHANES standardization
            # Using built-in standardization since we have small subset
            sequence = processor.extract_pat_sequences(
                actigraphy,
                seqn,
                normalize=True,      # Log transform
                standardize=True     # Use NHANES z-score
            )

            if sequence.shape != (10080,):
                continue

            sequences.append(sequence)

            # Get depression label
            subj_depression = depression[depression['SEQN'] == seqn]
            if len(subj_depression) > 0:
                label = int(subj_depression['depressed'].iloc[0])
                labels.append(label)
            else:
                sequences.pop()

        except Exception as e:
            logger.warning(f"Error processing subject {seqn}: {e}")
            continue

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    logger.info(f"Prepared {len(sequences)} sequences")
    logger.info(f"Class distribution: {np.bincount(labels)}")
    logger.info(f"Positive rate: {labels.mean():.2%}")

    # Three-way split: train/val/test
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # Then split train/val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )

    # Skip sklearn StandardScaler - using NHANES built-in standardization
    # This avoids variance collapse with small subset
    logger.info("Using NHANES built-in standardization")

    # Cache if requested
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test
        )
        logger.info(f"Cached data to {cache_file}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_balanced_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    use_sampler: bool = True
) -> DataLoader:
    """Create dataloader with optional balanced sampling for training."""

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)

    if shuffle and use_sampler:
        # Create balanced sampler
        class_counts = np.bincount(y)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: PATDepressionNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    save_dir: Path,
    use_sampler: bool = True
) -> dict[str, Any]:
    """
    Train model with two-tier optimization and best practices.

    Returns:
        Dictionary of training history and metrics
    """
    # Calculate positive weight for loss
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())

    # When using balanced sampler, use pos_weight=1.0
    if use_sampler:
        pos_weight = 1.0
        logger.info("Using balanced sampler with pos_weight=1.0")
    else:
        # Only calculate class imbalance when not using sampler
        num_pos = sum(all_labels)
        num_neg = len(all_labels) - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        logger.info(f"Natural distribution with pos_weight: {pos_weight:.2f}")
    
    pos_weight_tensor = torch.tensor(pos_weight).to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Two-tier optimizer
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = model.head.parameters()

    param_groups = [
        {"params": encoder_params, "lr": config["encoder_lr"], "name": "encoder"},
        {"params": head_params, "lr": config["head_lr"], "name": "head"}
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])

    # Learning rate scheduler
    if config.get("scheduler") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"] - config.get("warmup_epochs", 0),
            eta_min=1e-6
        )
    elif config.get("scheduler") == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    else:
        scheduler = None

    # Early stopping
    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_pr_auc": [],
        "val_f1": [],
        "lr_encoder": [],
        "lr_head": []
    }

    best_val_auc = 0.0
    best_model_path = save_dir / "best_model.pt"

    # Training loop
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_losses = []

        # Sanity check on first epoch
        if epoch == 0:
            with torch.no_grad():
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_X)
                    logger.info("First batch sanity check:")
                    logger.info(f" y  : {batch_y[:8].cpu().tolist()}")
                    logger.info(f"log : {logits[:8].cpu().tolist()}")
                    break

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        all_probs = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        # Calculate metrics
        val_auc = roc_auc_score(all_labels, all_probs)
        val_pr_auc = average_precision_score(all_labels, all_probs)
        val_preds = (np.array(all_probs) > 0.5).astype(int)
        val_f1 = f1_score(all_labels, val_preds)

        # Calculate logit separation
        pos_logits = [logit for logit, label in zip(all_logits, all_labels, strict=False) if label == 1]
        neg_logits = [logit for logit, label in zip(all_logits, all_labels, strict=False) if label == 0]

        if pos_logits and neg_logits:
            logit_sep = np.mean(pos_logits) - np.mean(neg_logits)
        else:
            logit_sep = 0.0

        # Update history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_f1"].append(val_f1)
        history["lr_encoder"].append(optimizer.param_groups[0]["lr"])
        history["lr_head"].append(optimizer.param_groups[1]["lr"])

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
            f"val_auc={val_auc:.4f}, val_pr_auc={val_pr_auc:.4f}, "
            f"val_f1={val_f1:.4f}, logit_sep={logit_sep:.4f}"
        )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_pr_auc': val_pr_auc,
                'config': config
            }, best_model_path)
            logger.info(f"Saved best model with AUC: {val_auc:.4f}")

        # Learning rate scheduling
        if scheduler is not None:
            if config.get("scheduler") == "cosine":
                scheduler.step()
            elif config.get("scheduler") == "plateau":
                scheduler.step(val_auc)

        # Early stopping
        if early_stopping(val_auc):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return {
        'history': history,
        'best_val_auc': best_val_auc,
        'best_epoch': checkpoint['epoch'],
        'final_metrics': {
            'val_auc': checkpoint['val_auc'],
            'val_pr_auc': checkpoint['val_pr_auc']
        }
    }


def evaluate_model(
    model: PATDepressionNet,
    test_loader: DataLoader,
    device: torch.device
) -> dict[str, float]:
    """Evaluate model on test set."""

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    test_auc = roc_auc_score(all_labels, all_probs)
    test_pr_auc = average_precision_score(all_labels, all_probs)
    test_preds = (np.array(all_probs) > 0.5).astype(int)
    test_f1 = f1_score(all_labels, test_preds)

    return {
        'test_auc': test_auc,
        'test_pr_auc': test_pr_auc,
        'test_f1': test_f1
    }


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--subset', type=int, help='Use subset of data for testing')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--model-size', default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--unfreeze-layers', type=int, default=1)
    parser.add_argument('--head-lr', type=float, default=5e-3)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--early-stopping-patience', type=int, default=25)
    parser.add_argument('--cache-dir', type=Path, default=Path('data/cache'))
    parser.add_argument('--output-dir', type=Path, default=Path('model_weights/pat/pytorch'))
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'none'])
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Number of warmup epochs for cosine scheduler')
    parser.add_argument('--checkpoint', type=Path, help='Load from checkpoint to continue training')
    parser.add_argument('--cache-only', action='store_true', help='Only build cache, skip training')
    parser.add_argument('--no-sampler', action='store_true', help='Disable WeightedRandomSampler for natural class distribution')
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'head_lr': args.head_lr,
            'encoder_lr': args.encoder_lr,
            'weight_decay': args.weight_decay,
            'grad_clip': args.grad_clip,
            'early_stopping_patience': args.early_stopping_patience,
            'scheduler': args.scheduler,
            'warmup_epochs': args.warmup_epochs
        }

    # Set device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = NHANESProcessor(
        data_dir=Path("data/nhanes/2013-2014"),
        output_dir=Path("data/nhanes/processed")
    )

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        processor,
        subset=args.subset,
        cache_dir=args.cache_dir
    )

    # If cache-only mode, exit here
    if args.cache_only:
        logger.info("Cache built successfully. Exiting.")
        return

    # Create dataloaders
    train_loader = create_balanced_dataloader(X_train, y_train, args.batch_size, shuffle=True, use_sampler=not args.no_sampler)
    val_loader = create_balanced_dataloader(X_val, y_val, args.batch_size, shuffle=False, use_sampler=False)
    test_loader = create_balanced_dataloader(X_test, y_test, args.batch_size, shuffle=False, use_sampler=False)

    # Create model
    logger.info(f"Creating PAT-{args.model_size.upper()} depression model...")
    model = PATDepressionNet(
        model_size=args.model_size,
        unfreeze_last_n=args.unfreeze_layers
    )

    # Load pretrained encoder weights BEFORE moving to device
    model_size_map = {'small': 'S', 'medium': 'M', 'large': 'L'}
    weights_path = Path(f"model_weights/pat/pretrained/PAT-{model_size_map[args.model_size]}_29k_weights.h5")
    if not model.load_pretrained_encoder(weights_path):
        logger.error("Failed to load pretrained weights")
        return

    # Move to device AFTER loading weights
    model = model.to(device)

    # Count trainable parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Trainable parameters - Encoder: {encoder_params:,}, Head: {head_params:,}")
    logger.info(f"Total parameters: {total_params:,}")

    # Train model
    logger.info("Starting training...")
    results = train_model(
        model, train_loader, val_loader,
        config, device, args.output_dir,
        use_sampler=not args.no_sampler
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    results['test_metrics'] = test_metrics

    # Save results
    results_file = args.output_dir / f"results_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
    print(f"Test AUC: {test_metrics['test_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['test_pr_auc']:.4f}")
    print(f"Test F1: {test_metrics['test_f1']:.4f}")
    print(f"Results saved to: {results_file}")
    print("="*60)


if __name__ == "__main__":
    main()
# noqa: E402
