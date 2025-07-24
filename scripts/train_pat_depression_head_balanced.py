#!/usr/bin/env python3
"""
Train PAT Depression Classification Head - Balanced Version

This script implements the fixes from the imbalanced learning checklist:
1. Proper data balancing with undersampling
2. BCE loss instead of Focal Loss with class weights
3. PR-AUC metric tracking
4. Proper threshold optimization
5. Balanced mini-batches
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import (
    NHANESProcessor,
)
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedTaskHead(nn.Module):
    """Task-specific head with proper initialization for imbalanced data."""

    def __init__(
        self,
        input_dim: int = 96,  # PAT-S uses 96-dim embeddings
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Single logit for BCE
        )

        # Proper initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better gradient flow."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # CRITICAL: Zero bias to start neutral
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)  # Shape: (batch_size,)


def prepare_balanced_data(
    nhanes_dir: Path,
    pat_model: Any,
    test_size: float = 0.2,
    val_size: float = 0.2,
    balance_strategy: str = 'undersample',
    random_seed: int = 42,
    subset: int = None
) -> tuple[dict, dict, dict]:
    """
    Prepare training data with proper balancing.

    Args:
        nhanes_dir: Directory with NHANES data
        pat_model: PAT model for feature extraction
        test_size: Test set proportion
        val_size: Validation set proportion
        balance_strategy: 'undersample', 'none', or 'weighted'
        random_seed: Random seed

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    logger.info("Loading NHANES data...")
    processor = NHANESProcessor(data_dir=nhanes_dir)

    # Load actigraphy and depression data
    actigraphy_df = processor.load_actigraphy("PAXMIN_H.xpt")
    depression_df = processor.load_depression_scores("DPQ_H.xpt")

    # Get participants with both actigraphy and depression data
    actigraphy_subjects = set(actigraphy_df['SEQN'].unique())
    depression_subjects = set(depression_df['SEQN'].unique())
    common_subjects = list(actigraphy_subjects & depression_subjects)

    logger.info(f"Found {len(common_subjects)} subjects with complete data")

    # Apply subset if specified
    if subset is not None:
        np.random.seed(random_seed)
        np.random.shuffle(common_subjects)
        common_subjects = common_subjects[:subset]
        logger.info(f"Using subset of {subset} subjects for smoke test")

    # Extract features for each subject
    embeddings_list = []
    labels_list = []
    subject_ids_list = []

    for i, subject_id in enumerate(common_subjects):
        if i % 100 == 0:
            logger.info(f"Processing subject {i+1}/{len(common_subjects)}")

        try:
            # Extract 7-day PAT sequences
            pat_sequences = processor.extract_pat_sequences(
                actigraphy_df,
                subject_id=subject_id,
                normalize=True
            )

            if pat_sequences.shape[0] < 7:
                continue

            # Use last 7 days
            sequence_7d = pat_sequences[-7:]
            activity_flat = sequence_7d.flatten()

            # Create PATSequence object
            pat_sequence = PATSequence(
                end_date=datetime.now().date(),
                activity_values=activity_flat,
                missing_days=[],
                data_quality_score=1.0
            )

            # Extract PAT embeddings
            embeddings = pat_model.extract_features(pat_sequence)

            # Get depression label
            subject_depression = depression_df[depression_df['SEQN'] == subject_id]
            phq9_score = subject_depression['PHQ9_total'].iloc[0]
            depression_label = int(phq9_score >= 10) if not np.isnan(phq9_score) else None

            if depression_label is not None:
                embeddings_list.append(embeddings)
                labels_list.append(depression_label)
                subject_ids_list.append(subject_id)

        except Exception as e:
            logger.warning(f"Error processing subject {subject_id}: {e}")
            continue

    # Convert to arrays
    X = np.array(embeddings_list)
    y = np.array(labels_list)
    subject_ids = np.array(subject_ids_list)

    logger.info(f"Extracted {len(X)} samples")
    logger.info(f"Class distribution: {np.bincount(y)} (negative, positive)")
    logger.info(f"Class ratio: 1:{np.bincount(y)[0] / np.bincount(y)[1]:.1f}")

    # First split: separate test set (stratified)
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, subject_ids,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    # Second split: train vs validation (stratified)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=y_temp
    )

    # Apply balancing strategy to training data only
    if balance_strategy == 'undersample':
        logger.info("Applying random undersampling to training data...")
        rus = RandomUnderSampler(random_state=random_seed)
        X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
        logger.info(f"After undersampling: {len(X_train_balanced)} samples")
        logger.info(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
        X_train, y_train = X_train_balanced, y_train_balanced

    logger.info(f"Final split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (
        {'X': X_train, 'y': y_train, 'ids': ids_train},
        {'X': X_val, 'y': y_val, 'ids': ids_val},
        {'X': X_test, 'y': y_test, 'ids': ids_test}
    )


def train_balanced_model(
    train_data: dict,
    val_data: dict,
    model_size: str,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 5e-3,
    patience: int = 10,
    device: str = 'cpu'
) -> tuple[Path, dict]:
    """
    Train depression classification head with balanced approach.

    Returns:
        Tuple of (model_path, metrics)
    """
    logger.info(f"Training balanced model for PAT-{model_size.upper()}")

    # Setup device
    if device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple M1 GPU acceleration")
    else:
        device = torch.device(device)

    # Convert data to tensors
    X_train = torch.FloatTensor(train_data['X']).to(device)
    y_train = torch.FloatTensor(train_data['y']).to(device)
    X_val = torch.FloatTensor(val_data['X']).to(device)
    y_val = torch.FloatTensor(val_data['y']).to(device)

    # Log class distributions
    train_pos = (y_train == 1).sum().item()
    train_neg = (y_train == 0).sum().item()
    val_pos = (y_val == 1).sum().item()
    val_neg = (y_val == 0).sum().item()

    logger.info(f"Training: {train_neg} neg, {train_pos} pos ({train_pos/(train_pos+train_neg)*100:.1f}%)")
    logger.info(f"Validation: {val_neg} neg, {val_pos} pos ({val_pos/(val_pos+val_neg)*100:.1f}%)")

    # Create model
    input_dim = X_train.shape[1]  # Should be 96 for PAT-S
    model = ImprovedTaskHead(input_dim=input_dim).to(device)

    # Loss function - plain BCE without class weights since data is balanced
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_pr_auc = 0.0
    patience_counter = 0
    metrics_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_probs = torch.sigmoid(val_outputs)

            # Calculate metrics
            y_val_np = y_val.cpu().numpy()
            probs_np = val_probs.cpu().numpy()

            # ROC-AUC
            roc_auc = roc_auc_score(y_val_np, probs_np)

            # PR-AUC (more informative for imbalanced data)
            precision, recall, _ = precision_recall_curve(y_val_np, probs_np)
            pr_auc = auc(recall, precision)

            # Basic accuracy with 0.5 threshold
            val_preds = (probs_np > 0.5).astype(int)
            val_acc = (val_preds == y_val_np).mean()

        # Update learning rate
        scheduler.step(val_loss)

        # Log progress
        avg_train_loss = np.mean(train_losses)
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}"
        )

        # Store metrics
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss.item(),
            'val_acc': val_acc,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        })

        # Early stopping based on PR-AUC
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            patience_counter = 0

            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

        # Additional monitoring every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Check logit separation
            pos_outputs = val_outputs[y_val == 1].cpu().numpy()
            neg_outputs = val_outputs[y_val == 0].cpu().numpy()

            if len(pos_outputs) > 0 and len(neg_outputs) > 0:
                pos_mean = pos_outputs.mean()
                neg_mean = neg_outputs.mean()
                separation = abs(pos_mean - neg_mean)
                logger.info(f"  Logit separation: {separation:.4f} (pos: {pos_mean:.4f}, neg: {neg_mean:.4f})")

    # Load best model
    model.load_state_dict(best_model_state)

    # Find optimal threshold on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_probs = torch.sigmoid(val_outputs).cpu().numpy()

    best_f1 = 0.0
    best_threshold = 0.5

    logger.info("Searching for optimal threshold...")
    for threshold in np.linspace(0.1, 0.9, 17):
        preds = (val_probs > threshold).astype(int)
        if len(np.unique(preds)) > 1:  # Avoid single-class predictions
            from sklearn.metrics import f1_score
            f1 = f1_score(y_val_np, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    logger.info(f"Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")

    # Final evaluation with best threshold
    final_preds = (val_probs > best_threshold).astype(int)

    from sklearn.metrics import classification_report
    report = classification_report(y_val_np, final_preds, output_dict=True)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("model_weights/pat/heads")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"pat_depression_{model_size}_balanced_{timestamp}.pt"

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': 128,
            'dropout': 0.3
        },
        'training_config': {
            'epochs': epoch + 1,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'balance_strategy': 'undersample'
        },
        'metrics': {
            'best_pr_auc': best_pr_auc,
            'best_threshold': best_threshold,
            'final_f1': best_f1,
            'classification_report': report,
            'metrics_history': metrics_history
        }
    }

    torch.save(save_dict, model_path)
    logger.info(f"Model saved to {model_path}")

    return model_path, save_dict['metrics']


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train PAT depression head with balanced approach"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=['small', 'medium', 'large'],
        default='small',
        help="Model size to train"
    )
    parser.add_argument(
        "--nhanes-dir",
        type=Path,
        default=Path("data/nhanes/2013-2014"),
        help="Directory containing NHANES data files"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device to use (cpu, cuda, or mps)"
    )
    parser.add_argument(
        "--balance-strategy",
        type=str,
        choices=['undersample', 'none'],
        default='undersample',
        help="Data balancing strategy"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use only a subset of subjects for smoke testing"
    )

    args = parser.parse_args()

    # Model configuration
    model_configs = {
        'small': {'model_file': 'PAT-S_29k_weights.h5', 'hidden_dim': 64},
        'medium': {'model_file': 'PAT-M_29k_weights.h5', 'hidden_dim': 128},
        'large': {'model_file': 'PAT-L_29k_weights.h5', 'hidden_dim': 256}
    }

    config = model_configs[args.model_size]

    # Load PAT model
    logger.info(f"Loading PAT-{args.model_size.upper()} model...")
    pat_model = PATModel(model_size=args.model_size)
    weights_path = Path(f"model_weights/pat/pretrained/{config['model_file']}")

    if not pat_model.load_pretrained_weights(weights_path):
        logger.error(f"Failed to load PAT-{args.model_size.upper()} weights")
        return

    # Prepare balanced data
    train_data, val_data, test_data = prepare_balanced_data(
        args.nhanes_dir,
        pat_model,
        balance_strategy=args.balance_strategy,
        subset=args.subset
    )

    # Train model
    model_path, metrics = train_balanced_model(
        train_data,
        val_data,
        args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Log final results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Best PR-AUC: {metrics['best_pr_auc']:.4f}")
    logger.info(f"Best threshold: {metrics['best_threshold']:.2f}")
    logger.info(f"Final F1 score: {metrics['final_f1']:.4f}")

    # Save summary
    summary_path = Path("model_weights/pat/heads/training_summary_balanced.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_size': args.model_size,
            'balance_strategy': args.balance_strategy,
            'metrics': metrics
        }, f, indent=2, default=str)

    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
