#!/usr/bin/env python3
"""
Train PAT Depression Classification Head

This script trains a binary classification head on top of PAT embeddings
to predict current depression state (PHQ-9 >= 10) from NHANES data.

Usage:
    python scripts/train_pat_depression_head.py \
        --nhanes-dir data/nhanes \
        --output-dir model_weights/pat/heads \
        --epochs 50 \
        --batch-size 32

The script:
1. Loads NHANES actigraphy and depression labels
2. Extracts PAT embeddings for each subject
3. Trains a simple MLP head for depression classification
4. Saves the trained weights for use in production
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import (
    NHANESProcessor,
)
from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
    PATPopulationTrainer,
)
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_training_data(
    nhanes_dir: Path,
    pat_model: Any,
    max_subjects: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data from NHANES.
    
    Args:
        nhanes_dir: Directory containing NHANES data files
        pat_model: Pre-trained PAT model for embedding extraction
        max_subjects: Maximum number of subjects to process (for testing)
        
    Returns:
        Tuple of (embeddings, labels, subject_ids)
    """
    logger.info("Loading NHANES data...")
    processor = NHANESProcessor()

    # Load actigraphy and depression data
    actigraphy_df = processor.load_actigraphy_data(nhanes_dir)
    depression_df = processor.load_depression_data(nhanes_dir)

    # Get unique subjects with both actigraphy and depression data
    actigraphy_subjects = set(actigraphy_df['SEQN'].unique())
    depression_subjects = set(depression_df['SEQN'].unique())
    common_subjects = list(actigraphy_subjects & depression_subjects)

    if max_subjects:
        common_subjects = common_subjects[:max_subjects]

    logger.info(f"Found {len(common_subjects)} subjects with complete data")

    # Extract features for each subject
    embeddings_list = []
    labels_list = []
    subject_ids = []

    for i, subject_id in enumerate(common_subjects):
        if i % 10 == 0:
            logger.info(f"Processing subject {i+1}/{len(common_subjects)}")

        try:
            # Extract 7-day PAT sequences
            pat_sequences = processor.extract_pat_sequences(
                actigraphy_df,
                subject_id=subject_id,
                normalize=True
            )

            if pat_sequences.shape[0] < 7:
                logger.warning(f"Subject {subject_id} has insufficient data")
                continue

            # Use last 7 days
            sequence_7d = pat_sequences[-7:]  # Shape: (7, 1440)

            # Create PATSequence object
            pat_sequence = PATSequence(
                activity_counts=sequence_7d,
                start_date=datetime.now().date() - timedelta(days=7),
                end_date=datetime.now().date()
            )

            # Extract PAT embeddings
            embeddings = pat_model.extract_features(pat_sequence)

            # Get depression label (PHQ-9 >= 10)
            subject_depression = depression_df[depression_df['SEQN'] == subject_id]
            phq9_score = subject_depression['PHQ9_TOTAL'].iloc[0]
            depression_label = int(phq9_score >= 10) if not np.isnan(phq9_score) else None

            if depression_label is not None:
                embeddings_list.append(embeddings)
                labels_list.append(depression_label)
                subject_ids.append(subject_id)

        except Exception as e:
            logger.warning(f"Error processing subject {subject_id}: {e}")
            continue

    # Convert to arrays
    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    subject_ids = np.array(subject_ids)

    logger.info(f"Prepared {len(embeddings)} samples")
    logger.info(f"Class distribution: {np.bincount(labels)}")

    return embeddings, labels, subject_ids


def train_depression_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2
) -> nn.Module:
    """
    Train depression classification head.
    
    Args:
        embeddings: PAT embeddings (N, 96)
        labels: Binary depression labels (N,)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        val_split: Validation split ratio
        
    Returns:
        Trained model
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=val_split, random_state=42, stratify=labels
    )

    logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize trainer
    trainer = PATPopulationTrainer(
        task_name="depression",
        input_dim=96,
        hidden_dim=64,
        learning_rate=learning_rate
    )

    # Training loop
    best_val_auc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        trainer.model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            loss = trainer.train_step(batch_X, batch_y)
            train_loss += loss

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        trainer.model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = trainer.model(batch_X)
                probs = torch.sigmoid(outputs).squeeze()
                val_preds.extend(probs.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)

        # Metrics
        val_auc = roc_auc_score(val_true, val_preds)
        val_acc = accuracy_score(val_true, val_preds > 0.5)

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val AUC: {val_auc:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = trainer.model.state_dict().copy()

    # Restore best model
    trainer.model.load_state_dict(best_model_state)
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")

    return trainer.model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train PAT depression classification head"
    )
    parser.add_argument(
        "--nhanes-dir",
        type=Path,
        default=Path("data/nhanes"),
        help="Directory containing NHANES data files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_weights/pat/heads"),
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to use (for testing)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load PAT model
    logger.info("Loading PAT model...")
    pat_model = PATModel(model_size="small")  # Use small model for efficiency
    weights_path = Path("model_weights/pat/pretrained/PAT-S_29k_weights.h5")
    if not pat_model.load_pretrained_weights(weights_path):
        logger.error("Failed to load PAT weights")
        return

    # Prepare training data
    embeddings, labels, subject_ids = prepare_training_data(
        args.nhanes_dir,
        pat_model,
        max_subjects=args.max_subjects
    )

    if len(embeddings) < 100:
        logger.error(f"Insufficient training data: {len(embeddings)} samples")
        return

    # Train model
    logger.info("Training depression classification head...")
    model = train_depression_head(
        embeddings,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Save model
    output_path = args.output_dir / "pat_depression_head.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 96,
            'hidden_dim': 64,
            'output_dim': 1,
            'task': 'depression_binary'
        },
        'training_info': {
            'n_samples': len(embeddings),
            'n_positive': int(labels.sum()),
            'n_negative': int((1 - labels).sum()),
            'epochs': args.epochs
        }
    }, output_path)

    logger.info(f"Model saved to {output_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
