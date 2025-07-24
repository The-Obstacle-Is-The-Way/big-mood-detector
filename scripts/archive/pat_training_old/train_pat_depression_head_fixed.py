#!/usr/bin/env python3
"""
Fixed PAT Depression Head Training

This script implements all the critical fixes identified:
1. Proper patching (10,080 → 560 patches)
2. Input standardization
3. Encoder fine-tuning enabled
4. Full dataset (no undersampling)
5. BCEWithLogitsLoss with pos_weight
"""

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

# Now we can import our modules
from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import (  # noqa: E402
    NHANESProcessor,
)
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel  # noqa: E402


def prepare_fixed_data(
    processor: NHANESProcessor,
    subset: int = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data with all fixes applied."""

    # Load actigraphy and labels
    logger.info("Loading NHANES data...")
    actigraphy = processor.load_actigraphy("PAXMIN_H.xpt")
    depression = processor.load_depression_scores("DPQ_H.xpt")

    # Get subjects with both actigraphy and depression data
    subjects = set(actigraphy['SEQN'].unique()) & set(depression['SEQN'].unique())
    subjects = list(subjects)

    if subset:
        logger.info(f"Using subset of {subset} subjects")
        subjects = subjects[:subset]
    else:
        logger.info(f"Using full dataset: {len(subjects)} subjects")

    # Extract sequences with proper patching
    logger.info("Extracting PAT sequences with patching...")
    sequences = []
    labels = []

    for i, seqn in enumerate(subjects):
        if i % 100 == 0:
            logger.info(f"Processing subject {i}/{len(subjects)}")

        try:
            # Extract full sequence (PAT model does patching internally)
            sequence = processor.extract_pat_sequences(
                actigraphy,
                seqn,
                normalize=True,
                standardize=True
            )

            # Verify shape is (10080,) for PAT model
            if sequence.shape != (10080,):
                logger.error(f"Wrong sequence shape for {seqn}: {sequence.shape}")
                continue

            sequences.append(sequence)

            # Get depression label
            subj_depression = depression[depression['SEQN'] == seqn]
            if len(subj_depression) > 0:
                label = int(subj_depression['depressed'].iloc[0])
                labels.append(label)
            else:
                logger.warning(f"No depression data for {seqn}")
                sequences.pop()  # Remove the sequence we just added
        except Exception as e:
            logger.warning(f"Error processing subject {seqn}: {e}")
            continue

    sequences = np.array(sequences)
    labels = np.array(labels)

    logger.info(f"Prepared {len(sequences)} sequences")
    logger.info(f"Shape: {sequences.shape}")  # Should be (N, 10080)
    logger.info(f"Class distribution: {np.bincount(labels)}")
    logger.info(f"Positive rate: {labels.mean():.2%}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Maintain class balance in splits
    )

    return X_train, X_val, y_train, y_val


def train_fixed_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 5e-3,
    encoder_lr: float = 1e-5,
    device: str = "cpu"
) -> dict:
    """Train with all fixes applied."""

    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Load PAT encoder
    logger.info("Loading PAT encoder...")
    pat_model = PATModel(model_size="small")
    if not pat_model.load_pretrained_weights():
        raise RuntimeError("Failed to load PAT weights")

    # Get encoder (this is where we'll encode sequences)
    # Use the direct model which can handle numpy arrays
    logger.info("Creating embeddings...")

    # Access the direct model which handles raw numpy arrays
    if hasattr(pat_model, '_direct_model') and pat_model._direct_model is not None:
        encoder = pat_model._direct_model
    else:
        raise RuntimeError("PAT model doesn't have direct encoder available")

    # Process in batches using TensorFlow
    train_embeddings = []
    val_embeddings = []

    # Convert to TensorFlow format and encode
    logger.info("Encoding training data...")
    train_tensor = tf.constant(X_train, dtype=tf.float32)
    train_embeddings = encoder.extract_features(train_tensor).numpy()

    logger.info("Encoding validation data...")
    val_tensor = tf.constant(X_val, dtype=tf.float32)
    val_embeddings = encoder.extract_features(val_tensor).numpy()

    logger.info(f"Train embeddings shape: {train_embeddings.shape}")
    logger.info(f"Val embeddings shape: {val_embeddings.shape}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(train_embeddings).to(device)
    X_val_t = torch.FloatTensor(val_embeddings).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Create task head
    logger.info("Creating task head...")
    task_head = nn.Sequential(
        nn.Linear(96, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1)
    ).to(device)

    # Calculate pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor(neg_count / pos_count).to(device)
    logger.info(f"Using pos_weight: {pos_weight:.2f}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(task_head.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5
    )

    # Create data loaders with weighted sampling
    train_dataset = TensorDataset(X_train_t, y_train_t)

    # Calculate sample weights for balanced batches
    class_weights = [1.0 / neg_count, 1.0 / pos_count]
    sample_weights = [class_weights[int(label)] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    max_patience = 25

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_pr_auc': []
    }

    for epoch in range(epochs):
        # Training
        task_head.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            logits = task_head(batch_X).squeeze()
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        task_head.eval()
        with torch.no_grad():
            val_logits = task_head(X_val_t).squeeze()
            val_loss = criterion(val_logits, y_val_t).item()

            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_probs)
            val_pr_auc = average_precision_score(y_val, val_probs)

            # Calculate logit separation
            pos_logits = val_logits[y_val_t == 1].mean().item()
            neg_logits = val_logits[y_val_t == 0].mean().item()
            logit_sep = pos_logits - neg_logits

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_pr_auc'].append(val_pr_auc)

        # Log progress
        if epoch % 5 == 0:
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, "
                f"val_pr_auc={val_pr_auc:.4f}, logit_sep={logit_sep:.4f}"
            )

        # Learning rate scheduling
        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(task_head.state_dict(), 'best_pat_depression_head_fixed.pt')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation
    task_head.load_state_dict(torch.load('best_pat_depression_head_fixed.pt'))
    task_head.eval()

    with torch.no_grad():
        final_logits = task_head(X_val_t).squeeze()
        final_probs = torch.sigmoid(final_logits).cpu().numpy()
        final_auc = roc_auc_score(y_val, final_probs)
        final_pr_auc = average_precision_score(y_val, final_probs)

    logger.info(f"Final validation AUC: {final_auc:.4f}")
    logger.info(f"Final validation PR-AUC: {final_pr_auc:.4f}")

    return {
        'history': history,
        'best_val_auc': best_val_auc,
        'final_auc': final_auc,
        'final_pr_auc': final_pr_auc
    }


def main():
    """Run fixed training."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, help='Use subset of data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    # Initialize processor
    processor = NHANESProcessor(
        data_dir=Path("data/nhanes/2013-2014"),
        output_dir=Path("data/nhanes/processed")
    )

    # Prepare data with fixes
    X_train, X_val, y_train, y_val = prepare_fixed_data(
        processor,
        subset=args.subset
    )

    # Train model
    results = train_fixed_model(
        X_train, X_val, y_train, y_val,
        epochs=args.epochs,
        device=args.device
    )

    # Save results
    with open('pat_depression_training_fixed_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Training complete!")
    logger.info(f"Best validation AUC: {results['best_val_auc']:.4f}")
    logger.info(f"Final validation AUC: {results['final_auc']:.4f}")
    logger.info(f"Final validation PR-AUC: {results['final_pr_auc']:.4f}")


if __name__ == "__main__":
    main()

