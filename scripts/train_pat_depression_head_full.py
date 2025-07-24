#!/usr/bin/env python3
"""
Train PAT Depression Classification Head - Full Version

This script trains binary classification heads on top of PAT embeddings
for all model sizes (S, M, L) with proper validation and metrics.

Based on the PAT paper findings:
- PAT-S: ~0.56 AUC
- PAT-M: ~0.56 AUC
- PAT-L: ~0.59 AUC
- PAT Conv-L: ~0.61 AUC (best)
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

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


def get_model_config(model_size: str) -> dict:
    """Get configuration for different model sizes."""
    configs = {
        'small': {
            'model_file': 'PAT-S_29k_weights.h5',
            'hidden_dim': 64,
            'params': '285K'
        },
        'medium': {
            'model_file': 'PAT-M_29k_weights.h5',
            'hidden_dim': 128,
            'params': '1M'
        },
        'large': {
            'model_file': 'PAT-L_29k_weights.h5',
            'hidden_dim': 256,
            'params': '2M'
        }
    }
    return configs[model_size]


def prepare_full_training_data(
    nhanes_dir: Path,
    pat_model: Any,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
    subset: int = None
) -> tuple[dict, dict, dict]:
    """
    Prepare training data with proper train/val/test splits.

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

    # Shuffle subjects
    np.random.seed(random_seed)
    np.random.shuffle(common_subjects)

    # Apply subset if specified (for smoke testing)
    if subset is not None:
        common_subjects = common_subjects[:subset]
        logger.info(f"Limiting to {subset} subjects for smoke test")

    # Extract features for each subject
    all_embeddings = []
    all_labels = []
    all_subject_ids = []

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
            sequence_7d = pat_sequences[-7:]  # Shape: (7, 1440)

            # Flatten 7x1440 to 1x10080 for PATSequence
            activity_flat = sequence_7d.flatten()  # Shape: (10080,)

            # Create PATSequence object
            pat_sequence = PATSequence(
                end_date=datetime.now().date(),
                activity_values=activity_flat,
                missing_days=[],  # No missing days if we have all 7
                data_quality_score=1.0  # Full data
            )

            # Extract PAT embeddings
            embeddings = pat_model.extract_features(pat_sequence)

            # Get depression label (PHQ-9 >= 10)
            subject_depression = depression_df[depression_df['SEQN'] == subject_id]
            phq9_score = subject_depression['PHQ9_total'].iloc[0]
            depression_label = int(phq9_score >= 10) if not np.isnan(phq9_score) else None

            if depression_label is not None:
                all_embeddings.append(embeddings)
                all_labels.append(depression_label)
                all_subject_ids.append(subject_id)

        except Exception as e:
            logger.warning(f"Error processing subject {subject_id}: {e}")
            continue

    # Convert to arrays
    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels)
    subject_ids = np.array(all_subject_ids)

    logger.info(f"Prepared {len(embeddings)} samples")
    if len(labels) > 0:
        logger.info(f"Class distribution: {np.bincount(labels.astype(int))} (negative, positive)")
    else:
        logger.error("No valid samples extracted!")
        raise ValueError("No valid training samples found")

    # Create train/val/test splits
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        embeddings, labels, subject_ids,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for the first split
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=y_temp
    )

    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (
        {'X': X_train, 'y': y_train, 'ids': ids_train},
        {'X': X_val, 'y': y_val, 'ids': ids_val},
        {'X': X_test, 'y': y_test, 'ids': ids_test}
    )



def train_enhanced_depression_head(
    train_data: dict,
    val_data: dict,
    model_size: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 10,
    device: str = 'cpu',
    manual_pos_weight: float = None
) -> tuple[Any, dict]:
    """
    Train depression classification head using PATPopulationTrainer.

    Returns:
        Tuple of (trained model path, training metrics)
    """
    logger.info(f"Training depression head for {model_size} with {epochs} epochs")

    # Use only training data - let trainer handle validation split
    # or use validation_split=0.0 if we want to use our existing split
    train_sequences = train_data['X']
    train_labels = train_data['y']

    # Initialize trainer with timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = PATPopulationTrainer(
        task_name=f"depression_{model_size}_{timestamp}",
        output_dir=Path("model_weights/pat/heads")
    )

    # Handle class imbalance (1:10 ratio typical in depression data)
    if manual_pos_weight is not None:
        pos_weight = manual_pos_weight
        logger.info(f"Using manual pos_weight: {pos_weight:.2f}")
    else:
        pos_weight = len(train_labels) / train_labels.sum() if train_labels.sum() > 0 else 1.0
        logger.info(f"Calculated pos_weight for class imbalance: {pos_weight:.2f}")

    # Note: Device handling moved to trainer's responsibility to avoid MPS issues

    # Train using the fine_tune method
    logger.info("Starting PAT fine-tuning...")
    try:
        metrics = trainer.fine_tune(
            sequences=train_sequences,
            labels=train_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=0.2,  # Let trainer create validation split
            device=device,  # Pass device for M1 GPU acceleration
            pos_weight=pos_weight  # Handle class imbalance
        )
    except TypeError:
        # Fallback if trainer doesn't support pos_weight
        logger.info("Trainer doesn't support pos_weight, training without class balancing")
        metrics = trainer.fine_tune(
            sequences=train_sequences,
            labels=train_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=0.2,
            device=device
        )

    # Log final metrics
    logger.info("Training completed!")
    logger.info(f"Final metrics: {metrics}")

    # Find the saved model path (PATPopulationTrainer saves it automatically)
    model_path = Path("model_weights/pat/heads") / f"pat_depression_{model_size}_{timestamp}.pt"

    return model_path, metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train PAT depression classification heads (all sizes)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=['small', 'medium', 'large', 'all'],
        default='medium',
        help="Model size to train (or 'all' for all sizes)"
    )
    parser.add_argument(
        "--nhanes-dir",
        type=Path,
        default=Path("data/nhanes/2013-2014"),
        help="Directory containing NHANES data files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_weights/pat/heads"),
        help="Directory to save trained models"
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
        "--device",
        type=str,
        default='cpu',
        help="Device to use (cpu or cuda or mps for M1)"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Subset of subjects to use for smoke testing (e.g., 500)"
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Manual positive class weight (overrides automatic calculation)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to train
    if args.model_size == 'all':
        model_sizes = ['small', 'medium', 'large']
    else:
        model_sizes = [args.model_size]

    # Check for M1 acceleration
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple M1 GPU acceleration")
    else:
        device = torch.device(args.device)

    # Store results for all models
    all_results = {}

    for model_size in model_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training PAT-{model_size.upper()} model")
        logger.info(f"{'='*60}")

        # Load PAT model
        config = get_model_config(model_size)
        pat_model = PATModel(model_size=model_size)
        weights_path = Path(f"model_weights/pat/pretrained/{config['model_file']}")

        if not pat_model.load_pretrained_weights(weights_path):
            logger.error(f"Failed to load PAT-{model_size.upper()} weights")
            continue

        # Prepare data (only do this once)
        if model_size == model_sizes[0]:
            train_data, val_data, test_data = prepare_full_training_data(
                args.nhanes_dir,
                pat_model,
                subset=args.subset
            )

        # Train model
        model_path, training_metrics = train_enhanced_depression_head(
            train_data,
            val_data,
            model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device
        )

        logger.info(f"Model training completed and saved to {model_path}")
        logger.info(f"Training results: {training_metrics}")

        # Store actual training metrics (no need to repack)
        all_results[model_size] = training_metrics

        # Log training results
        logger.info(f"\nTraining Results for PAT-{model_size.upper()}:")
        for metric, value in training_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")

    for model_size, results in all_results.items():
        auc = results.get('auc', 0.0)
        accuracy = results.get('accuracy', 0.0)
        logger.info(
            f"PAT-{model_size.upper()}: "
            f"AUC={auc:.4f}, "
            f"Accuracy={accuracy:.4f}"
        )

    # Save summary
    summary_path = args.output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'results': all_results,
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }
        }, f, indent=2)

    logger.info(f"\nTraining complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
