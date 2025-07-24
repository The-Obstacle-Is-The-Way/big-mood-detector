#!/usr/bin/env python3
"""
End-to-End PAT Depression Head Training

This implements true fine-tuning with gradient flow through the encoder,
not just linear probing.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

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
from big_mood_detector.infrastructure.fine_tuning.population_trainer import (  # noqa: E402
    PATPopulationTrainer,
)


def prepare_data(
    processor: NHANESProcessor,
    subset: int = None,
    test_size: float = 0.2,
    random_state: int = 42
):
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

    # Extract sequences with proper processing
    logger.info("Extracting PAT sequences...")
    sequences = []
    labels = []

    for i, seqn in enumerate(subjects):
        if i % 100 == 0:
            logger.info(f"Processing subject {i}/{len(subjects)}")

        try:
            # Extract full 10,080 sequence (PAT model does patching internally)
            sequence = processor.extract_pat_sequences(
                actigraphy,
                seqn,
                normalize=True,      # Log transform
                standardize=True     # Z-score normalize
            )

            # Verify shape
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
    logger.info(f"Shape: {sequences.shape}")
    logger.info(f"Class distribution: {np.bincount(labels)}")
    logger.info(f"Positive rate: {labels.mean():.2%}")

    # Split data (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    return X_train, X_val, y_train, y_val


def train_with_population_trainer(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    model_size: str = "small",
    epochs: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
    unfreeze_layers: int = 1,
    head_lr: float = 5e-3,
    encoder_lr: float = 1e-5
):
    """Train using the PATPopulationTrainer with proper fine-tuning."""

    logger.info("Initializing PAT Population Trainer...")

    # Create trainer with fine-tuning configuration
    trainer = PATPopulationTrainer(
        task_name="depression",
        output_dir=Path("model_weights/pat/heads"),
        base_model_path=Path(f"model_weights/pat/pretrained/PAT-{model_size.upper()}_29k_weights.h5")
    )

    # Calculate positive class weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = neg_count / pos_count
    logger.info(f"Using pos_weight: {pos_weight:.2f}")

    # Train with proper parameters
    logger.info("Starting end-to-end training with encoder fine-tuning...")
    metrics = trainer.fine_tune(
        sequences=X_train,
        labels=y_train,
        validation_data=(X_val, y_val),  # Added validation data
        epochs=epochs,
        batch_size=batch_size,
        head_learning_rate=head_lr,
        encoder_learning_rate=encoder_lr,
        unfreeze_layers=unfreeze_layers,
        pos_weight=pos_weight,
        device=device,
        early_stopping_patience=25
    )

    return metrics


def main():
    """Run end-to-end training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, help='Use subset of data for testing')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--model-size', default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--unfreeze-layers', type=int, default=1, help='Number of encoder layers to unfreeze')
    parser.add_argument('--head-lr', type=float, default=5e-3, help='Learning rate for head')
    parser.add_argument('--encoder-lr', type=float, default=1e-5, help='Learning rate for encoder')
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

    # Train with end-to-end approach
    metrics = train_with_population_trainer(
        X_train, X_val, y_train, y_val,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        unfreeze_layers=args.unfreeze_layers,
        head_lr=args.head_lr,
        encoder_lr=args.encoder_lr
    )

    # Save results
    results = {
        'model_size': args.model_size,
        'unfreeze_layers': args.unfreeze_layers,
        'head_lr': args.head_lr,
        'encoder_lr': args.encoder_lr,
        'final_metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    output_file = f'pat_depression_end_to_end_results_{args.model_size}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to {output_file}")

    # Show key metrics
    if 'best_val_auc' in metrics:
        logger.info(f"Best validation AUC: {metrics['best_val_auc']:.4f}")
    if 'best_val_pr_auc' in metrics:
        logger.info(f"Best validation PR-AUC: {metrics['best_val_pr_auc']:.4f}")


if __name__ == "__main__":
    main()

