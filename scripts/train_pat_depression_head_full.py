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
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_recall_fscore_support,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    random_seed: int = 42
) -> Tuple[dict, dict, dict]:
    """
    Prepare training data with proper train/val/test splits.
    
    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    logger.info("Loading NHANES data...")
    processor = NHANESProcessor(data_dir=nhanes_dir)
    
    # Load actigraphy and depression data
    actigraphy_df = processor.load_actigraphy("PAXHD_H.xpt")
    depression_df = processor.load_depression_scores("DPQ_H.xpt")
    
    # Get participants with both actigraphy and depression data
    actigraphy_subjects = set(actigraphy_df['SEQN'].unique())
    depression_subjects = set(depression_df['SEQN'].unique())
    common_subjects = list(actigraphy_subjects & depression_subjects)
    
    logger.info(f"Found {len(common_subjects)} subjects with complete data")
    
    # Shuffle subjects
    np.random.seed(random_seed)
    np.random.shuffle(common_subjects)
    
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
    logger.info(f"Class distribution: {np.bincount(labels)} (negative, positive)")
    
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


def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs).squeeze()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    brier = brier_score_loss(all_labels, all_probs)
    
    # Calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(
        all_labels, all_probs, n_bins=10
    )
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier_score': brier,
        'calibration': {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
    }


def train_enhanced_depression_head(
    train_data: dict,
    val_data: dict,
    model_size: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 10,
    device: str = 'cpu'
) -> Tuple[nn.Module, dict]:
    """
    Train depression classification head with validation and early stopping.
    
    Returns:
        Tuple of (trained model, training history)
    """
    config = get_model_config(model_size)
    
    # Convert to tensors
    X_train = torch.FloatTensor(train_data['X']).to(device)
    y_train = torch.FloatTensor(train_data['y']).to(device)
    X_val = torch.FloatTensor(val_data['X']).to(device)
    y_val = torch.FloatTensor(val_data['y']).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize trainer
    trainer = PATPopulationTrainer(
        task_name="depression",
        input_dim=96,
        hidden_dim=config['hidden_dim'],
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    trainer.model = trainer.model.to(device)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'val_metrics': [],
        'best_epoch': 0
    }
    
    # Early stopping
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    
    logger.info(f"Training {model_size} model ({config['params']} parameters)")
    
    for epoch in range(epochs):
        # Training
        trainer.model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            loss = trainer.train_step(batch_X, batch_y)
            train_loss += loss
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        val_metrics = evaluate_model(trainer.model, val_loader, device)
        history['val_metrics'].append(val_metrics)
        
        # Learning rate step
        scheduler.step()
        
        # Logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val Brier: {val_metrics['brier_score']:.4f}"
            )
        
        # Early stopping
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = trainer.model.state_dict().copy()
            history['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    trainer.model.load_state_dict(best_model_state)
    logger.info(f"Best validation AUC: {best_val_auc:.4f} at epoch {history['best_epoch']+1}")
    
    return trainer.model, history


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
                pat_model
            )
        
        # Train model
        model, history = train_enhanced_depression_head(
            train_data,
            val_data,
            model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device
        )
        
        # Evaluate on test set
        test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(test_data['X']),
                torch.FloatTensor(test_data['y'])
            ),
            batch_size=args.batch_size,
            shuffle=False
        )
        
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Save model and results
        output_path = args.output_dir / f"pat_{model_size}_depression_head.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'model_size': model_size,
                'input_dim': 96,
                'hidden_dim': config['hidden_dim'],
                'output_dim': 1,
                'task': 'depression_binary'
            },
            'training_info': {
                'n_train': len(train_data['y']),
                'n_val': len(val_data['y']),
                'n_test': len(test_data['y']),
                'n_positive_train': int(train_data['y'].sum()),
                'n_positive_val': int(val_data['y'].sum()),
                'n_positive_test': int(test_data['y'].sum()),
                'epochs': args.epochs,
                'best_epoch': history['best_epoch']
            },
            'test_metrics': test_metrics,
            'training_history': history
        }, output_path)
        
        logger.info(f"Model saved to {output_path}")
        
        # Store results
        all_results[model_size] = {
            'test_auc': test_metrics['auc'],
            'test_accuracy': test_metrics['accuracy'],
            'test_brier': test_metrics['brier_score'],
            'val_auc': max([m['auc'] for m in history['val_metrics']])
        }
        
        # Log test results
        logger.info(f"\nTest Results for PAT-{model_size.upper()}:")
        logger.info(f"  AUC: {test_metrics['auc']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  Brier Score: {test_metrics['brier_score']:.4f}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    
    for model_size, results in all_results.items():
        logger.info(
            f"PAT-{model_size.upper()}: "
            f"Test AUC={results['test_auc']:.4f}, "
            f"Val AUC={results['val_auc']:.4f}, "
            f"Brier={results['test_brier']:.4f}"
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