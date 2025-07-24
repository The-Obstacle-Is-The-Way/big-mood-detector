#!/usr/bin/env python3
"""
Fixed PAT-L Training Script
Main fixes:
1. Compute normalization statistics from actual training data
2. Use higher initial learning rate
3. Better pos_weight calculation
4. Monitoring of gradient norms
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import NHANESProcessor
from big_mood_detector.infrastructure.ml_models.pat_pytorch import PATDepressionNet


def prepare_data_with_proper_normalization(
    processor: NHANESProcessor,
    subset: int = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Prepare data with normalization computed from training set."""
    
    logger.info("Loading NHANES data...")
    
    # Load raw data
    actigraphy = processor.load_actigraphy("PAXMIN_H.xpt")
    depression = processor.load_depression_scores("DPQ_H.xpt")
    
    # Get subjects
    common_subjects = set(actigraphy['participant_id'].unique()) & set(depression.keys())
    logger.info(f"Found {len(common_subjects)} subjects with both data types")
    
    if subset:
        common_subjects = list(common_subjects)[:subset]
    
    # Prepare sequences WITHOUT normalization first
    X_raw = []
    y = []
    
    for subject_id in common_subjects:
        # Get subject data
        subject_data = actigraphy[actigraphy['participant_id'] == subject_id]
        
        # Process 7-day sequence
        sequences = []
        for day in range(1, 8):
            day_data = subject_data[subject_data['DayOfStudy'] == day]
            if len(day_data) > 0:
                # Create minute array
                minutes_array = np.zeros(1440, dtype=np.float32)
                minutes = day_data['AxisMinute'].values
                intensities = day_data['MIMS_UNIT'].values.astype(np.float32)
                
                valid_mask = (minutes >= 0) & (minutes < 1440)
                minutes_array[minutes[valid_mask]] = intensities[valid_mask]
                sequences.append(minutes_array)
        
        if len(sequences) == 7:
            # Concatenate to 10,080 minutes
            flat_sequence = np.concatenate(sequences)
            
            # Apply log transform (but not z-score yet)
            flat_sequence = np.log1p(flat_sequence)
            flat_sequence = np.clip(flat_sequence, 0, 10)
            
            X_raw.append(flat_sequence)
            
            # Get label
            label = 1 if depression.get(subject_id, 0) >= 10 else 0
            y.append(label)
    
    X_raw = np.array(X_raw, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    logger.info(f"Prepared {len(X_raw)} sequences with {sum(y)} positive samples ({100*sum(y)/len(y):.1f}%)")
    
    # Split data FIRST, then normalize based on training set
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Compute normalization statistics from TRAINING SET ONLY
    train_mean = X_train_raw.mean()
    train_std = X_train_raw.std()
    
    logger.info(f"Training set statistics (after log transform):")
    logger.info(f"  Mean: {train_mean:.4f}")
    logger.info(f"  Std: {train_std:.4f}")
    
    # Apply normalization using training statistics
    X_train = (X_train_raw - train_mean) / train_std
    X_val = (X_val_raw - train_mean) / train_std  # Use training stats!
    
    # Verify normalization
    logger.info(f"After normalization:")
    logger.info(f"  Train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    logger.info(f"  Val mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
    
    # Check that sequences have different means
    train_seq_means = X_train.mean(axis=1)
    logger.info(f"  Train sequence means - min: {train_seq_means.min():.4f}, max: {train_seq_means.max():.4f}, std: {train_seq_means.std():.4f}")
    
    return X_train, X_val, y_train, y_val, {'mean': train_mean, 'std': train_std}


def train_with_monitoring(
    X_train, X_val, y_train, y_val,
    model_size: str = "large",
    epochs: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
    learning_rate: float = 5e-3,  # Higher initial LR
    output_dir: Path = None
):
    """Train with gradient monitoring and better optimization."""
    
    logger.info(f"Using device: {device}")
    
    # Create model
    model = PATDepressionNet(
        model_size=model_size,
        unfreeze_last_n=0,
        hidden_dim=128,  # Simpler head first
        dropout=0.2
    )
    
    # Load pretrained weights
    weights_path = Path(f"model_weights/pat/pretrained/PAT-{model_size.upper()}_29k_weights.h5")
    if weights_path.exists():
        logger.info(f"Loading pretrained encoder from {weights_path}")
        success = model.encoder.load_tf_weights(weights_path)
        if not success:
            logger.warning("Failed to load pretrained weights!")
    
    model = model.to(device)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
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
    
    # Better pos_weight calculation
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f} (based on {pos_count:.0f} pos, {neg_count:.0f} neg)")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # OneCycle scheduler for better convergence
    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Training tracking
    best_val_auc = 0
    gradient_norms = []
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        batch_gradient_norms = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits.squeeze(), y_batch)
            loss.backward()
            
            # Monitor gradients BEFORE clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            batch_gradient_norms.append(total_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                          f"Loss={loss.item():.4f}, LR={current_lr:.2e}, "
                          f"Grad Norm={total_norm:.4f}")
        
        # Track gradient norms
        avg_grad_norm = np.mean(batch_gradient_norms)
        gradient_norms.append(avg_grad_norm)
        
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
        
        # Metrics
        val_auc = roc_auc_score(all_labels, all_preds)
        val_pr_auc = average_precision_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, np.array(all_preds) > 0.5)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"\nEpoch {epoch+1}/{epochs} Summary:")
        logger.info(f"  Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        logger.info(f"  Metrics - AUC: {val_auc:.4f}, PR-AUC: {val_pr_auc:.4f}, F1: {val_f1:.4f}")
        logger.info(f"  Avg Gradient Norm: {avg_grad_norm:.4f}")
        
        # Check for vanishing gradients
        if avg_grad_norm < 0.001:
            logger.warning("⚠️  Vanishing gradients detected! Consider increasing learning rate.")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if output_dir:
                checkpoint_path = output_dir / f"pat_{model_size}_fixed_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'gradient_norms': gradient_norms
                }, checkpoint_path)
                logger.info(f"✅ Saved best model with AUC: {val_auc:.4f}")
    
    return best_val_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, help='Use subset of data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--learning-rate', type=float, default=5e-3)
    parser.add_argument('--output-dir', type=str, 
                       default='model_weights/pat/pytorch/pat_l_fixed')
    args = parser.parse_args()
    
    # Initialize processor
    processor = NHANESProcessor(
        data_dir=Path("data/nhanes/2013-2014"),
        output_dir=Path("data/nhanes/processed")
    )
    
    # Prepare data with proper normalization
    X_train, X_val, y_train, y_val, norm_stats = prepare_data_with_proper_normalization(
        processor,
        subset=args.subset
    )
    
    # Train
    best_auc = train_with_monitoring(
        X_train, X_val, y_train, y_val,
        model_size='large',
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        learning_rate=args.learning_rate,
        output_dir=Path(args.output_dir)
    )
    
    logger.info(f"\n🎉 Training completed! Best validation AUC: {best_auc:.4f}")
    
    # Save summary
    output_dir = Path(args.output_dir)
    summary = {
        'model_size': 'large',
        'best_val_auc': best_auc,
        'normalization_stats': norm_stats,
        'config': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()