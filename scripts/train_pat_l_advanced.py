#!/usr/bin/env python3
"""
Advanced PAT-L Training with Progressive Unfreezing and Better Architecture
Implements:
1. Progressive unfreezing of encoder layers
2. 2-layer head with GELU activation
3. Cosine warm restarts with better scheduling
4. Differential learning rates
5. Data augmentation (time shifting)
6. Better logging and visualization
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import (
    NHANESProcessor,
)
from big_mood_detector.infrastructure.ml_models.pat_pytorch import (
    PATPyTorchEncoder,
)


class ImprovedPATDepressionNet(nn.Module):
    """
    Enhanced PAT model for depression with:
    - 2-layer head with GELU
    - Better initialization
    - Progressive unfreezing support
    """
    
    def __init__(
        self, 
        model_size: str = "large",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        unfreeze_last_n: int = 0
    ):
        super().__init__()
        
        # PAT encoder
        self.encoder = PATPyTorchEncoder(model_size=model_size)
        
        # Enhanced 2-layer head with GELU
        self.head = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in second layer
            nn.Linear(64, 1)
        )
        
        # Initialize head with Xavier/He initialization
        self._init_head()
        
        # Freeze encoder as specified
        self._freeze_encoder(unfreeze_last_n)
        
    def _init_head(self):
        """Initialize head layers with proper initialization."""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _freeze_encoder(self, unfreeze_last_n: int) -> None:
        """Freeze encoder parameters except last N transformer blocks."""
        # First freeze everything
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer blocks
        if unfreeze_last_n > 0:
            num_blocks = len(self.encoder.blocks)
            start_idx = max(0, num_blocks - unfreeze_last_n)
            
            for i in range(start_idx, num_blocks):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True
                    
            logger.info(f"Unfroze last {unfreeze_last_n} transformer blocks (from block {start_idx})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        embeddings = self.encoder(x)
        logits = self.head(embeddings)
        return logits
    
    def load_pretrained_encoder(self, weights_path: Path) -> bool:
        """Load pretrained encoder weights."""
        return self.encoder.load_tf_weights(weights_path)


class AugmentedActivityDataset(Dataset):
    """Dataset with time-shift augmentation for activity sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = True, max_shift: int = 60):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.max_shift = max_shift  # Max minutes to shift
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment and torch.rand(1).item() > 0.5:
            # Random time shift augmentation
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shift)
                # Zero out rolled-over values
                if shift > 0:
                    x[:shift] = 0
                else:
                    x[shift:] = 0
        
        return x, y


def get_differential_lr_groups(model: ImprovedPATDepressionNet, 
                              encoder_lr: float, 
                              head_lr: float) -> list:
    """Create parameter groups with differential learning rates."""
    encoder_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
    
    param_groups = []
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': encoder_lr})
    param_groups.append({'params': head_params, 'lr': head_lr})
    
    logger.info(f"Created {len(param_groups)} parameter groups")
    logger.info(f"Encoder params: {len(encoder_params)}, Head params: {len(head_params)}")
    
    return param_groups


def train_advanced(
    X_train, X_val, y_train, y_val,
    model_size: str = "large",
    epochs: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    head_lr: float = 1e-3,
    encoder_lr: float = 1e-5,
    unfreeze_at_epoch: int = 20,
    unfreeze_last_n: int = 2,
    output_dir: Path = None,
    augment: bool = True
):
    """Advanced training with progressive unfreezing and better optimization."""
    
    logger.info(f"Using device: {device}")
    
    # Stage 1: Create model with frozen encoder
    logger.info(f"Stage 1: Training with frozen encoder")
    model = ImprovedPATDepressionNet(
        model_size=model_size,
        hidden_dim=256,
        dropout=0.3,
        unfreeze_last_n=0  # Start frozen
    )
    
    # Load pretrained weights
    weights_path = Path(f"model_weights/pat/pytorch/pat_{model_size}_weights.pt")
    if weights_path.exists():
        logger.info(f"Loading pretrained encoder from {weights_path}")
        model.load_pretrained_encoder(weights_path)
    else:
        logger.warning(f"No pretrained weights found at {weights_path}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets with augmentation
    train_dataset = AugmentedActivityDataset(X_train, y_train, augment=augment)
    val_dataset = AugmentedActivityDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Calculate positive weight with smoothing
    pos_count = sum(y_train)
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / (pos_count + 1)], dtype=torch.float32)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Initial optimizer (head only)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=head_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training tracking
    best_val_auc = 0
    patience_counter = 0
    early_stopping_patience = 30
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_pr_auc': [], 'lr': []}
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Progressive unfreezing
        if epoch == unfreeze_at_epoch and unfreeze_last_n > 0:
            logger.info(f"\n🔓 Stage 2: Unfreezing last {unfreeze_last_n} encoder blocks")
            
            # Unfreeze encoder blocks
            model._freeze_encoder(unfreeze_last_n)
            
            # Recreate optimizer with differential LR
            param_groups = get_differential_lr_groups(model, encoder_lr, head_lr)
            optimizer = optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
            
            # Reset scheduler
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=encoder_lr / 10
            )
            
            # Reset patience counter
            patience_counter = 0
        
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
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
            
            # Log progress
            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
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
        val_auc = roc_auc_score(all_labels, all_preds)
        val_pr_auc = average_precision_score(all_labels, all_preds)
        
        # Calculate F1 with optimal threshold
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        val_f1 = f1_score(all_labels, np.array(all_preds) > optimal_threshold)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        history['val_pr_auc'].append(val_pr_auc)
        history['lr'].append(current_lr)
        
        logger.info(f"\nEpoch {epoch+1}/{epochs}:")
        logger.info(f"  Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        logger.info(f"  Metrics - AUC: {val_auc:.4f}, PR-AUC: {val_pr_auc:.4f}, F1: {val_f1:.4f}")
        logger.info(f"  Optimal threshold: {optimal_threshold:.3f}, LR: {current_lr:.2e}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_auc > best_val_auc:
            improvement = val_auc - best_val_auc
            best_val_auc = val_auc
            patience_counter = 0
            
            if output_dir:
                checkpoint_path = output_dir / f"pat_{model_size}_best_auc_{val_auc:.4f}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_pr_auc': val_pr_auc,
                    'val_f1': val_f1,
                    'optimal_threshold': optimal_threshold,
                    'history': history
                }, checkpoint_path)
                logger.info(f"✅ Saved best model! (improvement: +{improvement:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"\n⏹ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Plot progress every 10 epochs
        if epoch % 10 == 0 and output_dir:
            plot_training_history(history, output_dir, epoch)
    
    return best_val_auc, history


def plot_training_history(history: dict, output_dir: Path, epoch: int):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC
    axes[0, 1].plot(history['val_auc'], label='Val AUC', color='green')
    axes[0, 1].plot(history['val_pr_auc'], label='Val PR-AUC', color='orange')
    axes[0, 1].set_title('AUC Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LR')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # AUC zoomed in
    if len(history['val_auc']) > 10:
        axes[1, 1].plot(history['val_auc'][-20:])
        axes[1, 1].set_title('Recent AUC (last 20 epochs)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epoch_{epoch}.png', dpi=150)
    plt.close()


def prepare_data(processor: NHANESProcessor, subset: int = None):
    """Load and prepare NHANES data."""
    # Try cache first
    cache_path = Path(f"data/cache/nhanes_pat_data_subset{subset}.npz")
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        return data['X_train'], data['X_val'], data['y_train'], data['y_val']
    
    # Load data
    logger.info("Loading NHANES data...")
    actigraphy = processor.load_actigraphy("PAXMIN_H.xpt")
    depression = processor.load_depression_scores("DPQ_H.xpt")
    
    # Get subjects
    common_subjects = set(actigraphy['participant_id'].unique()) & set(depression.keys())
    logger.info(f"Found {len(common_subjects)} subjects with both data types")
    
    if subset:
        common_subjects = list(common_subjects)[:subset]
    
    # Prepare sequences
    X, y, valid_indices = processor.prepare_pad_sequences_and_labels(
        actigraphy=actigraphy,
        depression_scores=depression,
        subjects=common_subjects,
        sequence_length=10080
    )
    
    X = X[valid_indices]
    y = y[valid_indices]
    
    logger.info(f"Dataset: {len(X)} samples, {sum(y)} positive ({100*sum(y)/len(y):.1f}%)")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X_train=X_train, X_val=X_val, 
                       y_train=y_train, y_val=y_val)
    
    return X_train, X_val, y_train, y_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, help='Use subset of data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--unfreeze-at-epoch', type=int, default=20)
    parser.add_argument('--unfreeze-last-n', type=int, default=2)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--output-dir', type=str, 
                       default='model_weights/pat/pytorch/pat_l_advanced')
    args = parser.parse_args()
    
    # Initialize processor
    processor = NHANESProcessor(
        data_dir=Path("data/nhanes/2013-2014"),
        output_dir=Path("data/nhanes/processed")
    )
    
    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data(processor, args.subset)
    
    # Train
    best_auc, history = train_advanced(
        X_train, X_val, y_train, y_val,
        model_size='large',
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        head_lr=args.head_lr,
        encoder_lr=args.encoder_lr,
        unfreeze_at_epoch=args.unfreeze_at_epoch,
        unfreeze_last_n=args.unfreeze_last_n,
        output_dir=Path(args.output_dir),
        augment=not args.no_augment
    )
    
    logger.info(f"\n🎉 Training completed! Best validation AUC: {best_auc:.4f}")
    
    # Save final summary
    output_dir = Path(args.output_dir)
    summary = {
        'model_size': 'large',
        'training_type': 'progressive_unfreezing',
        'best_val_auc': best_auc,
        'epochs_trained': len(history['train_loss']),
        'config': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot final curves
    plot_training_history(history, output_dir, len(history['train_loss']))


if __name__ == "__main__":
    main()