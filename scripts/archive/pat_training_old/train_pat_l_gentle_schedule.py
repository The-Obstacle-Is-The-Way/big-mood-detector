#!/usr/bin/env python3
"""
PAT-L Training - Gentle LR Schedule
===================================

Goal: Bridge the 0.057 AUC gap (0.5633 → 0.620)
Issue: Previous run overfitted after epoch 7 due to aggressive cosine decay
Solution: Gentler LR schedule + plateau-based adjustments

Key Changes:
- Slower cosine decay (T_max=50 vs 30)  
- ReduceLROnPlateau backup
- Earlier stopping (patience=5)
- Same proven LRs: encoder=5e-5, head=5e-4
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from big_mood_detector.infrastructure.ml_models.pat_pytorch import PATPyTorchEncoder


class SimplePATDepressionModel(nn.Module):
    """PAT-L with simple linear head for depression."""
    
    def __init__(self, model_size: str = "large"):
        super().__init__()
        
        self.encoder = PATPyTorchEncoder(model_size=model_size)
        self.head = nn.Linear(96, 1)
        
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        logits = self.head(embeddings)
        return logits.squeeze()


def load_data():
    """Load the CORRECTED NHANES depression data."""
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    logger.info(f"Loading corrected data from {cache_path}")
    
    data = np.load(cache_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    
    logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Class balance - Train: {(y_train == 1).sum()}/{len(y_train)} positive")
    
    # Log statistics to verify normalization
    logger.info(f"Data statistics:")
    logger.info(f"  Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    logger.info(f"  Val - Mean: {X_val.mean():.6f}, Std: {X_val.std():.6f}")
    
    return X_train, X_val, y_train, y_val

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pat_l_gentle_schedule.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("PAT-L Training - Gentle LR Schedule")
    logger.info("Goal: Bridge 0.057 AUC gap (0.5633 → 0.620)")
    logger.info("Strategy: Slower cosine decay + plateau backup")
    logger.info("="*60)
    
    # Load the corrected data
    X_train, X_val, y_train, y_val = load_data()
    
    # Verify normalization is fixed
    train_mean = X_train.mean()
    train_std = X_train.std()
    
    if abs(train_mean - (-1.24)) < 0.01:
        logger.error("❌ BAD NORMALIZATION DETECTED!")
        logger.error("Data has fixed normalization (mean≈-1.24)")
        logger.error("This will cause AUC to stay at ~0.47")
        raise ValueError("Normalization issue not resolved!")
    
    logger.info("✅ Normalization looks good - proceeding with training")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model
    model = SimplePATDepressionModel(model_size="large")
    
    # Load pretrained weights
    weights_path = Path("model_weights/pat/pretrained/PAT-L_29k_weights.h5")
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    
    logger.info("Loading pretrained weights...")
    success = model.encoder.load_tf_weights(weights_path)
    if not success:
        raise RuntimeError("Failed to load pretrained weights!")
    
    model = model.to(device)
    
    # Log parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Create datasets
    batch_size = 32
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Loss with class weighting
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer - same proven LRs
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    
    logger.info("Optimizer config:")
    logger.info("  - Encoder LR: 5e-5 (proven effective)")
    logger.info("  - Head LR: 5e-4")
    
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 5e-5},
        {'params': head_params, 'lr': 5e-4}
    ])
    
    # GENTLE LR Scheduling
    logger.info("LR Schedule: Gentle cosine decay + plateau backup")
    primary_scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)  # Slower!
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
    
    # Training loop
    logger.info("\n" + "="*50)
    logger.info("Starting Training - Gentle Schedule")
    logger.info("Current best to beat: 0.5633 AUC")
    logger.info("Target: 0.620 AUC (gap: 0.057)")
    logger.info("="*50)
    
    best_auc = 0.0
    patience = 0
    max_patience = 5  # Stricter early stopping
    
    for epoch in range(40):  # Fewer epochs, focus on quality
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                probs = torch.sigmoid(output.squeeze())
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # Calculate AUC
        val_auc = roc_auc_score(val_targets, val_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        # Get current learning rates
        encoder_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}, "
                   f"Encoder LR={encoder_lr:.2e}, Head LR={head_lr:.2e}")
        
        # Step schedulers
        primary_scheduler.step()
        plateau_scheduler.step(val_auc)  # Plateau based on AUC
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            
            save_path = "model_weights/pat/pytorch/pat_l_gentle_best.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ Saved best model with AUC: {val_auc:.4f}")
            
            # Check if we hit target
            if val_auc >= 0.620:
                logger.info("🎯 TARGET REACHED! AUC >= 0.620")
                break
                
        else:
            patience += 1
            logger.info(f"No improvement. Patience: {patience}/{max_patience}")
            
            if patience >= max_patience:
                logger.info("Early stopping triggered")
                break
    
    logger.info("\n" + "="*50)
    logger.info("Training Complete!")
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    logger.info(f"Target was: 0.620 AUC")
    logger.info(f"Gap remaining: {0.620 - best_auc:.4f}")
    
    if best_auc >= 0.590:
        logger.info("✅ SUCCESS: Reached paper's PAT-L (FT) performance!")
    if best_auc >= 0.620:
        logger.info("🎯 EXCELLENT: Hit target performance!")
    
    logger.info("="*50)

if __name__ == "__main__":
    main() 