#!/usr/bin/env python3
"""
PAT-Conv-L SIMPLIFIED Training Script
=====================================

Based on what actually worked: Simple approach that got 0.5924 AUC.
No complex scheduler gymnastics, just straightforward training.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pat_conv_l_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import our existing model
from scripts.pat_training.train_pat_conv_l import (
    SimplePATConvLModel, load_data
)


def main():
    logger.info("="*60)
    logger.info("PAT-Conv-L SIMPLIFIED Training")
    logger.info("Target: 0.625 AUC (paper's Conv-L result)")
    logger.info("Strategy: Simple approach that already got 0.5924!")
    logger.info("="*60)
    
    # Load data
    X_train, X_val, y_train, y_val = load_data()
    
    # Verify data normalization
    logger.info(f"Data check - Train mean: {X_train.mean():.6f}, std: {X_train.std():.6f}")
    if abs(X_train.std() - 1.0) > 0.1:
        logger.error("DATA NORMALIZATION ISSUE DETECTED!")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = SimplePATConvLModel(model_size="large")
    
    # Load pretrained weights
    weights_path = Path("model_weights/pat/pretrained/PAT-L_29k_weights.h5")
    if weights_path.exists():
        logger.info("Loading pretrained weights...")
        model.encoder.load_tf_weights(weights_path)
        logger.info("✅ Loaded transformer weights")
    
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
    
    # Loss
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # SIMPLE OPTIMIZER - What worked before!
    base_lr = 1e-4  # This got us 0.5924
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    
    # Simple cosine scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * 15,  # 15 epochs
        eta_min=base_lr * 0.1  # Don't go to zero
    )
    
    logger.info(f"Optimizer: AdamW with LR={base_lr}")
    logger.info("Scheduler: Cosine annealing over 15 epochs")
    
    # Training loop
    best_auc = 0.0
    patience = 0
    max_patience = 10
    
    for epoch in range(15):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 20 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)}")
            
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item()
                
                probs = torch.sigmoid(output.squeeze())
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # Metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1:2d}: "
                   f"Train Loss={avg_train_loss:.4f}, "
                   f"Val Loss={avg_val_loss:.4f}, "
                   f"Val AUC={val_auc:.4f}, "
                   f"LR={current_lr:.2e}")
        
        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            
            save_path = "model_weights/pat/pytorch/pat_conv_l_simple_best.pth"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc
            }, save_path)
            
            logger.info(f"✅ Saved best model with AUC: {val_auc:.4f}")
            
            if val_auc >= 0.58:
                logger.info("🎯 EXCELLENT: Beat standard PAT-L!")
            if val_auc >= 0.60:
                logger.info("🚀 OUTSTANDING: Strong performance!")
            if val_auc >= 0.625:
                logger.info("🏆 TARGET REACHED! Matched paper's result!")
                break
        else:
            patience += 1
            if patience >= max_patience:
                logger.info("Early stopping")
                break
    
    logger.info(f"\nFinal best AUC: {best_auc:.4f}")
    logger.info(f"Gap to target: {0.625 - best_auc:.3f}")


if __name__ == "__main__":
    main()