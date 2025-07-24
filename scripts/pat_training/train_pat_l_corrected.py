#!/usr/bin/env python3
"""
PAT-L Training with Corrected Data
Uses the fixed cache with proper normalization.
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
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    
    # NO ADDITIONAL NORMALIZATION - data is already correct!
    
    return X_train, X_val, y_train, y_val


def train_corrected():
    """Train PAT-L with corrected data."""
    
    # Load data
    X_train, X_val, y_train, y_val = load_data()
    
    # Device
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
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Loss with class weighting
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Separate optimizers
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    
    logger.info("Setting up optimizers:")
    logger.info("  - Encoder LR: 2e-5")
    logger.info("  - Head LR: 5e-4")
    
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 2e-5},
        {'params': head_params, 'lr': 5e-4}
    ])
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # Training loop
    logger.info("\n" + "="*50)
    logger.info("Starting Training with CORRECTED Data")
    logger.info("Target: 0.620 AUC")
    logger.info("="*50)
    
    best_auc = 0
    patience = 0
    max_patience = 15
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                output = model(data)
                val_preds.extend(torch.sigmoid(output).cpu().numpy())
                val_targets.extend(target.numpy())
        
        val_auc = roc_auc_score(val_targets, val_preds)
        avg_loss = train_loss / len(train_loader)
        
        # Get current learning rates
        encoder_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}, "
                   f"Encoder LR={encoder_lr:.2e}, Head LR={head_lr:.2e}")
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'model_weights/pat/pytorch/pat_l_corrected_best.pth')
            logger.info(f"✅ Saved best model with AUC: {val_auc:.4f}")
            patience = 0
            
            if val_auc > 0.6:
                logger.info(f"🎯 Getting close to target! Current: {val_auc:.4f}, Target: 0.620")
        else:
            patience += 1
            if patience >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info("\n" + "="*50)
    logger.info(f"Training complete! Best AUC: {best_auc:.4f}")
    logger.info(f"Target from paper (n=2800): 0.620")
    if best_auc >= 0.615:
        logger.info("✅ SUCCESS! Achieved paper-level performance!")
    else:
        logger.info("⚠️  Below target. May need further tuning.")
    logger.info("="*50)
    
    # Save results
    results = {
        'best_auc': float(best_auc),
        'model': 'PAT-L',
        'method': 'FT (Corrected Data)',
        'encoder_lr': '2e-5',
        'head_lr': '5e-4',
        'total_params': total_params,
        'completed_at': datetime.now().isoformat()
    }
    
    with open('model_weights/pat/pytorch/pat_l_corrected_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Create output directory
    Path("model_weights/pat/pytorch").mkdir(parents=True, exist_ok=True)
    
    # Run training
    train_corrected()