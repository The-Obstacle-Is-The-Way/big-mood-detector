#!/usr/bin/env python3
"""
PAT-L Simple Linear Probing - EXACTLY as described in the paper
Target: 0.582 AUC for PAT-L (LP) on depression

Paper quote: "linear probing (LP), where we freeze PAT and only train the added layers"
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
    """Same architecture as FT version."""
    
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
    """Load and fix NHANES depression data."""
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    logger.info(f"Loading data from {cache_path}")
    
    data = np.load(cache_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    
    logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Fix normalization if needed
    train_means = X_train.mean(axis=1)
    if train_means.std() < 0.01:
        logger.warning("Fixing bad normalization...")
        X_train = X_train * 2.0 + 2.5
        X_val = X_val * 2.0 + 2.5
        
        train_mean = X_train.mean()
        train_std = X_train.std()
        
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        
        logger.info(f"Fixed normalization - Mean: {train_mean:.3f}, Std: {train_std:.3f}")
    
    return X_train, X_val, y_train, y_val


def train_simple_lp():
    """Train PAT-L with linear probing as per paper."""
    
    # Load data
    X_train, X_val, y_train, y_val = load_data()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
    
    # FREEZE ENCODER - this is the key difference from FT
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    
    # Log parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} (head only)")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Loss with class weighting
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer - only for head parameters
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)
    
    # Training loop
    logger.info("\n" + "="*50)
    logger.info("Starting Linear Probing (LP)")
    logger.info("="*50)
    
    best_auc = 0
    patience = 0
    max_patience = 10
    
    for epoch in range(50):  # Fewer epochs needed for LP
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
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
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'model_weights/pat/pytorch/pat_l_lp_best.pth')
            logger.info(f"✅ Saved best model with AUC: {val_auc:.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info("\n" + "="*50)
    logger.info(f"Training complete! Best AUC: {best_auc:.4f}")
    logger.info(f"Target from paper: 0.582 for PAT-L (LP)")
    logger.info("="*50)
    
    # Save results
    results = {
        'best_auc': float(best_auc),
        'model': 'PAT-L',
        'method': 'LP (Linear Probing)',
        'total_params': total_params,
        'trainable_params': trainable_params,
        'completed_at': datetime.now().isoformat()
    }
    
    with open('model_weights/pat/pytorch/pat_l_lp_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Create output directory
    Path("model_weights/pat/pytorch").mkdir(parents=True, exist_ok=True)
    
    # Run training
    train_simple_lp()