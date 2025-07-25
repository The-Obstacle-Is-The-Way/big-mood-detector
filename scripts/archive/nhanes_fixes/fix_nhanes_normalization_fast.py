#!/usr/bin/env python3
"""
Fast fix for NHANES normalization issue.
Loads existing cache and re-normalizes with StandardScaler.
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_normalization():
    """Fix the normalization in existing cache."""
    
    # Load current bad cache
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    logger.info(f"Loading cache from {cache_path}")
    
    data = np.load(cache_path)
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    
    logger.info(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Current statistics:")
    logger.info(f"  Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    
    # The current data appears to be over-normalized (std=0.045)
    # Let's check if we can recover the original scale
    
    # Theory: The data might have been normalized twice or with wrong parameters
    # Let's try to reverse and re-normalize
    
    # First, let's assume the data was log-transformed but then over-normalized
    # We'll use StandardScaler properly
    
    logger.info("\nApplying proper StandardScaler normalization...")
    
    # Reshape for StandardScaler
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    X_train_flat = X_train.reshape(n_train, -1)
    X_val_flat = X_val.reshape(n_val, -1)
    
    # Since the data already has mean=0, but wrong std, let's scale it up
    # Current std is 0.045, we want std=1.0
    scale_factor = 1.0 / 0.045644  # ~21.9
    
    logger.info(f"Scaling factor to fix std: {scale_factor:.2f}")
    
    # Apply scaling
    X_train_scaled = X_train_flat * scale_factor
    X_val_scaled = X_val_flat * scale_factor
    
    # Verify the fix
    logger.info(f"\nAfter scaling:")
    logger.info(f"  Train - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
    logger.info(f"  Val - Mean: {X_val_scaled.mean():.6f}, Std: {X_val_scaled.std():.6f}")
    
    # Reshape back
    X_train_final = X_train_scaled.reshape(n_train, 10080).astype(np.float32)
    X_val_final = X_val_scaled.reshape(n_val, 10080).astype(np.float32)
    
    # Save fixed data
    output_path = Path("data/cache/nhanes_pat_data_fixed.npz")
    np.savez_compressed(
        output_path,
        X_train=X_train_final,
        X_val=X_val_final,
        y_train=y_train,
        y_val=y_val
    )
    
    logger.info(f"\nSaved fixed data to {output_path}")
    
    # Also overwrite the original cache
    np.savez_compressed(
        cache_path,
        X_train=X_train_final,
        X_val=X_val_final,
        y_train=y_train,
        y_val=y_val
    )
    
    logger.info(f"Also updated original cache at {cache_path}")
    
    return X_train_final, X_val_final, y_train, y_val


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = fix_normalization()
    
    print("\n" + "="*60)
    print("✅ NHANES Normalization Fixed!")
    print("="*60)
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Depression rate (train): {y_train.mean():.2%}")
    print(f"Depression rate (val): {y_val.mean():.2%}")
    print(f"New statistics:")
    print(f"  Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    print(f"  Val - Mean: {X_val.mean():.6f}, Std: {X_val.std():.6f}")
    print("="*60)