#!/usr/bin/env python3
"""
Fix NHANES cache normalization by reversing bad normalization and applying StandardScaler.
The issue: Data was normalized with fixed values (mean=2.5, std=2.0) instead of computing from data.
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_cache():
    """Fix the normalization in NHANES cache."""
    
    # Restore backup first
    backup_path = Path("data/cache/nhanes_pat_data_subsetNone.npz.backup")
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    
    logger.info(f"Loading backup from {backup_path}")
    data = np.load(backup_path)
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    
    logger.info(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Current (bad) statistics:")
    logger.info(f"  Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    logger.info(f"  Val - Mean: {X_val.mean():.6f}, Std: {X_val.std():.6f}")
    
    # The issue: data was normalized with FIXED values instead of computed from data
    # Bad normalization: (x - 2.5) / 2.0
    # Current mean = -1.24 suggests original data mean was ~0.02
    
    logger.info("\nReversing bad normalization (mean=2.5, std=2.0)...")
    
    # Reverse: x_original = x_normalized * 2.0 + 2.5
    X_train_raw = X_train * 2.0 + 2.5
    X_val_raw = X_val * 2.0 + 2.5
    
    logger.info(f"After reversal:")
    logger.info(f"  Train - Mean: {X_train_raw.mean():.6f}, Std: {X_train_raw.std():.6f}")
    logger.info(f"  Val - Mean: {X_val_raw.mean():.6f}, Std: {X_val_raw.std():.6f}")
    
    # Now apply PROPER normalization using StandardScaler
    logger.info("\nApplying StandardScaler (fit on training data)...")
    
    # Reshape for StandardScaler
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    X_train_flat = X_train_raw.reshape(n_train, -1)
    X_val_flat = X_val_raw.reshape(n_val, -1)
    
    # Fit scaler on TRAINING data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    
    # Apply to validation using TRAINING statistics
    X_val_scaled = scaler.transform(X_val_flat)
    
    # Reshape back
    X_train_final = X_train_scaled.reshape(n_train, 10080).astype(np.float32)
    X_val_final = X_val_scaled.reshape(n_val, 10080).astype(np.float32)
    
    logger.info(f"\nFinal statistics (should be ~0 mean, ~1 std):")
    logger.info(f"  Train - Mean: {X_train_final.mean():.6f}, Std: {X_train_final.std():.6f}")
    logger.info(f"  Val - Mean: {X_val_final.mean():.6f}, Std: {X_val_final.std():.6f}")
    
    # Save corrected data
    np.savez_compressed(
        cache_path,
        X_train=X_train_final,
        X_val=X_val_final,
        y_train=y_train,
        y_val=y_val
    )
    
    logger.info(f"\nSaved corrected data to {cache_path}")
    
    # Also save scaler parameters for reference
    scaler_info = {
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'n_features': scaler.n_features_in_,
        'n_samples_seen': scaler.n_samples_seen_
    }
    
    return X_train_final, X_val_final, y_train, y_val, scaler_info


if __name__ == "__main__":
    X_train, X_val, y_train, y_val, scaler_info = fix_cache()
    
    print("\n" + "="*60)
    print("NHANES Cache Fixed with Proper Normalization!")
    print("="*60)
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Depression rate (train): {y_train.mean():.2%}")
    print(f"Depression rate (val): {y_val.mean():.2%}")
    print(f"\nFinal statistics:")
    print(f"  Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    print(f"  Val - Mean: {X_val.mean():.6f}, Std: {X_val.std():.6f}")
    print(f"\nScaler info:")
    print(f"  Original data mean: {scaler_info['mean'].mean():.6f}")
    print(f"  Original data std: {scaler_info['scale'].mean():.6f}")
    print("="*60)