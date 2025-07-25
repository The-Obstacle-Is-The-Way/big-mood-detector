#!/usr/bin/env python3
"""
Fix NHANES normalization using GLOBAL normalization across all timesteps.
The paper likely uses global normalization, not per-timestep.
"""

import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_with_global_norm():
    """Fix normalization using global statistics."""
    
    # Load backup
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
    
    # Reverse bad normalization
    logger.info("\nReversing bad normalization (mean=2.5, std=2.0)...")
    X_train_raw = X_train * 2.0 + 2.5
    X_val_raw = X_val * 2.0 + 2.5
    
    logger.info(f"After reversal:")
    logger.info(f"  Train - Mean: {X_train_raw.mean():.6f}, Std: {X_train_raw.std():.6f}")
    
    # GLOBAL normalization - compute stats across ALL data points
    logger.info("\nApplying GLOBAL normalization...")
    
    # Compute global statistics from training data
    global_mean = X_train_raw.mean()
    global_std = X_train_raw.std()
    
    logger.info(f"Global statistics from training:")
    logger.info(f"  Mean: {global_mean:.6f}")
    logger.info(f"  Std: {global_std:.6f}")
    
    # Apply global normalization
    X_train_final = ((X_train_raw - global_mean) / global_std).astype(np.float32)
    X_val_final = ((X_val_raw - global_mean) / global_std).astype(np.float32)
    
    logger.info(f"\nFinal statistics:")
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
    
    logger.info(f"\nSaved globally normalized data to {cache_path}")
    
    return X_train_final, X_val_final, y_train, y_val


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = fix_with_global_norm()
    
    print("\n" + "="*60)
    print("NHANES Cache Fixed with GLOBAL Normalization!")
    print("="*60)
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Depression rate (train): {y_train.mean():.2%}")
    print(f"Depression rate (val): {y_val.mean():.2%}")
    print(f"\nFinal statistics:")
    print(f"  Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    print(f"  Val - Mean: {X_val.mean():.6f}, Std: {X_val.std():.6f}")
    print("\nThis should now have std ~1.0 for proper training!")
    print("="*60)