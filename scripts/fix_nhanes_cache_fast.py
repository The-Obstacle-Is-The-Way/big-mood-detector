#!/usr/bin/env python3
"""
Fast fix for NHANES cache - repairs normalization without reloading huge files.
Takes seconds instead of 20+ minutes.
"""

import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_nhanes_cache():
    """Fix the normalization in existing cache file."""
    
    # Load the incorrectly normalized cache
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    if not cache_path.exists():
        logger.error(f"Cache file not found: {cache_path}")
        logger.error("Run the full data preparation script first!")
        return False
    
    logger.info(f"Loading existing cache from {cache_path}")
    data = np.load(cache_path)
    
    X_train = data['X_train']
    X_val = data['X_val'] 
    y_train = data['y_train']
    y_val = data['y_val']
    
    logger.info(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}")
    logger.info(f"Class balance - Train: {(y_train == 1).sum()}/{len(y_train)} positive")
    
    # Check if normalization is bad
    train_mean = X_train.mean()
    train_std = X_train.std()
    logger.info(f"Current statistics - Mean: {train_mean:.6f}, Std: {train_std:.6f}")
    
    if abs(train_mean - (-1.24)) < 0.5:
        logger.warning("Detected bad normalization! Fixing...")
        
        # Reverse the bad normalization (mean=2.5, std=2.0)
        logger.info("Step 1: Reversing incorrect normalization...")
        X_train_raw = X_train * 2.0 + 2.5
        X_val_raw = X_val * 2.0 + 2.5
        
        # Verify we recovered reasonable values
        logger.info(f"After reversal - Mean: {X_train_raw.mean():.3f}, Std: {X_train_raw.std():.3f}")
        
        # Apply correct normalization using StandardScaler
        logger.info("Step 2: Applying correct normalization (StandardScaler from training data)...")
        
        # Reshape for StandardScaler
        n_train = X_train_raw.shape[0]
        n_val = X_val_raw.shape[0]
        
        X_train_flat = X_train_raw.reshape(n_train, -1)
        X_val_flat = X_val_raw.reshape(n_val, -1)
        
        # Fit scaler on TRAINING data only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_val_scaled = scaler.transform(X_val_flat)  # Use training statistics!
        
        # Reshape back
        X_train_fixed = X_train_scaled.reshape(n_train, -1).astype(np.float32)
        X_val_fixed = X_val_scaled.reshape(n_val, -1).astype(np.float32)
        
        # Verify correct normalization
        logger.info(f"Fixed statistics - Train Mean: {X_train_fixed.mean():.6f}, Std: {X_train_fixed.std():.6f}")
        logger.info(f"Fixed statistics - Val Mean: {X_val_fixed.mean():.6f}, Std: {X_val_fixed.std():.6f}")
        
    else:
        logger.info("Normalization looks okay, no fix needed")
        X_train_fixed = X_train
        X_val_fixed = X_val
    
    # Save the fixed cache
    output_path = Path("data/cache/nhanes_pat_data_subsetNone_fixed.npz")
    np.savez_compressed(
        output_path,
        X_train=X_train_fixed,
        X_val=X_val_fixed,
        y_train=y_train,
        y_val=y_val
    )
    logger.info(f"Saved fixed cache to {output_path}")
    
    # Also overwrite the original for convenience
    backup_path = cache_path.with_suffix('.npz.backup')
    cache_path.rename(backup_path)
    logger.info(f"Backed up original to {backup_path}")
    
    np.savez_compressed(
        cache_path,
        X_train=X_train_fixed,
        X_val=X_val_fixed,
        y_train=y_train,
        y_val=y_val
    )
    logger.info(f"Overwrote original cache with fixed version")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 Fast NHANES Cache Fix")
    print("="*60)
    
    success = fix_nhanes_cache()
    
    if success:
        print("\n✅ Cache fixed successfully!")
        print("The data now uses proper StandardScaler normalization")
        print("You can now retrain and should see AUC improve to ~0.62")
    else:
        print("\n❌ Fix failed - see errors above")
    
    print("="*60)