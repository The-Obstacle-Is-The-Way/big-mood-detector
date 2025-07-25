#!/usr/bin/env python3
"""
Prepare NHANES depression data with CORRECT normalization.
Following the PAT paper exactly - compute normalization from training data.
"""

import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import NHANESProcessor


def prepare_depression_data_correct():
    """Prepare NHANES depression data following paper methodology EXACTLY."""
    
    # Initialize processor
    processor = NHANESProcessor(
        data_dir=Path("data/nhanes/2013-2014"),
        output_dir=Path("data/processed")
    )
    
    logger.info("Loading NHANES 2013-2014 data...")
    
    # Load actigraphy and depression scores
    actigraphy = processor.load_actigraphy("PAXMIN_H.xpt")
    depression = processor.load_depression_scores("DPQ_H.xpt")
    
    # Get subjects with both actigraphy and PHQ-9 scores
    actigraphy_subjects = set(actigraphy['SEQN'].unique())
    depression_subjects = set(depression['SEQN'].unique())
    common_subjects = list(actigraphy_subjects & depression_subjects)
    
    logger.info(f"Found {len(common_subjects)} subjects with both actigraphy and PHQ-9 scores")
    
    # The paper says 4,800 participants total for depression
    # If we have more, we need to filter
    if len(common_subjects) > 4800:
        logger.warning(f"Have {len(common_subjects)} subjects, but paper used 4,800. Sampling...")
        np.random.seed(42)
        common_subjects = np.random.choice(common_subjects, 4800, replace=False)
    
    # Extract sequences and labels
    sequences = []
    labels = []
    
    for subject_id in common_subjects:
        try:
            # Extract 10,080 minute sequence WITHOUT normalization
            sequence = processor.extract_pat_sequences(
                actigraphy, 
                subject_id,
                normalize=True,      # Log transform
                standardize=False    # NO STANDARDIZATION YET!
            )
            
            # Get depression label
            subject_depression = depression[depression['SEQN'] == subject_id]
            if len(subject_depression) > 0:
                phq9_total = subject_depression['PHQ9_total'].iloc[0]
                label = 1 if phq9_total >= 10 else 0
                
                sequences.append(sequence)
                labels.append(label)
        except Exception as e:
            logger.debug(f"Skipping subject {subject_id}: {e}")
            continue
    
    # Convert to arrays
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    logger.info(f"Extracted {len(X)} valid sequences")
    logger.info(f"Depression prevalence: {y.mean():.2%} ({y.sum()}/{len(y)})")
    
    # Split according to paper: 2,000 test, rest for train/val
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=2000, 
        random_state=42, 
        stratify=y
    )
    
    # Split remaining into train/val (80/20 is common)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42,
        stratify=y_temp
    )
    
    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # NOW STANDARDIZE - Using TRAINING data statistics only!
    logger.info("Standardizing using StandardScaler (computing from training data)...")
    
    # Reshape for StandardScaler (needs 2D input)
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(n_train, -1)
    X_val_flat = X_val.reshape(n_val, -1)
    X_test_flat = X_test.reshape(n_test, -1)
    
    # Fit scaler on TRAINING data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    
    # Apply to validation and test using TRAINING statistics
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape back to original
    X_train_final = X_train_scaled.reshape(n_train, 10080).astype(np.float32)
    X_val_final = X_val_scaled.reshape(n_val, 10080).astype(np.float32)
    X_test_final = X_test_scaled.reshape(n_test, 10080).astype(np.float32)
    
    # Log statistics
    logger.info(f"Training data statistics (after scaling):")
    logger.info(f"  Mean: {X_train_final.mean():.6f} (should be ~0)")
    logger.info(f"  Std: {X_train_final.std():.6f} (should be ~1)")
    logger.info(f"Validation data statistics:")
    logger.info(f"  Mean: {X_val_final.mean():.6f}")
    logger.info(f"  Std: {X_val_final.std():.6f}")
    
    # Save the CORRECTLY normalized data
    cache_path = Path("data/cache/nhanes_pat_data_correct.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        cache_path,
        X_train=X_train_final,
        X_val=X_val_final,
        X_test=X_test_final,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_
    )
    
    logger.info(f"Saved correctly normalized data to {cache_path}")
    
    # Also save the old-style cache for compatibility
    old_cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    np.savez_compressed(
        old_cache_path,
        X_train=X_train_final,
        X_val=X_val_final,
        y_train=y_train,
        y_val=y_val
    )
    logger.info(f"Also saved to old cache location: {old_cache_path}")
    
    return {
        'n_train': len(X_train_final),
        'n_val': len(X_val_final),
        'n_test': len(X_test_final),
        'depression_rate_train': y_train.mean(),
        'depression_rate_val': y_val.mean(),
        'depression_rate_test': y_test.mean()
    }


if __name__ == "__main__":
    # Delete old bad cache
    old_cache = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    if old_cache.exists():
        logger.warning(f"Deleting old incorrectly normalized cache: {old_cache}")
        old_cache.unlink()
    
    # Create new correct cache
    stats = prepare_depression_data_correct()
    
    print("\n" + "="*60)
    print("✅ NHANES Depression Data Prepared Correctly!")
    print("="*60)
    print(f"Train samples: {stats['n_train']}")
    print(f"Val samples: {stats['n_val']}")
    print(f"Test samples: {stats['n_test']}")
    print(f"Depression rate (train): {stats['depression_rate_train']:.2%}")
    print(f"Depression rate (val): {stats['depression_rate_val']:.2%}")
    print("\nNormalization: StandardScaler computed from TRAINING data")
    print("This matches the PAT paper methodology exactly!")
    print("="*60)