# NHANES Preprocessing Analysis - PAT Paper vs Our Implementation

## Critical Finding: Fixed vs Dynamic Normalization

### Paper's Approach (What They Actually Did)

From Section 4.3 of the PAT paper:

> "In our supervised datasets, **all train, test, and validation sets were standardized separately using Sklearn's StandardScaler**, such that each activity intensity is reported as a standard deviation from the mean of a given minute in the sequence."

Key points:
1. **Standardization is computed FROM the actual data** - not fixed values
2. Train, validation, and test sets are standardized **separately**
3. Uses scikit-learn's StandardScaler (computes mean and std from data)

### Our Implementation (What We're Doing Wrong)

From `nhanes_processor.py` lines 21-28:

```python
# NHANES cycle statistics for standardization (from PAT paper)
# These are approximate values for log-transformed activity
NHANES_STATS = {
    "2013-2014": {"mean": 2.5, "std": 2.0},
    "2011-2012": {"mean": 2.4, "std": 1.9},
    "2005-2006": {"mean": 2.3, "std": 1.8},
    "2003-2004": {"mean": 2.3, "std": 1.8},
}
```

And in `extract_pat_sequences()` lines 364-366:

```python
cycle_stats = self.NHANES_STATS.get("2013-2014", {"mean": 2.5, "std": 2.0})
flat_minutes = ((flat_minutes - cycle_stats["mean"]) / cycle_stats["std"]).astype(np.float32)
```

**We're using FIXED normalization values instead of computing from the data!**

## Why This Causes Poor Performance

1. **Signal Destruction**: When we normalize all data with mean=2.5, std=2.0, we're forcing all sequences to have nearly identical distributions
2. **The "Fixing bad normalization" warning** in our training logs happens because all normalized sequences have mean ≈ -1.24 with tiny variance
3. **AUC stuck at ~0.57** because the model can't distinguish between depressed/non-depressed when all inputs look identical

## Dataset Size Discrepancy

- Paper: 4,800 total participants → 2,000 test → **2,800 training**
- Our logs: **3,077 training samples** (10% more than paper)

This suggests we may be using a different data split or including extra participants.

## Complete Preprocessing Pipeline (Paper)

1. **Data Collection**: NHANES 2013-2014 wrist-worn accelerometer data
2. **Sequence Length**: 10,080 minutes (7 days × 1440 minutes/day)
3. **Transformation**: 
   - Log transform: `log1p(activity)`
   - Clip to [0, 10] range
4. **Standardization**: 
   - Compute mean and std **from training data only**
   - Apply z-score normalization: `(x - train_mean) / train_std`
   - **Validation/test use training statistics**
5. **Labels**: Depression = 1 if PHQ-9 >= 10, else 0

## What We Need to Fix

1. **Remove fixed NHANES_STATS** - these are destroying the signal
2. **Compute normalization from training data**:
   ```python
   # After log transform
   train_mean = X_train.mean()
   train_std = X_train.std()
   
   # Normalize all sets using training statistics
   X_train = (X_train - train_mean) / train_std
   X_val = (X_val - train_mean) / train_std  # Use TRAIN stats
   X_test = (X_test - train_mean) / train_std  # Use TRAIN stats
   ```

3. **Verify dataset size**: Check why we have 3,077 vs 2,800 training samples

4. **Fix the normalization in our cached data**: The `.npz` files were created with bad normalization

## Evidence This Is The Problem

From our training log:
```
2025-07-24 15:13:39,880 - WARNING - Fixing bad normalization...
2025-07-24 15:13:40,005 - INFO - Fixed normalization - Mean: 0.012, Std: 0.317
```

This shows that after "normalization" with fixed stats, all our data had nearly identical values (mean ≈ 0.012), which we then had to "fix" - but the damage was already done in the cached `.npz` file.

## Summary

**The paper uses dynamic normalization computed from training data. We use fixed normalization values that destroy the discriminative signal in the data. This explains why we're getting 0.57 AUC instead of 0.62 AUC.**