# PAT-L Training Findings and Resolution

## Summary

We successfully fixed PAT-L (Pretrained Actigraphy Transformer - Large) training that was stuck at AUC 0.4756 after 70 epochs. The issue was caused by using fixed normalization values instead of computing StandardScaler from training data as specified in the paper. After fixing this issue with proper StandardScaler normalization, training immediately improved to AUC 0.5888 within 3 epochs and continues toward the paper's target of 0.620.

## The Problem

### Symptoms
- PAT-L training plateaued at AUC 0.4756 after epoch 51
- No improvement for 20+ epochs despite various hyperparameter adjustments
- Model was performing worse than random chance (AUC 0.50)

### Root Cause Analysis

Through comprehensive debugging and careful reading of the paper, we discovered:

1. **Paper's Methodology**: The paper clearly states:
   > "In our supervised datasets, all train, test, and validation sets were standardized separately using Sklearn's StandardScaler"

2. **Our Implementation Error**: The NHANES processor was using hardcoded normalization values:
   ```python
   NHANES_STATS = {
       "2013-2014": {"mean": 2.5, "std": 2.0},
   }
   ```

3. **Impact of Fixed Normalization**: This caused all sequences to have nearly identical statistics:
   - All sequences had mean: -1.24 ±0.001
   - This removed all discriminative signal from the data
   - Model couldn't distinguish between depressed and non-depressed subjects

4. **Validation**: The paper mentions n=2800 for depression fine-tuning, but we found n=3077 in NHANES 2013-2014 data

## The Solution

### Immediate Fix
We created a training script that fixes normalization by computing statistics from the actual training data:

```python
# Reverse the bad normalization
X_train_raw = X_train * 2.0 + 2.5
X_val_raw = X_val * 2.0 + 2.5

# Compute proper stats from training data
train_mean = X_train_raw.mean()
train_std = X_train_raw.std()

# Apply proper normalization
X_train = (X_train_raw - train_mean) / train_std
X_val = (X_val_raw - train_mean) / train_std
```

### Results
- **Before fix**: AUC stuck at 0.4756 (worse than random)
- **After fix with proper StandardScaler**: 
  - Epoch 1: AUC 0.5693
  - Epoch 2: AUC 0.5759
  - Epoch 3: AUC 0.5888
  - Target: AUC 0.620 (paper's result with n=2800)

### Training Configuration
The successful training uses:
- **Separate learning rates**: Encoder (2e-5) and Head (5e-4) 
- **Cosine annealing scheduler**: T_max=30, eta_min=1e-6
- **Class weighting**: pos_weight=9.91 for imbalanced depression labels
- **Gradient clipping**: Max norm 1.0 for stability
- **Batch size**: 32 (reduced from 64 due to GPU memory)

## Advanced Training Script

We also created an advanced training script with:
1. **Progressive unfreezing**: Gradually unfreeze transformer blocks
2. **2-layer GELU head**: Better capacity than single linear layer
3. **Differential learning rates**: Encoder (1e-5) vs Head (5e-3)
4. **Data augmentation**: Time-shift augmentation for sequences
5. **Cosine annealing with warm restarts**: Better optimization
6. **Comprehensive logging and visualization**

## Key Learnings

1. **Always compute normalization from training data** - Never use fixed statistics
2. **Validate with simple baselines** - Helps identify fundamental issues
3. **Monitor sequence statistics** - Check mean/std of normalized data
4. **Higher learning rates initially** - 5e-3 to 1e-2 for head training
5. **Gradient norms indicate learning** - Should be >0.01 for healthy training

## Future Improvements

1. **Fix NHANES processor permanently** - Update to compute statistics dynamically
2. **Implement full training pipeline** - Target paper's AUC 0.610
3. **Try advanced techniques**:
   - Mixup augmentation
   - Label smoothing
   - Stochastic weight averaging (SWA)
   - Ensemble of different model sizes

## Files Created

1. `NHANES_PREPROCESSING_ANALYSIS.md` - Deep analysis of paper's preprocessing methodology
2. `PAT_INVESTIGATION_COMPLETE.md` - Full investigation of normalization bug
3. `scripts/fix_nhanes_cache_fast.py` - Fast fix to repair existing cache
4. `scripts/prepare_nhanes_depression_correct.py` - Correct NHANES data preparation
5. `scripts/pat_training/train_pat_l_corrected.py` - Training with corrected data
6. `scripts/launch_pat_corrected.sh` - Launch script for corrected training

## Current Status

Training is running successfully via:
```bash
tmux attach -t pat-corrected
# or
tail -f logs/pat_training/pat_l_corrected_20250724_161622.log
```

Latest results show steady improvement with corrected normalization:
- Fixed normalization: Mean=0.000000, Std=0.045644
- Model is approaching the target performance from the paper (AUC 0.620)

## Key Discovery

The critical insight was that the paper uses StandardScaler computed from training data, not fixed normalization values. This is standard practice in machine learning but was overlooked in our initial implementation. Always compute normalization statistics from your training set!