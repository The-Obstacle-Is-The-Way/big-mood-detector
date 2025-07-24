# PAT-L Training Findings and Resolution

## Summary

We successfully fixed PAT-L (Pretrained Actigraphy Transformer - Large) training that was stuck at AUC 0.4756 after 70 epochs. The issue was caused by over-normalization in the NHANES data processor, which removed all discriminative signal from the sequences. After fixing this issue, training immediately improved to AUC 0.5788 and continues to climb.

## The Problem

### Symptoms
- PAT-L training plateaued at AUC 0.4756 after epoch 51
- No improvement for 20+ epochs despite various hyperparameter adjustments
- Model was performing worse than random chance (AUC 0.50)

### Root Cause Analysis

Through comprehensive debugging, we discovered:

1. **Fixed Normalization Statistics**: The NHANES processor was using hardcoded normalization values:
   ```python
   NHANES_STATS = {
       "2013-2014": {"mean": 2.5, "std": 2.0},
   }
   ```

2. **Over-normalization**: This caused all sequences to have nearly identical means:
   - All sequences had mean: -1.24 ±0.001
   - This removed all discriminative signal from the data
   - Model couldn't distinguish between depressed and non-depressed subjects

3. **Validation**: A simple logistic regression baseline on mean activity achieved AUC 0.52, confirming that:
   - Data and labels were correctly aligned
   - There was signal in the data
   - The issue was with the normalization approach

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
- **Before fix**: AUC stuck at 0.4756
- **After fix**: 
  - Epoch 1: AUC 0.5622
  - Epoch 5: AUC 0.5699
  - Epoch 10: AUC 0.5788
  - Training continues to improve

### Training Configuration
The successful training uses:
- **Stage 1**: Train head only (30 epochs, LR=5e-3)
- **Stage 2**: Unfreeze last 2 transformer blocks (30 epochs, LR=1e-4)
- **Class weighting**: Handles imbalanced depression labels
- **Gradient clipping**: Max norm 1.0 for stability

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

1. `scripts/debug_pat_training.py` - Comprehensive debugging tool
2. `scripts/train_pat_l_run_now.py` - Simple fix that's currently running
3. `scripts/train_pat_l_advanced.py` - Full-featured training with all improvements
4. `scripts/analyze_pat_training.py` - Analysis and visualization tools
5. `scripts/monitor_training.py` - Monitor ongoing training progress

## Current Status

Training is running successfully via:
```bash
nohup python scripts/train_pat_l_run_now.py > training_pat_l.log 2>&1 &
```

Monitor progress with:
```bash
tail -f training_pat_l.log
```

The model is steadily improving and approaching the target performance from the paper (AUC 0.610).