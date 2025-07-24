# PAT Depression Head Training Plan

## Executive Summary

Based on the PAT paper review, we can train high-quality depression classification heads. The paper shows PAT-L achieved ~0.59 AUC for depression prediction, which aligns with our 0.64 AUC (suggesting our initial result might be reasonable).

## Key Findings from Literature

### 1. Model Performance by Size
From Table 2 in the paper:
- **PAT-S (285K params)**: 0.560 AUC for depression
- **PAT-M (1M params)**: 0.559 AUC for depression  
- **PAT-L (2M params)**: 0.589 AUC for depression
- **PAT Conv-L**: 0.610 AUC for depression (best)

The convolutional patch embeddings consistently outperform linear projections.

### 2. Optimal Training Configuration
From their experiments (Table 3):
- **Mask ratio**: 0.90 during pretraining (best)
- **Loss function**: MSE on all patches (not just masked)
- **Smoothing**: No improvement from smoothing data
- **Dataset sizes**: Works well even with 500 participants

### 3. Training Strategy
The paper shows:
- Models perform well even with limited data (500 participants)
- Larger models show more improvement with more data
- Fine-tuning outperforms linear probing

## Our Training Plan

### Phase 1: Validate Small Model (✅ Done)
- Trained PAT-S with limited data
- Achieved 0.64 AUC (reasonable given paper's 0.56-0.61 range)
- Weights saved to `model_weights/pat/heads/pat_depression_head.pt`

### Phase 2: Full Dataset Training (Recommended Next)

#### Step 1: Data Preparation
```python
# Use all available NHANES 2013-2014 data
# The paper mentions ~2,800 participants for depression dataset
# We have DPQ_H.xpt (depression) and PAXMIN_H.xpt (actigraphy)
```

#### Step 2: Train All Model Sizes
Given your M1 Pro capabilities:

1. **PAT-S (Small)** - ✅ Already done
   - 285K parameters
   - ~30 minutes training time
   - Baseline performance

2. **PAT-M (Medium)** - Recommended
   - 1M parameters  
   - ~2-3 hours on M1 Pro
   - Good balance of performance/speed

3. **PAT-L (Large)** - If time permits
   - 2M parameters
   - ~4-6 hours on M1 Pro
   - Best performance

4. **PAT Conv-L** - Highest priority
   - 2M parameters with conv embeddings
   - Best depression performance in paper
   - ~4-6 hours on M1 Pro

### Phase 3: Training Protocol

```bash
# For each model size:
python scripts/train_pat_depression_head.py \
    --model-size medium \
    --use-conv-embeddings \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --nhanes-dir data/nhanes/2013-2014 \
    --output-dir model_weights/pat/heads
```

### Phase 4: Validation Strategy

1. **Train/Val/Test Split**:
   - 60% training
   - 20% validation (for early stopping)
   - 20% test (held-out evaluation)

2. **Metrics to Track**:
   - AUC (primary)
   - Accuracy, Precision, Recall
   - Calibration (Brier score)

3. **Expected Performance**:
   - PAT-S: 0.55-0.60 AUC
   - PAT-M: 0.55-0.60 AUC
   - PAT-L: 0.58-0.62 AUC
   - PAT Conv-L: 0.60-0.65 AUC

### Phase 5: Model Selection

Choose best model based on:
1. Test set AUC
2. Calibration quality
3. Inference speed requirements

## Implementation Details

### Data Loading
```python
# Load all participants with both depression scores and actigraphy
# Filter: PHQ-9 total score available
# Binary label: PHQ-9 >= 10 (moderate depression)
```

### Training Configuration
```python
config = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'early_stopping_patience': 10,
    'scheduler': 'cosine',
    'warmup_steps': 500
}
```

### Hardware Optimization for M1 Pro
- Use Apple Metal acceleration if available
- Batch size 32 should fit in memory
- Enable mixed precision training
- Use gradient accumulation if needed

## Timeline

1. **Today**: Review plan, prepare environment
2. **Day 1**: Train PAT-M and PAT Conv-M (4-6 hours)
3. **Day 2**: Train PAT-L and PAT Conv-L (8-12 hours)
4. **Day 3**: Evaluate all models, select best
5. **Day 4**: Integration testing with TemporalEnsembleOrchestrator

## Success Criteria

1. **Minimum**: Match paper's performance (0.589 AUC)
2. **Target**: Achieve 0.60+ AUC with proper validation
3. **Stretch**: 0.65+ AUC with Conv-L model

## Risk Mitigation

1. **If training fails**: Start with smaller batch sizes
2. **If performance is poor**: Check data preprocessing
3. **If M1 struggles**: Use cloud resources (Colab Pro)

## Publishing Results

Once we achieve good performance:
1. Document training process
2. Create model card with performance metrics
3. Consider contributing to original PAT repository
4. Update our documentation with final results

## Next Steps

1. Review this plan
2. Set up training environment
3. Start with PAT-M training (best ROI)
4. Monitor progress and adjust as needed

The key insight from the paper is that even modest-sized models (PAT-M) can achieve good performance, and the convolutional variants consistently outperform standard versions. Let's focus on getting solid, reproducible results with proper validation.