# PAT Conv-L Depression Debugging Plan & Results Tracking

## Target
**Paper's PAT Conv-L Depression Results (n=2800):**
- PAT Conv-L (LP): 0.625 AUC
- PAT Conv-L (FT): 0.624 AUC

## Current Issue
- Getting 0.44 AUC in LP phase (WORSE than random!)
- Previous simple FT got 0.592 AUC (better but still 0.033 short)

## Debugging Hypotheses (Ranked by Likelihood)

### 1. ❌ Encoder Freezing Issue
**Hypothesis:** Encoder params not properly frozen or head not trainable
**Test:** Check requires_grad on all params
**Expected:** Encoder frozen, head trainable
**Result:** [PENDING]

### 2. ❌ Label Polarity/Inversion  
**Hypothesis:** Labels or predictions inverted (0.44 ≈ 1 - 0.56)
**Test:** Check label distribution and prediction distribution
**Expected:** ~10% positive labels (depression)
**Result:** [PENDING]

### 3. ❌ Head Initialization/LR Too High
**Hypothesis:** Random head + high LR → saturated sigmoid
**Test:** Check initial loss, gradient magnitudes
**Expected:** Initial BCE ~0.69, not 1.6+
**Result:** [PENDING]

### 4. ❌ Data Normalization Issue
**Hypothesis:** StandardScaler not applied correctly
**Test:** Check data statistics (mean≈0, std≈1)
**Expected:** Properly normalized features
**Result:** [PENDING]

### 5. ❌ Conv Embedding Architecture
**Hypothesis:** Conv implementation differs from paper
**Test:** Compare our Conv vs paper description
**Expected:** Matches paper spec
**Result:** [PENDING]

## Experiments to Run

### Experiment 1: Diagnostic Script
```python
# Run on current model to diagnose
with torch.no_grad():
    # Sample batch
    x_sample = X_val[:128]
    y_sample = y_val[:128]
    
    # Get predictions
    logits = model(x_sample.to(device)).cpu().numpy()
    probs = torch.sigmoid(logits).numpy()
    
    print(f"Label distribution - Mean: {y_sample.mean():.3f}")
    print(f"Logits - Mean: {logits.mean():.3f}, Std: {logits.std():.3f}")
    print(f"Probs - Mean: {probs.mean():.3f}")
    print(f"Manual AUC: {roc_auc_score(y_sample, logits):.4f}")
```

### Experiment 2: Parameter Check
```python
# Check which params are frozen
encoder_frozen = []
head_trainable = []
for name, param in model.named_parameters():
    if 'encoder' in name and not param.requires_grad:
        encoder_frozen.append(name)
    elif param.requires_grad:
        head_trainable.append(name)
        
print(f"Frozen encoder params: {len(encoder_frozen)}")
print(f"Trainable head params: {len(head_trainable)}")
print("Head params:", head_trainable)
```

### Experiment 3: Simple Baseline
- Train with JUST fine-tuning (no LP)
- Use our working setup (Adam, 1e-4/1e-3 LR)
- Should get ~0.59 AUC to confirm data is OK

### Experiment 4: Paper's Exact LP
- Fix encoder freezing properly
- Use paper's exact LRs
- AdamW with weight decay
- Monitor each epoch carefully

## Results Tracking

| Exp # | Setup | Epochs | Best Val AUC | Notes |
|-------|-------|--------|--------------|-------|
| 0 | Original FT (Adam, 1e-4/1e-3) | 11 | 0.5924 | Good baseline |
| 1 | Paper LP→FT (AdamW, broken freeze) | 5 | 0.4446 | BROKEN - worse than random |
| 2 | Fixed normalization (std=1.0) LP | 5 | 0.5442 | FIXED! Normalization was the issue |
| 3 | Fixed norm + FT phase | STALLED | 0.5442 | Scheduler bug: encoder LR stayed 0 |
| 4 | Fixed scheduler + 3 param groups | FAILED | 0.4461 | Scheduler still zeroing LRs |
| 5 | Initial run (Adam, 1e-4/1e-3) | SUCCESS | 0.5924 | Proves model is capable! |
| 6 | Final fix: force initial_lr | ABANDONED | - | Too complex, reverted to simple |
| 7 | Simple approach (train_pat_conv_l_simple.py) | RUNNING | TBD | Best baseline: 0.5924 AUC |
| 8 | Stable script (train_pat_stable.py) | RUNNING | TBD | SSOT for reproducible runs |

## Key Paper Details to Match
1. **Optimizer:** AdamW (β1=0.9, β2=0.95)
2. **Weight Decay:** 0.01 (except biases & LayerNorm)
3. **LP Phase:** 5 epochs, encoder frozen, head LR=5e-4
4. **FT Phase:** 15 epochs, encoder LR=3e-5, head LR=5e-4
5. **Warmup:** 10% of total steps
6. **Scheduler:** Cosine decay after warmup
7. **Gradient Clip:** 1.0
8. **Dropout:** 0.1
9. **Batch Size:** 32
10. **Gradient Accumulation:** 2

## Next Steps
1. Run diagnostic script on broken model
2. Fix most likely issue
3. Re-run and track results
4. Iterate until we hit 0.625 AUC