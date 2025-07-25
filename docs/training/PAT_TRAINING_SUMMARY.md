# PAT Training Summary - What We Learned

## Paper Facts (Verified)

### Depression Results from Paper (Table 5)
- **PAT-L (FT)**: 0.589 AUC 
- **PAT-L (LP)**: 0.582 AUC
- **PAT Conv-L (FT)**: 0.610 AUC ← Best result
- **PAT Conv-L (LP)**: 0.611 AUC

### Methodology from Paper
1. **Architecture**: "small feed forward layer and sigmoid activation"
2. **FT (Full Fine-Tuning)**: "we freeze no weights" - train everything
3. **LP (Linear Probing)**: "we freeze PAT and only train the added layers"
4. **NO mention of**: Two-stage training, complex heads, gradual unfreezing

## Our Training Attempts

### Attempt 1: Original simple approach
- Got to ~0.58 AUC but plateaued
- Used single Linear layer head (correct!)
- But learning rates might have been off

### Attempt 2: Two-stage "paper exact" (actually wrong!)
- Stage 1: 0.5754 (frozen encoder)
- Stage 2: 0.5314 (got WORSE when unfrozen)
- **Problem**: Paper doesn't do two-stage training!

## Key Insights

1. **We don't have Conv implementation** - We only have standard PAT-L
2. **Paper's approach is SIMPLE** - Just train everything (FT) or freeze encoder (LP)
3. **Our two-stage approach was wrong** - Made performance worse

## Clean Implementation

### Option 1: Full Fine-Tuning (FT)
```python
# Target: 0.589 AUC
model = PAT-L + Linear(96, 1)
optimizer = Adam(all_parameters, lr=1e-4)
# Train everything from start
```

### Option 2: Linear Probing (LP)
```python
# Target: 0.582 AUC
model = PAT-L + Linear(96, 1)
encoder.requires_grad = False  # Freeze
optimizer = Adam(head_only, lr=1e-3)
```

## Action Plan

1. **Run FT first** (`./scripts/launch_pat_l_clean.sh` → Choose 1)
   - Should achieve ~0.589 AUC
   - Simple, clean approach
   - 10-30 minutes on RTX 4090

2. **If FT works, try LP** for comparison
   - Should achieve ~0.582 AUC
   - Faster training (head only)

3. **Future: Implement Conv variant**
   - Need convolutional patch embeddings
   - Would enable 0.610 AUC target

## Why Our Previous Attempts Failed

1. **Overcomplicated the approach** - Two-stage training not in paper
2. **Wrong learning rates** - 5e-5 for encoder too low initially
3. **Misread the paper** - Thought Conv-L results applied to standard PAT-L

## Lessons Learned

- **Read the paper carefully** - FT means train everything, period
- **Start simple** - Paper used simple architecture for a reason
- **Check model variants** - Conv vs Standard are different models
- **Trust the baselines** - 0.589 for PAT-L (FT) is still good!

The clean scripts now implement EXACTLY what the paper describes. No tricks, no staging, just simple fine-tuning or linear probing.