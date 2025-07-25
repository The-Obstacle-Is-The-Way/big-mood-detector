# Email to PAT Researchers: PAT-Conv-L Depression Classification Results

Dear PAT Research Team,

I'm implementing your Pretrained Actigraphy Transformer (PAT) for depression classification using the NHANES 2013-2014 dataset. While I've made significant progress, I'm seeking your guidance to close the final performance gap.

## Current Results

**Model**: PAT-Conv-L (PAT-L with Conv1d patch embedding)  
**Best AUC**: 0.5929  
**Target (from your paper)**: 0.625  
**Gap**: 3.2%

## Implementation Details

### Data Preprocessing
```python
# NHANES 2013-2014 subset
# Binary classification: PHQ-9 >= 10 as positive class
# 7 days × 1440 minutes = 10,080 timesteps per sample

# Normalization (following your paper)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)  # Fit on training only
X_val_scaled = scaler.transform(X_val_flat)         # Apply to validation

# Verified: mean=0, std=1 after normalization
```

### Architecture
```python
# PAT-L backbone with Conv1d patch embedding
class SimplePATConvLModel(nn.Module):
    def __init__(self):
        # Conv1d patch embedding (replacing linear)
        self.conv_patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=96,      # PAT-L embed_dim
            kernel_size=9,        # patch_size
            stride=9
        )
        
        # Standard PAT-L transformer (from pretrained weights)
        self.encoder = PATEncoder(
            num_layers=12,
            embed_dim=96,
            num_heads=4,
            mlp_ratio=4.0
        )
        
        # Depression classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 1)
        )
```

### Training Configuration
```python
# Hyperparameters
batch_size = 32
learning_rate = 1e-4
optimizer = AdamW(lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
scheduler = CosineAnnealingLR(T_max=15_epochs, eta_min=1e-5)
loss = BCEWithLogitsLoss(pos_weight=9.91)  # Handle class imbalance

# Training details
- Epochs: 15 (early stopped at 12)
- Best AUC achieved at epoch 2
- Used pretrained PAT-L_29k_weights.h5
- Conv layer initialized randomly
- No data augmentation
- Single seed (42)
```

### Training Log
```
Epoch  1: Train Loss=1.3685, Val AUC=0.5739
Epoch  2: Train Loss=1.3056, Val AUC=0.5929 (BEST)
Epoch  3: Train Loss=1.3159, Val AUC=0.5662
...
Epoch 12: Train Loss=1.2472, Val AUC=0.5634 (stopped)
```

## Key Findings

1. **Normalization was critical** - Initially got 0.4756 AUC due to incorrect normalization
2. **Conv embedding helped** - PAT-Conv-L (0.5929) outperformed standard PAT-L (0.5888)
3. **Simple training worked best** - Complex LP→FT strategies performed worse
4. **Quick convergence** - Best result at epoch 2, then plateaued

## Questions

1. **Data Augmentation**: Did you use any augmentation techniques (temporal jittering, mixup, etc.) in your training?

2. **Multiple Seeds**: Were your reported results averaged over multiple random seeds?

3. **NHANES Subset**: Did you use the full NHANES 2013-2014 dataset or apply any filtering criteria?

4. **Training Duration**: How many epochs did you typically train for? Did you observe similar early peaking?

5. **Hyperparameter Ranges**: Could you share the ranges you explored for:
   - Learning rates (encoder vs. classification head)
   - Batch sizes
   - Weight decay values

6. **Architecture Details**: For the Conv variant, did you use any specific kernel sizes or multiple conv layers?

## Attempted Variations

- Learning rates: 3e-5 to 2e-4
- Separate LRs for encoder (3e-5) and head (1e-3)
- Linear probing → fine-tuning (performed worse)
- Different schedulers (StepLR, LambdaLR)
- Batch sizes: 16, 32, 64

## Environment

- PyTorch 2.1.0
- CUDA 12.1
- Python 3.12
- Hardware: NVIDIA GPU (24GB VRAM)

Any insights to help close the 3.2% gap would be greatly appreciated. I'm happy to share code or additional details if helpful.

Best regards,
[Your name]

---

**Attachments available on request:**
- Full training script (`train_pat_conv_l_simple.py`)
- Training logs
- Model architecture implementation
- Data preprocessing pipeline

---

## Files to Attach to Email

Please attach these files when sending the email:

1. **Training Script** (main implementation)
   ```
   scripts/pat_training/train_pat_conv_l_simple.py
   ```

2. **Training Log** (shows full progression)
   ```
   training/logs/pat_conv_l_v0.5929_20250725.log
   ```

3. **Achievement Summary** (detailed results)
   ```
   docs/training/PAT_CONV_L_ACHIEVEMENT.md
   ```

4. **Original Training Log** (raw output)
   ```
   docs/archive/pat_experiments/pat_conv_l_simple.log
   ```

5. **Optional: Model Architecture** (if they want implementation details)
   ```
   src/big_mood_detector/infrastructure/ml_models/pat_model.py
   ```

Total size: ~200KB (very reasonable for email)