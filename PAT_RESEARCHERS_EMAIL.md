# Email to PAT Researchers: PAT-Conv-L Depression Classification Results

Dear PAT Research Team,

I'm implementing your Pretrained Actigraphy Transformer (PAT) for depression classification using the NHANES 2013-2014 dataset. While I've made significant progress, I'm seeking your guidance to close the final performance gap.

## Current Results

**Model**: PAT-Conv-L (PAT-L with Conv1d patch embedding)  
**Best AUC**: 0.5929  
**Target (from your paper)**: 0.625  
**Gap**: 3.2%

## Implementation Details

### Complete Data Processing Pipeline

#### 1. NHANES Data Extraction
```python
# Data source: NHANES 2013-2014 (PAXMIN_H.xpt, PAXDAY_H.xpt)
# Sample selection criteria:
#   - Has valid 7-day actigraphy data (10,080 minutes)
#   - Has PHQ-9 depression screening scores
#   - Excluded subjects on benzodiazepines/SSRIs
#   - Final dataset: n=3,077 subjects

# Binary classification target:
#   - Depression: PHQ-9 >= 10 (per DSM-5 criteria)
#   - Class balance: ~9% positive (279 depressed / 3077 total)
```

#### 2. Activity Count Preprocessing
```python
# Step 1: Log transformation (as per paper)
activity_log = np.log(activity_counts + 1)

# Step 2: Reshape to sequences
# Input: 7 days × 1440 minutes/day = 10,080 timesteps
sequences = activity_log.reshape(-1, 10080)

# Step 3: StandardScaler normalization
from sklearn.preprocessing import StandardScaler

# CRITICAL: Fit scaler on TRAINING data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(n_train, -1))
X_val_scaled = scaler.transform(X_val.reshape(n_val, -1))

# Reshape back to sequences
X_train_final = X_train_scaled.reshape(n_train, 10080)
X_val_final = X_val_scaled.reshape(n_val, 10080)

# Verified statistics after normalization:
# Train: mean=0.000001, std=0.999998
# Val: mean=-0.000823, std=1.002341
```

#### 3. Critical Bug We Fixed
```python
# ❌ WRONG (caused AUC 0.4756 - worse than random):
HARDCODED_STATS = {"mean": 2.5, "std": 2.0}
X_normalized = (X - HARDCODED_STATS["mean"]) / HARDCODED_STATS["std"]
# This made all sequences identical!

# ✅ CORRECT (immediately improved to AUC 0.57+):
scaler = StandardScaler()
scaler.fit(X_train)  # Compute from actual training data
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Use training statistics
```

### Architecture
```python
# PAT-L backbone with Conv1d patch embedding
class PATConvLEncoder(PATPyTorchEncoder):
    def __init__(self):
        super().__init__(model_size="large")
        
        # Replace linear patch embedding with convolutional
        self.patch_embed = ConvPatchEmbedding(
            patch_size=9,      # Same as PAT-L
            embed_dim=96,      # PAT-L hidden dimension
            in_channels=1,     # Univariate time series
            kernel_size=9      # Conv kernel = patch size
        )

class ConvPatchEmbedding(nn.Module):
    """Key innovation: Conv1D instead of Linear projection"""
    def __init__(self, patch_size, embed_dim, in_channels=1, kernel_size=None):
        super().__init__()
        kernel_size = kernel_size or patch_size
        
        # 1D Convolution creates patch embeddings
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=patch_size,     # Non-overlapping patches
            padding=0
        )
    
    def forward(self, x):
        # Input: (batch, 10080)
        x = x.unsqueeze(1)  # Add channel dim: (batch, 1, 10080)
        x = self.conv(x)    # Conv: (batch, 96, 1120 patches)
        x = x.permute(0, 2, 1)  # Rearrange: (batch, 1120, 96)
        return x

# Full model with classification head
class SimplePATConvLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PATConvLEncoder()  # Our Conv variant
        self.head = nn.Linear(96, 1)      # Binary classification
    
    def forward(self, x):
        embeddings = self.encoder(x)  # (batch, 96)
        logits = self.head(embeddings)  # (batch, 1)
        return logits.squeeze()
```

### Training Configuration
```python
# Hyperparameters that worked best
batch_size = 32
base_lr = 1e-4  # Single LR for all parameters
optimizer = optim.AdamW(
    model.parameters(),
    lr=base_lr,
    betas=(0.9, 0.95),    # Paper's values
    weight_decay=0.01
)

# Learning rate schedule
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=len(train_loader) * 15,  # 15 epochs
    eta_min=base_lr * 0.1
)

# Loss function with class weighting
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~9.91
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Training details
- Total epochs: 15 (early stopped at 12)
- Best AUC: 0.5929 at epoch 2
- Pretrained weights: PAT-L_29k_weights.h5 (transformer only)
- Conv layer: Xavier uniform initialization
- Data augmentation: None (potential improvement area)
- Random seed: 42
- Hardware: NVIDIA GPU with 24GB VRAM
```

### Training Log
```
Epoch  1: Train Loss=1.3685, Val AUC=0.5739
Epoch  2: Train Loss=1.3056, Val AUC=0.5929 (BEST)
Epoch  3: Train Loss=1.3159, Val AUC=0.5662
...
Epoch 12: Train Loss=1.2472, Val AUC=0.5634 (stopped)
```

## Key Findings & Complete Journey

### 1. The Normalization Discovery (Critical!)
**Problem**: Model stuck at AUC 0.4756 for 70+ epochs (worse than random)
**Root Cause**: Used fixed normalization values instead of StandardScaler
**Impact**: All sequences became nearly identical (mean=-1.24±0.001)
**Solution**: Implemented proper StandardScaler → immediate jump to AUC 0.57+

### 2. Model Evolution
| Model | Architecture | Best AUC | Notes |
|-------|-------------|----------|-------|
| PAT-S | Standard Linear | 0.560 | Matches paper (0.560) |
| PAT-M | Standard Linear | 0.540 | Close to paper (0.559) |
| PAT-L | Standard Linear | 0.5888 | Below paper (0.610) |
| PAT-Conv-L | Conv1D patches | **0.5929** | Our best, still 3.2% gap |

### 3. Training Strategy Insights
- **Simple is better**: Basic AdamW outperformed complex LP→FT schemes
- **Early convergence**: Best results at epochs 2-3, then plateaued
- **Learning rate**: 1e-4 worked better than paper's 3e-5 for encoder
- **No augmentation**: We didn't use any (possible improvement area)

### 4. What Didn't Work
- Separate learning rates for encoder (3e-5) vs head (5e-4)
- Linear Probing → Fine-tuning (actually decreased performance)
- Progressive unfreezing of transformer blocks
- Different weight initializations for classification head

## Questions

### Data & Preprocessing
1. **NHANES Sample Size**: We found n=3,077 subjects with valid data, but your paper mentions n=2,800. Did you apply additional exclusion criteria beyond removing benzodiazepine/SSRI users?

2. **Activity Count Range**: After log(x+1) transformation, what was the typical range of values before StandardScaler? We observed ~0-8 range.

3. **Missing Data**: How did you handle missing minutes in the 7-day sequences? Zero-fill, interpolation, or exclusion?

### Training Methodology
4. **Data Augmentation**: The 3.2% gap suggests we're missing something. Did you use:
   - Temporal jittering or time warping?
   - Mixup/CutMix for time series?
   - Activity count noise injection?
   - Synthetic minority oversampling (SMOTE)?

5. **Multiple Seeds & Cross-Validation**: 
   - Were reported results averaged over multiple seeds?
   - Did you use k-fold cross-validation?
   - Any ensemble of different random initializations?

6. **Training Dynamics**:
   - We see best results at epoch 2-3 then plateau. Is this expected?
   - Did you use early stopping or train for fixed epochs?
   - Any learning rate warmup period?

### Architecture & Optimization
7. **Conv Implementation Details**:
   - Single Conv1D layer or multiple?
   - Any specific kernel initialization?
   - Dropout in the conv embedding?
   - Did you try different kernel sizes beyond patch_size?

8. **Optimization Tricks**:
   - Gradient accumulation for larger effective batch size?
   - Label smoothing for the imbalanced dataset?
   - Different optimizer (Lion, AdaFactor)?
   - Stochastic Weight Averaging (SWA)?

### Debugging Help
9. **Sanity Checks**: What validation metrics beyond AUC did you monitor?

10. **Common Pitfalls**: Any other preprocessing or training details that are easy to miss?

## Complete List of Attempted Variations

### Successful Approaches
- ✅ StandardScaler normalization (fixed the critical bug)
- ✅ Conv1D patch embedding (0.5929 > 0.5888)
- ✅ Simple training with single LR (1e-4)
- ✅ Class-weighted loss (pos_weight=9.91)

### Unsuccessful Attempts
- ❌ Paper's exact LP→FT strategy (worse performance)
- ❌ Separate LRs: encoder (3e-5) + head (5e-4)
- ❌ Progressive unfreezing of transformer blocks
- ❌ 2-layer classification head with GELU
- ❌ Different batch sizes (16, 64)
- ❌ StepLR and LambdaLR schedulers
- ❌ Higher learning rates (2e-4, 5e-4)
- ❌ Longer training (30+ epochs)

### Not Yet Tried (Potential Solutions)
- ⏳ Data augmentation techniques
- ⏳ Multiple random seeds ensemble
- ⏳ Different train/val splits
- ⏳ Focal loss for class imbalance
- ⏳ AdamW with different betas
- ⏳ Gradient accumulation
- ⏳ Mix of PAT sizes (S+M+L ensemble)

## Environment

- PyTorch 2.1.0
- CUDA 12.1
- Python 3.12
- Hardware: NVIDIA GPU (24GB VRAM)

## Our Implementation Context

### Full System Architecture
We're implementing PAT as part of a clinical bipolar mood prediction system that combines:
- **PAT**: Assesses current state from past 7 days (depression risk)
- **XGBoost**: Predicts future risk from circadian patterns (36 statistical features)
- **Temporal Ensemble**: Combines "now" (PAT) and "tomorrow" (XGBoost) predictions

### Code Quality & Testing
- 976 unit tests passing
- 90%+ code coverage
- Full type safety (mypy)
- Clean architecture with dependency injection
- Production-ready with FastAPI deployment

### Reproducibility
Our implementation is fully open source. Key files:
- `train_pat_conv_l_simple.py`: Exact training script
- `nhanes_processor.py`: Data preprocessing pipeline
- `pat_pytorch.py`: PyTorch model implementation
- All training logs and model checkpoints preserved

We've documented every step of our journey, including failures, which might help other researchers avoid similar pitfalls.

Any insights to help close the 3.2% gap would be greatly appreciated. I'm happy to share code, run specific experiments, or provide additional details if helpful.

Best regards,
[Your name]

P.S. Thank you for the excellent paper and pretrained weights. Even at 0.5929 AUC, PAT is providing valuable clinical insights in our mood prediction system.

---

**Attachments available on request:**
- Full training script (`train_pat_conv_l_simple.py`)
- Training logs
- Model architecture implementation
- Data preprocessing pipeline

---

## Anticipated Follow-up Questions

### "Why is your n=3,077 different from our n=2,800?"
We included all NHANES 2013-2014 subjects with:
- Complete 7-day actigraphy data
- Valid PHQ-9 scores
- Not on benzodiazepines/SSRIs

Possible differences:
- Different NHANES cycles?
- Additional quality control filters?
- Minimum wear time requirements?

### "Did you verify the weight conversion?"
Yes, we achieved near-perfect conversion:
- Max difference: 0.000006 between TensorFlow and PyTorch weights
- Verified layer-by-layer correspondence
- Attention patterns match expected behavior

### "What about the embedding extraction?"
We replicated your embedding extraction exactly:
- Global average pooling after transformer blocks
- Returns 96-dimensional features (PAT-L)
- No projection head for embedding extraction

### "Have you tried other depression datasets?"
- Currently focused on NHANES for direct comparison
- Our system uses Apple Health data for mood prediction
- Would be interested in collaborating on other datasets

---

## Technical Details for Reproducibility

### Exact Software Versions
```
python==3.12.0
pytorch==2.1.0+cu121
numpy==1.26.4
scikit-learn==1.3.2
pandas==2.1.4
```

### Hardware Specifications
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CUDA: 12.1
- Training time: ~3 minutes per epoch
- Peak memory: 4.2GB

### Data Processing Checksums
```python
# After StandardScaler normalization
X_train.shape: (2478, 10080)
X_val.shape: (599, 10080)
X_train.mean(): 0.000001
X_train.std(): 0.999998
```

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