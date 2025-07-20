# Fine-Tuning Implementation Strategy

## 📦 Reference Implementations Available

We have cloned critical repositories that provide ready-to-use components:

1. **mood_ml** - Pre-trained XGBoost models
   - `XGBoost_DE.pkl` - Depressive Episode (4.9MB)
   - `XGBoost_HME.pkl` - Hypomanic Episode (3.2MB)
   - `XGBoost_ME.pkl` - Manic Episode (1.6MB)

2. **Pretrained-Actigraphy-Transformer** - Official PAT implementation
   - NHANES data loaders
   - Fine-tuning notebooks
   - Model weights and architectures

3. **peft** - HuggingFace Parameter-Efficient Fine-Tuning
   - LoRA adapters for transformers
   - Memory-efficient personal calibration
   - Drop-in integration with PAT

4. **sleepfm-clinical** - Production training pipeline reference
   - PyTorch Lightning structure
   - Scalable data loaders
   - Clinical validation utilities

## 🚀 Two-Stage Fine-Tuning Pipeline

### Stage 1: Population Fine-Tuning (Task-Level)
**What**: Train task-specific heads on NHANES cohorts
**Why**: Transforms random weights → clinical recognition

```bash
# Depression detection head
big-mood ft-task \
  --pretrained weights/pat_base.pt \
  --dataset nhanes/PAT_depression.parquet \
  --label PHQ9>=10

# Benzodiazepine use detection
big-mood ft-task \
  --pretrained weights/pat_base.pt \
  --dataset nhanes/PAT_benzo.parquet \
  --label benzodiazepine

# SSRI use detection  
big-mood ft-task \
  --pretrained weights/pat_base.pt \
  --dataset nhanes/PAT_ssri.parquet \
  --label ssri_use
```

### Stage 2: Personal Calibration (User-Level)
**What**: Learn individual baseline + adapt to personal patterns
**Why**: Mania/bipolar shifts are relative to individual baseline

```bash
# Initial calibration with episode labels
big-mood calibrate \
  --episodes episodes.csv \
  --export apple_export.zip \
  --output models/user_123/

# Continuous prediction with personal model
big-mood predict \
  --export latest_export.zip \
  --model models/user_123/ \
  --baseline baseline.db
```

## 📊 NHANES Data Processing Pipeline

### 1. Convert XPT → Parquet with Labels
```python
# src/big_mood_detector/infrastructure/nhanes/processor.py
class NHANESProcessor:
    def process_cohort(self):
        # Load actigraphy data (PAXHD_H.xpt)
        actigraphy = pd.read_sas('nhanes/PAXHD_H.xpt')
        
        # Load depression scores (DPQ_H.xpt)
        depression = pd.read_sas('nhanes/DPQ_H.xpt')
        depression['PHQ9_total'] = depression[['DPQ010', 'DPQ020', ...]].sum(axis=1)
        depression['depressed'] = (depression['PHQ9_total'] >= 10).astype(int)
        
        # Load medications (RXQ_RX_H.xpt + RXQ_DRUG.xpt)
        medications = self.load_medications()
        medications['benzodiazepine'] = medications['drug_name'].apply(
            lambda x: self.is_benzodiazepine(x)
        )
        
        # Merge and create labeled dataset
        cohort = actigraphy.merge(depression, on='SEQN')
        cohort = cohort.merge(medications, on='SEQN')
        
        return cohort
```

### 2. Feature Engineering for PAT
```python
# src/big_mood_detector/domain/services/pat_features.py
class PATFeatureExtractor:
    def extract_sequences(self, actigraphy_df):
        """Convert NHANES 80Hz data to minute-level sequences"""
        # Aggregate to minute-level (1440 points/day)
        minute_activity = actigraphy_df.resample('1T').sum()
        
        # Create 60-minute sliding windows
        sequences = []
        for i in range(len(minute_activity) - 60):
            seq = minute_activity[i:i+60].values
            sequences.append(seq)
            
        return np.array(sequences)
```

## 🔬 Concrete Implementation from Reference Repos

### From mood_ml: XGBoost Feature Engineering
The 36 features used in the pre-trained models:
```python
# Sleep features (from example.csv)
- sleep_start, sleep_end, time_in_bed
- minutes_sleep, minutes_awake
- sleep_efficiency = minutes_sleep / time_in_bed

# Circadian features (calculated in Index_calculation.m)
- IS (Interdaily Stability)
- IV (Intradaily Variability)  
- RA (Relative Amplitude)
- L5, M10 (Least/Most active 5/10 hours)
- Sleep regularity index
- Social jet lag
```

### From PAT: Model Weights Available
- PAT-S: Small (PAT-S_29k_weights.h5)
- PAT-M: Medium (PAT-M_29k_weights.h5)
- PAT-L: Large (PAT-L_29k_weights.h5)

### From PEFT: Efficient Adaptation
```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA for PAT
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,  # Low rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Adapt attention layers only
)

# Wrap PAT with LoRA
model = get_peft_model(pat_model, peft_config)
# Only 0.1% of parameters are trainable!
```

## 🔧 Implementation Phases

### Phase 1: Population Fine-Tuning Infrastructure (Week 1)
- [ ] NHANES data loader (XPT → Parquet)
- [ ] Label extraction (PHQ-9, medications)
- [ ] PAT dataset class for NHANES
- [ ] Fine-tuning script with task heads
- [ ] Model weight storage/versioning

### Phase 2: Personal Calibration Pipeline (Week 2)
- [ ] Baseline extractor from Apple Health
- [ ] Episode labeling CLI interface
- [ ] Adapter layer implementation (LoRA/Linear)
- [ ] Personal model storage (SQLite + weights)
- [ ] Incremental learning support

### Phase 3: Inference Pipeline (Week 3)
- [ ] Model ensemble (PAT + XGBoost)
- [ ] Baseline deviation detection
- [ ] Confidence calibration
- [ ] Risk score generation
- [ ] Clinical report output

### Phase 4: Validation & Testing (Week 4)
- [ ] Cross-validation on NHANES holdout
- [ ] Synthetic personal data testing
- [ ] Performance benchmarking
- [ ] Edge case handling
- [ ] Documentation & examples

## 🎯 Key Implementation Details

### Model Architecture
```python
class PersonalizedPAT(nn.Module):
    def __init__(self, pretrained_path, num_tasks=3):
        super().__init__()
        # Frozen pre-trained backbone
        self.backbone = load_pat_backbone(pretrained_path)
        self.backbone.requires_grad_(False)
        
        # Task-specific heads (population fine-tuned)
        self.task_heads = nn.ModuleDict({
            'depression': nn.Linear(768, 1),
            'benzodiazepine': nn.Linear(768, 1),
            'ssri': nn.Linear(768, 1),
        })
        
        # Personal adapter (user-specific)
        self.adapter = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 768),
        )
        
    def forward(self, x, task='depression'):
        # Extract features
        features = self.backbone(x)
        
        # Apply personal adapter
        adapted = features + self.adapter(features)
        
        # Task-specific prediction
        return self.task_heads[task](adapted)
```

### Training Configuration
```yaml
# config/fine_tuning.yaml
population_training:
  batch_size: 256
  learning_rate: 1e-4
  epochs: 20
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  early_stopping_patience: 5

personal_calibration:
  batch_size: 32
  learning_rate: 1e-3
  epochs: 10
  adapter_dropout: 0.1
  min_labeled_days: 14
  baseline_window_days: 28
```

## 🔐 Privacy & Security

1. **Local-First Processing**
   - All personal fine-tuning on device
   - No raw health data leaves device
   - Optional encrypted cloud backup

2. **Model Isolation**
   ```
   models/
   ├── population/           # Shared task heads
   │   ├── pat_depression.pt
   │   ├── pat_benzo.pt
   │   └── pat_ssri.pt
   └── users/               # Personal adapters
       ├── user_123/
       │   ├── adapter.pt
       │   ├── baseline.db
       │   └── metadata.json
       └── user_456/
   ```

## 📈 Expected Performance

Based on PAT paper results:
- **Population fine-tuning**: 0.77-0.80 AUC (NHANES tasks)
- **+ Personal calibration**: 0.83-0.87 AUC (6-10pp improvement)
- **Ensemble with XGBoost**: 0.85-0.90 AUC

## 🚦 Success Metrics

1. **Technical**
   - Population training < 2 hours on GPU
   - Personal calibration < 5 minutes on CPU
   - Inference < 100ms per day

2. **Clinical**
   - Depression detection: >85% sensitivity
   - Medication adherence: >90% specificity
   - Episode prediction: 3-7 days early warning

## 🎉 Why This Is Amazing

1. **Exact replication**: We have the same NHANES data as the paper
2. **Clinical validity**: Published methodology with peer review
3. **Scalable approach**: Population models ship with app, personal models stay local
4. **Privacy-preserving**: No need for cloud training infrastructure
5. **Incremental updates**: Can improve with more labeled data over time

Let's build this beast! 🚀