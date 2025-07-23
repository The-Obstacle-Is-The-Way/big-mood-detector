# Phase 2 Comprehensive Implementation Plan

**Updated**: July 23, 2025
**Based on**: First principles analysis of PAT paper, XGBoost Seoul features, and existing infrastructure

## 🎯 Executive Summary

We're implementing PAT classification heads for mood prediction, building on:
1. **PAT Pre-training**: Already done on 20k+ NHANES participants (we have the weights)
2. **XGBoost Success**: 36 Seoul features achieving AUC 0.80-0.98 for mood episodes
3. **Our Infrastructure**: Fine-tuning code exists but needs adaptation

## 📊 Key Insights from Literature Review

### PAT Paper Findings
- **Best Configuration**: PAT-L with 90% mask ratio, MSE on all timesteps
- **Performance**: 8.8% improvement over baselines with only 500 training samples
- **Architecture**: Vision Transformer approach with patching (18-min patches optimal)
- **Training Time**: <6 hours on Google Colab Pro ($10/month)
- **Depression Detection**: AUC 0.610 (PAT-L) vs 0.586 (CNN-3D)

### XGBoost Seoul Features (Our Current Production)
- **36 Features**: Sleep, activity, circadian, and statistical measures
- **Performance**: Depression AUC 0.80, Mania AUC 0.98, Hypomania AUC 0.95
- **Limitation**: Requires feature engineering, misses temporal patterns

### Gap Analysis
1. **We Have**: PAT embeddings (96-dim), XGBoost predictions
2. **We Need**: Classification heads to map embeddings → mood predictions
3. **Advantage**: PAT captures long-range temporal dependencies XGBoost misses

## 🏗️ Architecture Decisions

### 1. Reuse Existing Infrastructure
```python
# We already have:
src/big_mood_detector/infrastructure/
├── fine_tuning/
│   ├── nhanes_processor.py      # NHANES data loading
│   ├── personal_calibrator.py   # Individual adaptation
│   └── population_trainer.py    # Training pipeline
└── ml_models/
    └── pat_model.py             # PAT wrapper (embeddings only)
```

### 2. Classification Head Design
Based on PAT paper's approach:
```python
class PATClassificationHead(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=256, num_classes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, embeddings):
        return self.mlp(embeddings)
```

### 3. Training Data Strategy
The PAT paper used NHANES labels for:
- Depression (PHQ-9 scores)
- Sleep disorders
- SSRI/Benzodiazepine usage

We'll map these to our 3-class problem:
- **Normal**: PHQ-9 < 5, no medications
- **Depression**: PHQ-9 ≥ 10 or antidepressants
- **Mania/Hypomania**: Benzodiazepines or mood stabilizers (proxy)

## 📋 Implementation Plan (Avoiding Redundancy)

### Phase 2.1: Adapt Existing NHANES Processor (2 hours)
**File**: `infrastructure/fine_tuning/nhanes_processor.py`
- [x] Already loads NHANES XPT files
- [x] Already extracts medication usage
- [ ] Add PHQ-9 depression score extraction
- [ ] Add activity data extraction for PAT sequences
- [ ] Map to our 3-class labels

### Phase 2.2: Create Classification Head Module (2 hours)
**New File**: `infrastructure/ml_models/pat_classification_head.py`
```python
# Separate from pat_model.py to maintain single responsibility
# Uses PyTorch to match PAT's TensorFlow models
# Includes confidence calibration
```

### Phase 2.3: Extend PAT Model (1 hour)
**Update**: `infrastructure/ml_models/pat_model.py`
```python
def predict_mood(self, sequence: PATSequence) -> MoodPrediction:
    """New method using classification heads"""
    embeddings = self.extract_features(sequence)
    logits = self.classification_head(embeddings)
    return self._logits_to_mood_prediction(logits)
```

### Phase 2.4: Training Pipeline (3 hours)
**Reuse**: `infrastructure/fine_tuning/population_trainer.py`
- Adapt for PAT instead of XGBoost
- Add early stopping based on validation AUC
- Implement confidence calibration
- Save trained heads to `model_weights/pat/`

### Phase 2.5: Integration Tests (2 hours)
- Test with existing ensemble orchestrator
- Verify predictions differ from XGBoost
- Check confidence scores are calibrated
- Performance benchmarks (should be <100ms)

## 🚨 Critical Considerations

### 1. Data Privacy & Ethics
- NHANES is public but check usage terms
- Don't include participant IDs in logs
- Document consent/IRB considerations

### 2. Model Versioning
```yaml
# model_weights/pat/metadata.yaml
version: "1.0.0"
pretrained_base: "pat_large_90mask"
fine_tuning_date: "2025-07-24"
training_samples: 2800
validation_auc:
  depression: 0.62
  mania: 0.58
  normal: 0.65
```

### 3. Backward Compatibility
- Keep `extract_features()` method unchanged
- Add feature flag for predictions:
```python
if settings.ENABLE_PAT_PREDICTIONS:
    result.pat_prediction = pat_model.predict_mood(sequence)
else:
    result.pat_prediction = None
```

### 4. Performance Optimization
- Cache PAT sequences per user/date
- Batch predictions when possible
- Consider quantization for edge deployment

## 📊 Success Metrics

1. **Technical**:
   - [ ] All existing tests pass
   - [ ] PAT predictions != XGBoost predictions
   - [ ] Inference time <100ms
   - [ ] Memory usage <500MB

2. **Clinical**:
   - [ ] Depression detection AUC >0.60
   - [ ] Ensemble AUC improves by >2%
   - [ ] Confidence calibration ECE <0.1

3. **Engineering**:
   - [ ] No code duplication
   - [ ] Follows SOLID principles
   - [ ] Comprehensive test coverage
   - [ ] Clear documentation

## 🔄 Iterative Improvements

### V1 (This Sprint)
- Basic classification heads
- Simple ensemble averaging
- Minimal viable confidence

### V2 (Next Sprint)
- Attention-based ensemble
- Temporal consistency constraints
- Personal calibration

### V3 (Future)
- Multi-task learning (sleep + mood)
- Uncertainty quantification
- Federated learning ready

## 📚 References to Keep Handy

1. **PAT Reference Implementation**: 
   `reference_repos/Pretrained-Actigraphy-Transformer/Fine-tuning/`

2. **Original Papers**:
   - PAT: `docs/literature/converted_markdown/pretrained-actigraphy-transformer/`
   - XGBoost: `docs/literature/converted_markdown/xgboost-mood/`

3. **Our Current Models**:
   - XGBoost: `model_weights/xgboost/`
   - PAT base: `model_weights/pat/`

## 🎬 Next Steps

1. **Review** this plan with the team
2. **Create** feature branch if not done
3. **Start** with TDD - write failing tests first
4. **Implement** incrementally with small PRs
5. **Validate** with clinical collaborators

---

*Remember: We're not just adding features, we're advancing mental health care through thoughtful engineering.*