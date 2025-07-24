# Phase 2 Comprehensive Implementation Plan

**Updated**: July 23, 2025 (REVISED after literature review)
**Based on**: Careful analysis of PAT paper capabilities and limitations

## 🚨 CRITICAL REVISION: PAT's Actual Capabilities

After careful literature review, PAT was trained on **5 binary tasks**, NOT 3-class mood prediction:
1. **Depression** (PHQ-9 ≥ 10): AUC 0.589
2. **Benzodiazepine usage**: AUC 0.767 (proxy for mood stabilization, NOT direct mania detection)
3. **SSRI usage**: AUC 0.700
4. **Sleep disorders**: AUC 0.632
5. **Sleep abnormalities**: AUC 0.665

**PAT CANNOT distinguish hypomania from mania** - it was never trained on this distinction.

## 🎯 Executive Summary (Revised with Temporal Insight)

We're implementing PAT classification heads with a critical temporal distinction:
1. **PAT**: Assesses CURRENT state (depression NOW based on past 7 days)
2. **XGBoost**: Predicts TOMORROW's risk (mood episode in next 24 hours)
3. **Ensemble**: Parallel temporal assessments, NOT averaged!

### The Clinical Game-Changer
- **PAT answers**: "Is this person depressed RIGHT NOW?"
- **XGBoost answers**: "Will this person have a mood episode TOMORROW?"
- **Together**: Complete temporal picture for intervention timing

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

## 🏗️ Architecture Decisions (Revised)

### 1. Binary Classification Heads - Matching PAT's Training
```python
# Create separate binary heads for each PAT task
class PATBinaryHead(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)  # Binary output
        )
    
    def forward(self, embeddings):
        return self.mlp(embeddings)
```

### 2. What We'll Actually Implement
- **Depression Head**: Predict PHQ-9 ≥ 10 (matches PAT training)
- **Medication Head**: Predict benzodiazepine usage (proxy signal only)
- **NOT implementing hypomania/mania heads** - PAT has no training for this

### 3. Updated Domain Model
```python
@dataclass
class MoodPrediction:
    # Required fields (XGBoost always provides these)
    depression_risk: float
    hypomanic_risk: float
    manic_risk: float
    confidence: float
    
    # Optional PAT-specific fields
    pat_depression_score: Optional[float] = None
    pat_medication_proxy: Optional[float] = None
```

### 4. Temporal Ensemble Strategy (NEW!)
```python
# NO AVERAGING! Keep temporal contexts separate
assessment = {
    "current_state": {
        "depression": pat_depression_now,  # Based on past 7 days
        "on_benzodiazepine": pat_benzo_now,
        "on_ssri": pat_ssri_now
    },
    "tomorrow_forecast": {
        "depression": xgb_depression_tomorrow,
        "hypomania": xgb_hypomania_tomorrow,
        "mania": xgb_mania_tomorrow
    }
}
```
- **Current vs Future**: Never mix these timeframes!
- **Clinical value**: Immediate intervention vs preventive action
- **Both needed**: A patient can be stable now but at risk tomorrow

## 📋 Implementation Plan (Revised for Binary Heads)

### Phase 2.1: NHANES Binary Labels (COMPLETED)
**File**: `infrastructure/fine_tuning/nhanes_processor.py`
- [x] PHQ-9 depression score extraction
- [x] Medication usage extraction
- [x] Create binary labels (not 3-class)
- [x] PAT sequence extraction

### Phase 2.2: Binary Classification Heads (IN PROGRESS)
**File**: `infrastructure/ml_models/pat_classification_head.py`
```python
# Two separate binary heads:
# 1. Depression head (PHQ-9 >= 10)
# 2. Benzodiazepine head (medication proxy)
# Use sigmoid activation, not softmax
# Binary cross-entropy loss, not categorical
```

### Phase 2.3: Update Domain Model (TODO)
**File**: `domain/services/mood_predictor.py`
- Add optional PAT-specific fields
- Keep XGBoost fields as required
- Document which model provides which prediction

### Phase 2.4: Fix PAT Predictor Interface (TODO)
**File**: `domain/services/pat_predictor.py`
```python
class PATPredictorInterface(ABC):
    @abstractmethod
    def predict_depression(self, embeddings: np.ndarray) -> float:
        """Binary depression prediction (0-1)"""
        
    @abstractmethod
    def predict_medication_proxy(self, embeddings: np.ndarray) -> float:
        """Binary benzodiazepine prediction (0-1)"""
```

### Phase 2.5: Update Ensemble Logic (TODO)
**File**: `application/use_cases/predict_mood_ensemble_use_case.py`
- Combine PAT depression score with XGBoost depression risk
- Leave hypomanic/manic risks to XGBoost alone
- Add medication proxy as supplementary information

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

## ⚠️ PAT Limitations (Critical to Document)

Based on the literature, PAT **CANNOT**:
1. **Distinguish hypomania from mania** - it was never trained on this
2. **Directly detect manic episodes** - benzodiazepine is only a proxy
3. **Provide 3-class mood predictions** - it only does binary classification
4. **Replace XGBoost for bipolar spectrum** - XGBoost has much better AUCs for mania/hypomania

PAT **CAN** help with:
1. **Depression detection** (though AUC 0.589 is modest)
2. **Medication usage patterns** (benzodiazepine AUC 0.767)
3. **Long-range temporal patterns** that XGBoost might miss
4. **Complementary signal** for ensemble methods

## 🎬 Next Steps

1. **Refactor our code** to binary heads (not 3-class)
2. **Update tests** to match binary outputs
3. **Document limitations** clearly in code and docs
4. **Retrain** on NHANES with proper binary labels
5. **Evaluate** ensemble performance vs XGBoost alone

---

*Remember: Clinical accuracy requires honest assessment of model capabilities. PAT adds value for depression detection but cannot replace XGBoost for the full bipolar spectrum.*