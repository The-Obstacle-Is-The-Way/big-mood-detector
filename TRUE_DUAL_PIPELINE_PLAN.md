# True Dual Pipeline Implementation Plan
Generated: 2025-07-23  
Updated: 2025-07-23 (After GitHub issue analysis)

## Executive Summary

After analyzing GitHub issues (#25, #27, #40, #50) and documentation, the core problem is clear: we have a **pseudo-ensemble** where both "models" are actually XGBoost. PAT only provides embeddings as features, not predictions. This plan details how to implement a true dual pipeline with independent models making complementary predictions.

## The Current Problem

### What We Have (Pseudo-Ensemble)
```python
# Current "ensemble" - Both predictions use XGBoost!
xgboost_pred = xgboost.predict(seoul_features)  # Prediction 1
pat_enhanced_pred = xgboost.predict(seoul[:20] + pat_embeddings[:16])  # Prediction 2
final = 0.6 * xgboost_pred + 0.4 * pat_enhanced_pred
```

**This is NOT a true ensemble** - it's the same XGBoost model making predictions on two different feature sets!

### What We Need (True Dual Pipeline)
```python
# True ensemble - Two independent models
xgboost_pred = xgboost.predict(seoul_features)  # Model 1
pat_pred = pat_classifier.predict(pat_embeddings)  # Model 2 (needs training!)
final = weighted_average(xgboost_pred, pat_pred)
```

## Understanding the Vision

From the documentation and GitHub issues analysis:
1. **XGBoost**: Predicts tomorrow's risk using 30-day Seoul features (working)
2. **PAT**: Should assess current state using 7-day activity patterns (only embeddings work)
3. **Ensemble**: Should combine both perspectives (currently broken - both use XGBoost)

### Key GitHub Issues
- **#25**: Temporal window mismatch - XGBoost predicts 24hr ahead, PAT analyzes current
- **#27**: PAT not providing predictions - only embeddings, no classification heads
- **#40**: XGBoost predict_proba missing for JSON models
- **#50**: Performance optimization (mostly fixed with OptimizedAggregationPipeline)

## The Roadmap

### Phase 1: Unclog the Pipeline (1 day)
**Goal**: Separate concerns so we can develop independently

#### 1.1 Refactor EnsembleOrchestrator (CRITICAL CHANGE)

**Current Problem** (lines 214-260 in predict_mood_ensemble_use_case.py):
```python
def _predict_with_pat(self, statistical_features, activity_records, prediction_date):
    # PROBLEM: This still uses XGBoost!
    pat_features = self.pat_model.extract_features(sequence)
    enhanced_features = np.concatenate([statistical_features[:20], pat_features[:16]])
    return self.xgboost_predictor.predict(enhanced_features)  # XGBoost again!
```

**New Clean Separation**:
```python
class EnsembleOrchestrator:
    def _extract_pat_embeddings(self, activity_records, prediction_date):
        """Extract PAT embeddings only - no prediction yet."""
        sequence = self.pat_builder.build_sequence(activity_records, end_date=prediction_date)
        return self.pat_model.extract_features(sequence)
    
    def predict(self, statistical_features, activity_records, prediction_date):
        # Parallel execution
        xgboost_future = executor.submit(self._predict_xgboost, statistical_features)
        pat_future = executor.submit(self._extract_pat_embeddings, activity_records, prediction_date)
        
        # Get independent results
        xgboost_pred = xgboost_future.result()  # MoodPrediction
        pat_embeddings = pat_future.result()    # np.array(96,)
        
        # For now, PAT only returns embeddings
        return EnsemblePrediction(
            xgboost_prediction=xgboost_pred,
            pat_embeddings=pat_embeddings,
            pat_prediction=None,  # Not available yet
            ensemble_prediction=xgboost_pred  # Only XGBoost for now
        )
```

#### 1.2 Update Pipeline Config
```python
@dataclass
class PipelineConfig:
    use_xgboost: bool = True
    use_pat_embeddings: bool = True  # Renamed from include_pat_sequences
    use_pat_classifier: bool = False  # Future: when we have classification head
    ensemble_mode: str = "xgboost_only"  # Options: xgboost_only, pat_enhanced, true_ensemble
```

### Phase 2: Create PAT Training Infrastructure (1 week)

#### 2.1 NHANES Data Processor (DATA ALREADY DOWNLOADED!)

**Great news**: NHANES data files are already in `/Users/ray/Downloads/`!
```bash
# Move to correct location:
mv /Users/ray/Downloads/*.xpt data/nhanes/2013-2014/

# Files we have:
# - PAXHD_H.xpt     ✅ Physical Activity Monitor - Header
# - PAXMIN_H.xpt    ✅ Physical Activity Monitor - Minute data  
# - DPQ_H.xpt       ✅ Depression Questionnaire (PHQ-9)
# - RXQ_DRUG.xpt    ✅ Prescription Drug Information
# - RXQ_RX_H.xpt    ✅ Prescription Medications
```

```python
class NHANESProcessor:
    def prepare_training_data(self):
        # 1. Load actigraphy sequences
        sequences = self.load_pat_sequences()  # 7-day windows
        
        # 2. Load labels
        depression_scores = self.load_phq9_scores()
        medications = self.load_medications()
        
        # 3. Create binary labels
        labels = self.create_mood_labels(depression_scores, medications)
        
        # 4. Extract PAT embeddings
        embeddings = self.extract_pat_embeddings(sequences)
        
        return embeddings, labels
```

#### 2.2 PAT Classification Head
```python
class PATClassificationHead(tf.keras.Model):
    def __init__(self, input_dim=96, hidden_dim=64):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        
        # Three binary classifiers
        self.depression_head = tf.keras.layers.Dense(1, activation='sigmoid')
        self.hypomanic_head = tf.keras.layers.Dense(1, activation='sigmoid')
        self.manic_head = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, embeddings):
        x = self.dense1(embeddings)
        x = self.dropout(x)
        x = self.dense2(x)
        
        return {
            'depression': self.depression_head(x),
            'hypomanic': self.hypomanic_head(x),
            'manic': self.manic_head(x)
        }
```

#### 2.3 PAT Fine-Tuning Pipeline
```python
class PATFineTuner:
    def __init__(self, pat_model, classification_head):
        self.pat_model = pat_model
        self.classifier = classification_head
        
    def train(self, sequences, labels):
        # Option 1: Freeze PAT, train classifier only
        self.pat_model.trainable = False
        
        # Option 2: Fine-tune end-to-end (requires more data)
        # self.pat_model.trainable = True
        
        # Training loop
        for epoch in range(epochs):
            embeddings = self.pat_model.extract_features_batch(sequences)
            predictions = self.classifier(embeddings)
            loss = self.compute_loss(predictions, labels)
            # ... optimization step
```

### Phase 3: Integrate True Ensemble (3 days)

#### 3.1 Update PAT Model
```python
class PATModel:
    def __init__(self):
        self.encoder = None  # Existing PAT encoder
        self.classifier = None  # New classification head
        
    def predict_mood(self, sequence):
        # Extract embeddings
        embeddings = self.extract_features(sequence)
        
        # Classify
        if self.classifier is None:
            raise RuntimeError("No classifier loaded - call load_classifier() first")
            
        predictions = self.classifier(embeddings)
        
        return MoodPrediction(
            depression_risk=float(predictions['depression']),
            hypomanic_risk=float(predictions['hypomanic']),
            manic_risk=float(predictions['manic']),
            confidence=self._calculate_confidence(predictions)
        )
```

#### 3.2 Fix EnsembleOrchestrator
```python
def _predict_with_pat(self, activity_records, prediction_date):
    """True PAT prediction, not enhanced XGBoost."""
    # Build sequence
    sequence = self.pat_builder.build_sequence(activity_records, end_date=prediction_date)
    
    # Get PAT's own prediction
    pat_prediction = self.pat_model.predict_mood(sequence)
    
    return pat_prediction  # Independent prediction!
```

#### 3.3 Update Ensemble Logic
```python
def predict(self, ...):
    # Two independent predictions
    xgboost_pred = self._predict_xgboost(seoul_features)
    pat_pred = self._predict_with_pat(activity_records, date)
    
    # True ensemble
    if self.config.ensemble_mode == "true_ensemble":
        ensemble_pred = self._calculate_ensemble(xgboost_pred, pat_pred)
    else:
        ensemble_pred = xgboost_pred  # Fallback
    
    return EnsemblePrediction(
        xgboost_prediction=xgboost_pred,
        pat_prediction=pat_pred,  # Now available!
        ensemble_prediction=ensemble_pred,
        temporal_context={
            'xgboost': 'next_24_hours',
            'pat': 'current_state'
        }
    )
```

## Implementation Order

### Week 1: Foundation
1. **Day 1**: Refactor EnsembleOrchestrator to cleanly separate pipelines
2. **Day 2**: Create NHANES data processor and verify data loading
3. **Day 3-4**: Build PAT classification head architecture
4. **Day 5**: Implement training pipeline with frozen encoder

### Week 2: Training & Integration
1. **Day 1-2**: Train classification heads on NHANES data
2. **Day 3**: Evaluate performance, tune hyperparameters
3. **Day 4**: Integrate trained classifier into PAT model
4. **Day 5**: Update ensemble to use true dual predictions

### Week 3: Testing & Documentation
1. **Day 1-2**: Comprehensive testing of dual pipeline
2. **Day 3**: Update all documentation to reflect true ensemble
3. **Day 4**: Performance benchmarking
4. **Day 5**: Deployment preparation

## Success Criteria

### Phase 1 (Unclogging)
- [ ] EnsembleOrchestrator returns embeddings separately
- [ ] XGBoost predictions work independently
- [ ] PAT embeddings extracted without affecting XGBoost
- [ ] Clear separation of concerns

### Phase 2 (Training)
- [ ] NHANES data loads and processes correctly
- [ ] Classification head trains successfully
- [ ] Validation metrics comparable to paper
- [ ] Saved models can be loaded

### Phase 3 (Integration)
- [ ] PAT makes independent mood predictions
- [ ] True ensemble averages two models
- [ ] Temporal context clearly labeled
- [ ] All tests pass

## Key Decisions

### 1. Training Strategy
- **Option A**: Freeze encoder, train classifier only (faster, less data needed)
- **Option B**: Fine-tune end-to-end (better performance, needs more data)
- **Recommendation**: Start with A, move to B if needed

### 2. Label Generation
- Use PHQ-9 scores for depression (threshold ≥ 10)
- Use medication data for mania (mood stabilizers, antipsychotics)
- Consider NHANES sleep disorder questions

### 3. Ensemble Weights
- Start with 60/40 (XGBoost/PAT)
- Optimize based on validation performance
- Consider dynamic weighting based on data quality

### 4. Temporal Alignment (Critical from Issue #25)
- **XGBoost**: Keep as 24-hour forecast
- **PAT**: Train for current state assessment
- **Output**: Clearly label temporal windows in predictions

## Risks & Mitigations

### Risk 1: Insufficient Training Data
- **Mitigation**: Use data augmentation (time shifting, noise)
- **Mitigation**: Transfer learning from similar tasks

### Risk 2: PAT Performance Below XGBoost
- **Mitigation**: Keep XGBoost-only mode as fallback
- **Mitigation**: Use PAT for specific scenarios (e.g., rapid cycling)

### Risk 3: Breaking Changes
- **Mitigation**: Feature flag for ensemble modes
- **Mitigation**: Comprehensive testing before release

## Next Steps

1. **Today**: Review plan with team, get buy-in
2. **Tomorrow**: Start Phase 1 refactoring
3. **This Week**: Complete unclogging and data prep
4. **Next Week**: Begin training experiments

## Summary

The current "ensemble" is misleading - it's XGBoost predicting on two feature sets. After analyzing GitHub issues and documentation:

**What's Broken**:
1. PAT can't make predictions (Issue #27) - only provides embeddings
2. Temporal confusion (Issue #25) - mixing forecast vs current state  
3. "Ensemble" is actually XGBoost twice with different features
4. XGBoost predict_proba missing for JSON models (Issue #40)

**The Fix**:
1. Separate the pipelines cleanly (Phase 1)
2. Train PAT classification heads with NHANES data (Phase 2)
3. Implement true dual-model ensemble (Phase 3)

This will give us complementary perspectives:
- **XGBoost**: Tomorrow's risk based on 30-day patterns
- **PAT**: Current state based on 7-day activity

Together, they'll provide both immediate assessment and predictive warning.