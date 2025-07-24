# PAT (Pre-trained Actigraphy Transformer) Feature Analysis
Generated: 2025-07-23

## What is PAT?

PAT is a transformer-based foundation model specifically designed for wearable movement data (actigraphy). It's pre-trained on data from 29,307 participants from NHANES datasets using masked autoencoding.

## PAT Input Requirements

### Raw Data Format
- **Input**: 7 consecutive days of minute-level activity data
- **Shape**: 10,080 values (7 days × 24 hours × 60 minutes)
- **Type**: Step counts per minute from Apple Health
- **No aggregation**: Uses raw minute-level data, not statistical summaries

### Data Flow
```
Apple Health Activity Records (STEP_COUNT)
    ↓
ActivitySequenceExtractor (domain/services/)
    ↓
Daily sequences [1440 values per day]
    ↓
PATSequenceBuilder (combines 7 days)
    ↓
PATSequence object (10,080 values)
    ↓
PAT Transformer Model
    ↓
96-dimensional embedding vector
```

## PAT Architecture Details

### Model Variants
1. **PAT-S (Small)**: 285K parameters, patch size 18
2. **PAT-M (Medium)**: 1M parameters, patch size 18
3. **PAT-L (Large)**: 1.99M parameters, patch size 9

### Patching Strategy
- Breaks 10,080 minutes into patches to handle long sequences
- PAT-S/M: 560 patches (18 minutes each)
- PAT-L: 1,120 patches (9 minutes each)
- Uses positional embeddings to maintain temporal information

### Processing Steps
1. **Normalization**: Z-score normalization of activity values
2. **Patching**: Divide into fixed-size patches
3. **Embedding**: Linear projection to embedding dimension
4. **Transformer**: Multi-head self-attention layers
5. **Pooling**: Average over sequence dimension
6. **Output**: 96-dimensional feature vector

## How PAT Differs from XGBoost

### Temporal Windows
- **XGBoost**: 30-day statistical summaries
- **PAT**: 7-day raw sequences

### Feature Types
- **XGBoost**: Hand-crafted features (sleep %, circadian phase, etc.)
- **PAT**: Learned representations from pre-training

### Input Granularity
- **XGBoost**: Daily aggregates → statistics
- **PAT**: Minute-level raw data

### Prediction Approach
- **XGBoost**: Rule-based on clinical features
- **PAT**: Pattern recognition from population data

## PAT Integration in Ensemble

### Ensemble Prediction Flow
```python
# 1. Build PAT sequence
sequence = pat_builder.build_sequence(
    activity_records, 
    end_date=target_date
)

# 2. Extract embeddings
pat_embeddings = pat_model.extract_features(sequence)  # 96 dims

# 3. Enhance with statistical features
enhanced = concat([stats[:20], pat_embeddings[:16]])  # 36 total

# 4. Predict with XGBoost on enhanced features
pat_prediction = xgboost.predict(enhanced)

# 5. Combine with regular XGBoost prediction
final = 0.6 * xgboost_pred + 0.4 * pat_pred
```

### Why This Works
- PAT captures complex temporal patterns
- XGBoost provides interpretable clinical rules
- Combination leverages both approaches

## PAT's Clinical Value

### What PAT Learns
From the paper's attention visualizations:
- Late wake times (noon) for benzodiazepine users
- Early morning activity patterns
- Weekly behavioral cycles
- Disrupted circadian rhythms

### Advantages
1. **No feature engineering**: Learns directly from data
2. **Long-range dependencies**: Can relate Tuesday night to Thursday morning
3. **Population knowledge**: Pre-trained on 29K participants
4. **Minimal data needs**: Works with just 7 days

### Clinical Performance
From the PAT paper:
- Benzodiazepine prediction: AUC 0.767
- SSRI prediction: AUC 0.700
- Sleep disorders: AUC 0.632
- Depression: AUC 0.610

## Key Implementation Files

1. **Model**: `infrastructure/ml_models/pat_model.py`
   - Model loading and inference
   - Feature extraction logic

2. **Sequence Building**: `domain/services/pat_sequence_builder.py`
   - Combines daily sequences
   - Handles missing data

3. **Activity Extraction**: `domain/services/activity_sequence_extractor.py`
   - Converts records to minute arrays
   - Calculates PAT (Principal Activity Time)

4. **Ensemble**: `application/use_cases/predict_mood_ensemble_use_case.py`
   - Combines PAT and XGBoost predictions
   - Handles timeouts and failures

## Critical Points for Our Fix

1. **PAT is independent**: Doesn't use `ClinicalFeatureExtractor` or `AggregationPipeline`
2. **Direct from records**: Works directly with `ActivityRecord` objects
3. **No feature mapping needed**: PAT handles its own data transformation
4. **Already working**: Current implementation is correct

## Validation Checklist

- [x] PAT loads pre-trained weights correctly
- [x] Activity sequences extracted properly (10,080 values)
- [x] Model produces 96-dim embeddings
- [x] Ensemble combines predictions correctly
- [ ] Performance matches paper benchmarks (need more data)