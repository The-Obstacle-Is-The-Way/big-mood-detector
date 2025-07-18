# Model Integration Guide

This guide explains how the Big Mood Detector integrates pretrained models for mood prediction.

## Overview

The system uses two types of models in an ensemble approach:

1. **PAT (Pretrained Actigraphy Transformer)**: Deep learning models that extract learned representations from 7-day activity sequences
2. **XGBoost Models**: Gradient boosting models that predict mood episodes from 36 engineered features

## Model Architecture

```
Raw Data → Feature Extraction → Models → Predictions
    │              │                │          │
    │              │                │          └─> MoodPrediction
    │              │                │                 ├─ depression_risk
    │              │                │                 ├─ hypomanic_risk
    │              │                │                 ├─ manic_risk
    │              │                │                 └─ confidence
    │              │                │
    │              │                ├─> XGBoost Models (Required)
    │              │                │     ├─ depression_model.pkl
    │              │                │     ├─ hypomanic_model.pkl
    │              │                │     └─ manic_model.pkl
    │              │                │
    │              │                └─> PAT Models (Optional)
    │              │                      ├─ PAT-S (285K params)
    │              │                      ├─ PAT-M (1M params)
    │              │                      └─ PAT-L (1.99M params)
    │              │
    │              ├─> 36 Statistical Features
    │              │     ├─ Sleep features (18)
    │              │     └─ Circadian features (18)
    │              │
    │              └─> PAT Features (96-dim)
    │                    └─ Deep representations
    │
    └─> Apple Health XML / JSON
          ├─ Sleep records
          ├─ Activity records
          └─ Heart rate records
```

## Setting Up Models

### 1. Directory Structure

```bash
model_weights/
├── pat/
│   ├── pretrained/
│   │   ├── PAT-S_29k_weights.h5  # Small model
│   │   ├── PAT-M_29k_weights.h5  # Medium model
│   │   └── PAT-L_29k_weights.h5  # Large model
│   └── finetuned/
│       └── (your custom models)
└── xgboost/
    ├── pretrained/
    │   ├── depression_model.pkl
    │   ├── hypomanic_model.pkl
    │   └── manic_model.pkl
    └── finetuned/
        └── (your custom models)
```

### 2. Obtaining Models

#### PAT Models
```bash
# Copy from reference repository
cp reference_repos/Pretrained-Actigraphy-Transformer/model_weights/*.h5 \
   model_weights/pat/pretrained/

# Or download using script
python scripts/download_model_weights.py --model pat
```

#### XGBoost Models
```bash
# Copy from reference repository
cp reference_repos/mood_ml/XGBoost_DE.pkl model_weights/xgboost/pretrained/depression_model.pkl
cp reference_repos/mood_ml/XGBoost_HME.pkl model_weights/xgboost/pretrained/hypomanic_model.pkl
cp reference_repos/mood_ml/XGBoost_ME.pkl model_weights/xgboost/pretrained/manic_model.pkl
```

## Usage Examples

### Basic XGBoost Prediction

```python
from pathlib import Path
from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostMoodPredictor

# Load models
predictor = XGBoostMoodPredictor()
predictor.load_models(Path("model_weights/xgboost/pretrained"))

# Make prediction with 36 features
features = extract_features(health_data)  # Your feature extraction
prediction = predictor.predict(features)

print(f"Depression risk: {prediction.depression_risk:.1%}")
print(f"Hypomanic risk: {prediction.hypomanic_risk:.1%}")
print(f"Manic risk: {prediction.manic_risk:.1%}")
```

### PAT Feature Extraction

```python
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
from big_mood_detector.domain.services.pat_sequence_builder import PATSequenceBuilder

# Initialize PAT model
pat_model = PATModel(model_size="medium")
pat_model.load_pretrained_weights(Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5"))

# Build 7-day sequence
builder = PATSequenceBuilder()
sequence = builder.build_sequence(activity_records, end_date=date.today())

# Extract deep features
pat_features = pat_model.extract_features(sequence)
```

### Full Pipeline

```python
from big_mood_detector.application.mood_prediction_pipeline import MoodPredictionPipeline

# Process health data
pipeline = MoodPredictionPipeline()
features_df = pipeline.process_health_export(
    Path("apple_export/export.xml"),
    output_path=Path("output/features.csv"),
    start_date=date(2025, 1, 1),
    end_date=date(2025, 5, 31)
)

# Load predictor and make predictions
predictor = XGBoostMoodPredictor()
predictor.load_models(Path("model_weights/xgboost/pretrained"))

for date, features in features_df.iterrows():
    prediction = predictor.predict(features.values)
    print(f"{date}: {prediction.highest_risk_type} ({prediction.highest_risk_value:.1%})")
```

## Model Details

### PAT Models

Based on: "AI Foundation Models for Wearable Movement Data" (Ruan et al., 2024)

- **Input**: 10,080 minutes (7 days) of activity data
- **Architecture**: Transformer with patch embeddings
- **Pretraining**: Masked autoencoding on 29,307 NHANES participants
- **Output**: 96-dimensional feature vector

Model variants:
- **PAT-S**: 285K params, 18-min patches, 1 encoder layer
- **PAT-M**: 1M params, 18-min patches, 2 encoder layers
- **PAT-L**: 1.99M params, 9-min patches, 4 encoder layers

### XGBoost Models

Based on: "Accurately predicting mood episodes using wearable data" (Yun et al., 2022)

- **Input**: 36 engineered features
- **Architecture**: Gradient boosted trees
- **Training**: Seoul National University Hospital patients
- **Output**: Binary classification (episode probability)

Features (36 total):
1. Sleep percentage (mean, SD, Z-score)
2. Sleep amplitude (mean, SD, Z-score)
3. Long sleep windows (6 features)
4. Short sleep windows (6 features)
5. Circadian rhythm metrics (6 features)

## Performance Metrics

### PAT Models (from paper)
- Benzodiazepine detection: 0.767 AUC
- SSRI detection: 0.700 AUC
- Sleep disorder: 0.632 AUC
- Depression: 0.610 AUC

### XGBoost Models (from paper)
- Depression: 0.80-0.98 AUC
- Hypomanic: 0.75-0.92 AUC
- Manic: 0.78-0.95 AUC

## Ensemble Strategies

The system supports multiple ensemble approaches:

1. **Feature-level fusion**: Concatenate PAT features with statistical features
2. **Score-level fusion**: Weighted average of model predictions
3. **Confidence-based selection**: Use most confident model
4. **Hierarchical**: Use PAT for screening, XGBoost for diagnosis

Example ensemble:
```python
# Combine PAT and statistical features
combined_features = np.concatenate([pat_features, statistical_features])

# Or weighted prediction averaging
final_risk = 0.7 * xgboost_risk + 0.3 * pat_risk
```

## Troubleshooting

### Common Issues

1. **"Model not loaded" error**
   - Check model files exist in correct directory
   - Verify file permissions
   - Ensure correct file extensions (.h5 for PAT, .pkl for XGBoost)

2. **XGBoost version warnings**
   - Expected for older models
   - Models still work correctly
   - Can re-save with newer XGBoost if needed

3. **TensorFlow not found**
   - PAT requires TensorFlow
   - Install with: `pip install tensorflow>=2.10.0`

4. **Feature mismatch**
   - Ensure exactly 36 features for XGBoost
   - Check feature order matches expected names

## Next Steps

1. **Validate on your data**: Test predictions against known outcomes
2. **Fine-tune models**: Adapt to your specific population
3. **Implement ensemble**: Combine models for better performance
4. **Monitor performance**: Track prediction accuracy over time
5. **Clinical validation**: Work with clinicians to validate predictions

## References

1. Ruan et al. (2024). "AI Foundation Models for Wearable Movement Data in Mental Health Research"
2. Yun et al. (2022). "Accurately predicting mood episodes in mood disorder patients using wearable sleep and circadian rhythm features"
3. PAT Repository: https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer/
4. XGBoost Repository: https://github.com/mcqeen1207/mood_ml