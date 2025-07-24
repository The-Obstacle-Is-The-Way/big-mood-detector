# Model Labeling Requirements for Big Mood Detector

This document clarifies the labeling requirements for the XGBoost and PAT (Pretrained Actigraphy Transformer) models used in Big Mood Detector, based on analysis of the source papers and codebase.

## Executive Summary

**CORRECTED Finding**: Only XGBoost works out-of-the-box for mood prediction. PAT requires additional work.

- **XGBoost** ✅: Pre-trained on clinical cohort data with mood episodes already labeled by psychiatrists. Full model weights included.
- **PAT** ⚠️: Only the encoder is available - outputs embeddings, NOT mood predictions. Classification heads are NOT included.
- **User Labels**: Required for PAT mood predictions, optional for XGBoost personalization

## XGBoost Model Details

### Source Paper Analysis
From "Accurately predicting mood episodes in mood disorder patients using wearable sleep and circadian rhythm features" (Lim et al., 2024):

#### Training Data
- **168 mood disorder patients** from clinical settings
- **108 patients (64%)** experienced mood episode recurrences during follow-up
- **235 total mood episodes**: 175 depressive, 39 hypomanic, 21 manic
- **587 days average clinical follow-up**, 267 days wearable data per patient

#### Pre-training Status
- **Models are already trained** on the clinical cohort
- **No user labeling required** - episodes were labeled by psychiatrists during clinical follow-up
- **Ready for immediate use** on new patient data

#### Prediction Capabilities
- **Next-day prediction** - predicts mood episodes for the following day
- **High accuracy**: AUC 0.80 (depression), 0.98 (mania), 0.95 (hypomania)
- **Key predictor**: Circadian phase Z-score (individual's deviation from their baseline)

#### Model Requirements
- **Input**: 60-day window of sleep-wake data (or 30-day minimum)
- **Features**: 36 sleep and circadian rhythm features derived from actigraphy
- **Personal baselines**: Automatically calibrated to individual patterns

## PAT Model Details

### Source Paper Analysis
From "AI Foundation Models for Wearable Movement Data in Mental Health Research" (Ruan et al., 2024):

#### Pre-training Data
- **29,307 participants** from NHANES (US national health survey)
- **Masked autoencoder pre-training** on unlabeled actigraphy data
- **Fine-tuned on labeled subsets** for specific tasks (depression, medication use, sleep disorders)

#### Pre-training Status
- **Foundation model approach** - encoder pre-trained on massive unlabeled dataset
- **Task-specific heads were fine-tuned** on labeled NHANES data **BUT NOT INCLUDED IN REPOSITORY**
- **User labeling IS REQUIRED** to train classification heads for mood predictions

#### Prediction Capabilities
- **Multiple mental health outcomes**: Depression (PHQ-9 ≥ 10), medication usage, sleep disorders
- **Works with small datasets**: Performs well even with 500 labeled samples
- **Transfer learning**: Can be fine-tuned for new tasks if needed

#### Model Requirements
- **Input**: One week (10,080 minutes) of actigraphy data
- **Lightweight**: <2M parameters, runs on Google Colab
- **Explainable**: Attention weights show which time periods influence predictions

## Labeling System in Big Mood Detector

### Available Episode Types
The system supports labeling these mood episode types:
- **Depressive episodes** (including major, minor, and brief)
- **Manic episodes**
- **Hypomanic episodes**
- **Mixed episodes** (both depressive and manic/hypomanic features)

### Label Command Usage
```bash
# Label a depressive episode
python -m big_mood_detector label episode \
  --date-range 2024-01-15:2024-01-30 \
  --mood depressive \
  --severity 7 \
  --rater-id clinician_001

# Mark baseline (stable) periods
python -m big_mood_detector label baseline \
  --start 2024-02-01 \
  --end 2024-03-01 \
  --notes "Patient reported feeling stable"
```

### When Labels Are Used

1. **Model Evaluation**: Compare predictions against ground truth
2. **Population Fine-tuning**: Adapt models to specific cohorts
3. **Personal Calibration**: Improve individual accuracy over time
4. **Research**: Build datasets for future model development

## Practical Implications

### For Immediate Use
Users can get XGBoost predictions immediately, but PAT requires additional work:

```bash
# XGBoost predictions work immediately
python -m big_mood_detector process path/to/health/export \
  --model xgboost \
  --predict \
  --report

# PAT only outputs embeddings (no mood predictions)
# Ensemble mode will fail without PAT classification heads
```

### For Enhanced Accuracy
While not required, labels can improve accuracy through:

1. **Personal calibration**: System learns individual patterns
2. **Population adaptation**: Fine-tune for specific demographics
3. **Validation**: Measure real-world performance

### Label Collection Strategy
If you choose to collect labels:

1. **Start minimal**: Label only clear episodes initially
2. **Focus on extremes**: Prioritize labeling severe episodes
3. **Include baselines**: Mark stable periods for contrast
4. **Iterate**: Add labels as patterns become clearer

## Technical Implementation

### How Models Use Labels

1. **XGBoost**: 
   - Pre-trained weights are loaded from `model_weights/`
   - Personal calibration updates feature baselines
   - Population fine-tuning retrains final layers

2. **PAT**:
   - Foundation model loaded from `model_weights/`
   - Task-specific heads can be fine-tuned
   - Transfer learning preserves general patterns

### Storage and Management
- Labels stored in SQLite database
- Episode repository pattern for clean architecture
- Import/export functionality for data sharing

## Conclusion

**IMPORTANT CORRECTION**: Only the XGBoost models provide immediate clinical utility without requiring labeled data. The PAT model requires additional work:

- **XGBoost** ✅: Ready for immediate use with full mood prediction capabilities
- **PAT** ⚠️: Only provides embeddings - requires either:
  1. Fine-tuning with ~500 labeled episodes (as shown in paper)
  2. Obtaining pre-trained classification heads from original authors
  3. Using embeddings for clustering/anomaly detection only

The labeling functionality is:
- **Optional** for XGBoost (improves personalization)
- **Required** for PAT mood predictions (unless you obtain authors' heads)
- **Valuable** for continuous improvement and validation

**Recommendation**: Start with XGBoost-only predictions while working to obtain PAT classification heads or collecting labels for fine-tuning.