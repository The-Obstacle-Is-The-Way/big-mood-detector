# Fine-Tuning Dossier: Personalized Mood Prediction Models

## Executive Summary

This document outlines the complete system architecture for implementing fine-tuning capabilities in the Big Mood Detector. The core challenge is that pre-trained models (PAT and XGBoost) require personalized calibration with labeled mood episodes to provide accurate predictions for individual users.

## Core Problem Statement

Pre-trained models suffer from several limitations:
1. **Population-based training**: Models trained on aggregate data miss individual variations
2. **Lack of episode labels**: Without knowing when a user experienced mood episodes, models cannot calibrate to individual patterns
3. **Individual variability**: Circadian rhythms, activity patterns, and sleep needs vary significantly between individuals
4. **Clinical accuracy requirements**: False positives/negatives have serious implications for mental health monitoring

## System Components Overview

### 1. Data Collection Pipeline

#### 1.1 Baseline Data Requirements
- **Minimum duration**: 30-60 days of continuous data
- **Data types required**:
  - Sleep records (duration, stages, efficiency)
  - Activity sequences (minute-level, 1440 points/day)
  - Heart rate patterns
  - Circadian markers (light exposure if available)

#### 1.2 Episode Labeling Sources
- **Clinical records**: Hospital/psychiatrist notes
- **Self-reported**: User annotations via CLI/UI
- **Healthcare notes**: Uploaded clinical documentation
- **Automated detection**: Anomaly detection for potential episodes

### 2. Labeling Infrastructure

#### 2.1 CLI Annotation Tool
```bash
mood-detector label --date 2024-03-15 --episode "hypomanic" --severity 3 --notes "Decreased sleep, increased energy"
mood-detector label --date-range 2024-03-10:2024-03-17 --episode "depressive" --severity 4
mood-detector label --baseline --date-range 2024-01-01:2024-02-01
```

#### 2.2 Healthcare Document Parser
- **Supported formats**: PDF, TXT, DOCX, HL7 CDA
- **NLP extraction**: Dates, episode types, severities, medications
- **Validation**: Cross-reference with sensor data availability

#### 2.3 Web Interface
- **Timeline visualization**: Interactive calendar with sensor data overlay
- **Episode marking**: Click-and-drag to mark episode periods
- **Severity scales**: DSM-5 aligned rating scales
- **Note attachment**: Clinical notes and context

### 3. Feature Engineering for Personalization

#### 3.1 Individual Baseline Features
- **Personalized circadian phase**: Individual DLMO estimation
- **Activity patterns**: Personal activity distribution
- **Sleep architecture**: Individual sleep stage percentages
- **Heart rate zones**: Personalized HR thresholds

#### 3.2 Deviation Features
- **Sleep deviation**: Current vs personal baseline
- **Activity anomalies**: Unusual patterns for individual
- **Circadian disruption**: Phase shifts from baseline
- **Social rhythm metrics**: Routine consistency

### 4. Model Architecture

#### 4.1 PAT Fine-Tuning
- **Architecture**: Frozen backbone + personalized head
- **Training data**: 60-minute activity sequences with labels
- **Output**: Episode probability time series
- **Transfer learning**: Keep pre-trained weights, train final layers

#### 4.2 XGBoost Incremental Learning
- **Approach**: Boosting from pre-trained model
- **Features**: 36 engineered features + personal deviations
- **Calibration**: Platt scaling for probability adjustment
- **Update frequency**: Weekly with new labeled data

#### 4.3 Ensemble Strategy
- **Model weighting**: Based on individual performance
- **Confidence calibration**: Uncertainty quantification
- **Disagreement handling**: Conservative predictions when models disagree

### 5. Privacy & Security

#### 5.1 Data Protection
- **Local processing**: All fine-tuning on user device
- **Encryption**: AES-256 for stored models and labels
- **Model isolation**: Separate model files per user
- **Audit logging**: Track all labeling and training events

#### 5.2 Clinical Data Handling
- **HIPAA compliance**: Encrypted storage and transmission
- **Access control**: Role-based permissions
- **Data retention**: Configurable retention policies
- **Export capabilities**: Clinical-grade reports

### 6. Validation & Monitoring

#### 6.1 Cross-Validation Strategy
- **Time-series splits**: Respect temporal ordering
- **Episode-stratified**: Ensure episodes in train/test
- **Baseline periods**: Separate validation on stable periods

#### 6.2 Performance Metrics
- **Sensitivity/Specificity**: For clinical thresholds
- **Episode prediction accuracy**: Days before episode
- **False positive rate**: Critical for user trust
- **Calibration plots**: Probability reliability

#### 6.3 Drift Detection
- **Feature drift**: Monitor feature distributions
- **Prediction drift**: Track prediction patterns
- **Retraining triggers**: Automated based on performance

### 7. User Experience

#### 7.1 Onboarding Flow
1. Initial data collection (30 days)
2. Baseline establishment wizard
3. Historical episode labeling
4. Model training and validation
5. Personalized insights activation

#### 7.2 Continuous Improvement
- **Feedback loops**: Easy episode confirmation/rejection
- **Model updates**: Transparent retraining notifications
- **Performance dashboards**: Show model accuracy over time

### 8. Technical Infrastructure

#### 8.1 Storage Requirements
- **Model storage**: ~50MB per user (PAT + XGBoost + metadata)
- **Label database**: SQLite with encrypted fields
- **Feature cache**: 1 year rolling window (~100MB)

#### 8.2 Compute Requirements
- **Initial training**: 10-15 minutes on modern CPU
- **Incremental updates**: 1-2 minutes weekly
- **Inference**: <100ms for daily predictions

#### 8.3 Integration Points
- **Background scheduler**: Weekly model updates
- **API endpoints**: Label submission, model status
- **Export formats**: FHIR, CSV, clinical reports

## Implementation Phases

### Phase 1: MVP (Weeks 1-4)
- CLI labeling tool
- Basic PAT fine-tuning
- Local model storage
- Validation metrics

### Phase 2: Clinical Integration (Weeks 5-8)
- Healthcare document parser
- XGBoost incremental learning
- Ensemble predictions
- Clinical report generation

### Phase 3: User Experience (Weeks 9-12)
- Web labeling interface
- Automated retraining
- Performance monitoring
- Feedback mechanisms

### Phase 4: Scale & Optimize (Weeks 13-16)
- Multi-user support
- Cloud sync options
- Advanced privacy features
- Clinical trial readiness

## Success Criteria

1. **Clinical accuracy**: >85% sensitivity, >90% specificity
2. **Early detection**: 3-7 days before clinical presentation
3. **User engagement**: >80% weekly labeling compliance
4. **Model stability**: <5% performance degradation over 6 months
5. **Processing efficiency**: <15min training, <100ms inference

## Risk Mitigation

1. **Insufficient labels**: Implement active learning to request labels for uncertain periods
2. **Model overfitting**: Regularization and ensemble methods
3. **Privacy concerns**: Local-first architecture with optional sync
4. **Clinical liability**: Clear disclaimers and healthcare provider integration
5. **Technical complexity**: Phased rollout with extensive testing

## Conclusion

This fine-tuning system transforms population-based models into personalized mood prediction tools. By combining multiple labeling sources, advanced feature engineering, and careful model adaptation, we can achieve clinical-grade accuracy while maintaining user privacy and system performance.