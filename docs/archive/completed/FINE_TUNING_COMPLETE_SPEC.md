# Complete Fine-Tuning Specification & Implementation Plan

## 🎯 Overview

We have everything needed to implement a two-stage fine-tuning pipeline:
1. **Population fine-tuning** using NHANES data (we have locally)
2. **Personal calibration** using individual health exports + episode labels

## 📁 Available Resources

### Pre-trained Models
- **XGBoost**: `mood_ml/XGBoost_{DE,HME,ME}.pkl` (ready to use)
- **PAT weights**: `PAT-{S,M,L}_29k_weights.h5` (needs task heads)

### Data
- **NHANES**: Local XPT files with PHQ-9, medications, actigraphy
- **Features**: 36 sleep/circadian features (from mood_ml)

### Libraries
- **PEFT**: HuggingFace parameter-efficient fine-tuning (LoRA)
- **Reference code**: mood_ml notebooks, sleepfm-clinical pipelines

## 🏗️ Implementation Architecture

```
src/big_mood_detector/
├── infrastructure/
│   ├── fine_tuning/
│   │   ├── __init__.py
│   │   ├── nhanes_processor.py      # XPT → Parquet with labels
│   │   ├── population_trainer.py    # Task-specific fine-tuning
│   │   ├── personal_calibrator.py   # User-level adaptation
│   │   └── model_registry.py        # Model version management
│   └── ml/
│       ├── pat_model.py             # PAT with PEFT adapters
│       └── xgboost_incremental.py   # XGBoost continuous learning
├── domain/
│   └── services/
│       └── fine_tuning_service.py   # Orchestration logic
└── interfaces/
    └── cli/
        └── fine_tune_commands.py    # CLI for labeling & training
```

## 🔄 Data Flow

### Stage 1: Population Fine-Tuning (One-time, ships with app)
```python
# 1. Process NHANES
nhanes_processor = NHANESProcessor()
cohort_df = nhanes_processor.process_cohort(
    actigraphy='PAXHD_H.xpt',
    depression='DPQ_H.xpt', 
    medications=['RXQ_RX_H.xpt', 'RXQ_DRUG.xpt']
)

# 2. Extract features (36 sleep/circadian)
features = extract_mood_ml_features(cohort_df)  # Matches mood_ml

# 3. Fine-tune models
pat_trainer = PATPopulationTrainer()
pat_trainer.fine_tune(features, labels=cohort_df['PHQ9>=10'])
pat_trainer.save('models/population/pat_depression.pt')

xgb_model = joblib.load('mood_ml/XGBoost_DE.pkl')
# Already fine-tuned! Just ship as-is
```

### Stage 2: Personal Calibration (Per-user, on-device)
```python
# 1. Baseline extraction
baseline = BaselineExtractor()
baseline.process_apple_export('export.zip')
user_features = baseline.get_features()  # Same 36 features

# 2. Episode labeling
labels = EpisodeLabeler()
labels.add_episode('2024-03-15', 'hypomanic', severity=3)
labels.add_baseline('2024-01-01', '2024-02-01')

# 3. Personal adaptation
# PAT: Small LoRA adapter
pat_calibrator = PersonalPATCalibrator(
    base_model='models/population/pat_depression.pt',
    peft_config=LoraConfig(r=8, target_modules=['query', 'value'])
)
pat_calibrator.fit(user_features, labels)
pat_calibrator.save(f'models/users/{user_id}/pat_adapter.pt')

# XGBoost: Incremental boosting
xgb_calibrator = PersonalXGBoostCalibrator(
    base_model='mood_ml/XGBoost_DE.pkl'
)
xgb_calibrator.incremental_fit(user_features, labels, sample_weight=2.0)
xgb_calibrator.save(f'models/users/{user_id}/xgb_personal.pkl')
```

## 🛠️ TDD Implementation Plan

### Week 1: NHANES Processing & Population Fine-Tuning
```bash
# Day 1-2: NHANES data pipeline
pytest tests/unit/infrastructure/fine_tuning/test_nhanes_processor.py -v
- test_load_xpt_files
- test_merge_cohorts
- test_label_extraction
- test_feature_engineering

# Day 3-4: Population training
pytest tests/unit/infrastructure/fine_tuning/test_population_trainer.py -v
- test_pat_fine_tuning
- test_model_serialization
- test_validation_metrics

# Day 5: Integration
pytest tests/integration/test_population_pipeline.py -v
- test_end_to_end_nhanes_training
```

### Week 2: Personal Calibration Pipeline
```bash
# Day 1-2: Baseline extraction
pytest tests/unit/domain/services/test_baseline_extractor.py -v
- test_sleep_baseline_calculation
- test_circadian_baseline
- test_deviation_features

# Day 3-4: Personal adaptation
pytest tests/unit/infrastructure/fine_tuning/test_personal_calibrator.py -v
- test_pat_lora_adapter
- test_xgboost_incremental
- test_calibration_metrics

# Day 5: CLI integration
pytest tests/unit/interfaces/cli/test_fine_tune_commands.py -v
- test_label_command
- test_calibrate_command
- test_predict_with_personal_model
```

### Week 3: Model Management & Inference
```bash
# Day 1-2: Model registry
pytest tests/unit/infrastructure/fine_tuning/test_model_registry.py -v
- test_model_versioning
- test_fallback_to_population
- test_model_updates

# Day 3-4: Ensemble predictions
pytest tests/unit/domain/services/test_ensemble_predictor.py -v
- test_pat_xgboost_ensemble
- test_confidence_calibration
- test_disagreement_handling

# Day 5: E2E validation
pytest tests/integration/test_personalized_predictions.py -v
- test_full_pipeline_with_labels
- test_performance_vs_baseline
```

## 📊 Performance Targets

### Population Models
- Training time: <2 hours on GPU (one-time)
- Model size: ~100MB (all task heads)
- AUC: 0.77-0.80 (matching paper)

### Personal Calibration
- Adaptation time: <5 minutes on CPU
- Adapter size: <1MB per user
- AUC improvement: +6-10pp (0.83-0.87)

### Inference
- Prediction time: <100ms per day
- Memory usage: <200MB total
- Battery impact: <1% daily

## 🔒 Privacy & Clinical Safety

1. **Data Protection**
   - All personal data stays on device
   - Models encrypted at rest
   - No cloud training required

2. **Clinical Validation**
   - Require minimum 30 days baseline
   - Confidence intervals on predictions
   - Clear uncertainty communication

3. **Fallback Strategy**
   - Population model if <30 days data
   - Conservative predictions when uncertain
   - Healthcare provider integration

## ✅ Success Criteria

1. **Technical**
   - [ ] NHANES pipeline processes 7k participants
   - [ ] Personal calibration <5min on M1 Mac
   - [ ] Model updates don't break predictions

2. **Clinical**
   - [ ] Replicate paper AUC (0.77 population)
   - [ ] Achieve +6pp with personal data
   - [ ] <10% false positive rate

3. **User Experience**
   - [ ] Simple CLI for labeling
   - [ ] Clear improvement metrics
   - [ ] Transparent model behavior

## 🚀 Next Steps

1. Start with NHANES processor (TDD)
2. Implement population fine-tuning
3. Add personal calibration layer
4. Create CLI commands
5. Validate against paper metrics

We have all the pieces - let's build this!