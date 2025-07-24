# Phase 2 TDD Plan: PAT Classification Heads

**Created**: July 23, 2025
**Status**: Ready to implement
**Approach**: Test-Driven Development (TDD)

## 🎯 Goal
Enable PAT to make independent mood predictions by training classification heads on NHANES data.

## 📋 TDD Implementation Order

### 1. PAT Prediction Interface (30 min)
**Test First**: `test_pat_predictor_interface.py`
```python
def test_pat_predictor_returns_mood_prediction():
    """PAT predictor should return standard MoodPrediction object"""
    predictor = PATPredictorInterface()
    embeddings = np.random.rand(96)
    prediction = predictor.predict_from_embeddings(embeddings)
    
    assert isinstance(prediction, MoodPrediction)
    assert 0 <= prediction.depression_risk <= 1
    assert 0 <= prediction.hypomanic_risk <= 1
    assert 0 <= prediction.manic_risk <= 1
```

**Implementation**: Create interface in `domain/services/pat_predictor.py`

### 2. NHANES Data Loader (45 min)
**Test First**: `test_nhanes_loader.py`
```python
def test_load_nhanes_actigraphy_data():
    """Should load and parse NHANES actigraphy files"""
    loader = NHANESLoader()
    data = loader.load_actigraphy("test_data/nhanes_sample.csv")
    
    assert len(data) > 0
    assert "activity_counts" in data.columns
    assert "timestamp" in data.columns

def test_load_nhanes_mood_labels():
    """Should load corresponding mood/depression labels"""
    loader = NHANESLoader()
    labels = loader.load_labels("test_data/nhanes_labels.csv")
    
    assert "participant_id" in labels.columns
    assert "phq9_score" in labels.columns  # Depression score
```

**Implementation**: `infrastructure/data_loaders/nhanes_loader.py`

### 3. Training Data Preparation (45 min)
**Test First**: `test_training_data_prep.py`
```python
def test_prepare_pat_training_data():
    """Convert NHANES data to PAT training format"""
    prep = PATTrainingDataPrep()
    sequences, labels = prep.prepare_data(nhanes_data, nhanes_labels)
    
    assert sequences.shape[1] == 1440  # Minutes per day
    assert len(sequences) == len(labels)
    assert all(label in ["normal", "depressed", "manic"] for label in labels)
```

### 4. Classification Head Model (1 hour)
**Test First**: `test_pat_classification_head.py`
```python
def test_classification_head_architecture():
    """Classification head should map 96-dim embeddings to 3 classes"""
    head = PATClassificationHead(input_dim=96, num_classes=3)
    
    embeddings = torch.rand(32, 96)  # Batch of 32
    logits = head(embeddings)
    
    assert logits.shape == (32, 3)
    
def test_classification_head_training():
    """Should be trainable with standard loss"""
    head = PATClassificationHead()
    optimizer = torch.optim.Adam(head.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Mock training step
    embeddings = torch.rand(32, 96)
    labels = torch.randint(0, 3, (32,))
    
    logits = head(embeddings)
    loss = criterion(logits, labels)
    loss.backward()
    
    assert loss.item() > 0
```

### 5. Training Pipeline (1.5 hours)
**Test First**: `test_pat_training_pipeline.py`
```python
def test_train_classification_heads():
    """End-to-end training pipeline"""
    pipeline = PATTrainingPipeline()
    
    # Load data
    train_data = pipeline.prepare_training_data("nhanes_path")
    
    # Train model
    trained_model = pipeline.train(
        train_data,
        epochs=2,  # Small for testing
        batch_size=32
    )
    
    # Verify model was trained
    assert trained_model.get_accuracy() > 0.3  # Better than random
    assert Path("model_weights/pat/classification_head.pth").exists()
```

### 6. Integration with Ensemble (1 hour)
**Test First**: `test_ensemble_with_pat_predictions.py`
```python
def test_ensemble_combines_independent_predictions():
    """Ensemble should combine XGBoost and PAT predictions"""
    orchestrator = EnsembleOrchestrator(
        xgboost_predictor=mock_xgboost,
        pat_model=trained_pat_with_heads
    )
    
    result = orchestrator.predict(
        statistical_features=seoul_features,
        activity_records=activity_data
    )
    
    assert result.xgboost_prediction is not None
    assert result.pat_prediction is not None  # No longer None!
    assert result.pat_prediction != result.xgboost_prediction
    assert "pat_prediction" in result.models_used
```

## 🛠️ Implementation Details

### Directory Structure
```
infrastructure/
├── data_loaders/
│   └── nhanes_loader.py
├── ml_models/
│   ├── pat_classification_head.py
│   └── pat_predictor_impl.py
└── training/
    ├── pat_training_pipeline.py
    └── training_config.yaml
```

### Key Decisions
1. Use PyTorch for classification heads (matches PAT)
2. Simple MLP architecture initially (can improve later)
3. Store trained weights separately from pre-trained PAT
4. Version control training configurations
5. Log all training metrics for reproducibility

## ⏱️ Time Estimate
- Total: ~5 hours of focused TDD
- Can be split across multiple sessions
- Each test should be written before implementation

## 🚨 Important Notes
1. NHANES data is public but check usage guidelines
2. Keep training code separate from inference
3. Maintain backward compatibility
4. Document confidence calculation methodology
5. Plan for A/B testing infrastructure

## ✅ Success Criteria
- [ ] All tests pass
- [ ] PAT makes independent predictions
- [ ] Ensemble combines both model outputs
- [ ] Performance benchmarks maintained
- [ ] CI/CD remains green