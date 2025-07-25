# XGBoost Models

## Status
XGBoost models are part of the temporal ensemble system but are not included in the repository due to:
- File size (PKL files can be large)
- Security (pickle files can contain arbitrary code)
- Focus on PAT-Conv-L for initial MVP

## Models Needed
For the full temporal ensemble, we need:
- `depression_model.pkl` - Depression risk prediction
- `mania_model.pkl` - Mania risk prediction  
- `hypomania_model.pkl` - Hypomania risk prediction

## How to Obtain

### Option 1: Download Pre-trained (When Available)
```bash
# Future: Download from model registry
# curl -O https://models.bigmooddetector.com/xgboost/v1.0/models.tar.gz
# tar -xzf models.tar.gz -C model_weights/xgboost/pretrained/
```

### Option 2: Train Your Own
```python
from big_mood_detector.infrastructure.fine_tuning.population_trainer import PopulationTrainer

trainer = PopulationTrainer(task_name="depression")
trainer.train_xgboost(X_train, y_train)
trainer.save_model("model_weights/xgboost/production/depression_model.pkl")
```

## Integration Status

- ✅ PAT-Conv-L (0.5929 AUC) - Current state assessment
- ⏳ XGBoost - Future risk prediction (not required for MVP)
- 🔮 Temporal Ensemble - Combines both (future feature)

For MVP, we're using PAT-Conv-L only. XGBoost integration can be added later without breaking changes.