# Ensemble Model Weight Configuration

As of 2025-07-19, the ensemble model weights have been moved from hardcoded values to configurable settings, allowing for A/B testing and easy adjustment without code changes.

## Configuration Options

The ensemble model combines predictions from XGBoost and PAT models using weighted averaging. The weights can be configured via:

1. **Environment Variables** (highest priority)
   ```bash
   export ENSEMBLE_XGBOOST_WEIGHT=0.65
   export ENSEMBLE_PAT_WEIGHT=0.35
   ```

2. **Settings File** (.env)
   ```env
   ENSEMBLE_XGBOOST_WEIGHT=0.65
   ENSEMBLE_PAT_WEIGHT=0.35
   ENSEMBLE_PAT_TIMEOUT=10.0
   ENSEMBLE_XGBOOST_TIMEOUT=5.0
   ```

3. **Default Values**
   - XGBoost weight: 0.6 (60%)
   - PAT weight: 0.4 (40%)
   - PAT timeout: 10.0 seconds
   - XGBoost timeout: 5.0 seconds

## Validation

The system validates that ensemble weights sum to 1.0 (±0.001 for floating point tolerance). Invalid configurations will raise a validation error on startup.

## Usage Examples

### Docker Deployment
```bash
docker run -e ENSEMBLE_XGBOOST_WEIGHT=0.7 -e ENSEMBLE_PAT_WEIGHT=0.3 big-mood-detector
```

### A/B Testing Different Weights
```python
# Test configuration 1: Favor XGBoost
os.environ["ENSEMBLE_XGBOOST_WEIGHT"] = "0.8"
os.environ["ENSEMBLE_PAT_WEIGHT"] = "0.2"

# Test configuration 2: Balanced
os.environ["ENSEMBLE_XGBOOST_WEIGHT"] = "0.5"
os.environ["ENSEMBLE_PAT_WEIGHT"] = "0.5"
```

### Programmatic Access
```python
from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import EnsembleConfig

# Load from settings
config = EnsembleConfig.from_settings()
print(f"XGBoost weight: {config.xgboost_weight}")
print(f"PAT weight: {config.pat_weight}")
```

## Implementation Details

- Settings defined in: `src/big_mood_detector/infrastructure/settings/config.py`
- Used by: `EnsembleOrchestrator` in `predict_mood_ensemble_use_case.py`
- Validation: Pydantic model validator ensures weights sum to 1.0

## Research Notes

The default 60/40 split (XGBoost/PAT) was determined through empirical testing and aligns with the clinical accuracy metrics:
- XGBoost provides strong baseline predictions with AUC 0.80-0.98
- PAT adds temporal pattern recognition for improved early detection

Future improvements could include:
- Dynamic weight adjustment based on data quality
- Per-condition weights (depression vs mania)
- Confidence-based weighting