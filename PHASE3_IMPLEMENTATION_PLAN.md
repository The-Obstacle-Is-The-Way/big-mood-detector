# Phase 3 Implementation Plan: Fixing the Ensemble

## Problem Statement

The current "ensemble" is a lie. It just returns XGBoost predictions and ignores PAT completely. We need to fix this by implementing proper temporal separation.

## Current Code Analysis

### 1. The Fake Ensemble (`predict_mood_ensemble_use_case.py`)

```python
# Lines 203-207: This is NOT an ensemble!
ensemble_pred = xgboost_pred if xgboost_pred else MoodPrediction(
    depression_probability=0.33,
    hypomania_probability=0.33,
    mania_probability=0.34,
    confidence=0.0
)
```

**Problems:**
- Just returns XGBoost prediction
- PAT embeddings extracted but unused
- No actual ensemble logic
- Ignores temporal context

### 2. The Missing Pieces

```python
# What we have:
pat_embeddings = pat_encoder.encode(...)  # ✅ Works
xgboost_pred = xgboost_model.predict(...)  # ✅ Works

# What's missing:
pat_predictions = pat_predictor.predict(pat_embeddings)  # ❌ No predictor!
temporal_assessment = orchestrator.combine(...)  # ❌ No temporal logic!
```

### 3. The Incorrect Integration

Current flow:
```
Health Data → Features → [XGBoost → Prediction] → "Ensemble" → Result
                      ↘ [PAT → Embeddings → ???]
```

Should be:
```
Health Data → Features → [XGBoost → Future Risk]    → Temporal  → Result
                      ↘ [PAT → Current State]    ↗   Assessment
```

## Implementation Strategy

### Phase 3.1: Add Deprecation Warnings (Don't Break Anything)

```python
# predict_mood_ensemble_use_case.py
class EnsembleOrchestrator:
    def predict(self, ...):
        warnings.warn(
            "EnsembleOrchestrator is deprecated. Use TemporalEnsembleOrchestrator "
            "for proper temporal separation of current state (PAT) and future risk (XGBoost).",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing code continues working
```

### Phase 3.2: Implement Temporal Orchestrator

1. **Create the orchestrator** (following TDD tests already written)
2. **Wire up PAT predictions** using trained classification heads
3. **Keep temporal contexts separate** - no averaging!
4. **Add clinical decision logic** based on temporal patterns

### Phase 3.3: Integration Points

#### A. Pipeline Integration

```python
# mood_prediction_pipeline.py
class MoodPredictionPipeline:
    def __init__(self, use_temporal_ensemble=False):
        if use_temporal_ensemble:
            self.orchestrator = TemporalEnsembleOrchestrator(...)
        else:
            self.orchestrator = EnsembleOrchestrator(...)  # Legacy
```

#### B. API Integration

```python
# api/routes.py
@router.post("/api/v1/temporal-assessment")  # NEW
async def temporal_assessment(...):
    # Returns TemporalMoodAssessment

@router.post("/api/v1/predict")  # EXISTING - keep working!
async def predict(...):
    # Returns MoodPrediction (legacy format)
```

#### C. CLI Integration

```python
# cli/predict_command.py
@click.option('--temporal/--legacy', default=False, 
              help='Use temporal ensemble (NOW vs TOMORROW separation)')
def predict(temporal: bool):
    if temporal:
        # Use new temporal orchestrator
    else:
        # Use legacy ensemble
```

## Code Changes Required

### 1. Fix Ensemble Logic

**FROM:**
```python
# Just returns XGBoost
ensemble_pred = xgboost_pred if xgboost_pred else default
```

**TO:**
```python
# Temporal separation
current_state = self._assess_current_state(pat_embeddings)
future_risk = self._predict_future_risk(xgboost_features)
return TemporalMoodAssessment(
    current_state=current_state,
    future_risk=future_risk,
    assessment_timestamp=datetime.now(),
    user_id=user_id
)
```

### 2. Enable PAT Predictions

**ADD:**
```python
class TemporalEnsembleOrchestrator:
    def _assess_current_state(self, pat_embeddings):
        # Use trained PAT heads
        depression_prob = self.pat_depression_head(pat_embeddings)
        medication_prob = self.pat_medication_head(pat_embeddings)
        
        return CurrentMoodState(
            depression_probability=depression_prob,
            is_depressed=depression_prob > 0.5,
            medication_proxy_score=medication_prob,
            confidence=self._calculate_confidence(pat_embeddings),
            assessment_reason="Based on last 7 days of activity"
        )
```

### 3. Update Response Format

**Legacy (keep working):**
```json
{
    "prediction": {
        "depression_probability": 0.3,
        "hypomania_probability": 0.6,
        "mania_probability": 0.1
    }
}
```

**New Temporal:**
```json
{
    "temporal_assessment": {
        "current_state": {
            "depression_probability": 0.7,
            "is_depressed": true,
            "assessment_window": "past_7_days"
        },
        "future_risk": {
            "depression_probability": 0.3,
            "hypomania_probability": 0.6,
            "mania_probability": 0.1,
            "prediction_window": "next_24_hours"
        }
    }
}
```

## Testing Strategy

### 1. Unit Tests
- [x] `test_temporal_ensemble_orchestrator.py` - Already written!
- [ ] Add tests for deprecation warnings
- [ ] Add tests for feature flags

### 2. Integration Tests
- [ ] Test pipeline with both orchestrators
- [ ] Test API with both endpoints
- [ ] Test CLI with both flags

### 3. E2E Tests
- [ ] Full flow with temporal ensemble
- [ ] Backward compatibility with legacy

## Rollout Plan

### Stage 1: Silent Release (Week 1)
- Deploy with feature flag OFF
- Log temporal predictions without returning
- Compare accuracy

### Stage 2: Beta (Week 2)
- Enable for 10% of users
- Monitor for issues
- Gather feedback

### Stage 3: General Availability (Week 3)
- Enable for all users
- Keep legacy endpoint active
- Update documentation

### Stage 4: Deprecation (Month 2)
- Mark legacy as deprecated
- Plan sunset date
- Migrate remaining users

## Key Files to Change

1. **Deprecate**: `predict_mood_ensemble_use_case.py`
2. **Create**: `temporal_ensemble_orchestrator.py`
3. **Update**: `mood_prediction_pipeline.py`
4. **Update**: `api/routes.py`
5. **Update**: `cli/predict_command.py`
6. **Create**: `train_pat_heads.py`

## Success Metrics

1. **No Regressions**: Legacy API continues working
2. **Temporal Accuracy**: Clear NOW vs TOMORROW distinction
3. **Performance**: < 50ms additional latency
4. **Adoption**: 80% users prefer temporal view
5. **Clinical Value**: Reduced false alarms from temporal context

## Next Action

Start with implementing the `TemporalEnsembleOrchestrator` since we already have tests written. This follows TDD and ensures we build exactly what's needed.