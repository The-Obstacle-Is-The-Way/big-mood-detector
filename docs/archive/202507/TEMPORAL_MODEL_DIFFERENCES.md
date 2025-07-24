# Temporal Model Differences: XGBoost vs PAT

**Date:** 2025-07-20  
**Status:** CRITICAL - Impacts Clinical Decision Support  
**Priority:** HIGH

## Validated Temporal Differences from Literature

### XGBoost (Nature Digital Medicine 2024)
- **Prediction Window:** **24-hour forecast** (next-day prediction)
- **Input Window:** 30-60 days of historical data
- **Quote from paper:** "These features enabled accurate **next-day predictions** for depressive, manic, and hypomanic episodes"
- **Clinical Use:** Forward-looking early warning system

### PAT (Dartmouth 2024)
- **Prediction Window:** **Current state assessment** (not forecasting)
- **Input Window:** 7 days (one week) of minute-by-minute activity
- **Quote from paper:** "The input is actigraphy and the label indicates whether a participant is taking benzodiazepines"
- **Clinical Use:** Present-moment risk assessment based on recent behavior

## Critical Implications

### 1. **These Models Answer Different Questions**
```
XGBoost: "What is tomorrow's mood risk based on the past month?"
PAT:     "What is the current mood state based on the past week?"
```

### 2. **Clinical Decision Support Impact**
- **XGBoost**: Provides time for preventive interventions (24-hour lead time)
- **PAT**: Confirms current state, useful for immediate clinical decisions
- **Combined**: Could provide both current assessment AND future prediction

### 3. **Temporal Mismatch in v0.2.0**
Current "ensemble" mixes:
- Features from different time windows (30 days vs 7 days)
- Predictions with different temporal meanings
- This could confuse clinical interpretation

## Visual Representation

```
Timeline:
<-- 30 days --|-- 7 days --|-- Today --|-- Tomorrow -->

XGBoost:      [============ Input ============]
                                               └─> Predicts

PAT:                        [=== Input ===]
                                     └─> Assesses

Current v0.2.0: Mixes both inputs, only XGBoost predicts tomorrow
```

## Safety Concerns for CDS

### 1. **Misaligned Risk Windows**
- CDS might show "High depression risk" without clarifying if it's:
  - Tomorrow's risk (XGBoost)
  - Current state (PAT would show if working)
  - Some undefined mixture (current v0.2.0)

### 2. **Intervention Timing Confusion**
- 24-hour warning allows preventive action
- Current state assessment confirms need for immediate intervention
- Mixing these could lead to inappropriate clinical responses

### 3. **Label-Prediction Mismatch**
- XGBoost trained on "next-day mood episodes"
- PAT trained on "current medication use/depression status"
- Different ground truths = different clinical meanings

## Recommendations

### Immediate Actions
1. **Document temporal differences** in all user-facing outputs
2. **Clarify prediction windows** in clinical reports
3. **Add warnings** about temporal interpretation

### For v0.3.0 True Ensemble
1. **Keep predictions separate** initially
2. **Label each prediction** with its temporal window
3. **Consider sequential use**: PAT for "now", XGBoost for "tomorrow"
4. **Train new heads** if needed for temporal alignment

### CDS Integration
```python
# Proposed clear output structure
{
    "current_state": {
        "source": "PAT",
        "window": "past_7_days",
        "depression_risk": 0.65,
        "confidence": 0.80
    },
    "forecast": {
        "source": "XGBoost", 
        "window": "next_24_hours",
        "depression_risk": 0.72,
        "confidence": 0.85
    },
    "clinical_recommendation": {
        "immediate_action": "Based on current state...",
        "preventive_action": "For tomorrow's risk..."
    }
}
```

## Key Takeaways

1. **XGBoost = Tomorrow**, PAT = Today (when properly implemented)
2. **Current v0.2.0 is temporally confused** - mixes time windows
3. **CDS must clearly distinguish** prediction windows
4. **True ensemble in v0.3.0** should preserve temporal clarity

---

*This temporal mismatch is not a bug but a fundamental architectural consideration that must be addressed for safe clinical deployment.*