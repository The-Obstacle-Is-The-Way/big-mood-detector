# Domain Services Integration Audit - DETAILED FINDINGS

**Date**: 2025-07-19
**Status**: Comprehensive Analysis Complete

## Executive Summary

After deep investigation, the "orphaned" services are not actually orphaned - they follow a sophisticated facade pattern where specialized services are accessed through `clinical_interpreter.py`. However, several advanced features remain unintegrated:

1. **Feature Engineering Orchestrator** - Built but not connected
2. **Personal Baseline Persistence** - Z-scores calculated but baselines reset each run
3. **Clinical Decision Engine** - Duplicate implementation should be removed
4. **DI Container** - Missing service registrations

## Service Architecture Discovery

### 1. The Facade Pattern (WORKING CORRECTLY)

```
API Routes
    ↓
clinical_interpreter.py (Facade)
    ↓
├── biomarker_interpreter.py ✓
├── dsm5_criteria_evaluator.py ✓
├── early_warning_detector.py ✓
├── episode_interpreter.py ✓
├── risk_level_assessor.py ✓
└── treatment_recommender.py ✓
```

These 6 services ARE integrated through the clinical_interpreter facade used in API routes.

### 2. Services Truly Awaiting Integration

| Service | Purpose | Current Status | Integration Point |
|---------|---------|----------------|-------------------|
| `feature_engineering_orchestrator.py` | Advanced feature extraction with validation | Built, tested, NOT used | Should replace direct clinical_extractor calls |
| `clinical_decision_engine.py` | Duplicate of clinical_interpreter | Unused, should be removed | N/A - DELETE |

### 3. Personal Baseline Integration Status

**What Exists:**
- ✅ `BaselineExtractor` class with methods for sleep, activity, circadian baselines
- ✅ `PersonalCalibrator` for model fine-tuning
- ✅ Z-score calculations in `advanced_feature_engineering.py`
- ✅ Infrastructure for personal calibration in pipeline config

**What's Missing:**
- ❌ Baseline persistence (resets every run)
- ❌ BaselineExtractor not connected to feature extraction
- ❌ User-specific baseline storage in database
- ❌ Automatic baseline updates with new data

### 4. Dependency Injection Gaps

**Currently Registered:**
```python
# Domain services in DI container
- SleepWindowAnalyzer ✓
- ActivitySequenceExtractor ✓
- CircadianRhythmAnalyzer ✓
- DLMOCalculator ✓
- SparseDataHandler ✓
- ClinicalFeatureExtractor ✓
```

**Missing Registrations:**
```python
# Should be added to container
- clinical_interpreter
- feature_engineering_orchestrator
- BaselineExtractor
- PersonalCalibrator
```

## Detailed Service Analysis

### 1. Clinical Services (via clinical_interpreter.py)

**Location**: `src/big_mood_detector/domain/services/clinical_interpreter.py`
**Usage**: `src/big_mood_detector/interfaces/api/clinical_routes.py:12`

The clinical_interpreter acts as a facade that delegates to:

1. **biomarker_interpreter.py** (248 lines)
   - Interprets physiological markers
   - Maps features to clinical significance
   - Used at line 202 of clinical_interpreter

2. **dsm5_criteria_evaluator.py** (195 lines)
   - Evaluates DSM-5 criteria for mood episodes
   - Checks symptom duration and severity
   - Used at line 204 of clinical_interpreter

3. **early_warning_detector.py** (216 lines)
   - Detects prodromal symptoms
   - Identifies pattern changes before episodes
   - Used at line 206 of clinical_interpreter

4. **episode_interpreter.py** (234 lines)
   - Interprets mood episodes from predictions
   - Provides clinical context
   - Used at line 200 of clinical_interpreter

5. **risk_level_assessor.py** (470 lines)
   - Calculates risk scores
   - Provides confidence intervals
   - Used at line 205 of clinical_interpreter

6. **treatment_recommender.py** (324 lines)
   - Suggests interventions based on risk
   - Provides evidence-based recommendations
   - Used at line 203 of clinical_interpreter

### 2. Feature Engineering Orchestrator (NOT INTEGRATED)

**Location**: `src/big_mood_detector/domain/services/feature_engineering_orchestrator.py`
**Lines**: 603
**Purpose**: Provides structured feature extraction with:
- Validation and error handling
- Missing data strategies
- Anomaly detection
- Feature quality metrics

**Should Replace**: Direct calls to clinical_feature_extractor in:
- `process_health_data_use_case.py:446`

### 3. Personal Baseline System

**Current Z-score Implementation** (`advanced_feature_engineering.py:385-399`):
```python
# Calculates z-scores but uses in-memory baselines
sleep_z = (features.sleep_duration - baseline.sleep_mean) / baseline.sleep_std
```

**BaselineExtractor Implementation** (`personal_calibrator.py:45-156`):
- Provides proper 30-day rolling baselines
- Calculates personal activity patterns
- Extracts circadian rhythm baselines

**Integration Gap**: BaselineExtractor methods should be called from advanced_feature_engineering.py

## Critical Finding: Z-Score Normalization

From the literature review:
- **XGBoost paper**: "Circadian phase Z score" was the #1 predictor
- **Current implementation**: Calculates Z-scores but doesn't persist baselines
- **Result**: Loses personalization between runs

## Implementation Plan

### Phase 1: Remove Duplication ✅ COMPLETED
1. ✅ Migrated unique methods from `clinical_decision_engine.py` to `clinical_interpreter.py`
2. ✅ Updated `ClinicalAssessment`, `LongitudinalAssessment`, and `InterventionDecision` dataclasses
3. ✅ Deleted `clinical_decision_engine.py`
4. ✅ Deleted old test file and verified all tests pass

### Phase 2: Complete DI Registration
1. Register clinical_interpreter in container
2. Register feature_engineering_orchestrator
3. Register PersonalCalibrator and BaselineExtractor
4. Update injection points to use container

### Phase 3: Integrate Feature Engineering Orchestrator
1. Replace direct clinical_feature_extractor calls
2. Add orchestrator to process_health_data_use_case
3. Update tests for new integration

### Phase 4: Implement Baseline Persistence
1. Create baseline repository for database storage
2. Connect BaselineExtractor to advanced_feature_engineering
3. Load user baselines at pipeline start
4. Save updated baselines after processing

### Phase 5: Enable Personal Calibration
1. Wire PersonalCalibrator into main pipeline
2. Implement automatic baseline updates
3. Add continuous model adaptation
4. Create user-specific threshold adjustments

## Verification Checklist

- [ ] All 29 domain services have clear integration paths
- [ ] No duplicate services remain
- [ ] DI container registers all services
- [ ] Personal baselines persist between runs
- [ ] Feature engineering orchestrator is integrated
- [ ] Z-scores use personal baselines, not population
- [ ] Models can be fine-tuned per user
- [ ] Continuous adaptation is implemented

## Next Steps

1. Start with Phase 1 (remove duplication)
2. Write integration tests for each phase
3. Implement one phase at a time
4. Verify with clinical accuracy metrics

---
*This completes the comprehensive audit of domain services integration*