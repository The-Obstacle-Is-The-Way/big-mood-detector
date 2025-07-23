# Phase 2 Action Plan: PAT Classification Heads

**Date**: July 23, 2025
**Status**: Ready to Start
**Branch**: `feature/pat-classification-phase2`

## ✅ Phase 1 Completed
- Separated XGBoost and PAT pipelines
- PAT now only extracts embeddings (no predictions)
- XGBoost only sees statistical features
- All tests updated and CI green

## 🎯 Phase 2 Objectives
1. Create NHANES data processing pipeline
2. Train PAT classification heads for mood prediction
3. Update ensemble to combine two independent predictions
4. Implement proper confidence scoring

## 📋 Immediate Next Steps

### 1. Create Feature Branch
```bash
git checkout development
git pull origin development
git checkout -b feature/pat-classification-phase2
```

### 2. Documentation Cleanup
- [ ] Archive completed plans to `docs/archive/`
- [ ] Update CLAUDE.md with Phase 1 completion
- [ ] Create focused Phase 2 TDD plan

### 3. TDD Implementation Order
1. **Test**: PAT classification head interface
2. **Test**: NHANES data loader
3. **Test**: PAT training pipeline
4. **Test**: Ensemble combination logic
5. **Test**: End-to-end prediction flow

## 🏗️ Architecture Decisions
- Keep training code separate from inference
- Use same MoodPrediction interface for consistency
- Store trained heads in `model_weights/pat/`
- Version control training configs

## ⚠️ Critical Considerations
- Ensure NHANES data privacy compliance
- Maintain backward compatibility
- Document confidence calculation methodology
- Plan for A/B testing old vs new ensemble

## 📈 Success Metrics
- PAT makes independent predictions
- Ensemble confidence improves
- All tests remain green
- Performance benchmarks maintained