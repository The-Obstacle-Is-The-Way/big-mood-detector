---
name: v0.3.0 Epic - True Ensemble Implementation
about: Track progress on implementing true dual-model predictions
title: '[EPIC] v0.3.0 - True Ensemble with PAT Classification Heads'
labels: 'epic, enhancement, v0.3.0'
assignees: ''
---

## 🎯 Epic Goal
Transform the current "PAT-enhanced XGBoost" into a true ensemble where both XGBoost and PAT make independent mood predictions.

## 📋 Current State (v0.2.0)
- ✅ XGBoost: Fully functional mood predictions
- ⚠️ PAT: Outputs embeddings only (no classification heads)
- 🔄 "Ensemble": Concatenates features, only XGBoost predicts

## 🚀 Target State (v0.3.0)
- ✅ XGBoost: Independent mood predictions
- ✅ PAT: Independent mood predictions via classification heads
- ✅ True Ensemble: Weighted voting between both models

## 📌 Sub-tasks

### Phase 1: Enable PAT Predictions
- [ ] Process NHANES data with `nhanes_processor.py`
- [ ] Extract PAT embeddings for all participants
- [ ] Train depression classification head (PHQ-9 ≥ 10)
- [ ] Integrate head into `PATModel` class
- [ ] Add `predict_mood()` method to PAT

### Phase 2: Implement True Ensemble
- [ ] Update `EnsembleOrchestrator` to use dual predictions
- [ ] Implement weighted voting (60/40 default)
- [ ] Add fallback logic if one model fails
- [ ] Update CLI flags (--model xgboost/pat/ensemble)
- [ ] Performance benchmarks

### Phase 3: Testing & Documentation
- [ ] Unit tests for PAT predictions
- [ ] Integration tests for true ensemble
- [ ] Update all documentation
- [ ] Add migration guide from v0.2.0
- [ ] Clinical validation notes

## 📊 Success Criteria
- [ ] PAT makes independent mood predictions
- [ ] Ensemble combines both model outputs
- [ ] Performance: Ensemble ≥ XGBoost alone
- [ ] Tests: >90% coverage maintained
- [ ] Docs: Accurate and complete

## 🔗 Resources
- [Honest State of v0.2.0](../../docs/HONEST_STATE_OF_V0.2.0.md)
- [v0.3.0 Roadmap](../../docs/ROADMAP_V0.3.0.md)
- [PAT Fine-tuning Guide](../../docs/PAT_FINE_TUNING_ROADMAP.md)
- [NHANES Data](../../data/nhanes/README.md)

## 📅 Timeline
- Week 1: NHANES processing and head training
- Week 2: Integration and ensemble implementation
- Week 3: Testing, documentation, and release

## 💬 Discussion
Use comments below to track progress, ask questions, or propose changes to the plan.