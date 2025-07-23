# Roadmap: v0.3.0 - True Ensemble Implementation

## Current State (v0.2.0)

### What Works ✅
- **XGBoost**: Fully functional mood predictions (0.80-0.98 AUC)
- **PAT**: Provides activity embeddings (not predictions)
- **"Ensemble"**: Actually XGBoost with PAT features concatenated

### What's Misleading ⚠️
- Not a true ensemble (only XGBoost predicts)
- PAT can't make mood predictions without classification heads
- Marketing doesn't match implementation

## v0.3.0 Goals: Deliver True Ensemble

### 1. Get PAT Classification Heads
**Option A: Contact Authors (1-2 days)**
```
Franklin Ruan <franklin.y.ruan.24@dartmouth.edu>
Request: Pre-trained depression classification heads
```

**Option B: Train Our Own (1 week)**
- Use NHANES data (already downloaded!)
- Train depression detection head
- Validate on held-out data

### 2. Implement True Dual Predictions
```python
# Current (v0.2.0):
prediction = xgboost.predict(concat([xgb_features, pat_embeddings]))

# Target (v0.3.0):
xgb_pred = xgboost.predict(xgb_features)        # 0.75 depression
pat_pred = pat_with_head.predict(activity)      # 0.82 depression  
ensemble = weighted_avg(xgb_pred, pat_pred)      # 0.78 depression
```

### 3. Update Documentation
- Clarify v0.2.0 capabilities honestly
- Document upgrade path to v0.3.0
- Set correct expectations

## Implementation Plan

### Phase 1: Quick Wins (This Week)
1. [ ] Process NHANES data with existing processor
2. [ ] Create simple PAT classification head
3. [ ] Test on depression detection task
4. [ ] Document current limitations clearly

### Phase 2: True Ensemble (Next 2 Weeks)
1. [ ] Wire PAT predictions into ensemble
2. [ ] Implement weighted voting
3. [ ] Add model selection CLI flags
4. [ ] Update all documentation

### Phase 3: Advanced Features (Month 2)
1. [ ] Train heads for mania/hypomania
2. [ ] Personal fine-tuning pipeline
3. [ ] Model performance dashboard
4. [ ] Clinical validation studies

## Success Metrics

- [ ] PAT makes independent predictions
- [ ] True ensemble outperforms XGBoost alone
- [ ] Documentation matches reality
- [ ] Users understand exactly what they're getting

## FAQ

**Q: Is v0.2.0 broken?**
A: No, it works! XGBoost predictions are valid. PAT just adds features.

**Q: Should we delay release?**
A: No. Ship with honest docs, then upgrade to true ensemble.

**Q: What about the 36 features?**
A: XGBoost uses 36 engineered features. The "ensemble" creates 36 features by combining 20 XGBoost + 16 PAT.

**Q: Can we train for anxiety?**
A: Yes, if you have anxiety labels (GAD-7). NHANES doesn't have this.

## Next Actions

1. **Today**: Update README to clarify current capabilities
2. **Tomorrow**: Start NHANES processing for PAT heads
3. **This Week**: Get first PAT predictions working
4. **Next Week**: Ship true ensemble in v0.3.0

---

*This roadmap turns a documentation crisis into a clear upgrade path.*