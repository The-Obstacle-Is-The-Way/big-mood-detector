---
name: PAT Fine-tuning Task
about: Help implement true ensemble predictions for v0.3.0
title: '[PAT] '
labels: 'enhancement, good first issue, v0.3.0'
assignees: ''
---

## Context
v0.2.0 only provides XGBoost predictions. PAT currently outputs embeddings but cannot make mood predictions without classification heads. We need to train these heads to enable true ensemble predictions.

## Task
Help implement PAT classification heads for mood prediction.

## Current State
- PAT encoder: ✅ Working (outputs 96-dim embeddings)
- NHANES data: ✅ Downloaded to `data/nhanes/2013-2014/`
- Classification heads: ❌ Missing

## Steps
1. [ ] Process NHANES data using `nhanes_processor.py`
2. [ ] Extract PAT embeddings for all participants
3. [ ] Train classification head for depression (PHQ-9 >= 10)
4. [ ] Integrate head into `PATModel` class
5. [ ] Update ensemble to use dual predictions
6. [ ] Add tests and documentation

## Resources
- [PAT Fine-tuning Roadmap](../../docs/PAT_FINE_TUNING_ROADMAP.md)
- [NHANES Processor Code](../../src/big_mood_detector/infrastructure/fine_tuning/nhanes_processor.py)
- [PAT Paper](https://arxiv.org/abs/2411.15240)

## Success Criteria
- [ ] PAT can make mood predictions independently
- [ ] Ensemble combines XGBoost and PAT predictions
- [ ] Tests pass with >90% coverage
- [ ] Documentation updated

## Questions?
Feel free to ask in the comments or check the roadmap document\!
EOF < /dev/null