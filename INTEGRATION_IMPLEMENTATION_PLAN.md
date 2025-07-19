# Domain Services Integration - Implementation Plan

**Created**: 2025-07-19
**Objective**: Fully integrate all domain services and enable personal baseline normalization

## Overview

This plan provides step-by-step tasks to complete the integration of all domain services, with a focus on:
1. Removing duplicate code
2. Completing dependency injection
3. Integrating the feature engineering orchestrator
4. Implementing persistent personal baselines
5. Enabling user-specific model calibration

## Phase 1: Remove Duplication (1 day)

### Tasks:
- [ ] Delete `src/big_mood_detector/domain/services/clinical_decision_engine.py`
- [ ] Update test file `tests/unit/domain/test_clinical_decision_engine.py` to test clinical_interpreter instead
- [ ] Search for any imports of clinical_decision_engine and update them
- [ ] Run all tests to ensure nothing breaks
- [ ] Commit with message: "refactor: remove duplicate clinical_decision_engine service"

### Verification:
```bash
# Should return no results
grep -r "clinical_decision_engine" src/
# All tests should pass
make test
```

## Phase 2: Complete Dependency Injection (2 days)

### Tasks:
- [ ] Add clinical_interpreter to DI container:
  ```python
  # In infrastructure/di/container.py after line 481
  @provider
  def provide_clinical_interpreter(self) -> ClinicalInterpreter:
      return ClinicalInterpreter()
  ```

- [ ] Add feature_engineering_orchestrator to container:
  ```python
  @provider
  def provide_feature_engineering_orchestrator(self) -> FeatureEngineeringOrchestrator:
      return FeatureEngineeringOrchestrator()
  ```

- [ ] Add BaselineExtractor to container:
  ```python
  @provider
  def provide_baseline_extractor(self) -> BaselineExtractor:
      return BaselineExtractor(baseline_window_days=30)
  ```

- [ ] Add PersonalCalibrator factory to container:
  ```python
  @provider
  def provide_personal_calibrator_factory(self) -> Callable:
      def factory(user_id: str, model_type: str) -> PersonalCalibrator:
          return PersonalCalibrator(
              user_id=user_id,
              model_type=model_type,
              output_dir=self.settings.MODEL_DIR / "personal"
          )
      return factory
  ```

- [ ] Update API routes to use injected clinical_interpreter
- [ ] Update use cases to accept orchestrator via dependency injection
- [ ] Write tests for all new DI providers
- [ ] Commit: "feat: complete dependency injection for all domain services"

## Phase 3: Integrate Feature Engineering Orchestrator (3 days)

### Current State:
```python
# process_health_data_use_case.py:446
features = self.clinical_extractor.extract_features(aggregated_data)
```

### Target State:
```python
# Use orchestrator instead
orchestration_result = self.feature_orchestrator.orchestrate_extraction(
    sleep_data=aggregated_data.sleep_summaries,
    activity_data=aggregated_data.activity_summaries,
    heart_data=aggregated_data.heart_summaries
)
features = orchestration_result.features
quality_report = orchestration_result.quality_report
```

### Tasks:
- [ ] Add feature_orchestrator parameter to MoodPredictionPipeline.__init__
- [ ] Inject orchestrator in process_health_data_use_case.py
- [ ] Replace direct clinical_extractor calls with orchestrator
- [ ] Handle orchestration_result.quality_report in pipeline
- [ ] Update pipeline result to include feature quality metrics
- [ ] Add orchestrator validation for anomaly detection
- [ ] Write integration tests for orchestrated extraction
- [ ] Update existing tests to use orchestrator
- [ ] Commit: "feat: integrate feature engineering orchestrator into pipeline"

## Phase 4: Implement Baseline Persistence (4 days)

### Task 1: Create Baseline Repository
- [ ] Create `domain/repositories/baseline_repository.py` interface
- [ ] Create `infrastructure/repositories/sqlite_baseline_repository.py`
- [ ] Define baseline schema:
  ```python
  class UserBaseline:
      user_id: str
      feature_name: str
      mean: float
      std: float
      sample_count: int
      last_updated: datetime
      window_days: int
  ```
- [ ] Add migrations for baseline table
- [ ] Write repository tests

### Task 2: Connect BaselineExtractor to Advanced Feature Engineering
- [ ] Modify `advanced_feature_engineering.py` to accept BaselineExtractor
- [ ] Replace in-memory baseline calculation with:
  ```python
  # Load existing baselines
  user_baselines = baseline_repo.get_user_baselines(user_id)
  
  # Calculate using BaselineExtractor
  sleep_baseline = baseline_extractor.extract_sleep_baseline(sleep_data)
  activity_baseline = baseline_extractor.extract_activity_baseline(activity_data)
  
  # Persist updated baselines
  baseline_repo.update_baselines(user_id, baselines)
  ```
- [ ] Update z-score calculations to use persisted baselines
- [ ] Handle case when no baseline exists (first run)
- [ ] Write tests for baseline persistence

### Task 3: Add User Context to Pipeline
- [ ] Add user_id parameter to pipeline configuration
- [ ] Thread user_id through all feature extraction calls
- [ ] Update CLI commands to accept --user-id flag
- [ ] Modify API to extract user_id from auth context
- [ ] Commit: "feat: implement persistent personal baselines"

## Phase 5: Enable Personal Calibration (5 days)

### Task 1: Wire PersonalCalibrator into Pipeline
- [ ] Add personal_calibrator to pipeline when user_id is provided:
  ```python
  if config.user_id:
      config.personal_calibrator = calibrator_factory(config.user_id, "xgboost")
      config.enable_personal_calibration = True
  ```
- [ ] Load existing personal model if available
- [ ] Pass calibrator to ensemble orchestrator
- [ ] Update predictions to use calibrated probabilities

### Task 2: Implement Automatic Baseline Updates
- [ ] Create background task for baseline updates
- [ ] Implement sliding window baseline calculation
- [ ] Add baseline drift detection
- [ ] Create alerts for significant baseline changes
- [ ] Store baseline history for trending

### Task 3: Continuous Model Adaptation
- [ ] Implement incremental learning for XGBoost
- [ ] Add PAT LoRA adapter fine-tuning
- [ ] Create feedback loop from labeled episodes
- [ ] Implement model performance tracking
- [ ] Add A/B testing for personal vs population models

### Task 4: User-Specific Thresholds
- [ ] Calculate personal risk thresholds from history
- [ ] Implement adaptive threshold adjustment
- [ ] Add confidence intervals for personal predictions
- [ ] Create user preference settings for sensitivity
- [ ] Commit: "feat: complete personal calibration system"

## Testing Strategy

### Unit Tests:
- [ ] Test each service in isolation
- [ ] Mock dependencies appropriately
- [ ] Achieve >90% coverage for new code

### Integration Tests:
- [ ] Test full pipeline with personal calibration
- [ ] Verify baseline persistence across runs
- [ ] Test orchestrator with various data scenarios
- [ ] Validate DI container wiring

### End-to-End Tests:
- [ ] Process sample data for multiple users
- [ ] Verify personal baselines are maintained
- [ ] Check model adaptation improves accuracy
- [ ] Test API endpoints with user context

## Performance Considerations

- [ ] Profile baseline calculations for large datasets
- [ ] Optimize database queries for baseline retrieval
- [ ] Implement caching for frequently accessed baselines
- [ ] Add batch processing for multiple users
- [ ] Monitor memory usage with personal models

## Documentation Updates

- [ ] Update ARCHITECTURE.md with new service relationships
- [ ] Document personal calibration in user guide
- [ ] Add API documentation for user_id parameter
- [ ] Create migration guide for existing users
- [ ] Update CLAUDE.md with integration patterns

## Success Criteria

1. **All services integrated**: No orphaned services remain
2. **Baselines persist**: User baselines maintained between runs
3. **Personal accuracy**: Individual predictions outperform population model
4. **Clean architecture**: All services properly injected via DI
5. **Test coverage**: >90% coverage maintained
6. **Performance**: No significant slowdown with personal features

## Rollout Plan

1. **Week 1**: Complete Phases 1-2 (cleanup and DI)
2. **Week 2**: Complete Phase 3 (orchestrator integration)
3. **Week 3**: Complete Phase 4 (baseline persistence)
4. **Week 4-5**: Complete Phase 5 (personal calibration)
5. **Week 6**: Testing, documentation, and deployment

---

## Quick Start for Development

```bash
# 1. Create feature branch
git checkout -b feature/complete-domain-integration

# 2. Start with Phase 1
rm src/big_mood_detector/domain/services/clinical_decision_engine.py
make test

# 3. Continue with each phase...
```

Remember: Each phase should be a separate PR for easier review!