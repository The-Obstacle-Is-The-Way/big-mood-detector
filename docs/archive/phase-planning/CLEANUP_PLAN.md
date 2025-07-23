# Comprehensive Cleanup Plan - Post Phase 1 Refactor
Generated: 2025-07-23

## Objective
Ensure the codebase is clean, all tests pass, and there's no confusion about baselines, types, or linting after merging the dual pipeline refactor.

## 1. Test Suite Health Check

### 1.1 Run All Tests
```bash
# Full test suite
make test

# Specific test categories
pytest -m unit          # Fast unit tests
pytest -m integration   # Integration tests
pytest -m ml           # ML model tests
pytest -m clinical     # Clinical validation tests
```

### 1.2 Fix Known Failing Tests
- [ ] `test_pipeline_without_ensemble` - Was already failing before refactor
- [ ] Check for any xfail/skip markers that can be removed

### 1.3 Remove Duplicate/Obsolete Tests
- [ ] Consolidate refactor tests with original tests
- [ ] Remove tests for deprecated `pat_enhanced_prediction` behavior

## 2. Baseline System Verification

### 2.1 Verify Baseline Persistence
```bash
# Run baseline tests
pytest tests/integration/test_baseline_persistence_pipeline.py -v

# Check baseline files are created
ls -la data/baselines/
```

### 2.2 Update Baseline Documentation
- [ ] Document how baselines work with new ensemble structure
- [ ] Update CLAUDE.md with baseline usage

## 3. Type Checking & Linting

### 3.1 Fix All Type Issues
```bash
# Run type checking
make type-check

# Common issues to fix:
# - Remove Any types
# - Add proper return type annotations
# - Fix Optional vs | None usage
```

### 3.2 Complete Linting Cleanup
```bash
# Check all lint issues
make lint

# Auto-fix what we can
ruff check --fix .

# Specific areas needing attention:
# - scripts/ folder (85 errors)
# - Unused imports
# - Line length violations
```

### 3.3 Update Pre-commit Hooks
- [ ] Ensure pre-commit runs on all files
- [ ] Add mypy to pre-commit if not present

## 4. Documentation Updates

### 4.1 Architecture Documentation
- [ ] Update ARCHITECTURE_OVERVIEW.md with true dual pipeline
- [ ] Update DUAL_PIPELINE_ARCHITECTURE.md to reflect reality
- [ ] Remove/consolidate duplicate PAT status files

### 4.2 API Documentation
- [ ] Document new EnsemblePrediction fields
- [ ] Update API endpoints to expose temporal_context
- [ ] Add examples of new response format

### 4.3 Developer Guide
- [ ] Update CONTRIBUTING.md with new test structure
- [ ] Document how to add new models to ensemble
- [ ] Add troubleshooting guide for common issues

## 5. Code Organization

### 5.1 Remove Dead Code
- [ ] Remove unused _predict_with_pat method (keep for now as deprecated)
- [ ] Clean up duplicate PAT documentation files
- [ ] Remove commented-out code

### 5.2 Consolidate Similar Files
```bash
# PAT documentation files to consolidate:
- PAT_INTEGRATION_PLAN.md
- PAT_INTEGRATION_STATUS.md 
- PAT_FINAL_STATUS.md
- docs/PAT_FINE_TUNING_ROADMAP.md
```

### 5.3 Move Test Files
- [ ] Move slow tests to tests/integration/
- [ ] Ensure test organization matches package structure

## 6. CI/CD Updates

### 6.1 Update GitHub Actions
- [ ] Ensure all tests run in CI
- [ ] Add type checking to CI
- [ ] Update coverage requirements

### 6.2 Fix Pre-push Hooks
- [ ] Update .pre-commit-config.yaml
- [ ] Ensure hooks don't block on minor issues
- [ ] Add --unsafe-fixes flag where appropriate

## 7. Performance Verification

### 7.1 Benchmark Current Performance
```bash
# Run performance tests
pytest -m performance -v

# Key metrics to verify:
# - XML parsing: >40k records/second
# - Aggregation: <20s for 365 days
# - API response: <200ms average
```

### 7.2 Memory Profiling
```bash
# Profile memory usage
mprof run python src/big_mood_detector/main.py process large_file.xml
mprof plot
```

## 8. Integration Testing

### 8.1 End-to-End Tests
- [ ] Test CLI with both JSON and XML inputs
- [ ] Test API with ensemble predictions
- [ ] Verify Docker deployment works

### 8.2 Backwards Compatibility
- [ ] Ensure old API responses still work
- [ ] Test model weight loading
- [ ] Verify configuration migration

## 9. Final Verification Checklist

### Before Pushing to Main
- [ ] All tests pass (except known issues)
- [ ] Zero type errors
- [ ] Lint errors < 10 (excluding scripts/)
- [ ] Documentation is accurate
- [ ] Performance benchmarks pass
- [ ] API responses are correct
- [ ] Docker builds successfully

### Quality Gates
```bash
# Run full quality check
make quality

# Should see:
# - Tests: PASSED
# - Type Check: PASSED  
# - Linting: PASSED (with exceptions)
# - Coverage: >90%
```

## 10. Next Steps After Cleanup

1. **Tag Release**: Create v0.2.4 with refactored ensemble
2. **Update CHANGELOG**: Document all changes
3. **Plan Phase 2**: PAT classification head training
4. **Create Issues**: For any remaining technical debt

## Execution Order

### Day 1: Testing & Types
1. Fix failing tests
2. Run type checker and fix issues
3. Update test organization

### Day 2: Linting & Docs
1. Complete lint cleanup
2. Update all documentation
3. Consolidate duplicate files

### Day 3: Integration & Performance
1. Run full integration tests
2. Benchmark performance
3. Final quality check

### Day 4: Release Prep
1. Update changelog
2. Create release tag
3. Update deployment docs

## Success Criteria

- **Tests**: 100% pass rate (excluding xfail)
- **Types**: 0 mypy errors
- **Lint**: <10 errors (scripts excluded)
- **Docs**: All accurate and up-to-date
- **Performance**: Meets all benchmarks
- **Integration**: E2E tests pass

---

This cleanup ensures we have a solid foundation before implementing Phase 2 (PAT classification heads).