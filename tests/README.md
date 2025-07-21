# Test Suite Documentation

## Test Organization

Our test suite follows a hierarchical structure based on test scope and purpose:

```
tests/
├── unit/               # Fast, isolated tests of individual components
├── integration/        # Tests of multiple components working together
├── e2e/               # Full end-to-end user workflow tests
└── conftest.py        # Shared fixtures and configuration
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual functions, classes, and modules in isolation
- Mock external dependencies (databases, APIs, file systems)
- Should run in <1 second per test
- No network calls, no file I/O, no database connections

**Structure:**
```
unit/
├── domain/           # Pure business logic tests
├── application/      # Use case and service tests
├── infrastructure/   # Infrastructure implementation tests
└── interfaces/       # CLI and API handler tests
```

### Integration Tests (`tests/integration/`)
- Test interactions between multiple components
- May use real databases, file systems, or external services
- Focus on specific subsystem integration
- Run in Docker containers when needed

**Structure:**
```
integration/
├── api/             # API endpoint integration tests
├── storage/         # Database and repository tests
├── pipeline/        # ML pipeline integration tests
├── ml/              # Model ensemble tests
└── (root level)     # Cross-cutting integration tests
```

**Root-level integration tests** are for cross-cutting concerns:
- Dependency injection validation
- Memory constraints
- API contract compliance
- System-wide behavior

### E2E Tests (`tests/e2e/`)
- Test complete user workflows from start to finish
- Require full application stack running
- Simulate real user interactions
- May take several minutes to run

**Examples:**
- Process health data → train model → make predictions
- Upload data via API → view results → export report

## Test Markers

We use pytest markers to categorize and filter tests:

```python
@pytest.mark.unit          # Unit tests (default)
@pytest.mark.integration   # Integration tests
@pytest.mark.ml           # Machine learning tests
@pytest.mark.clinical     # Clinical validation tests
@pytest.mark.slow         # Tests taking >5 seconds
@pytest.mark.heavy        # Tests requiring large files/models
@pytest.mark.flaky        # Known intermittent failures
```

Run specific categories:
```bash
pytest -m unit           # Only unit tests
pytest -m "not slow"     # Skip slow tests
pytest -m "ml and not heavy"  # ML tests without heavy models
```

## Skip vs XFail

### Use `@pytest.mark.skip` when:
- External dependency not available (database, model weights)
- Test requires specific environment (GPU, large memory)
- Feature temporarily disabled
- Test needs redesign (document reason)

```python
@pytest.mark.skip(reason="Requires PAT model weights")
@pytest.mark.skipif(not has_gpu(), reason="Requires GPU")
```

### Use `@pytest.mark.xfail` when:
- Known bug exists (link to issue)
- Feature partially implemented
- Test is flaky but important

```python
@pytest.mark.xfail(reason="Issue #40: XGBoost predict_proba missing")
```

## Writing Tests

### Naming Conventions
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<what_it_tests>`

### Test Structure
```python
def test_sleep_aggregation_handles_fragmented_sleep(self):
    """Test that fragmented sleep periods are correctly merged."""
    # Arrange
    sleep_records = create_fragmented_sleep_records()
    
    # Act
    result = SleepAggregator().aggregate(sleep_records)
    
    # Assert
    assert result.total_duration == expected_duration
    assert result.efficiency > 0.7
```

### Fixtures
Common fixtures are in `conftest.py`:
- `sample_health_data`: Small test dataset
- `ml_models`: Mock ML models
- `test_db`: In-memory test database
- `clinical_thresholds`: Test clinical config

## Performance Guidelines

### Unit Tests
- Should complete in <100ms
- Use mocks for slow operations
- Avoid file I/O

### Integration Tests
- Should complete in <5 seconds
- Use test containers when needed
- Clean up resources in teardown

### E2E Tests
- May take up to 5 minutes
- Run separately in CI
- Document expected runtime

## Coverage Requirements

- Overall: 75% minimum (currently 77%)
- New features: 90% minimum
- Critical paths: 95% minimum

Check coverage:
```bash
make coverage
# or
pytest --cov=big_mood_detector --cov-report=html
```

## Common Patterns

### Testing Async Code
```python
@pytest.mark.asyncio
async def test_async_endpoint(self):
    async with AsyncClient(app) as client:
        response = await client.get("/health")
        assert response.status_code == 200
```

### Testing with Real Data
```python
@pytest.mark.skipif(
    not Path("data/export.xml").exists(),
    reason="Requires Apple Health export"
)
def test_with_real_data(self):
    # Test with actual user data
```

### Testing ML Models
```python
@pytest.fixture
def mock_xgboost_model(mocker):
    model = mocker.Mock()
    model.predict_proba.return_value = [[0.2, 0.8]]
    return model
```

## Debugging Failed Tests

1. **Run with verbose output**: `pytest -vvs path/to/test.py`
2. **Use debugger**: `pytest --pdb` (drops to debugger on failure)
3. **Check test isolation**: Run test alone vs in suite
4. **Review logs**: `pytest --log-cli-level=DEBUG`

## CI/CD Integration

Tests run automatically on:
- Every push to PR
- Before merging to main
- Nightly full test suite

Fast feedback loop:
1. Unit tests first (fail fast)
2. Integration tests if units pass
3. E2E tests on merge to main

## Maintenance

### Adding New Tests
1. Choose appropriate directory based on scope
2. Use existing patterns and fixtures
3. Add markers for categorization
4. Ensure tests are deterministic

### Refactoring Tests
1. Keep test intent clear
2. Extract common setup to fixtures
3. Group related tests in classes
4. Document complex test scenarios

### Removing Tests
Only remove tests when:
- Feature is permanently removed
- Test duplicates another test
- Test no longer provides value

Always document why a test was removed in the commit message.