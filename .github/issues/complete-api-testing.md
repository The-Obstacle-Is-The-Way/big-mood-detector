# Complete API Testing and Fix Response Codes

## Current State
- API endpoints are implemented but integration tests are failing
- Response codes are incorrect (200 instead of 201 for POST)
- Test database pollution between test runs
- Missing test coverage for some endpoints

## Required Fixes

### 1. Response Code Corrections
```python
# Fix in routes/labels.py
@router.post("/episodes", response_model=EpisodeResponse, status_code=201)
@router.delete("/episodes/{episode_id}", status_code=204)
```

### 2. Test Isolation
- Add pytest fixture to clean database between tests
- Use temporary database for testing
- Ensure no state leaks between test cases

### 3. Missing Test Coverage
- File upload endpoints
- Batch processing
- Clinical interpretation routes
- Export functionality

## Acceptance Criteria
- [ ] All API integration tests pass
- [ ] Correct HTTP status codes returned
- [ ] No test pollution/isolation issues
- [ ] 100% endpoint coverage
- [ ] Response validation matches OpenAPI spec

@claude Please fix the API test issues by:
1. Adding proper test database isolation
2. Correcting all HTTP response status codes
3. Ensuring all tests pass consistently
4. Adding missing test coverage for uncovered endpoints