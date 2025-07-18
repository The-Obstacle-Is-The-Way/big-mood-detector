# Fix API Response Codes and Test Isolation

## Problem Description

The API integration tests revealed several issues:

1. **Incorrect HTTP Status Codes**
   - POST endpoints return 200 instead of 201 (Created)
   - DELETE endpoints may not return 204 (No Content)

2. **Test Database Pollution**
   - Labels persist between test runs
   - Tests fail because they expect empty database but find existing data

3. **Missing Validations**
   - Date range validation not working (end before start should fail)
   - No validation for overlapping episodes

## Current Test Failures

```
FAILED test_create_episode_success - assert 200 == 201
FAILED test_list_episodes_empty - Episodes from previous tests still exist
FAILED test_invalid_date_range - Returns 200 instead of 422 for invalid dates
```

## Proposed Solutions

### 1. Fix Response Codes
```python
# In labels.py
@router.post("/episodes", response_model=EpisodeResponse, status_code=201)  # Add status_code
async def create_episode(request: EpisodeCreateRequest) -> EpisodeResponse:
    ...

@router.delete("/episodes/{episode_id}", status_code=204)  # Add status_code
async def delete_episode(episode_id: str) -> None:
    ...
```

### 2. Test Isolation
```python
# Add test fixture for database cleanup
@pytest.fixture(autouse=True)
def clean_test_db():
    """Clean test database before each test."""
    if os.path.exists("labels.db"):
        os.remove("labels.db")
    yield
    if os.path.exists("labels.db"):
        os.remove("labels.db")
```

### 3. Add Date Validation
```python
# In EpisodeCreateRequest validator
@validator('end_date')
def validate_date_range(cls, v, values):
    if 'start_date' in values and v and v < values['start_date']:
        raise ValueError('end_date must be after start_date')
    return v
```

## Test Coverage Needed

- [ ] Test proper status codes for all HTTP methods
- [ ] Test database isolation between test runs
- [ ] Test date range validation
- [ ] Test overlapping episode detection

@claude Please implement these fixes with the following approach:
1. Start with test database isolation to ensure clean test runs
2. Fix the HTTP status codes in the API routes
3. Add proper validation for date ranges
4. Ensure all tests pass consistently