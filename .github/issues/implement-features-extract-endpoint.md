# Implement /features/extract API Endpoint

## Context

The API integration tests expect a `/features/extract` endpoint that doesn't exist. This endpoint would allow users to upload health data and get extracted features directly via API.

## Current State
- Test exists but is skipped: `test_extract_features_from_file`
- No route implementation in API
- Feature extraction logic exists in use cases

## Implementation Plan

### 1. Create Route
```python
# src/big_mood_detector/interfaces/api/routes/features.py
from fastapi import APIRouter, UploadFile, File
from big_mood_detector.application.use_cases.process_health_data_use_case import ProcessHealthDataUseCase

router = APIRouter(prefix="/api/v1/features", tags=["features"])

@router.post("/extract")
async def extract_features(file: UploadFile = File(...)):
    """Extract clinical features from uploaded health data."""
    # Save uploaded file temporarily
    # Process with use case
    # Return feature dict
```

### 2. Response Model
```python
class FeatureExtractionResponse(BaseModel):
    features: dict[str, float]
    metadata: dict[str, Any]
    processing_time_seconds: float
    feature_count: int
```

### 3. Integration Points
- Reuse existing `ProcessHealthDataUseCase`
- Support both JSON and XML formats
- Return same 36 features used by models

## Benefits
- Direct feature extraction without full processing
- Useful for debugging and analysis
- Enables feature caching for repeat predictions
- Supports real-time feature monitoring

## Test Coverage
- Upload JSON file → extract features
- Upload XML file → extract features
- Invalid file format → 422 error
- Large file handling → streaming support

@claude Please implement the /features/extract endpoint:
1. Create the features router with extract endpoint
2. Handle file uploads for JSON/XML formats
3. Return extracted features in structured format
4. Add proper error handling and validation
5. Enable the skipped test and ensure it passes