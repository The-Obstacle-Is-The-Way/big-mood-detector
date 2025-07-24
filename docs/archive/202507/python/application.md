# Application Layer API Reference

The application layer orchestrates domain logic and coordinates between different parts of the system through use cases.

## Use Cases

### ProcessHealthDataUseCase

Main orchestrator for processing raw health data into clinical features.

```python
from big_mood_detector.application.use_cases import ProcessHealthDataUseCase
from big_mood_detector.infrastructure.parsers import ParserFactory

# Initialize with dependencies
use_case = ProcessHealthDataUseCase(
    parser_factory=ParserFactory(),
    sleep_repo=sleep_repository,
    activity_repo=activity_repository,
    heart_rate_repo=hr_repository
)

# Process health data file
result = use_case.execute(
    file_path="/path/to/export.xml",
    file_type="apple_health"
)

# Access results
print(f"Processed {result.record_count} records")
print(f"Date range: {result.start_date} to {result.end_date}")
print(f"Sleep sessions: {result.sleep_count}")
```

### PredictMoodUseCase

Generates mood predictions from extracted features.

```python
from big_mood_detector.application.use_cases import PredictMoodUseCase

# Initialize with ML model
use_case = PredictMoodUseCase(
    predictor=mood_predictor,
    threshold_config=clinical_thresholds
)

# Make prediction
prediction = use_case.execute(features={
    "mean_sleep_duration": 8.2,
    "std_sleep_duration": 1.1,
    "mean_efficiency": 0.89,
    # ... other features
})

print(f"Depression risk: {prediction.depression_risk:.2f}")
print(f"Clinical interpretation: {prediction.clinical_summary}")
```

### PredictMoodEnsembleUseCase

Ensemble prediction combining multiple models.

```python
from big_mood_detector.application.use_cases import PredictMoodEnsembleUseCase

# Initialize ensemble
use_case = PredictMoodEnsembleUseCase(
    orchestrator=ensemble_orchestrator
)

# Prepare data
ensemble_input = EnsembleInput(
    features=feature_vector,  # 36 features
    pat_sequence=pat_sequence  # 7-day activity
)

# Get ensemble prediction
result = use_case.execute(ensemble_input)

print(f"XGBoost depression: {result.xgboost_predictions['depression']:.2f}")
print(f"PAT depression: {result.pat_predictions['depression']:.2f}")
print(f"Ensemble depression: {result.ensemble_predictions['depression']:.2f}")
```

### ExtractFeaturesUseCase

Extracts clinical features from processed health records.

```python
from big_mood_detector.application.use_cases import ExtractFeaturesUseCase

use_case = ExtractFeaturesUseCase(
    sleep_calculator=sleep_feature_calculator,
    activity_extractor=activity_sequence_extractor,
    circadian_analyzer=circadian_rhythm_analyzer
)

# Extract all features
features = use_case.execute(
    sleep_records=sleep_records,
    activity_records=activity_records,
    heart_rate_records=hr_records
)

# Features include:
# - Basic sleep (duration, efficiency, timing)
# - Circadian metrics (IS, IV, RA, L5, M10)
# - Activity patterns (steps, sedentary time)
# - Heart rate features (resting, variability)
```

## Services

### DataParsingService

Coordinates parsing of different health data formats.

```python
from big_mood_detector.application.services import DataParsingService

service = DataParsingService(parser_factory)

# Parse Apple Health XML
records = service.parse_file(
    file_path="/path/to/export.xml",
    file_type="apple_health"
)

# Parse Health Auto Export JSON
records = service.parse_file(
    file_path="/path/to/data.json",
    file_type="health_auto_export"
)
```

### FeatureEngineeringService

Manages feature extraction and transformation.

```python
from big_mood_detector.application.services import FeatureEngineeringService

service = FeatureEngineeringService()

# Generate feature vector for ML
feature_vector = service.create_feature_vector(
    sleep_features=sleep_features,
    activity_features=activity_features,
    circadian_features=circadian_features
)

# Validate features
is_valid = service.validate_features(feature_vector)
```

### LabelService

Manages clinical labels and episode annotations.

```python
from big_mood_detector.application.services import LabelService

service = LabelService(label_repository)

# Create new label
label_id = service.create_label(
    user_id="patient123",
    start_date=date(2024, 1, 10),
    end_date=date(2024, 1, 17),
    episode_type="depression",
    severity="moderate",
    notes="Clinical assessment"
)

# Query labels
labels = service.get_labels_for_period(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 2, 1)
)

# Export for training
training_data = service.export_training_data()
```

## Orchestrators

### EnsembleOrchestrator

Coordinates multiple ML models for ensemble predictions.

```python
from big_mood_detector.application.orchestrators import EnsembleOrchestrator

orchestrator = EnsembleOrchestrator(
    xgboost_predictor=xgboost_model,
    pat_model=pat_transformer,
    weights={"xgboost": 0.6, "pat": 0.4}
)

# Run ensemble prediction
result = orchestrator.predict(
    features=feature_vector,
    pat_sequence=activity_sequence
)

# Access individual and combined predictions
print(f"Individual: XGB={result.xgboost}, PAT={result.pat}")
print(f"Ensemble: {result.ensemble}")
```

## DTOs (Data Transfer Objects)

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    record_count: int
    start_date: date
    end_date: date
    sleep_count: int
    activity_count: int
    processing_time: float
    errors: List[str]
```

### PredictionResult

```python
@dataclass
class PredictionResult:
    depression_risk: float
    hypomanic_risk: float
    manic_risk: float
    risk_level: str  # "low", "moderate", "high"
    confidence: float
    clinical_summary: str
    recommendations: List[str]
```

### FeatureSet

```python
@dataclass
class FeatureSet:
    sleep_features: Dict[str, float]
    activity_features: Dict[str, float]
    circadian_features: Dict[str, float]
    heart_rate_features: Dict[str, float]
    
    def to_vector(self) -> List[float]:
        """Convert to ML-ready feature vector"""
```

## Usage Patterns

### Complete Processing Pipeline

```python
# 1. Process raw health data
process_use_case = ProcessHealthDataUseCase(dependencies)
processing_result = process_use_case.execute(
    file_path="/path/to/export.xml",
    file_type="apple_health"
)

# 2. Extract clinical features
feature_use_case = ExtractFeaturesUseCase(calculators)
features = feature_use_case.execute(
    sleep_records=processing_result.sleep_records,
    activity_records=processing_result.activity_records
)

# 3. Generate predictions
predict_use_case = PredictMoodEnsembleUseCase(orchestrator)
prediction = predict_use_case.execute(features)

# 4. Store results with labels
label_service.create_prediction_record(
    prediction=prediction,
    features=features,
    metadata=processing_result.metadata
)
```

### Batch Processing

```python
# Process multiple files
batch_processor = BatchProcessingService(
    process_use_case=process_use_case,
    feature_use_case=feature_use_case,
    predict_use_case=predict_use_case
)

results = batch_processor.process_directory(
    directory="/path/to/exports",
    output_dir="/path/to/results"
)

print(f"Processed {len(results)} files")
print(f"Predictions saved to CSV and JSON formats")
```

## Error Handling

All use cases follow consistent error handling:

```python
try:
    result = use_case.execute(input_data)
except ValidationError as e:
    # Invalid input data
    logger.error(f"Validation failed: {e}")
except ProcessingError as e:
    # Processing failed
    logger.error(f"Processing error: {e}")
except InfrastructureError as e:
    # External system error
    logger.error(f"Infrastructure error: {e}")
```