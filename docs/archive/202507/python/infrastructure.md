# Infrastructure Layer API Reference

The infrastructure layer provides concrete implementations for data access, ML models, and external integrations.

## Parsers

### StreamingXMLParser

High-performance streaming parser for large Apple Health XML exports.

```python
from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser

parser = StreamingXMLParser()

# Parse large XML file with minimal memory usage
records = parser.parse(
    file_path="/path/to/export.xml",
    callback=lambda record: process_record(record)
)

# Or collect all records
all_records = []
parser.parse(
    file_path="/path/to/export.xml", 
    callback=all_records.append
)
```

### JSONHealthParser

Parser for Health Auto Export JSON format.

```python
from big_mood_detector.infrastructure.parsers.json import JSONHealthParser

parser = JSONHealthParser()

# Parse JSON file
records = parser.parse("/path/to/health_data.json")

# Access typed records
for record in records:
    if record.record_type == "sleep":
        print(f"Sleep: {record.duration_hours} hours")
```

### ParserFactory

Factory for creating appropriate parser based on file type.

```python
from big_mood_detector.infrastructure.parsers import ParserFactory

factory = ParserFactory()

# Automatically detect parser type
parser = factory.create_parser("apple_health")  # Returns StreamingXMLParser
parser = factory.create_parser("health_auto_export")  # Returns JSONHealthParser
```

## ML Models

### XGBoostMoodPredictor

Production XGBoost model for mood prediction.

```python
from big_mood_detector.infrastructure.ml_models import XGBoostMoodPredictor

predictor = XGBoostMoodPredictor()

# Load pretrained models
predictor.load_models("/path/to/model_weights/xgboost/pretrained")

# Make predictions
features = np.array([...])  # 36 features
predictions = predictor.predict(features)

print(f"Depression: {predictions['depression']:.2f}")
print(f"Hypomanic: {predictions['hypomanic']:.2f}")
print(f"Manic: {predictions['manic']:.2f}")
```

### DirectPATModel

Pretrained Actigraphy Transformer for activity sequences.

```python
from big_mood_detector.infrastructure.ml_models import DirectPATModel

# Initialize model (requires TensorFlow)
model = DirectPATModel(model_dir="/path/to/model_weights/pat")

# Prepare 7-day activity sequence
activity_sequence = np.array([...])  # Shape: (10080,)

# Get predictions
predictions = model.predict(activity_sequence)
```

### PATModelStub

Fallback when TensorFlow is not available.

```python
from big_mood_detector.infrastructure.ml_models import PATModelStub

# Returns baseline predictions when TF unavailable
stub = PATModelStub()
predictions = stub.predict(activity_sequence)
# Returns: {"depression": 0.33, "hypomanic": 0.33, "manic": 0.34}
```

## Repositories

### FileSleepRepository

File-based storage for sleep records.

```python
from big_mood_detector.infrastructure.repositories import FileSleepRepository

repo = FileSleepRepository(base_path="/data/sleep")

# Save records
repo.save(sleep_record)
repo.save_batch(sleep_records)

# Query records
records = repo.get_by_date_range(
    start=date(2024, 1, 1),
    end=date(2024, 1, 31)
)
```

### SQLiteEpisodeRepository

SQLite storage for labeled episodes.

```python
from big_mood_detector.infrastructure.repositories import SQLiteEpisodeRepository

repo = SQLiteEpisodeRepository(db_path="/data/labels.db")

# Create episode
episode_id = repo.create_episode(
    start_date=date(2024, 1, 10),
    end_date=date(2024, 1, 17),
    episode_type="depression",
    severity=3,
    confidence=0.85
)

# Query episodes
episodes = repo.get_episodes_by_rater("clinician_001")
stats = repo.get_episode_stats()
```

### InMemoryLabelRepository

In-memory storage for testing and development.

```python
from big_mood_detector.infrastructure.repositories import InMemoryLabelRepository

repo = InMemoryLabelRepository()

# Full CRUD operations
label_id = repo.create(label)
label = repo.get(label_id)
repo.update(label_id, updated_label)
repo.delete(label_id)
```

## Background Tasks

### TaskQueue

Redis-based task queue for async processing.

```python
from big_mood_detector.infrastructure.background import TaskQueue

queue = TaskQueue(redis_url="redis://localhost:6379")

# Enqueue task
task_id = queue.enqueue(
    "process_health_file",
    file_path="/uploads/export.xml",
    user_id="user123"
)

# Check status
status = queue.get_status(task_id)
print(f"Task {task_id}: {status}")
```

### Worker

Background worker for processing tasks.

```python
from big_mood_detector.infrastructure.background import Worker

worker = Worker(queue=task_queue)

# Register task handlers
worker.register_handler("process_health_file", process_file_handler)
worker.register_handler("generate_report", generate_report_handler)

# Start worker
worker.start()  # Runs until interrupted
```

## Fine-Tuning

### NHANESProcessor

Processes NHANES data for model training.

```python
from big_mood_detector.infrastructure.fine_tuning import NHANESProcessor

processor = NHANESProcessor()

# Process NHANES activity data
features, labels = processor.process_paxmin_file(
    paxmin_path="/data/NHANES/PAXMIN_J.XPT",
    demo_path="/data/NHANES/DEMO_J.XPT"
)

# Ready for model training
print(f"Samples: {len(features)}")
print(f"Features per sample: {features.shape[1]}")
```

### PersonalCalibrator

Fine-tunes models with personal data.

```python
from big_mood_detector.infrastructure.fine_tuning import PersonalCalibrator

calibrator = PersonalCalibrator(base_model=xgboost_model)

# Calibrate with labeled personal data
calibrated_model = calibrator.calibrate(
    features=personal_features,
    labels=personal_labels,
    calibration_type="platt_scaling"
)

# Save calibrated model
calibrator.save_calibration("/models/personal/calibrated.pkl")
```

## Monitoring

### FileWatcher

Monitors directories for new health data files.

```python
from big_mood_detector.infrastructure.monitoring import FileWatcher

watcher = FileWatcher(
    watch_dir="/uploads",
    handler=process_new_file,
    extensions=[".xml", ".json", ".zip"]
)

# Start watching
watcher.start()  # Non-blocking
watcher.stop()   # Stop watching
```

## Settings

### Settings

Application configuration management.

```python
from big_mood_detector.infrastructure.settings import Settings

settings = Settings()

# Access configuration
print(f"Model path: {settings.MODEL_WEIGHTS_PATH}")
print(f"Upload dir: {settings.UPLOAD_DIR}")
print(f"Log level: {settings.LOG_LEVEL}")

# Environment-specific
if settings.ENVIRONMENT == "production":
    settings.validate_production_config()
```

## Utilities

### Logging

Structured logging with context.

```python
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)

# Structured logging
logger.info(
    "Processing file",
    file_path=file_path,
    size_mb=file_size_mb,
    user_id=user_id
)

# With timing
with logger.timer("parse_xml"):
    records = parser.parse(file_path)
```

### Dependency Injection

Container for managing dependencies.

```python
from big_mood_detector.infrastructure.di import Container

container = Container()

# Register dependencies
container.register(
    SleepRepository, 
    FileSleepRepository,
    base_path="/data/sleep"
)

# Resolve dependencies
use_case = container.resolve(ProcessHealthDataUseCase)
```

## Example: Complete Infrastructure Setup

```python
# Initialize infrastructure
from big_mood_detector.infrastructure import (
    Settings,
    Container,
    get_module_logger
)

# Configure
settings = Settings()
container = Container()
logger = get_module_logger(__name__)

# Set up repositories
container.register(SleepRepository, FileSleepRepository, settings.DATA_DIR)
container.register(ActivityRepository, FileActivityRepository, settings.DATA_DIR)
container.register(LabelRepository, SQLiteEpisodeRepository, settings.LABELS_DB_PATH)

# Set up ML models
xgboost = XGBoostMoodPredictor()
xgboost.load_models(settings.XGBOOST_MODEL_PATH)
container.register_instance(XGBoostMoodPredictor, xgboost)

# Set up background processing
if settings.ENABLE_ASYNC_UPLOAD:
    queue = TaskQueue(settings.REDIS_URL)
    container.register_instance(TaskQueue, queue)

# Ready for use
logger.info("Infrastructure initialized", environment=settings.ENVIRONMENT)
```