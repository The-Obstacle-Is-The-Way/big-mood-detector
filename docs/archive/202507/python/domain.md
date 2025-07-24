# Domain Layer API Reference

The domain layer contains the core business logic and entities that represent the fundamental concepts of mood detection.

## Entities

### SleepRecord

Represents a single sleep session with clinical metrics.

```python
from big_mood_detector.domain.entities import SleepRecord
from datetime import datetime, timedelta

record = SleepRecord(
    start_time=datetime.now() - timedelta(hours=8),
    end_time=datetime.now(),
    sleep_duration_hours=7.5,
    deep_sleep_ratio=0.25,
    rem_sleep_ratio=0.22,
    awake_ratio=0.08,
    heart_rate_avg=58.0,
    heart_rate_min=48.0,
    respiratory_rate=14.0
)
```

### ActivityRecord

Represents hourly activity data with step counts and calories.

```python
from big_mood_detector.domain.entities import ActivityRecord

activity = ActivityRecord(
    timestamp=datetime.now(),
    date="2024-01-15",
    hour=14,
    steps=523,
    active_calories=45.2,
    heart_rate_avg=72.0,
    distance_km=0.4
)
```

### HeartRateRecord

Represents heart rate measurements with variability metrics.

```python
from big_mood_detector.domain.entities import HeartRateRecord

hr = HeartRateRecord(
    timestamp=datetime.now(),
    value=65.0,
    motion_context="resting",
    hrv_sdnn=45.0
)
```

### Label

Represents clinical labels for mood episodes.

```python
from big_mood_detector.domain.entities import Label
from datetime import date

label = Label(
    start_date=date(2024, 1, 10),
    end_date=date(2024, 1, 17),
    episode_type="depression",
    severity="moderate",
    confidence=0.85,
    source="clinician",
    notes="Patient reported low mood and fatigue"
)
```

## Services

### SleepWindowAnalyzer

Analyzes and merges sleep episodes based on clinical criteria.

```python
from big_mood_detector.domain.services import SleepWindowAnalyzer

analyzer = SleepWindowAnalyzer(merge_threshold_hours=3.75)

# Merge sleep episodes within 3.75 hour windows
merged_episodes = analyzer.merge_sleep_episodes(sleep_records)

# Analyze sleep patterns
patterns = analyzer.analyze_patterns(merged_episodes)
```

### ActivitySequenceExtractor

Extracts minute-level activity sequences for ML processing.

```python
from big_mood_detector.domain.services import ActivitySequenceExtractor

extractor = ActivitySequenceExtractor()

# Generate 7-day activity sequence (10,080 values)
sequence = extractor.extract_week_sequence(
    activity_records,
    end_date=date(2024, 1, 15)
)

# Get daily summaries
daily_stats = extractor.get_daily_summaries(activity_records)
```

### SleepFeatureCalculator

Calculates clinical sleep features from sleep records.

```python
from big_mood_detector.domain.services import SleepFeatureCalculator

calculator = SleepFeatureCalculator()

# Calculate all sleep features
features = calculator.calculate_features(sleep_records)
# Returns: mean_duration, std_duration, efficiency, timing metrics
```

### CircadianRhythmAnalyzer

Analyzes circadian rhythm stability and patterns.

```python
from big_mood_detector.domain.services import CircadianRhythmAnalyzer

analyzer = CircadianRhythmAnalyzer()

# Calculate circadian metrics
metrics = analyzer.calculate_metrics(activity_records)
# Returns: IS, IV, RA, L5, M10 values
```

## Value Objects

### TimeRange

Immutable time range representation.

```python
from big_mood_detector.domain.value_objects import TimeRange

range = TimeRange(
    start=datetime(2024, 1, 1, 22, 0),
    end=datetime(2024, 1, 2, 6, 30)
)

duration_hours = range.duration_hours
overlaps = range.overlaps(other_range)
```

### ClinicalThreshold

Immutable clinical threshold values.

```python
from big_mood_detector.domain.value_objects import ClinicalThreshold

threshold = ClinicalThreshold(
    depression=0.5,
    hypomanic=0.3,
    manic=0.2
)
```

## Repository Interfaces

### SleepRepository

Abstract interface for sleep data persistence.

```python
from big_mood_detector.domain.repositories import SleepRepository

class MySleepRepo(SleepRepository):
    def save(self, record: SleepRecord) -> None:
        # Implementation
        pass
    
    def get_by_date_range(self, start: date, end: date) -> List[SleepRecord]:
        # Implementation
        pass
```

### ActivityRepository

Abstract interface for activity data persistence.

```python
from big_mood_detector.domain.repositories import ActivityRepository

class MyActivityRepo(ActivityRepository):
    def save_batch(self, records: List[ActivityRecord]) -> None:
        # Implementation
        pass
    
    def get_hourly_data(self, date: date) -> List[ActivityRecord]:
        # Implementation
        pass
```

## Design Patterns

The domain layer follows these patterns:

1. **Entity Pattern**: Rich domain objects with behavior
2. **Value Object Pattern**: Immutable objects for concepts without identity
3. **Repository Pattern**: Abstract interfaces for data access
4. **Domain Service Pattern**: Business logic that doesn't belong to entities

## Usage Example

```python
# Complete domain workflow
from big_mood_detector.domain.services import (
    SleepWindowAnalyzer,
    ActivitySequenceExtractor,
    SleepFeatureCalculator
)

# Analyze sleep patterns
sleep_analyzer = SleepWindowAnalyzer()
merged_sleep = sleep_analyzer.merge_sleep_episodes(sleep_records)

# Extract activity sequences
activity_extractor = ActivitySequenceExtractor()
activity_seq = activity_extractor.extract_week_sequence(
    activity_records,
    end_date=date.today()
)

# Calculate clinical features
sleep_calculator = SleepFeatureCalculator()
sleep_features = sleep_calculator.calculate_features(merged_sleep)

# Features ready for ML prediction
print(f"Sleep efficiency: {sleep_features['mean_efficiency']}")
print(f"Activity pattern: {len(activity_seq.activity_values)} minutes tracked")
```