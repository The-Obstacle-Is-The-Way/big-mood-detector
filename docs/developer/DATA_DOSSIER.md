# Big Mood Detector - Data Requirements Dossier

## Executive Summary

This dossier documents the data requirements for the dual-model pipeline (XGBoost + PAT) based on peer-reviewed research. The system requires Apple HealthKit data to be processed into two distinct formats:
1. **36 daily features** for XGBoost mood prediction
2. **10,080-minute activity sequences** for PAT transformer analysis

## Model Requirements

### XGBoost Model (Seoul National Study)
**Paper**: "Accurately predicting mood episodes using wearable sleep and circadian rhythm features" (Nature Digital Medicine, 2024)

#### Required Features (36 total):
1. **Sleep Indexes (10 base features)**:
   - Sleep amplitude (coefficient of variation of wake amounts)
   - Sleep percentage (daily fraction of sleep period)
   - Sleep window analysis:
     - Long window count (>3.75h)
     - Long window length
     - Long window sleep time
     - Long window wake time
     - Short window count (<3.75h)
     - Short window length
     - Short window sleep time
     - Short window wake time

2. **Circadian Indexes (2 base features)**:
   - Circadian phase (DLMO estimate from mathematical modeling)
   - Circadian amplitude (CBT rhythm strength)

3. **Feature Processing**:
   - Each of the 12 base features generates:
     - Mean (MN)
     - Standard deviation (SD)
     - Z-score
   - Total: 12 × 3 = 36 features

#### Key Implementation Details:
- Sleep windows are aggregated if <1h apart
- Sleep window date assignment based on midpoint relative to midnight
- Circadian phase estimated by subtracting 7h from CBT minimum
- Light profile assumed: 250 lux when awake, 0 lux when sleeping

### PAT Model (Dartmouth Study)
**Paper**: "AI Foundation Models for Wearable Movement Data" (2024)

#### Required Input:
- **Sequence Length**: 10,080 minutes (7 days × 24h × 60m)
- **Data Format**: Minute-level activity intensity values
- **Patch Size**: 18 minutes per patch
- **Total Patches**: 560 tokens (10,080 ÷ 18)

#### Key Implementation Details:
- Activity intensity can be derived from step counts or active energy
- Missing minutes filled with zeros
- Standardization required (z-scores relative to dataset mean/std)
- Model expects continuous 7-day sequences

## Current Implementation Status

### ✅ Completed Components

1. **Data Parsing**:
   - XML/JSON parsers for Apple HealthKit
   - Sleep states (including new Apple Health values)
   - Activity records with time bounds
   - Heart rate and HRV data

2. **Daily Aggregation**:
   - `SleepAggregator`: Total hours, efficiency, fragmentation
   - `ActivityAggregator`: Total steps, active hours, peak activity
   - `HeartRateAggregator`: Resting HR, HRV (avg & min)

3. **Feature Framework**:
   - `AdvancedFeatureEngineer` with 36-feature structure
   - Basic circadian calculations (simplified)
   - Temporal features (7-day rolling windows)
   - Z-score normalization

### ❌ Missing Components

#### For XGBoost:
1. **Sleep Window Analysis**:
   - Need to implement 3.75h threshold classification
   - Window aggregation logic (<1h apart)
   - Date assignment based on midpoint

2. **Advanced Circadian Features**:
   - Interdaily Stability (IS) - currently simplified
   - Intradaily Variability (IV) - currently simplified
   - Relative Amplitude (RA) - using sleep efficiency proxy
   - L5/M10 activity levels - need hourly resolution

3. **Mathematical Circadian Modeling**:
   - Light profile generation from sleep/wake
   - Core Body Temperature (CBT) rhythm simulation
   - DLMO estimation

#### For PAT:
1. **Activity Sequence Extraction**:
   - No minute-level data extraction implemented
   - No time-bounded to minute array conversion
   - No activity distribution logic

2. **Sequence Assembly**:
   - No 7-day rolling window generation
   - No patching mechanism
   - No standardization pipeline

## Raw Data Formats

### Apple Health XML Export (`export.xml`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData>
<HealthData locale="en_US">
  <ExportDate value="2024-01-20 10:00:00 -0800"/>
  
  <!-- Sleep Records -->
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" 
          sourceName="Apple Watch" 
          value="HKCategoryValueSleepAnalysisAsleepCore"
          startDate="2024-01-01 23:00:00 -0800" 
          endDate="2024-01-02 03:00:00 -0800"
          creationDate="2024-01-02 03:00:00 -0800"/>
  
  <!-- Newer Apple Health Sleep States -->
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          value="HKCategoryValueSleepAnalysisAsleepREM"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          value="HKCategoryValueSleepAnalysisAsleepDeep"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          value="HKCategoryValueSleepAnalysisAsleepUnspecified"/>
  
  <!-- Activity Records -->
  <Record type="HKQuantityTypeIdentifierStepCount" 
          sourceName="iPhone" 
          value="1000" 
          startDate="2024-01-01 10:00:00 -0800" 
          endDate="2024-01-01 11:00:00 -0800"
          unit="count"/>
  
  <Record type="HKQuantityTypeIdentifierActiveEnergyBurned"
          value="45.5"
          unit="kcal"/>
  
  <Record type="HKQuantityTypeIdentifierFlightsClimbed"
          value="10"
          unit="count"/>
  
  <!-- Heart Rate Records -->
  <Record type="HKQuantityTypeIdentifierHeartRate" 
          sourceName="Apple Watch" 
          value="65" 
          startDate="2024-01-01 06:00:00 -0800" 
          endDate="2024-01-01 06:01:00 -0800"
          unit="count/min">
    <MetadataEntry key="HKMetadataKeyHeartRateMotionContext" 
                   value="HKHeartRateMotionContextSedentary"/>
  </Record>
  
  <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
          value="45"
          unit="ms"/>
</HealthData>
```

### Health Auto Export JSON Format

```json
// Sleep Analysis.json
[
  {
    "startDate": "2024-01-01 23:00:00",
    "endDate": "2024-01-02 07:00:00",
    "value": "Core",
    "sourceName": "Apple Watch",
    "duration": 480  // minutes
  }
]

// Step Count.json
[
  {
    "date": "2024-01-01",
    "qty": 8543,
    "unit": "count",
    "source": "iPhone"
  }
]

// Heart Rate.json
[
  {
    "date": "2024-01-01 06:00:00",
    "qty": 65,
    "unit": "count/min",
    "source": "Apple Watch"
  }
]
```

## Data Extraction Requirements

### For XGBoost (Daily Features)

| Data Type | What We Need | Current Status |
|-----------|--------------|----------------|
| **Sleep** | Start/end times, sleep states, fragmentation | ✅ Parsed |
| **Sleep Windows** | Episodes >10min, <1h gaps merged, 3.75h threshold | ❌ Not implemented |
| **Activity** | Hourly step counts for L5/M10 | ⚠️ Have daily only |
| **Heart Rate** | Resting HR, HRV values | ✅ Parsed |
| **Circadian** | Light profile → CBT model → DLMO | ❌ Not implemented |

### For PAT (Minute Sequences)

| Data Type | What We Need | Current Status |
|-----------|--------------|----------------|
| **Activity** | Steps/energy per minute for 7 days | ❌ Not implemented |
| **Time Resolution** | 10,080 data points (1 per minute) | ❌ Not implemented |
| **Distribution** | Algorithm to spread activity across time bounds | ❌ Not implemented |

## Current Processor Architecture

### 1. Parser Layer (`infrastructure/parsers/`)

```python
# XML Parsing Flow
SleepParser.parse_to_entities(xml) → List[SleepRecord]
ActivityParser.parse_to_entities(xml) → List[ActivityRecord]
HeartRateParser.parse_to_entities(xml) → List[HeartRateRecord]

# Domain Entities Created
SleepRecord(
    source_name="Apple Watch",
    start_date=datetime(2024,1,1,23,0),
    end_date=datetime(2024,1,2,7,0),
    state=SleepState.ASLEEP_CORE
)

ActivityRecord(
    source_name="iPhone",
    start_date=datetime(2024,1,1,10,0),
    end_date=datetime(2024,1,1,11,0),
    activity_type=ActivityType.STEP_COUNT,
    value=1000.0,
    unit="count"
)
```

### 2. Aggregation Layer (`domain/services/`)

```python
# Daily Aggregation
SleepAggregator.aggregate_daily(records) → Dict[date, DailySleepSummary]
ActivityAggregator.aggregate_daily(records) → Dict[date, DailyActivitySummary]
HeartRateAggregator.aggregate_daily(records) → Dict[date, DailyHeartSummary]

# Daily Summary Example
DailySleepSummary(
    date=date(2024,1,2),
    total_sleep_hours=8.0,
    sleep_efficiency=0.85,
    sleep_fragmentation_index=0.15,
    earliest_bedtime=time(23,0),
    latest_wake_time=time(7,0),
    is_clinically_significant=False
)
```

### 3. Feature Engineering Layer

```python
# Current Implementation
FeatureExtractionService.extract_features() → Dict[date, ClinicalFeatures]
FeatureExtractionService.extract_advanced_features() → Dict[date, AdvancedFeatures]

# 36 Features Generated
AdvancedFeatures.to_ml_features() → np.array(shape=(36,))
```

## Processing Gaps & Implementation Plan

### Gap 1: Sleep Window Analysis
**Current**: Simple sleep duration calculation
**Needed**: Window classification with 3.75h threshold
```python
# TODO(gh-107): Implement in SleepAggregator
def _create_sleep_windows(self, records: List[SleepRecord]) -> List[SleepWindow]:
    # 1. Merge episodes <10min apart
    # 2. Group windows <1h apart
    # 3. Classify as long (>3.75h) or short
    # 4. Assign date based on midpoint
```

### Gap 2: Minute-Level Activity
**Current**: Time-bounded records (e.g., 1000 steps from 10-11am)
**Needed**: 10,080 minute array
```python
# TODO(gh-108): Create new ActivitySequenceExtractor
def extract_minute_sequence(self, records: List[ActivityRecord], days: int = 7) -> np.array:
    # 1. Create empty 10,080 minute array
    # 2. Distribute each record's value across its time bounds
    # 3. Handle overlaps and gaps
    # 4. Return standardized sequence
```

### Gap 3: Circadian Modeling
**Current**: Simple phase calculation
**Needed**: Mathematical CBT model
```python
# TODO(gh-109): Implement CircadianPacemakerModel
def estimate_circadian_rhythm(self, sleep_wake_pattern: List[bool]) -> CircadianMetrics:
    # 1. Convert sleep/wake to light profile (250/0 lux)
    # 2. Run pacemaker differential equations
    # 3. Extract CBT rhythm
    # 4. Calculate DLMO (CBTmin - 7 hours)
```

## Test Status (2025-07-16)

### Current Test Results
- **Total Tests**: 170
- **Passing**: 165
- **Failing**: 1 (requires real data files)
- **Skipped**: 4 (large files / missing JSON)
- **Warnings**: 5 (division by zero in IV calculation)

### Fixed Issues
- ✅ HRV clinical note detection (added min_hrv tracking)
- ✅ Heart rate parser test (corrected motion context format)
- ✅ Linting issues (Union → |, whitespace cleanup)
- ⚠️ Type checking still has errors in advanced_feature_engineering.py

## Next Implementation Steps

### Priority 1: Sleep Window Analysis
The Seoul study requires sleep window analysis with specific rules:
1. Merge sleep episodes <10 minutes apart
2. Group windows <1 hour apart  
3. Classify as long (>3.75h) or short (<3.75h)
4. Track window count, length, sleep time, wake time

### Priority 2: Minute-Level Activity for PAT
PAT needs 10,080 minute-level activity values:
1. Create ActivitySequenceExtractor service
2. Distribute time-bounded records across minutes
3. Handle overlaps and gaps
4. Standardize output

### Priority 3: Complete Circadian Features
Implement proper calculations for:
- Interdaily Stability (IS)
- Intradaily Variability (IV) 
- Relative Amplitude (RA)
- L5/M10 activity levels

## Version History

- **v0.1** (2025-07-16): Initial dossier creation
- **v0.2** (2025-07-16): Added raw data formats and processor details
- **v0.3** (2025-07-16): Updated test status and implementation priorities

---
*This is a living document - update as implementation progresses*

## Implementation Priorities

### Phase 1: Complete XGBoost Pipeline
1. Implement sleep window analysis
2. Add mathematical circadian modeling
3. Refine IS/IV/RA calculations
4. Add L5/M10 with hourly aggregation

### Phase 2: Build PAT Pipeline
1. Create minute-level activity extractor
2. Implement activity distribution algorithm
3. Build 7-day sequence assembler
4. Add standardization and patching

### Phase 3: Integration
1. Dual model orchestrator
2. Ensemble prediction logic
3. API endpoints

## Test Coverage Requirements

### Unit Tests:
- Sleep window classification (3.75h threshold)
- Circadian phase calculation validation
- Activity distribution across minutes
- Sequence assembly correctness

### Integration Tests:
- 36-feature vector generation
- 10,080-minute sequence generation
- End-to-end pipeline validation

## References

1. Seoul National Study: [Nature Digital Medicine 2024]
2. Dartmouth PAT Study: [AI Foundation Models 2024]
3. Apple HealthKit Documentation
4. MATLAB Reference: `Index_calculation.m` in `reference_repos/mood_ml`

---
*Last Updated: 2025-07-16*
*Status: Active Development*