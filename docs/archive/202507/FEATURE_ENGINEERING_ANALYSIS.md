# AdvancedFeatureEngineering Analysis

## Current State (658 lines - God Class)

The `AdvancedFeatureEngineer` class violates Single Responsibility Principle by handling:

### 1. Sleep Feature Calculation (Lines ~256-303)
- Sleep regularity index calculation
- Interdaily stability (IS) calculation
- Intradaily variability (IV) calculation
- Relative amplitude (RA) calculation
- Sleep window percentage calculations
- Sleep timing variance calculations

### 2. Circadian Rhythm Feature Calculation (Lines ~304-369)
- L5 (Least active 5 hours) calculation
- M10 (Most active 10 hours) calculation
- Circadian phase advance/delay detection
- DLMO (Dim Light Melatonin Onset) estimation
- Core body temperature nadir estimation

### 3. Activity Pattern Feature Calculation (Lines ~370-403)
- Activity fragmentation analysis
- Sedentary bout statistics
- Activity intensity ratio calculation
- Step count distribution analysis

### 4. Temporal Feature Calculation (Lines ~439-476)
- 7-day rolling window statistics
- Mean and standard deviation tracking
- Cross-domain temporal patterns

### 5. Normalization and Baseline Management (Lines ~404-438, 593-624)
- Z-score calculations
- Individual baseline tracking
- Population baseline management
- Cross-individual normalization

### 6. Clinical Pattern Detection (Lines ~477-515)
- Hypersomnia pattern detection
- Insomnia pattern detection
- Phase shift detection
- Pattern irregularity assessment
- Composite mood risk scoring

## Proposed Refactoring Strategy

### Phase 2.2: Extract SleepFeatureCalculator
```python
class SleepFeatureCalculator:
    """Calculates sleep-specific features from daily summaries."""
    def calculate_regularity_index(self, sleep_summaries: list[DailySleepSummary]) -> float
    def calculate_interdaily_stability(self, sleep_summaries: list[DailySleepSummary]) -> float
    def calculate_intradaily_variability(self, sleep_summaries: list[DailySleepSummary]) -> float
    def calculate_relative_amplitude(self, sleep_summaries: list[DailySleepSummary]) -> float
    def calculate_sleep_window_percentages(self, sleep_summaries: list[DailySleepSummary]) -> tuple[float, float]
    def calculate_timing_variances(self, sleep_summaries: list[DailySleepSummary]) -> tuple[float, float]
```

### Phase 2.3: Extract CircadianFeatureCalculator
```python
class CircadianFeatureCalculator:
    """Calculates circadian rhythm features."""
    def calculate_l5_m10_metrics(self, activity_summaries: list[DailyActivitySummary]) -> L5M10Result
    def calculate_phase_shifts(self, sleep_summaries: list[DailySleepSummary]) -> PhaseShiftResult
    def estimate_dlmo(self, sleep_summaries: list[DailySleepSummary]) -> datetime
    def estimate_core_temp_nadir(self, sleep_summaries: list[DailySleepSummary]) -> datetime
```

### Phase 2.4: Extract ActivityFeatureCalculator
```python
class ActivityFeatureCalculator:
    """Calculates activity pattern features."""
    def calculate_fragmentation(self, activity_summaries: list[DailyActivitySummary]) -> float
    def calculate_sedentary_bouts(self, activity_summaries: list[DailyActivitySummary]) -> SedentaryBoutStats
    def calculate_intensity_ratio(self, activity_summaries: list[DailyActivitySummary]) -> float
```

### Phase 2.5: Extract TemporalFeatureCalculator
```python
class TemporalFeatureCalculator:
    """Calculates rolling window temporal features."""
    def calculate_rolling_statistics(self, values: list[float], window_days: int) -> RollingStats
    def calculate_temporal_features(self, summaries: list[Any], metrics: list[str]) -> TemporalFeatures
```

### Phase 2.6: Extract NormalizationService
```python
class NormalizationService:
    """Handles individual and population normalization."""
    def track_individual_baseline(self, metric: str, value: float) -> None
    def calculate_zscore(self, metric: str, value: float) -> float
    def update_population_baseline(self, metric: str, values: list[float]) -> None
```

### Phase 2.7: Extract ClinicalPatternDetector
```python
class ClinicalPatternDetector:
    """Detects clinical patterns from features."""
    def detect_hypersomnia(self, sleep_features: SleepFeatures) -> bool
    def detect_insomnia(self, sleep_features: SleepFeatures) -> bool
    def detect_phase_shifts(self, circadian_features: CircadianFeatures) -> PhaseShiftPattern
    def calculate_mood_risk_score(self, all_features: AdvancedFeatures) -> float
```

### Phase 2.8: Create FeatureEngineeringOrchestrator
```python
class FeatureEngineeringOrchestrator:
    """Orchestrates all feature calculators."""
    def __init__(self, 
                 sleep_calculator: SleepFeatureCalculator,
                 circadian_calculator: CircadianFeatureCalculator,
                 activity_calculator: ActivityFeatureCalculator,
                 temporal_calculator: TemporalFeatureCalculator,
                 normalization_service: NormalizationService,
                 pattern_detector: ClinicalPatternDetector):
        # Dependency injection of all services
    
    def extract_advanced_features(self, ...) -> AdvancedFeatures:
        # Orchestrate all calculators
```

## Benefits
1. **Single Responsibility**: Each calculator handles one domain
2. **Testability**: Each component can be tested in isolation
3. **Maintainability**: Changes to sleep features don't affect activity features
4. **Reusability**: Calculators can be used independently
5. **Dependency Injection**: Easy to mock for testing
6. **Strategy Pattern**: Different calculation strategies can be swapped

## Testing Strategy
- Write comprehensive tests for each calculator BEFORE extraction (TDD)
- Ensure AdvancedFeatureEngineer still works during refactoring (Facade pattern)
- Test orchestrator integration after all components extracted