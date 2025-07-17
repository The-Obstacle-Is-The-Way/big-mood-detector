# Design Patterns Catalog - Big Mood Detector

## Overview
This catalog documents all design patterns implemented during the refactoring, with concrete examples and rationale.

## 🏭 Creational Patterns

### Factory Pattern
**Purpose**: Create objects without specifying exact classes

#### Implementation: InterpolationStrategyFactory
```python
class InterpolationStrategyFactory:
    @staticmethod
    def create_strategy(method: str) -> InterpolationStrategy:
        strategies = {
            "linear": LinearInterpolationStrategy,
            "forward_fill": ForwardFillInterpolationStrategy,
            "circadian_spline": CircadianSplineInterpolationStrategy,
            "none": NoInterpolationStrategy,
        }
        return strategies[method]()
```

#### Implementation: ParserFactory (in DataParsingService)
```python
def get_parser_for_path(self, file_path: Path) -> HealthDataParser:
    if file_path.suffix == ".xml":
        return self._xml_parser
    elif file_path.is_dir():
        return self._json_parser
```

### Dependency Injection
**Purpose**: Inject dependencies rather than creating them

#### Example: MoodPredictionPipeline
```python
def __init__(
    self,
    data_parsing_service: DataParsingService | None = None,
    aggregation_pipeline: AggregationPipeline | None = None,
):
    self.data_parsing_service = data_parsing_service or DataParsingService()
    self.aggregation_pipeline = aggregation_pipeline or AggregationPipeline()
```

## 🌉 Structural Patterns

### Facade Pattern
**Purpose**: Provide simplified interface to complex subsystem

#### Implementation: ClinicalDecisionEngine
```python
class ClinicalDecisionEngine:
    """Facade for clinical decision-making"""
    
    def assess_patient(self, features: FeatureSet) -> ClinicalAssessment:
        # Coordinates multiple services behind simple interface
        dsm5_result = self.dsm5_evaluator.evaluate(features)
        risk_level = self.risk_assessor.assess(features)
        warnings = self.warning_detector.detect(features)
        
        return self._create_assessment(dsm5_result, risk_level, warnings)
```

#### Implementation: FeatureEngineeringOrchestrator
```python
class FeatureEngineeringOrchestrator:
    """Facade for feature engineering"""
    
    def extract_all_features(self, data: HealthData) -> UnifiedFeatureSet:
        sleep = self.sleep_calculator.calculate(data)
        circadian = self.circadian_calculator.calculate(data)
        activity = self.activity_calculator.calculate(data)
        temporal = self.temporal_calculator.calculate(data)
        
        return UnifiedFeatureSet(sleep, circadian, activity, temporal)
```

### Adapter Pattern
**Purpose**: Allow incompatible interfaces to work together

#### Implementation: StreamingXMLParser
```python
class StreamingXMLParser:
    """Adapts XML parsing to domain entities"""
    
    def parse_file(self, path: Path) -> Iterator[DomainEntity]:
        for event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag == "Record":
                entity = self._adapt_to_domain(elem)
                if entity:
                    yield entity
```

## 🎭 Behavioral Patterns

### Strategy Pattern
**Purpose**: Define family of algorithms, make them interchangeable

#### Implementation: InterpolationStrategy
```python
class InterpolationStrategy(Protocol):
    def interpolate(self, df: pd.DataFrame, limit: int) -> pd.DataFrame:
        ...

class LinearInterpolationStrategy:
    def interpolate(self, df: pd.DataFrame, limit: int) -> pd.DataFrame:
        return df.interpolate(method="linear", limit=limit)

class CircadianSplineInterpolationStrategy:
    def interpolate(self, df: pd.DataFrame, limit: int) -> pd.DataFrame:
        # Custom spline that respects circadian patterns
        return self._circadian_aware_spline(df, limit)
```

#### Implementation: RiskAssessmentStrategy
```python
class RiskAssessmentStrategy(Protocol):
    def assess_risk(self, indicators: ClinicalIndicators) -> RiskLevel:
        ...

class DSM5RiskStrategy:
    def assess_risk(self, indicators: ClinicalIndicators) -> RiskLevel:
        # Assess based on DSM-5 criteria
        
class MLRiskStrategy:
    def assess_risk(self, indicators: ClinicalIndicators) -> RiskLevel:
        # Assess using ML model
```

### Template Method Pattern
**Purpose**: Define skeleton of algorithm, subclasses override specific steps

#### Implementation: BaseFeatureCalculator
```python
class BaseFeatureCalculator:
    def calculate_features(self, data: HealthData) -> FeatureSet:
        # Template method
        validated = self._validate_data(data)
        raw_features = self._extract_raw_features(validated)
        normalized = self._normalize_features(raw_features)
        return self._create_feature_set(normalized)
    
    def _extract_raw_features(self, data: HealthData) -> dict:
        # Subclasses override this
        raise NotImplementedError
```

### Observer Pattern
**Purpose**: Define one-to-many dependency between objects

#### Implementation: ProgressCallback
```python
class DataParsingService:
    def parse_health_data(
        self, 
        file_path: Path,
        progress_callback: Callable[[str, float], None] | None = None
    ):
        if progress_callback:
            progress_callback("Starting parsing", 0.0)
        
        # ... parsing logic ...
        
        if progress_callback:
            progress_callback("Parsing complete", 1.0)
```

### Chain of Responsibility Pattern
**Purpose**: Pass request along chain of handlers

#### Implementation: ClinicalRuleChain
```python
class ClinicalRule(Protocol):
    def evaluate(self, context: ClinicalContext) -> RuleResult | None:
        ...

class ManicEpisodeRule:
    def __init__(self, next_rule: ClinicalRule | None = None):
        self.next_rule = next_rule
    
    def evaluate(self, context: ClinicalContext) -> RuleResult | None:
        if self._is_manic_episode(context):
            return RuleResult(type="manic", confidence=0.9)
        elif self.next_rule:
            return self.next_rule.evaluate(context)
        return None
```

### Value Object Pattern
**Purpose**: Immutable objects representing domain concepts

#### Implementation: ClinicalThreshold
```python
@dataclass(frozen=True)
class ClinicalThreshold:
    """Immutable clinical threshold"""
    name: str
    min_value: float
    max_value: float
    unit: str
    
    def is_within_range(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value
```

#### Implementation: TimeRange
```python
@dataclass(frozen=True)
class TimeRange:
    """Immutable time range"""
    start: datetime
    end: datetime
    
    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("End must be after start")
    
    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600
```

## 🔄 Integration Patterns

### Repository Pattern
**Purpose**: Encapsulate data access logic

#### Implementation: HealthDataRepository
```python
class HealthDataRepository(Protocol):
    def get_sleep_records(self, start: date, end: date) -> list[SleepRecord]:
        ...
    
    def get_activity_records(self, start: date, end: date) -> list[ActivityRecord]:
        ...
```

### Unit of Work Pattern
**Purpose**: Maintain list of objects affected by business transaction

#### Implementation: FeatureExtractionSession
```python
class FeatureExtractionSession:
    def __init__(self):
        self._extracted_features = []
        self._errors = []
    
    def add_features(self, features: FeatureSet):
        self._extracted_features.append(features)
    
    def commit(self) -> list[FeatureSet]:
        # Validate and return all features
        return self._extracted_features
```

## 📊 Pattern Benefits Matrix

| Pattern | Testability | Extensibility | Maintainability | Performance |
|---------|-------------|---------------|-----------------|-------------|
| Strategy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Facade | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Factory | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| DI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Value Object | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 When to Use Each Pattern

### Strategy Pattern
- Multiple algorithms for same purpose
- Algorithm selection at runtime
- Avoiding conditional statements

### Facade Pattern
- Simplifying complex subsystems
- Providing high-level interface
- Reducing coupling between layers

### Factory Pattern
- Object creation based on conditions
- Hiding concrete implementations
- Supporting multiple product families

### Dependency Injection
- Improving testability
- Reducing coupling
- Supporting different configurations

### Value Object
- Representing domain concepts
- Ensuring immutability
- Preventing invalid states

---
*This catalog serves as a reference for current and future developers working on the Big Mood Detector system.*