# 🚀 Big Mood Detector - Clean Architecture Refactoring Roadmap

## 📋 Executive Summary

This roadmap addresses critical architectural violations in the Big Mood Detector codebase. We have identified 4 god classes (>700 lines each) that violate SOLID principles and need immediate decomposition.

## 🎯 Refactoring Goals

1. **Achieve Single Responsibility Principle** - Each class should have one reason to change
2. **Reduce class sizes to <200 lines** - Improve maintainability and testability
3. **Apply design patterns** - Strategy, Factory, Observer, Command where appropriate
4. **Maintain 100% backward compatibility** - All existing tests must pass
5. **Follow TDD approach** - Write tests first for new extracted classes

## 📊 Current State Analysis

### Critical Files Requiring Refactoring

| File | Current Size | Target Size | Violations |
|------|-------------|-------------|------------|
| `clinical_interpreter.py` | 929 lines | <200 lines | God class, 7+ responsibilities |
| `mood_prediction_pipeline.py` | 957 lines | <150 lines | Orchestration monster |
| `advanced_feature_engineering.py` | 659 lines | <150 lines | Feature factory explosion |
| `sparse_data_handler.py` | 709 lines | <200 lines | Multiple strategies mixed |

## 🔨 Phase 1: Clinical Interpreter Decomposition (Week 1)

### Current State
```
ClinicalInterpreter (929 lines)
├── DSM-5 Evaluation
├── Risk Assessment
├── Biomarker Interpretation
├── Treatment Recommendations
├── Early Warning Detection
├── Confidence Adjustment
└── Trend Analysis
```

### Target State
```
ClinicalDecisionEngine (Facade) <150 lines
├── DSM5CriteriaEvaluator <150 lines
├── RiskLevelAssessor <150 lines
├── BiomarkerInterpreter (existing) ✓
├── TreatmentRecommender (existing) ✓
├── EarlyWarningDetector <150 lines
├── ConfidenceAdjuster <100 lines
└── TrendAnalyzer <100 lines
```

### Implementation Steps

#### 1.1 Extract DSM5CriteriaEvaluator
```python
class DSM5CriteriaEvaluator:
    """Evaluates mood episodes against DSM-5 criteria."""
    
    def evaluate_episode_duration(self, episode_type, days, hospitalization)
    def evaluate_symptom_count(self, symptoms, episode_type)
    def evaluate_functional_impairment(self, severity_indicators)
    def generate_dsm5_summary(self, evaluation_results)
```

#### 1.2 Extract RiskLevelAssessor
```python
class RiskLevelAssessor:
    """Determines clinical risk levels based on multiple factors."""
    
    def assess_depression_risk(self, phq_score, biomarkers)
    def assess_mania_risk(self, asrm_score, biomarkers)
    def assess_mixed_state_risk(self, combined_scores)
    def calculate_composite_risk(self, all_factors)
```

#### 1.3 Extract EarlyWarningDetector
```python
class EarlyWarningDetector:
    """Detects early warning signs of mood episodes."""
    
    def detect_depression_warnings(self, sleep_changes, activity_changes)
    def detect_mania_warnings(self, sleep_reduction, activity_increase)
    def evaluate_warning_severity(self, warning_signs, duration)
    def trigger_intervention_rules(self, warnings, consecutive_days)
```

## 🔧 Phase 2: Feature Engineering Refactoring (Week 2)

### Current State
```
AdvancedFeatureEngineer (659 lines)
├── Sleep Features (150+ lines)
├── Circadian Features (200+ lines)
├── Activity Features (150+ lines)
└── Temporal Features (100+ lines)
```

### Target State
```
FeatureEngineeringOrchestrator (Facade) <100 lines
├── SleepFeatureCalculator <150 lines
├── CircadianFeatureCalculator <150 lines
├── ActivityFeatureCalculator <150 lines
├── TemporalFeatureCalculator <100 lines
└── FeatureNormalizer <100 lines
```

### Strategy Pattern Implementation
```python
from abc import ABC, abstractmethod

class FeatureCalculationStrategy(ABC):
    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> dict[str, float]:
        pass

class SleepFeatureCalculator(FeatureCalculationStrategy):
    def calculate_features(self, data: pd.DataFrame) -> dict[str, float]:
        return {
            'sleep_duration_mean': self._calculate_duration_mean(data),
            'sleep_efficiency': self._calculate_efficiency(data),
            'sleep_fragmentation': self._calculate_fragmentation(data),
            # ... other sleep features
        }
```

## 🚀 Phase 3: Pipeline Orchestration Cleanup (Week 3)

### Current State
```
MoodPredictionPipeline (957 lines)
├── Data Parsing Logic
├── Aggregation Logic
├── Feature Engineering
├── ML Model Invocation
└── Result Formatting
```

### Target State
```
MoodPredictionOrchestrator <150 lines
├── DataParsingService <100 lines
├── AggregationPipeline <150 lines
├── FeatureEngineeringOrchestrator (from Phase 2)
├── MLModelExecutor <100 lines
└── ResultFormatter <50 lines
```

### Dependency Injection Pattern
```python
class MoodPredictionOrchestrator:
    def __init__(
        self,
        parser: DataParsingService,
        aggregator: AggregationPipeline,
        feature_engineer: FeatureEngineeringOrchestrator,
        ml_executor: MLModelExecutor,
        formatter: ResultFormatter
    ):
        self.parser = parser
        self.aggregator = aggregator
        self.feature_engineer = feature_engineer
        self.ml_executor = ml_executor
        self.formatter = formatter
```

## 🎨 Phase 4: Sparse Data Handler Strategies (Week 4)

### Current State
```
SparseDataHandler (709 lines)
├── Multiple Interpolation Methods
├── Data Density Analysis
├── Alignment Algorithms
└── Quality Scoring
```

### Target State
```
DataDensityManager <100 lines
├── InterpolationStrategy (Abstract)
│   ├── LinearInterpolator
│   ├── SplineInterpolator
│   └── ForwardFillInterpolator
├── DensityAnalyzer <100 lines
├── DataAligner <100 lines
└── QualityScorer <100 lines
```

## 🏗️ Phase 5: Design Pattern Implementation (Week 5)

### 5.1 Factory Pattern for Recommendations
```python
class RecommendationFactory:
    @staticmethod
    def create_recommendation(
        episode_type: str,
        severity: str
    ) -> TreatmentRecommendation:
        if episode_type == "manic":
            return ManicRecommendationBuilder().build(severity)
        elif episode_type == "depressive":
            return DepressiveRecommendationBuilder().build(severity)
```

### 5.2 Observer Pattern for Thresholds
```python
class ThresholdMonitor(Subject):
    def __init__(self):
        self._observers = []
        self._thresholds = {}
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def notify(self, threshold_event):
        for observer in self._observers:
            observer.update(threshold_event)
```

### 5.3 Command Pattern for ML Models
```python
class MLCommand(ABC):
    @abstractmethod
    def execute(self, features: pd.DataFrame) -> MoodPrediction:
        pass

class XGBoostPredictCommand(MLCommand):
    def __init__(self, model):
        self.model = model
    
    def execute(self, features: pd.DataFrame) -> MoodPrediction:
        return self.model.predict(features)
```

## 📈 Success Metrics

### Code Quality Metrics
- [ ] All classes < 200 lines
- [ ] Cyclomatic complexity < 10 per method
- [ ] Test coverage > 90%
- [ ] No methods > 50 lines

### Architecture Metrics
- [ ] Single Responsibility achieved for all classes
- [ ] Dependency injection used throughout
- [ ] Design patterns properly implemented
- [ ] Clean separation of concerns

### Performance Metrics
- [ ] No performance regression
- [ ] Memory usage stable or improved
- [ ] Processing speed maintained

## 🚦 Implementation Guidelines

### 1. Test-Driven Development
- Write tests for new classes FIRST
- Ensure existing tests still pass
- Add integration tests for facades

### 2. Incremental Refactoring
- One class at a time
- Maintain backward compatibility
- Use adapter pattern if needed

### 3. Code Review Process
- Each extraction requires review
- Performance benchmarks before/after
- Documentation updates required

### 4. Rollback Strategy
- Keep old classes until migration complete
- Feature flags for gradual rollout
- Parallel testing of old vs new

## 📝 Documentation Requirements

For each refactored component:
1. Update class docstrings
2. Add architecture decision records (ADRs)
3. Update dependency diagrams
4. Create migration guides

## 🎯 Definition of Done

A refactoring phase is complete when:
- [ ] All target classes created and tested
- [ ] Original god class replaced with facade
- [ ] All existing tests pass
- [ ] New unit tests achieve >90% coverage
- [ ] Performance benchmarks show no regression
- [ ] Documentation updated
- [ ] Code review approved

## 📅 Timeline

| Phase | Duration | Dependencies | Risk Level |
|-------|----------|--------------|------------|
| Phase 1 | 1 week | None | High (core functionality) |
| Phase 2 | 1 week | Phase 1 | Medium |
| Phase 3 | 1 week | Phase 2 | High (main pipeline) |
| Phase 4 | 1 week | None | Low |
| Phase 5 | 1 week | Phases 1-4 | Low |

## 🚨 Risk Mitigation

### High Risk Areas
1. **Clinical Interpreter** - Core business logic
   - Mitigation: Extensive testing, gradual migration
2. **Mood Pipeline** - Main execution path
   - Mitigation: Feature flags, parallel execution

### Medium Risk Areas
1. **Feature Engineering** - ML model dependencies
   - Mitigation: Validate feature parity

### Low Risk Areas
1. **Design Patterns** - New abstractions
   - Mitigation: Optional initially

## ✅ Next Steps

1. Review and approve this roadmap
2. Set up feature flags for gradual rollout
3. Begin Phase 1.1: Extract DSM5CriteriaEvaluator
4. Create ADR for architectural decisions
5. Set up performance benchmarking

---

*This is a living document. Update as refactoring progresses.*