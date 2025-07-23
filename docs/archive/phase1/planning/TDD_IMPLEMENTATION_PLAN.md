# TDD Implementation Plan: Fixing XGBoost Feature Pipeline
Generated: 2025-07-23

## Overview

We'll fix the XGBoost feature generation bug using Test-Driven Development (TDD) with professional best practices.

## TDD Cycle: Red → Green → Refactor

### Phase 1: RED (Write Failing Tests)

#### Test 1: Verify Seoul Feature Generation
```python
# tests/integration/test_seoul_feature_generation.py
import pytest
from datetime import date, timedelta
import numpy as np

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
    DailyFeatures
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord


class TestSeoulFeatureGeneration:
    """Test that AggregationPipeline generates correct Seoul features."""
    
    def test_generates_all_36_seoul_features(self, sample_health_data):
        """Verify all 36 Seoul features are generated with correct names."""
        # Given: 30 days of health data
        pipeline = AggregationPipeline()
        
        # When: Generate features
        daily_features = pipeline.aggregate_daily_features(
            sleep_records=sample_health_data['sleep'],
            activity_records=sample_health_data['activity'],
            heart_records=sample_health_data['heart'],
            start_date=date(2025, 6, 1),
            end_date=date(2025, 6, 30)
        )
        
        # Then: Should have features for each day
        assert len(daily_features) > 0
        
        # And: Each day should have all 36 Seoul features
        first_day_features = daily_features[0]
        feature_dict = first_day_features.to_dict()
        
        expected_features = [
            # Sleep percentage (3)
            "sleep_percentage_MN", "sleep_percentage_SD", "sleep_percentage_Z",
            # Sleep amplitude (3)
            "sleep_amplitude_MN", "sleep_amplitude_SD", "sleep_amplitude_Z",
            # Long sleep windows (12)
            "long_num_MN", "long_num_SD", "long_num_Z",
            "long_len_MN", "long_len_SD", "long_len_Z",
            "long_ST_MN", "long_ST_SD", "long_ST_Z",
            "long_WT_MN", "long_WT_SD", "long_WT_Z",
            # Short sleep windows (12)
            "short_num_MN", "short_num_SD", "short_num_Z",
            "short_len_MN", "short_len_SD", "short_len_Z",
            "short_ST_MN", "short_ST_SD", "short_ST_Z",
            "short_WT_MN", "short_WT_SD", "short_WT_Z",
            # Circadian (6)
            "circadian_amplitude_MN", "circadian_amplitude_SD", "circadian_amplitude_Z",
            "circadian_phase_MN", "circadian_phase_SD", "circadian_phase_Z"
        ]
        
        for feature in expected_features:
            assert feature in feature_dict, f"Missing Seoul feature: {feature}"
        
        # Verify exactly 36 features (plus date)
        assert len(feature_dict) == 37  # 36 features + date

    def test_seoul_features_have_valid_values(self, sample_health_data):
        """Verify Seoul features have reasonable values."""
        # Given: Health data
        pipeline = AggregationPipeline()
        
        # When: Generate features
        daily_features = pipeline.aggregate_daily_features(
            sleep_records=sample_health_data['sleep'],
            activity_records=sample_health_data['activity'],
            heart_records=sample_health_data['heart'],
            start_date=date(2025, 6, 1),
            end_date=date(2025, 6, 30)
        )
        
        # Then: Values should be reasonable
        feature_dict = daily_features[0].to_dict()
        
        # Sleep percentage should be between 0 and 1
        assert 0 <= feature_dict["sleep_percentage_MN"] <= 1
        
        # Standard deviations should be non-negative
        assert feature_dict["sleep_percentage_SD"] >= 0
        
        # Z-scores should typically be between -3 and 3
        assert -5 <= feature_dict["sleep_percentage_Z"] <= 5
```

#### Test 2: XGBoost Model Integration
```python
# tests/integration/test_xgboost_prediction_pipeline.py
import pytest
import numpy as np

from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader
from big_mood_detector.application.services.aggregation_pipeline import DailyFeatures


class TestXGBoostPredictionPipeline:
    """Test XGBoost models work with AggregationPipeline features."""
    
    def test_xgboost_accepts_aggregation_features(self, sample_daily_features):
        """Verify XGBoost can predict using DailyFeatures."""
        # Given: DailyFeatures from AggregationPipeline
        daily_features: DailyFeatures = sample_daily_features
        feature_dict = daily_features.to_dict()
        
        # And: Loaded XGBoost models
        xgboost_loader = XGBoostModelLoader()
        xgboost_loader.load_all_models(Path("model_weights/xgboost"))
        
        # When: Convert to array and predict
        feature_array = xgboost_loader.dict_to_array(feature_dict)
        prediction = xgboost_loader.predict(feature_array)
        
        # Then: Should get valid prediction
        assert prediction is not None
        assert 0 <= prediction.depression_risk <= 1
        assert 0 <= prediction.hypomanic_risk <= 1
        assert 0 <= prediction.manic_risk <= 1
        assert 0 <= prediction.confidence <= 1
        
    def test_xgboost_fails_with_clinical_features(self):
        """Verify XGBoost fails with wrong features (current bug)."""
        # Given: Clinical features (wrong type)
        from big_mood_detector.domain.services.clinical_feature_extractor import (
            ClinicalFeatureExtractor
        )
        
        extractor = ClinicalFeatureExtractor()
        clinical_features = extractor.extract_seoul_features(
            sleep_records=[],  # Would need real data
            activity_records=[],
            heart_records=[],
            target_date=date.today()
        )
        
        # When: Try to predict
        xgboost_loader = XGBoostModelLoader()
        xgboost_loader.load_all_models(Path("model_weights/xgboost"))
        
        # Then: Should fail (this demonstrates the bug)
        with pytest.raises(Exception) as exc_info:
            feature_array = clinical_features.to_xgboost_features()
            xgboost_loader.predict(np.array(feature_array))
        
        assert "missing fields" in str(exc_info.value).lower()
```

#### Test 3: End-to-End Prediction Test
```python
# tests/integration/test_end_to_end_prediction.py
class TestEndToEndPrediction:
    """Test complete prediction pipeline with real data structure."""
    
    def test_prediction_pipeline_with_aggregation(self, real_health_data_path):
        """Test full pipeline: Data → Aggregation → XGBoost → Prediction."""
        # Given: Real health data
        from big_mood_detector.application.use_cases.process_health_data_use_case import (
            MoodPredictionPipeline,
            PipelineConfig
        )
        
        config = PipelineConfig(
            include_pat_sequences=False,  # Test XGBoost only
            use_aggregation_pipeline=True  # NEW: Force correct pipeline
        )
        
        pipeline = MoodPredictionPipeline(config=config)
        
        # When: Process and predict
        result = pipeline.process_apple_health_file(
            file_path=real_health_data_path,
            start_date=date(2025, 6, 1),
            end_date=date(2025, 6, 30)
        )
        
        # Then: Should get valid predictions
        assert result.daily_predictions is not None
        assert len(result.daily_predictions) > 0
        
        # And: Predictions should be reasonable
        for date, prediction in result.daily_predictions.items():
            assert 0 <= prediction['depression_risk'] <= 1
            assert prediction['depression_risk'] != 0.5  # Not default
```

### Phase 2: GREEN (Make Tests Pass)

#### Fix 1: Update PipelineConfig
```python
# src/big_mood_detector/application/use_cases/process_health_data_use_case.py
@dataclass
class PipelineConfig:
    """Configuration for mood prediction pipeline."""
    include_pat_sequences: bool = False
    min_days_required: int = 1
    enable_personal_calibration: bool = True
    user_id: str | None = None
    use_aggregation_pipeline: bool = True  # NEW: Control feature generation
```

#### Fix 2: Update MoodPredictionPipeline
```python
class MoodPredictionPipeline:
    def _create_predictor(self):
        """Create appropriate predictor based on config."""
        if self.config.include_pat_sequences and PAT_AVAILABLE:
            # Ensemble mode
            return self._create_ensemble_predictor()
        else:
            # XGBoost only - use aggregation pipeline
            if self.config.use_aggregation_pipeline:
                return self._create_aggregation_based_predictor()
            else:
                # Legacy mode (will fail with current models)
                return self._create_clinical_based_predictor()
    
    def _create_aggregation_based_predictor(self):
        """Create predictor using AggregationPipeline (correct for XGBoost)."""
        from big_mood_detector.application.services.aggregation_pipeline import (
            AggregationPipeline
        )
        
        self.aggregation_pipeline = AggregationPipeline()
        self.xgboost_loader = XGBoostModelLoader()
        self.xgboost_loader.load_all_models(self.model_dir)
        
        return self._predict_with_aggregation
    
    def _predict_with_aggregation(self, records, target_date):
        """Predict using aggregation pipeline features."""
        # Generate Seoul features
        daily_features = self.aggregation_pipeline.aggregate_daily_features(
            sleep_records=records['sleep'],
            activity_records=records['activity'],
            heart_records=records.get('heart', []),
            start_date=target_date - timedelta(days=30),
            end_date=target_date
        )
        
        if not daily_features:
            return None
            
        # Get latest features
        latest_features = daily_features[-1]
        feature_dict = latest_features.to_dict()
        
        # Predict
        feature_array = self.xgboost_loader.dict_to_array(feature_dict)
        return self.xgboost_loader.predict(feature_array)
```

### Phase 3: REFACTOR (Clean Up)

#### Refactoring 1: Extract Feature Generation Strategy
```python
# src/big_mood_detector/application/services/feature_generation_strategy.py
from abc import ABC, abstractmethod

class FeatureGenerationStrategy(ABC):
    """Abstract strategy for feature generation."""
    
    @abstractmethod
    def generate_features(self, records, target_date):
        """Generate features for prediction."""
        pass

class AggregationFeatureStrategy(FeatureGenerationStrategy):
    """Generate Seoul statistical features via aggregation."""
    
    def __init__(self):
        self.pipeline = AggregationPipeline()
    
    def generate_features(self, records, target_date):
        daily_features = self.pipeline.aggregate_daily_features(
            sleep_records=records['sleep'],
            activity_records=records['activity'],
            heart_records=records.get('heart', []),
            start_date=target_date - timedelta(days=30),
            end_date=target_date
        )
        return daily_features[-1].to_dict() if daily_features else None

class ClinicalFeatureStrategy(FeatureGenerationStrategy):
    """Generate clinical interpretation features."""
    
    def __init__(self):
        self.extractor = ClinicalFeatureExtractor()
    
    def generate_features(self, records, target_date):
        # Implementation for clinical features
        pass
```

## Professional Best Practices Applied

### 1. SOLID Principles
- **S**: Single Responsibility - Each class has one job
- **O**: Open/Closed - Strategies can be added without modifying existing code
- **L**: Liskov Substitution - Strategies are interchangeable
- **I**: Interface Segregation - Clean, focused interfaces
- **D**: Dependency Inversion - Depend on abstractions

### 2. Design Patterns
- **Strategy Pattern**: For feature generation
- **Factory Pattern**: For predictor creation
- **Repository Pattern**: Already in use

### 3. Testing Best Practices
- **Arrange-Act-Assert**: Clear test structure
- **Test Isolation**: Each test independent
- **Meaningful Names**: Tests describe behavior
- **Edge Cases**: Test boundaries and failures

### 4. Code Quality
- **Type Hints**: Full typing throughout
- **Documentation**: Clear docstrings
- **Error Handling**: Graceful failures
- **Logging**: Structured logging for debugging

## Implementation Schedule

### Day 1: Write Tests (2-3 hours)
1. Create all failing tests
2. Verify tests fail for right reasons
3. Document expected behavior

### Day 2: Implementation (3-4 hours)
1. Add configuration option
2. Implement aggregation-based predictor
3. Make tests pass

### Day 3: Refactoring (2-3 hours)
1. Extract strategies
2. Clean up code
3. Update documentation

### Day 4: Integration Testing (2 hours)
1. Test with real data
2. Verify performance
3. Update user-facing docs

## Success Criteria

1. All tests pass
2. XGBoost predictions work without errors
3. No regression in existing functionality
4. Code coverage remains >90%
5. Documentation updated

## Risk Mitigation

1. **Feature flag**: Can toggle between pipelines
2. **Backwards compatibility**: Old pipeline still available
3. **Comprehensive tests**: Catch any regressions
4. **Staged rollout**: Test with subset of users first

This TDD approach ensures we fix the bug correctly while maintaining code quality and preventing regressions.