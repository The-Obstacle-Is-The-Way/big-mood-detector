"""
Unit tests for XGBoost feature validation and error handling.

These tests ensure the XGBoost pipeline produces valid feature vectors
and handles errors gracefully.
"""

import math
from datetime import date, datetime, timedelta, UTC
from unittest.mock import Mock, MagicMock
import pytest

from big_mood_detector.application.pipelines.xgboost_pipeline import (
    XGBoostPipeline,
    XGBoostResult,
)
from big_mood_detector.application.validators.pipeline_validators import XGBoostValidator
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType


class TestXGBoostFeatureValidation:
    """Test XGBoost feature vector validation."""
    
    @pytest.fixture
    def mock_feature_extractor(self) -> Mock:
        """Mock feature extractor that returns valid features."""
        mock = Mock()
        
        # Create mock Seoul features
        mock_seoul = Mock()
        mock_seoul.to_xgboost_features.return_value = [0.5] * 36  # Valid 36 features
        
        # Create mock clinical features
        mock_features = Mock()
        mock_features.seoul_features = mock_seoul
        
        mock.extract_clinical_features.return_value = mock_features
        return mock
    
    @pytest.fixture
    def mock_predictor(self) -> Mock:
        """Mock XGBoost predictor."""
        mock = Mock()
        mock.predict_mood_episodes.return_value = {
            "depression": {"probability": 0.15, "risk_level": "low"},
            "mania": {"probability": 0.98, "risk_level": "high"},  # High mania risk!
            "hypomania": {"probability": 0.45, "risk_level": "medium"},
        }
        return mock
    
    @pytest.fixture
    def pipeline(self, mock_feature_extractor: Mock, mock_predictor: Mock) -> XGBoostPipeline:
        """Create pipeline with mocks."""
        return XGBoostPipeline(
            feature_extractor=mock_feature_extractor,
            predictor=mock_predictor,
            validator=XGBoostValidator(),
        )
    
    def test_validates_feature_vector_length(
        self,
        pipeline: XGBoostPipeline,
        mock_feature_extractor: Mock,
    ) -> None:
        """Test that pipeline validates feature vector has exactly 36 features."""
        # Create invalid feature vector with wrong length
        mock_seoul = Mock()
        mock_seoul.to_xgboost_features.return_value = [0.5] * 35  # Only 35 features!
        mock_features = Mock()
        mock_features.seoul_features = mock_seoul
        mock_feature_extractor.extract_clinical_features.return_value = mock_features
        
        # Create minimal valid data
        sleep_records = [
            SleepRecord(
                source_name="Test",
                start_date=datetime(2025, 5, i+1, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 5, i+2, 6, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            )
            for i in range(30)  # 30 days
        ]
        
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2025, 5, 30),
        )
        
        # Should return None due to invalid feature vector
        assert result is None
    
    def test_detects_nan_in_features(
        self,
        pipeline: XGBoostPipeline,
        mock_feature_extractor: Mock,
    ) -> None:
        """Test that pipeline detects NaN values in feature vector."""
        # Create feature vector with NaN
        features = [0.5] * 35 + [float('nan')]
        mock_seoul = Mock()
        mock_seoul.to_xgboost_features.return_value = features
        mock_features = Mock()
        mock_features.seoul_features = mock_seoul
        mock_feature_extractor.extract_clinical_features.return_value = mock_features
        
        # Create minimal valid data
        sleep_records = [
            SleepRecord(
                source_name="Test",
                start_date=datetime(2025, 5, i+1, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 5, i+2, 6, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            )
            for i in range(30)
        ]
        
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2025, 5, 30),
        )
        
        assert result is None
    
    def test_detects_inf_in_features(
        self,
        pipeline: XGBoostPipeline,
        mock_feature_extractor: Mock,
    ) -> None:
        """Test that pipeline detects infinite values in feature vector."""
        # Create feature vector with inf
        features = [0.5] * 35 + [float('inf')]
        mock_seoul = Mock()
        mock_seoul.to_xgboost_features.return_value = features
        mock_features = Mock()
        mock_features.seoul_features = mock_seoul
        mock_feature_extractor.extract_clinical_features.return_value = mock_features
        
        # Create minimal valid data
        sleep_records = [
            SleepRecord(
                source_name="Test",
                start_date=datetime(2025, 5, i+1, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 5, i+2, 6, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            )
            for i in range(30)
        ]
        
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2025, 5, 30),
        )
        
        assert result is None
    
    def test_successful_mania_detection(
        self,
        pipeline: XGBoostPipeline,
        mock_predictor: Mock,
    ) -> None:
        """Test successful prediction with high mania risk."""
        # Create 30 days of data
        sleep_records = []
        activity_records = []
        
        for i in range(30):
            # Reduced sleep (mania indicator)
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=datetime(2025, 5, i+1, 2, 0, tzinfo=UTC),  # Late sleep
                    end_date=datetime(2025, 5, i+1, 5, 0, tzinfo=UTC),   # Only 3 hours!
                    state=SleepState.ASLEEP,
                )
            )
            
            # High activity (mania indicator)
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=datetime(2025, 5, i+1, 6, 0, tzinfo=UTC),
                    end_date=datetime(2025, 5, i+1, 23, 0, tzinfo=UTC),
                    activity_type=ActivityType.STEP_COUNT,
                    value=20000.0,  # Very high step count
                    unit="count",
                )
            )
        
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            target_date=date(2025, 5, 30),
        )
        
        assert result is not None
        assert isinstance(result, XGBoostResult)
        assert result.mania_probability == 0.98  # High mania risk!
        assert result.highest_risk_episode == "mania"
        assert "High risk for mania" in result.clinical_interpretation
        assert result.confidence_level in ["high", "medium", "low"]
        
        # Verify predictor was called with valid features
        mock_predictor.predict_mood_episodes.assert_called_once()
        call_args = mock_predictor.predict_mood_episodes.call_args
        features = call_args[1]['features']
        assert len(features) == 36
        assert all(not math.isnan(f) and not math.isinf(f) for f in features)
    
    def test_filters_records_to_date_range(
        self,
        pipeline: XGBoostPipeline,
        mock_feature_extractor: Mock,
    ) -> None:
        """Test that pipeline filters records to only the needed date range."""
        # Create 100 days of data
        all_sleep_records = []
        for i in range(100):
            all_sleep_records.append(
                SleepRecord(
                    source_name="Test",
                    start_date=datetime(2025, 4, 1, 22, 0, tzinfo=UTC) + timedelta(days=i),
                    end_date=datetime(2025, 4, 2, 6, 0, tzinfo=UTC) + timedelta(days=i),
                    state=SleepState.ASLEEP,
                )
            )
        
        # Process with target date that should only use last 60 days
        result = pipeline.process(
            sleep_records=all_sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2025, 7, 9),
        )
        
        # Check that feature extractor was called with filtered records
        mock_feature_extractor.extract_clinical_features.assert_called_once()
        call_args = mock_feature_extractor.extract_clinical_features.call_args
        filtered_sleep = call_args[1]['sleep_records']
        
        # Should have at most 60 days of records
        assert len(filtered_sleep) <= 60
        # All records should be within 60 days of target
        for record in filtered_sleep:
            days_diff = (date(2025, 7, 9) - record.start_date.date()).days
            assert days_diff < 60
    
    def test_probabilities_are_valid(
        self,
        pipeline: XGBoostPipeline,
        mock_predictor: Mock,
    ) -> None:
        """Test that all probabilities are between 0 and 1."""
        # Set up predictor with edge case probabilities
        mock_predictor.predict_mood_episodes.return_value = {
            "depression": {"probability": 0.0, "risk_level": "low"},
            "mania": {"probability": 1.0, "risk_level": "high"},
            "hypomania": {"probability": 0.5, "risk_level": "medium"},
        }
        
        # Create minimal data
        sleep_records = [
            SleepRecord(
                source_name="Test",
                start_date=datetime(2025, 5, i+1, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 5, i+2, 6, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            )
            for i in range(30)
        ]
        
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2025, 5, 30),
        )
        
        assert result is not None
        # Check all probabilities are valid
        assert 0.0 <= result.depression_probability <= 1.0
        assert 0.0 <= result.mania_probability <= 1.0
        assert 0.0 <= result.hypomania_probability <= 1.0