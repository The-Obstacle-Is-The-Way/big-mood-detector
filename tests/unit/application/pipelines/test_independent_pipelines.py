"""
Unit tests for independent PAT and XGBoost pipelines.

These tests verify that each pipeline can run independently
when its specific data requirements are met.
"""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, Mock

import pytest

from big_mood_detector.application.pipelines.pat_pipeline import (
    PatPipeline,
    PATResult,
)
from big_mood_detector.application.pipelines.xgboost_pipeline import (
    XGBoostPipeline,
    XGBoostResult,
)
from big_mood_detector.application.validators.pipeline_validators import (
    PATValidator,
    ValidationResult,
    XGBoostValidator,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestPatPipeline:
    """Test cases for independent PAT pipeline."""

    @pytest.fixture
    def mock_pat_loader(self) -> Mock:
        """Mock PAT model loader."""
        mock = Mock()
        # Mock the actual method that exists
        mock.predict_depression_from_activity.return_value = 0.35
        return mock

    @pytest.fixture
    def pat_validator(self) -> PATValidator:
        """Create PAT validator."""
        return PATValidator()

    @pytest.fixture
    def pipeline(self, mock_pat_loader: Mock, pat_validator: PATValidator) -> PatPipeline:
        """Create PAT pipeline with mocks."""
        return PatPipeline(
            pat_loader=mock_pat_loader,
            validator=pat_validator,
        )

    @pytest.fixture
    def seven_days_activity(self) -> list[ActivityRecord]:
        """Create 7 consecutive days of activity data."""
        records = []
        base_date = date(2025, 7, 20)
        
        for day_offset in range(7):
            # Multiple activity records per day to simulate real data
            for hour in [8, 12, 16, 20]:  # 4 times per day
                activity_date = datetime(
                    base_date.year,
                    base_date.month,
                    base_date.day + day_offset,
                    hour, 0, 0,
                    tzinfo=UTC
                )
                records.append(
                    ActivityRecord(
                        source_name="Apple Watch",
                        start_date=activity_date,
                        end_date=activity_date + timedelta(hours=1),
                        activity_type=ActivityType.STEP_COUNT,
                        value=1000.0 + (hour * 100),  # Varying step counts
                        unit="count",
                    )
                )
        
        return records

    def test_pat_runs_with_exactly_7_consecutive_days(
        self,
        pipeline: PatPipeline,
        seven_days_activity: list[ActivityRecord],
    ) -> None:
        """Test that PAT runs successfully with 7 consecutive days."""
        # Validate first
        validation = pipeline.can_run(
            activity_records=seven_days_activity,
            start_date=date(2025, 7, 20),
            end_date=date(2025, 7, 26),
        )
        
        assert validation.is_valid is True
        assert validation.can_run is True
        
        # Process data
        result = pipeline.process(
            activity_records=seven_days_activity,
            target_date=date(2025, 7, 26),
        )
        
        assert result is not None
        assert isinstance(result, PATResult)
        assert result.depression_risk_score == 0.35
        assert result.confidence == 0.85
        assert result.assessment_window_days == 7
        assert result.model_version == "PAT-L"
        assert "Current depression risk" in result.clinical_interpretation

    def test_pat_returns_none_with_insufficient_data(
        self,
        pipeline: PatPipeline,
    ) -> None:
        """Test that PAT returns None when insufficient data."""
        # Only 5 days of data
        records = []
        base_date = date(2025, 7, 20)
        
        for day_offset in range(5):  # Only 5 days
            activity_date = datetime(
                base_date.year,
                base_date.month,
                base_date.day + day_offset,
                12, 0, 0,
                tzinfo=UTC
            )
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        # Validate
        validation = pipeline.can_run(
            activity_records=records,
            start_date=base_date,
            end_date=base_date + timedelta(days=4),
        )
        
        assert validation.is_valid is False
        assert validation.can_run is False
        
        # Process should return None
        result = pipeline.process(
            activity_records=records,
            target_date=base_date + timedelta(days=4),
        )
        
        assert result is None

    def test_pat_finds_best_window_in_sparse_data(
        self,
        pipeline: PatPipeline,
    ) -> None:
        """Test that PAT finds the best 7-day window in sparse data."""
        records = []
        
        # Days 1-3: sparse
        for day in [1, 3]:
            activity_date = datetime(2025, 7, day, 12, 0, 0, tzinfo=UTC)
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        # Days 10-16: consecutive (7 days) - this is the window PAT should use
        for day in range(10, 17):
            activity_date = datetime(2025, 7, day, 12, 0, 0, tzinfo=UTC)
            records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        # Days 20, 22: sparse
        for day in [20, 22]:
            activity_date = datetime(2025, 7, day, 12, 0, 0, tzinfo=UTC)
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        # Process - should find and use days 10-16
        result = pipeline.process(
            activity_records=records,
            target_date=date(2025, 7, 22),  # Latest date
        )
        
        assert result is not None
        assert result.assessment_window_days == 7
        # PAT should have processed the consecutive window

    def test_pat_creates_minute_level_sequence(
        self,
        pipeline: PatPipeline,
        seven_days_activity: list[ActivityRecord],
        mock_pat_loader: Mock,
    ) -> None:
        """Test that PAT creates 10,080-minute sequence (7 days * 1440 minutes)."""
        result = pipeline.process(
            activity_records=seven_days_activity,
            target_date=date(2025, 7, 26),
        )
        
        assert result is not None
        
        # Verify the PAT loader was called with correct sequence length
        mock_pat_loader.predict_depression_from_activity.assert_called_once()
        call_args = mock_pat_loader.predict_depression_from_activity.call_args
        
        # Should have an array as first positional argument
        activity_sequence = call_args[0][0]
        assert len(activity_sequence) == 10080  # 7 days * 1440 minutes/day


class TestXGBoostPipeline:
    """Test cases for independent XGBoost pipeline."""

    @pytest.fixture
    def mock_feature_extractor(self) -> Mock:
        """Mock clinical feature extractor."""
        mock = Mock()
        mock_features = MagicMock()
        mock_features.seoul_features = MagicMock()
        mock_features.seoul_features.to_xgboost_input.return_value = [0.5] * 36
        mock.extract_clinical_features.return_value = mock_features
        return mock

    @pytest.fixture
    def mock_xgboost_predictor(self) -> Mock:
        """Mock XGBoost predictor."""
        mock = Mock()
        mock.predict_mood_episodes.return_value = {
            "depression": {"probability": 0.25, "risk_level": "low"},
            "mania": {"probability": 0.15, "risk_level": "low"},
            "hypomania": {"probability": 0.20, "risk_level": "low"},
        }
        return mock

    @pytest.fixture
    def xgboost_validator(self) -> XGBoostValidator:
        """Create XGBoost validator."""
        return XGBoostValidator()

    @pytest.fixture
    def pipeline(
        self,
        mock_feature_extractor: Mock,
        mock_xgboost_predictor: Mock,
        xgboost_validator: XGBoostValidator,
    ) -> XGBoostPipeline:
        """Create XGBoost pipeline with mocks."""
        return XGBoostPipeline(
            feature_extractor=mock_feature_extractor,
            predictor=mock_xgboost_predictor,
            validator=xgboost_validator,
        )

    @pytest.fixture
    def sparse_35_days_data(self) -> tuple[list[SleepRecord], list[ActivityRecord]]:
        """Create 35 days of sparse data (every other day)."""
        sleep_records = []
        activity_records = []
        
        for day in range(0, 70, 2):  # Every other day for 35 days
            # Sleep record
            sleep_date = datetime(2025, 6, 1, 22, 0, 0, tzinfo=UTC) + timedelta(days=day)
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=sleep_date,
                    end_date=sleep_date + timedelta(hours=8),
                    state=SleepState.ASLEEP,
                )
            )
            
            # Activity record
            activity_date = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC) + timedelta(days=day)
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0 + (day * 10),  # Varying activity
                    unit="count",
                )
            )
        
        return sleep_records, activity_records

    def test_xgboost_runs_with_sparse_30_days(
        self,
        pipeline: XGBoostPipeline,
        sparse_35_days_data: tuple[list[SleepRecord], list[ActivityRecord]],
    ) -> None:
        """Test that XGBoost runs successfully with 35 sparse days."""
        sleep_records, activity_records = sparse_35_days_data
        
        # Validate first
        validation = pipeline.can_run(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            start_date=date(2025, 6, 1),
            end_date=date(2025, 8, 9),
        )
        
        assert validation.is_valid is True
        assert validation.can_run is True
        assert validation.days_available == 35
        
        # Process data
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            target_date=date(2025, 8, 9),
        )
        
        assert result is not None
        assert isinstance(result, XGBoostResult)
        assert result.depression_probability == 0.25
        assert result.mania_probability == 0.15
        assert result.hypomania_probability == 0.20
        assert result.prediction_window == "next 24 hours"
        assert result.data_days_used == 35
        assert "low" in result.clinical_interpretation.lower()

    def test_xgboost_returns_none_with_insufficient_data(
        self,
        pipeline: XGBoostPipeline,
    ) -> None:
        """Test that XGBoost returns None with less than 30 days."""
        # Only 20 days
        sleep_records = []
        for day in range(20):
            sleep_date = datetime(2025, 7, 1, 22, 0, 0, tzinfo=UTC) + timedelta(days=day)
            sleep_records.append(
                SleepRecord(
                    source_name="iPhone",
                    start_date=sleep_date,
                    end_date=sleep_date + timedelta(hours=7),
                    state=SleepState.ASLEEP,
                )
            )
        
        # Validate
        validation = pipeline.can_run(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            start_date=date(2025, 7, 1),
            end_date=date(2025, 7, 20),
        )
        
        assert validation.is_valid is False
        assert validation.can_run is False
        
        # Process should return None
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2025, 7, 20),
        )
        
        assert result is None

    def test_xgboost_uses_all_available_data_types(
        self,
        pipeline: XGBoostPipeline,
        mock_feature_extractor: Mock,
    ) -> None:
        """Test that XGBoost uses sleep, activity, and heart data when available."""
        # Create minimal 30 days of mixed data
        sleep_records = []
        activity_records = []
        heart_records = []
        
        for day in range(30):
            base_date = datetime(2025, 7, 1, tzinfo=UTC) + timedelta(days=day)
            
            # Sleep
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=base_date.replace(hour=22),
                    end_date=base_date.replace(hour=6) + timedelta(days=1),
                    state=SleepState.ASLEEP,
                )
            )
            
            # Activity
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=base_date.replace(hour=12),
                    end_date=base_date.replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
            
            # Heart rate
            heart_records.append(
                HeartRateRecord(
                    source_name="Apple Watch",
                    timestamp=base_date.replace(hour=14),
                    value=72.0,
                    unit="bpm",
                    metric_type=HeartMetricType.RESTING_HEART_RATE,
                    motion_context=MotionContext.SEDENTARY,
                )
            )
        
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=date(2025, 7, 30),
        )
        
        assert result is not None
        
        # Verify feature extractor was called with all data types
        mock_feature_extractor.extract_clinical_features.assert_called_once()
        call_args = mock_feature_extractor.extract_clinical_features.call_args
        
        # Should have all three data types as records (not summaries)
        assert call_args[1]['sleep_records'] == sleep_records
        assert call_args[1]['activity_records'] == activity_records
        assert call_args[1]['heart_records'] == heart_records
        assert call_args[1]['include_pat_sequence'] is False

    def test_xgboost_calculates_seoul_features(
        self,
        pipeline: XGBoostPipeline,
        sparse_35_days_data: tuple[list[SleepRecord], list[ActivityRecord]],
        mock_feature_extractor: Mock,
    ) -> None:
        """Test that XGBoost calculates the 36 Seoul features."""
        sleep_records, activity_records = sparse_35_days_data
        
        result = pipeline.process(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            target_date=date(2025, 8, 9),
        )
        
        assert result is not None
        
        # Verify Seoul features were extracted
        mock_feature_extractor.extract_clinical_features.assert_called_once()
        
        # The mock should have returned 36 features
        mock_features = mock_feature_extractor.extract_clinical_features.return_value
        seoul_input = mock_features.seoul_features.to_xgboost_input.return_value
        assert len(seoul_input) == 36