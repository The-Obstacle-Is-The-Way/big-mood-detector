"""
Unit tests for ProcessWithIndependentPipelinesUseCase.

Tests the orchestration of independent PAT and XGBoost pipelines.
"""

import pytest
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from big_mood_detector.application.use_cases.process_with_independent_pipelines import (
    ProcessWithIndependentPipelinesUseCase,
    IndependentPipelineResult,
)
from big_mood_detector.application.pipelines.pat_pipeline import PATResult
from big_mood_detector.application.pipelines.xgboost_pipeline import XGBoostResult
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestProcessWithIndependentPipelines:
    """Test the independent pipelines use case."""
    
    @pytest.fixture
    def mock_data_parsing_service(self) -> Mock:
        """Mock data parsing service."""
        mock = Mock()
        
        # Create test data
        sleep_records = []
        activity_records = []
        
        # 35 days of sparse data for XGBoost
        for day in range(0, 70, 2):  # Every other day
            sleep_date = datetime(2025, 6, 1, 22, 0, 0, tzinfo=UTC) + timedelta(days=day)
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=sleep_date,
                    end_date=sleep_date + timedelta(hours=8),
                    state=SleepState.ASLEEP,
                )
            )
            
        # Last 7 consecutive days of activity for PAT
        base_date = date(2025, 7, 20)
        for day in range(7):
            activity_date = datetime(
                base_date.year,
                base_date.month,
                base_date.day + day,
                12, 0, 0,
                tzinfo=UTC
            )
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=activity_date,
                    end_date=activity_date + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
        
        mock.parse_health_data.return_value = {
            "sleep_records": sleep_records,
            "activity_records": activity_records,
            "heart_rate_records": [],
            "errors": [],
        }
        
        return mock
    
    @pytest.fixture
    def mock_pat_pipeline(self) -> Mock:
        """Mock PAT pipeline."""
        mock = Mock()
        
        # Mock validation
        validation_result = Mock()
        validation_result.can_run = True
        validation_result.message = "PAT ready"
        mock.can_run.return_value = validation_result
        
        # Mock process result
        mock.process.return_value = PATResult(
            depression_risk_score=0.35,
            confidence=0.85,
            assessment_window_days=7,
            model_version="PAT-L",
            clinical_interpretation="Current depression risk: Moderate risk",
            window_start_date=date(2025, 7, 20),
            window_end_date=date(2025, 7, 26),
        )
        
        return mock
    
    @pytest.fixture
    def mock_xgboost_pipeline(self) -> Mock:
        """Mock XGBoost pipeline."""
        mock = Mock()
        
        # Mock validation
        validation_result = Mock()
        validation_result.can_run = True
        validation_result.message = "XGBoost ready"
        mock.can_run.return_value = validation_result
        
        # Mock process result
        mock.process.return_value = XGBoostResult(
            depression_probability=0.25,
            mania_probability=0.15,
            hypomania_probability=0.20,
            prediction_window="next 24 hours",
            data_days_used=35,
            clinical_interpretation="Low risk for mood episodes in next 24 hours",
            highest_risk_episode="stable",
            confidence_level="high",
        )
        
        return mock
    
    @pytest.fixture
    def use_case(
        self,
        mock_data_parsing_service: Mock,
        mock_pat_pipeline: Mock,
        mock_xgboost_pipeline: Mock,
    ) -> ProcessWithIndependentPipelinesUseCase:
        """Create use case with mocked dependencies."""
        return ProcessWithIndependentPipelinesUseCase(
            data_parsing_service=mock_data_parsing_service,
            pat_pipeline=mock_pat_pipeline,
            xgboost_pipeline=mock_xgboost_pipeline,
        )
    
    def test_execute_with_both_pipelines_available(
        self,
        use_case: ProcessWithIndependentPipelinesUseCase,
        mock_pat_pipeline: Mock,
        mock_xgboost_pipeline: Mock,
    ) -> None:
        """Test execution when both pipelines have sufficient data."""
        # Execute
        result = use_case.execute(
            file_path=Path("test_export.xml"),
            target_date=date(2025, 7, 26),
        )
        
        # Verify result
        assert isinstance(result, IndependentPipelineResult)
        assert result.pat_available is True
        assert result.xgboost_available is True
        
        # Check PAT result
        assert result.pat_result is not None
        assert result.pat_result.depression_risk_score == 0.35
        assert "7/20" in result.pat_message or "2025-07-20" in result.pat_message
        
        # Check XGBoost result
        assert result.xgboost_result is not None
        assert result.xgboost_result.depression_probability == 0.25
        assert "35 days" in result.xgboost_message
        
        # Check temporal ensemble
        assert "temporal_windows" in result.temporal_ensemble
        assert "current_state" in result.temporal_ensemble["temporal_windows"]
        assert "future_risk" in result.temporal_ensemble["temporal_windows"]
        
        # Verify pipelines were called
        mock_pat_pipeline.process.assert_called_once()
        mock_xgboost_pipeline.process.assert_called_once()
    
    def test_execute_with_only_pat_available(
        self,
        use_case: ProcessWithIndependentPipelinesUseCase,
        mock_pat_pipeline: Mock,
        mock_xgboost_pipeline: Mock,
    ) -> None:
        """Test when only PAT has sufficient data."""
        # Mock XGBoost validation failure
        validation_result = Mock()
        validation_result.can_run = False
        validation_result.message = "Need at least 30 days, found 20"
        mock_xgboost_pipeline.can_run.return_value = validation_result
        
        # Execute
        result = use_case.execute(
            file_path=Path("test_export.xml"),
            target_date=date(2025, 7, 26),
        )
        
        # Verify result
        assert result.pat_available is True
        assert result.xgboost_available is False
        assert result.pat_result is not None
        assert result.xgboost_result is None
        assert "30 days" in result.xgboost_message
        
        # Check temporal ensemble
        assert "current_state" in result.temporal_ensemble["temporal_windows"]
        assert "future_risk" not in result.temporal_ensemble["temporal_windows"]
        assert "30 days of data for predictive" in str(result.temporal_ensemble["recommendations"])
    
    def test_execute_with_only_xgboost_available(
        self,
        use_case: ProcessWithIndependentPipelinesUseCase,
        mock_pat_pipeline: Mock,
        mock_xgboost_pipeline: Mock,
    ) -> None:
        """Test when only XGBoost has sufficient data."""
        # Mock PAT validation failure
        validation_result = Mock()
        validation_result.can_run = False
        validation_result.message = "No 7 consecutive days found"
        mock_pat_pipeline.can_run.return_value = validation_result
        
        # Execute
        result = use_case.execute(
            file_path=Path("test_export.xml"),
            target_date=date(2025, 7, 26),
        )
        
        # Verify result
        assert result.pat_available is False
        assert result.xgboost_available is True
        assert result.pat_result is None
        assert result.xgboost_result is not None
        assert "consecutive days" in result.pat_message
        
        # Check temporal ensemble
        assert "current_state" not in result.temporal_ensemble["temporal_windows"]
        assert "future_risk" in result.temporal_ensemble["temporal_windows"]
        assert "activity tracking" in str(result.temporal_ensemble["recommendations"])
    
    def test_execute_with_no_pipelines_available(
        self,
        use_case: ProcessWithIndependentPipelinesUseCase,
        mock_pat_pipeline: Mock,
        mock_xgboost_pipeline: Mock,
    ) -> None:
        """Test when neither pipeline has sufficient data."""
        # Mock both validations failing
        pat_validation = Mock()
        pat_validation.can_run = False
        pat_validation.message = "Insufficient consecutive days"
        mock_pat_pipeline.can_run.return_value = pat_validation
        
        xgboost_validation = Mock()
        xgboost_validation.can_run = False
        xgboost_validation.message = "Need at least 30 days"
        mock_xgboost_pipeline.can_run.return_value = xgboost_validation
        
        # Execute
        result = use_case.execute(
            file_path=Path("test_export.xml"),
            target_date=date(2025, 7, 26),
        )
        
        # Verify result
        assert result.pat_available is False
        assert result.xgboost_available is False
        assert result.pat_result is None
        assert result.xgboost_result is None
        
        # Check temporal ensemble
        assert "Insufficient data" in result.temporal_ensemble["clinical_summary"]
        assert len(result.temporal_ensemble["recommendations"]) > 0
    
    def test_temporal_ensemble_interpretations(
        self,
        use_case: ProcessWithIndependentPipelinesUseCase,
    ) -> None:
        """Test various temporal ensemble interpretations."""
        # Test 1: Current depression + future depression
        pat_result = PATResult(
            depression_risk_score=0.7,  # High current
            confidence=0.9,
            assessment_window_days=7,
            model_version="PAT-L",
            clinical_interpretation="High risk",
            window_start_date=date(2025, 7, 20),
            window_end_date=date(2025, 7, 26),
        )
        
        xgboost_result = XGBoostResult(
            depression_probability=0.8,  # High future
            mania_probability=0.1,
            hypomania_probability=0.1,
            prediction_window="next 24 hours",
            data_days_used=35,
            clinical_interpretation="High depression risk",
            highest_risk_episode="depression",
            confidence_level="high",
        )
        
        ensemble = use_case._create_temporal_ensemble(
            pat_result, xgboost_result, date(2025, 7, 26)
        )
        
        assert "Currently experiencing elevated depression" in ensemble["clinical_summary"]
        assert "Immediate clinical intervention" in ensemble["clinical_summary"]
        assert "Contact mental health provider" in str(ensemble["recommendations"])
        
        # Test 2: Current stable + future mania risk
        pat_result.depression_risk_score = 0.2  # Low current
        xgboost_result.depression_probability = 0.1
        xgboost_result.mania_probability = 0.7  # High mania
        xgboost_result.highest_risk_episode = "mania"
        
        ensemble = use_case._create_temporal_ensemble(
            pat_result, xgboost_result, date(2025, 7, 26)
        )
        
        assert "Elevated risk for mania" in ensemble["clinical_summary"]
        assert "Monitor for decreased sleep" in str(ensemble["recommendations"])
    
    def test_data_summary_in_result(
        self,
        use_case: ProcessWithIndependentPipelinesUseCase,
    ) -> None:
        """Test that data summary is included in result."""
        result = use_case.execute(
            file_path=Path("test_export.xml"),
            target_date=date(2025, 7, 26),
        )
        
        assert "data_summary" in result.__dict__
        assert result.data_summary["sleep_days"] == 35  # Every other day for 70 days
        assert result.data_summary["activity_days"] == 7  # Last 7 consecutive
        assert result.data_summary["heart_days"] == 0
        assert result.data_summary["total_records"] > 0
    
    def test_execute_with_no_data(
        self,
        use_case: ProcessWithIndependentPipelinesUseCase,
        mock_data_parsing_service: Mock,
    ) -> None:
        """Test execution when no health data is found."""
        # Mock empty data
        mock_data_parsing_service.parse_health_data.return_value = {
            "sleep_records": [],
            "activity_records": [],
            "heart_rate_records": [],
            "errors": [],
        }
        
        result = use_case.execute(
            file_path=Path("empty_export.xml"),
            target_date=date(2025, 7, 26),
        )
        
        assert result.pat_available is False
        assert result.xgboost_available is False
        assert "No activity records" in result.pat_message
        assert "No health records" in result.xgboost_message