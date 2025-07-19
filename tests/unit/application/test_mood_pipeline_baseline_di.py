"""
Test MoodPredictionPipeline baseline repository dependency injection.

TDD approach: Write failing tests first, then make them pass.
"""
import pytest
from unittest.mock import Mock, create_autospec, patch
from pathlib import Path

from big_mood_detector.infrastructure.di.container import Container
from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
)


class TestMoodPipelineBaselineDI:
    """Test baseline repository dependency injection in MoodPredictionPipeline."""
    
    def test_current_behavior_creates_repository_directly(self, tmp_path, monkeypatch):
        """
        Document current behavior: pipeline creates FileBaselineRepository directly.
        This test should PASS, showing the problem we need to fix.
        """
        # Change to tmp directory to avoid polluting project
        monkeypatch.chdir(tmp_path)
        
        # Create config with personal calibration
        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "test_user"
        
        # Pipeline creates FileBaselineRepository directly
        pipeline = MoodPredictionPipeline(config=config)
        
        # Verify it created the directory and repository
        assert (tmp_path / "data" / "baselines").exists()
        assert pipeline.clinical_extractor.baseline_repository is not None
    
    def test_pipeline_should_accept_baseline_repository(self):
        """
        Test that pipeline accepts baseline_repository parameter.
        This should now PASS after our implementation.
        """
        # Create mock repository
        mock_repo = create_autospec(BaselineRepositoryInterface)
        
        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "test_user"
        
        # This should now work!
        pipeline = MoodPredictionPipeline(
            config=config,
            baseline_repository=mock_repo
        )
        
        # Verify the clinical extractor uses our mock repository
        assert pipeline.clinical_extractor.baseline_repository is mock_repo
    
    def test_pipeline_should_use_di_container(self):
        """
        Test that pipeline uses DI container for baseline repository.
        This should now PASS after our implementation.
        """
        # Setup DI container with mock repository
        mock_repo = create_autospec(BaselineRepositoryInterface)
        container = Container()
        container.register_singleton(BaselineRepositoryInterface, lambda: mock_repo)
        
        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "test_user"
        
        # This should now work!
        pipeline = MoodPredictionPipeline(
            config=config,
            di_container=container
        )
        
        # Verify the clinical extractor uses the repository from DI container
        assert pipeline.clinical_extractor.baseline_repository is mock_repo