"""
Integration Smoke Test for DI Container

Tests that the DI container can successfully wire up the entire application
with real components (minus external dependencies like databases).
"""

from pathlib import Path

import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)
from big_mood_detector.domain.services.clinical_assessment_service import (
    ClinicalAssessmentService,
)
from big_mood_detector.domain.services.clinical_interpreter import ClinicalInterpreter
from big_mood_detector.infrastructure.di.container import setup_dependencies


@pytest.mark.integration
class TestDIIntegrationSmoke:
    """Smoke tests for dependency injection setup."""

    @pytest.fixture
    def settings(self, tmp_path):
        """Mock settings for DI setup."""
        from types import SimpleNamespace

        return SimpleNamespace(
            DATA_DIR=Path(__file__).parent.parent.parent / "data",
            data_dir=Path(__file__).parent.parent.parent / "data",
        )

    def test_can_resolve_clinical_interpreter(self, settings):
        """Test that ClinicalInterpreter can be resolved with all dependencies."""
        # Given: A configured DI container
        container = setup_dependencies(settings)

        # When: We resolve ClinicalInterpreter
        interpreter = container.resolve(ClinicalInterpreter)

        # Then: It should be properly initialized
        assert interpreter is not None
        assert hasattr(interpreter, 'clinical_assessment_service')
        assert hasattr(interpreter, 'longitudinal_assessment_service')
        assert hasattr(interpreter, 'intervention_evaluation_service')

    def test_can_resolve_mood_prediction_pipeline(self, settings):
        """Test that the main use case can be resolved."""
        # Given: A configured DI container
        container = setup_dependencies(settings)

        # When: We resolve the main pipeline
        pipeline = container.resolve(MoodPredictionPipeline)

        # Then: It should be properly initialized
        assert pipeline is not None
        assert hasattr(pipeline, 'process_apple_health_file')

    def test_clinical_assessment_service_integration(self, settings):
        """Test that ClinicalAssessmentService works end-to-end."""
        # Given: A configured DI container
        container = setup_dependencies(settings)
        service = container.resolve(ClinicalAssessmentService)

        # When: We make a clinical assessment
        assessment = service.make_clinical_assessment(
            mood_scores={"phq": 15.0, "asrm": 2.0},
            biomarkers={"sleep_hours": 9.5, "activity_steps": 1800},
            clinical_context={
                "symptom_days": 16,
                "functional_impairment": True,
            },
        )

        # Then: We should get a valid assessment
        assert assessment is not None
        assert assessment.primary_diagnosis == "depressive_episode"
        assert assessment.risk_level == "high"
        assert assessment.meets_dsm5_criteria is True
        assert len(assessment.treatment_options) > 0

    def test_all_services_are_singletons(self, settings):
        """Test that all services are registered as singletons."""
        # Given: A configured DI container
        container = setup_dependencies(settings)

        # When: We resolve services multiple times
        interpreter1 = container.resolve(ClinicalInterpreter)
        interpreter2 = container.resolve(ClinicalInterpreter)

        assessment1 = container.resolve(ClinicalAssessmentService)
        assessment2 = container.resolve(ClinicalAssessmentService)

        # Then: They should be the same instance (singleton)
        assert interpreter1 is interpreter2
        assert assessment1 is assessment2

    def test_container_includes_logging(self, settings, capsys):
        """Test that container includes proper logging for config path resolution."""
        # When: We setup dependencies
        container = setup_dependencies(settings)

        # Then: Container should be properly configured
        # and we should see logging output (captured by capsys)
        captured = capsys.readouterr()

        # Verify we see the loading_clinical_config event in stdout
        assert "loading_clinical_config" in captured.out or "clinical_config_not_found" in captured.out

        # And container should have registered services
        assert container._services  # Non-empty services dict
