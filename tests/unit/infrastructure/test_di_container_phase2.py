"""
Test DI Container Phase 2 - New Services Registration

Tests for registering the new services extracted from clinical_interpreter.
"""

import pytest

class TestDIContainerPhase2:
    """Test new service registrations for Phase 2."""

    @pytest.fixture
    def container(self):
        """Create a fresh container for testing."""
        return Container()

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for dependency setup."""
        from pathlib import Path
        from types import SimpleNamespace

        return SimpleNamespace(
            data_dir=Path("test_data"),
            DATA_DIR=Path("test_data"),
        )

    def test_clinical_assessment_service_not_registered_by_default(self, container):
        """Test that ClinicalAssessmentService is not registered by default."""
        from big_mood_detector.domain.services.clinical_assessment_service import ClinicalAssessmentService
        from big_mood_detector.infrastructure.di.container import DependencyNotFoundError

        with pytest.raises(DependencyNotFoundError):
            container.resolve(ClinicalAssessmentService)

    def test_longitudinal_assessment_service_not_registered_by_default(self, container):
        """Test that LongitudinalAssessmentService is not registered by default."""
        from big_mood_detector.domain.services.longitudinal_assessment_service import LongitudinalAssessmentService
        from big_mood_detector.infrastructure.di.container import DependencyNotFoundError

        with pytest.raises(DependencyNotFoundError):
            container.resolve(LongitudinalAssessmentService)

    def test_intervention_evaluation_service_not_registered_by_default(self, container):
        """Test that InterventionEvaluationService is not registered by default."""
        from big_mood_detector.domain.services.intervention_evaluation_service import InterventionEvaluationService
        from big_mood_detector.infrastructure.di.container import DependencyNotFoundError

        with pytest.raises(DependencyNotFoundError):
            container.resolve(InterventionEvaluationService)

    def test_clinical_interpreter_not_registered_by_default(self, container):
        """Test that ClinicalInterpreter is not registered by default."""
        from big_mood_detector.domain.services.clinical_interpreter import ClinicalInterpreter
        from big_mood_detector.infrastructure.di.container import DependencyNotFoundError

        with pytest.raises(DependencyNotFoundError):
            container.resolve(ClinicalInterpreter)

    def test_register_clinical_assessment_service(self, container, clinical_config):
        """Test registering ClinicalAssessmentService as singleton."""
        from big_mood_detector.domain.services.clinical_thresholds import ClinicalThresholdsConfig
        from big_mood_detector.domain.services.dsm5_criteria_evaluator import DSM5CriteriaEvaluator
        from big_mood_detector.domain.services.clinical_assessment_service import ClinicalAssessmentService
        from big_mood_detector.domain.services.treatment_recommender import TreatmentRecommender

        # Given: Required dependencies are registered

        container.register_singleton(ClinicalThresholdsConfig, clinical_config)
        container.register_singleton(DSM5CriteriaEvaluator)
        container.register_singleton(TreatmentRecommender)

        # When: We register ClinicalAssessmentService
        container.register_singleton(ClinicalAssessmentService)

        # Then: We can resolve it and it's a singleton
        service1 = container.resolve(ClinicalAssessmentService)
        service2 = container.resolve(ClinicalAssessmentService)

        assert service1 is not None
        assert service1 is service2  # Singleton

    def test_register_all_new_services(self, container, clinical_config):
        """Test registering all new services together."""
        from big_mood_detector.domain.services.clinical_thresholds import ClinicalThresholdsConfig
        from big_mood_detector.domain.services.clinical_assessment_service import ClinicalAssessmentService
        from big_mood_detector.domain.services.early_warning_detector import EarlyWarningDetector
        from big_mood_detector.domain.services.treatment_recommender import TreatmentRecommender
        from big_mood_detector.domain.services.intervention_evaluation_service import InterventionEvaluationService
        from big_mood_detector.domain.services.longitudinal_assessment_service import LongitudinalAssessmentService
        from big_mood_detector.domain.services.dsm5_criteria_evaluator import DSM5CriteriaEvaluator

        # Given: Required dependencies

        container.register_singleton(ClinicalThresholdsConfig, clinical_config)
        container.register_singleton(DSM5CriteriaEvaluator)
        container.register_singleton(TreatmentRecommender)
        container.register_singleton(EarlyWarningDetector)

        # When: We register all new services
        container.register_singleton(ClinicalAssessmentService)
        container.register_singleton(LongitudinalAssessmentService)
        container.register_singleton(InterventionEvaluationService)

        # Then: We can resolve all of them
        assessment_service = container.resolve(ClinicalAssessmentService)
        longitudinal_service = container.resolve(LongitudinalAssessmentService)
        intervention_service = container.resolve(InterventionEvaluationService)

        assert assessment_service is not None
        assert longitudinal_service is not None
        assert intervention_service is not None

    def test_clinical_interpreter_with_dependencies(self, container, clinical_config):
        """Test ClinicalInterpreter can be resolved with all its dependencies."""
        from big_mood_detector.domain.services.clinical_thresholds import ClinicalThresholdsConfig
        from big_mood_detector.domain.services.clinical_assessment_service import ClinicalAssessmentService
        from big_mood_detector.domain.services.early_warning_detector import EarlyWarningDetector
        from big_mood_detector.domain.services.clinical_interpreter import ClinicalInterpreter
        from big_mood_detector.domain.services.treatment_recommender import TreatmentRecommender
        from big_mood_detector.domain.services.episode_interpreter import EpisodeInterpreter
        from big_mood_detector.domain.services.longitudinal_assessment_service import LongitudinalAssessmentService
        from big_mood_detector.domain.services.intervention_evaluation_service import InterventionEvaluationService
        from big_mood_detector.domain.services.dsm5_criteria_evaluator import DSM5CriteriaEvaluator
        from big_mood_detector.domain.services.biomarker_interpreter import BiomarkerInterpreter
        from big_mood_detector.domain.services.risk_level_assessor import RiskLevelAssessor

        # Given: All required services are registered

        # Register config first (dependency of many services)
        container.register_singleton(ClinicalThresholdsConfig, clinical_config)

        # Register all required services
        container.register_singleton(EpisodeInterpreter)
        container.register_singleton(BiomarkerInterpreter)
        container.register_singleton(TreatmentRecommender)
        container.register_singleton(DSM5CriteriaEvaluator)
        container.register_singleton(RiskLevelAssessor)
        container.register_singleton(EarlyWarningDetector)
        container.register_singleton(ClinicalAssessmentService)
        container.register_singleton(LongitudinalAssessmentService)
        container.register_singleton(InterventionEvaluationService)

        # When: We register and resolve ClinicalInterpreter
        container.register_singleton(ClinicalInterpreter)
        interpreter = container.resolve(ClinicalInterpreter)

        # Then: It should be properly initialized with all services
        assert interpreter is not None
        assert hasattr(interpreter, "clinical_assessment_service")
        assert hasattr(interpreter, "longitudinal_assessment_service")
        assert hasattr(interpreter, "intervention_evaluation_service")

    def test_setup_dependencies_includes_new_services(self, mock_settings):
        """Test that setup_dependencies registers our new services."""
        from big_mood_detector.domain.services.clinical_assessment_service import ClinicalAssessmentService
        from big_mood_detector.domain.services.clinical_interpreter import ClinicalInterpreter
        from big_mood_detector.domain.services.intervention_evaluation_service import InterventionEvaluationService
        from big_mood_detector.infrastructure.di.container import (
            DependencyNotFoundError,
            setup_dependencies,
        )
        from big_mood_detector.domain.services.longitudinal_assessment_service import LongitudinalAssessmentService

        # Clear any existing container
        import big_mood_detector.infrastructure.di.container as di_module

        di_module._container = None
        di_module.get_container.cache_clear()

        # When: We setup dependencies
        container = setup_dependencies(mock_settings)

        # Then: New services should be registered
        # Note: This will fail initially, driving us to update setup_dependencies
        try:
            container.resolve(ClinicalAssessmentService)
            container.resolve(LongitudinalAssessmentService)
            container.resolve(InterventionEvaluationService)
            container.resolve(ClinicalInterpreter)
        except DependencyNotFoundError:
            pytest.fail("New services not registered in setup_dependencies")

    def test_clinical_interpreter_methods_work_after_di(
        from big_mood_detector.domain.services.clinical_thresholds import ClinicalThresholdsConfig
        from big_mood_detector.domain.services.clinical_assessment_service import ClinicalAssessmentService
        from big_mood_detector.domain.services.early_warning_detector import EarlyWarningDetector
        from big_mood_detector.domain.services.clinical_interpreter import ClinicalInterpreter
        from big_mood_detector.domain.services.treatment_recommender import TreatmentRecommender
        from big_mood_detector.domain.services.episode_interpreter import EpisodeInterpreter
        from big_mood_detector.domain.services.longitudinal_assessment_service import LongitudinalAssessmentService
        from big_mood_detector.domain.services.intervention_evaluation_service import InterventionEvaluationService
        from big_mood_detector.domain.services.dsm5_criteria_evaluator import DSM5CriteriaEvaluator
        from big_mood_detector.domain.services.biomarker_interpreter import BiomarkerInterpreter
        from big_mood_detector.domain.services.risk_level_assessor import RiskLevelAssessor

        self, container, clinical_config
    ):
        """Test that ClinicalInterpreter methods work after DI resolution."""
        # Given: Full DI setup

        container.register_singleton(ClinicalThresholdsConfig, clinical_config)
        container.register_singleton(EpisodeInterpreter)
        container.register_singleton(BiomarkerInterpreter)
        container.register_singleton(TreatmentRecommender)
        container.register_singleton(DSM5CriteriaEvaluator)
        container.register_singleton(RiskLevelAssessor)
        container.register_singleton(EarlyWarningDetector)
        container.register_singleton(ClinicalAssessmentService)
        container.register_singleton(LongitudinalAssessmentService)
        container.register_singleton(InterventionEvaluationService)
        container.register_singleton(ClinicalInterpreter)

        # When: We resolve and use ClinicalInterpreter
        interpreter = container.resolve(ClinicalInterpreter)

        # Then: All delegated methods should work
        assessment = interpreter.make_clinical_assessment(
            mood_scores={"phq": 15.0, "asrm": 2.0},
            biomarkers={"sleep_hours": 10.5, "activity_steps": 1500},
            clinical_context={
                "symptom_days": 16,
                "functional_impairment": True,
            },
        )

        assert assessment is not None
        assert assessment.primary_diagnosis == "depressive_episode"

    def test_register_feature_engineering_orchestrator(self, container):
        """Test registering FeatureEngineeringOrchestrator."""
        from big_mood_detector.domain.services.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
        from big_mood_detector.infrastructure.di.container import DependencyNotFoundError

        # Should not be registered by default
        with pytest.raises(DependencyNotFoundError):
            container.resolve(FeatureEngineeringOrchestrator)

        # Register the orchestrator
        container.register_singleton(FeatureEngineeringOrchestrator)

        # Should resolve successfully
        orchestrator = container.resolve(FeatureEngineeringOrchestrator)
        assert orchestrator is not None
        assert isinstance(orchestrator, FeatureEngineeringOrchestrator)

        # Should be singleton
        orchestrator2 = container.resolve(FeatureEngineeringOrchestrator)
        assert orchestrator is orchestrator2
