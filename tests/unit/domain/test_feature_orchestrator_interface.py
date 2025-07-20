"""
Test Feature Orchestrator Interface

Tests for the feature orchestrator interface following Interface Segregation Principle.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Protocol

import pytest

from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.feature_types import (
    CompletenessReport,
    FeatureValidationResult,
    UnifiedFeatureSet,
)
from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary


class TestFeatureOrchestratorInterface:
    """Test the feature orchestrator interface definition."""

    def test_feature_orchestrator_protocol_definition(self):
        """Test that we can define a protocol for feature orchestration."""

        class FeatureOrchestratorProtocol(Protocol):
            """Protocol defining the interface for feature orchestration."""

            def extract_features_for_date(
                self,
                target_date: date,
                sleep_data: list[DailySleepSummary],
                activity_data: list[DailyActivitySummary],
                heart_data: list[DailyHeartSummary],
                lookback_days: int = 30,
            ) -> UnifiedFeatureSet:
                """Extract features for a specific date."""
                ...

            def validate_features(
                self, features: UnifiedFeatureSet
            ) -> FeatureValidationResult:
                """Validate extracted features."""
                ...

            def generate_completeness_report(
                self,
                sleep_data: list[DailySleepSummary],
                activity_data: list[DailyActivitySummary],
                heart_data: list[DailyHeartSummary],
            ) -> CompletenessReport:
                """Generate data completeness report."""
                ...

        # Protocol should be defined without errors
        assert FeatureOrchestratorProtocol is not None

    def test_feature_extraction_interface(self):
        """Test the feature extraction interface."""

        class FeatureExtractorInterface(ABC):
            """Interface for feature extraction."""

            @abstractmethod
            def extract_features_for_date(
                self,
                target_date: date,
                sleep_data: list[DailySleepSummary],
                activity_data: list[DailyActivitySummary],
                heart_data: list[DailyHeartSummary],
                lookback_days: int = 30,
            ) -> UnifiedFeatureSet:
                """Extract features for a specific date."""
                pass

            @abstractmethod
            def extract_features_batch(
                self,
                start_date: date,
                end_date: date,
                sleep_data: list[DailySleepSummary],
                activity_data: list[DailyActivitySummary],
                heart_data: list[DailyHeartSummary],
                lookback_days: int = 30,
            ) -> list[UnifiedFeatureSet]:
                """Extract features for a date range."""
                pass

        # Interface should be defined without errors
        assert FeatureExtractorInterface is not None

        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            FeatureExtractorInterface()

    def test_feature_validator_interface(self):
        """Test the feature validation interface."""

        class FeatureValidatorInterface(ABC):
            """Interface for feature validation."""

            @abstractmethod
            def validate_features(
                self, features: UnifiedFeatureSet
            ) -> FeatureValidationResult:
                """Validate feature quality and completeness."""
                pass

            @abstractmethod
            def generate_completeness_report(
                self,
                sleep_data: list[DailySleepSummary],
                activity_data: list[DailyActivitySummary],
                heart_data: list[DailyHeartSummary],
            ) -> CompletenessReport:
                """Generate data completeness report."""
                pass

        # Interface should be defined without errors
        assert FeatureValidatorInterface is not None

    def test_feature_exporter_interface(self):
        """Test the feature export interface."""
        from typing import Any

        class FeatureExporterInterface(ABC):
            """Interface for feature export functionality."""

            @abstractmethod
            def export_features_to_dict(
                self, feature_sets: list[UnifiedFeatureSet]
            ) -> list[dict[str, Any]]:
                """Export features to dictionary format."""
                pass

            @abstractmethod
            def get_feature_importance(self) -> dict[str, float]:
                """Get feature importance scores."""
                pass

        # Interface should be defined without errors
        assert FeatureExporterInterface is not None

    def test_orchestrator_implements_all_interfaces(self):
        """Test that orchestrator can implement all interfaces."""

        # Define all interfaces
        class FeatureExtractorInterface(ABC):
            @abstractmethod
            def extract_features_for_date(
                self,
                target_date: date,
                sleep_data: list[DailySleepSummary],
                activity_data: list[DailyActivitySummary],
                heart_data: list[DailyHeartSummary],
                lookback_days: int = 30,
            ) -> UnifiedFeatureSet:
                pass

        class FeatureValidatorInterface(ABC):
            @abstractmethod
            def validate_features(
                self, features: UnifiedFeatureSet
            ) -> FeatureValidationResult:
                pass

        # Mock implementation
        class MockOrchestrator(FeatureExtractorInterface, FeatureValidatorInterface):
            def extract_features_for_date(
                self,
                target_date: date,
                sleep_data: list[DailySleepSummary],
                activity_data: list[DailyActivitySummary],
                heart_data: list[DailyHeartSummary],
                lookback_days: int = 30,
            ) -> UnifiedFeatureSet:
                # Mock implementation
                from big_mood_detector.domain.services.feature_types import (
                    ActivityFeatureSet,
                    CircadianFeatureSet,
                    ClinicalFeatureSet,
                    SleepFeatureSet,
                    TemporalFeatureSet,
                    UnifiedFeatureSet,
                )

                return UnifiedFeatureSet(
                    date=target_date,
                    sleep_features=SleepFeatureSet(
                        total_sleep_hours=7.5,
                        sleep_efficiency=0.85,
                        sleep_regularity_index=80.0,
                        interdaily_stability=0.8,
                        intradaily_variability=0.2,
                        relative_amplitude=0.9,
                        short_sleep_window_pct=0.1,
                        long_sleep_window_pct=0.1,
                        sleep_onset_variance=0.5,
                        wake_time_variance=0.5,
                    ),
                    circadian_features=CircadianFeatureSet(
                        l5_value=10.0,
                        m10_value=50.0,
                        circadian_phase_advance=0.0,
                        circadian_phase_delay=0.0,
                        circadian_amplitude=40.0,
                        phase_angle=0.0,
                    ),
                    activity_features=ActivityFeatureSet(
                        total_steps=8000.0,
                        activity_fragmentation=0.3,
                        sedentary_bout_mean=60.0,
                        sedentary_bout_max=180.0,
                        activity_intensity_ratio=0.5,
                        activity_rhythm_strength=0.7,
                    ),
                    temporal_features=TemporalFeatureSet(
                        sleep_7day_mean=7.5,
                        sleep_7day_std=0.5,
                        activity_7day_mean=8000.0,
                        activity_7day_std=1000.0,
                        hr_7day_mean=65.0,
                        hr_7day_std=5.0,
                        sleep_trend_slope=0.0,
                        activity_trend_slope=0.0,
                        sleep_momentum=0.0,
                        activity_momentum=0.0,
                    ),
                    clinical_features=ClinicalFeatureSet(
                        is_hypersomnia_pattern=False,
                        is_insomnia_pattern=False,
                        is_phase_advanced=False,
                        is_phase_delayed=False,
                        is_irregular_pattern=False,
                        mood_risk_score=0.3,
                    ),
                )

            def validate_features(
                self, features: UnifiedFeatureSet
            ) -> FeatureValidationResult:
                # Mock implementation
                return FeatureValidationResult(
                    is_valid=True,
                    missing_domains=[],
                    quality_score=0.95,
                    warnings=[],
                )

        # Should be able to instantiate
        orchestrator = MockOrchestrator()
        assert orchestrator is not None

        # Should implement both interfaces
        assert isinstance(orchestrator, FeatureExtractorInterface)
        assert isinstance(orchestrator, FeatureValidatorInterface)
