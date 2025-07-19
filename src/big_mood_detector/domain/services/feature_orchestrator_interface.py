"""
Feature Orchestrator Interfaces

Following Interface Segregation Principle (ISP) from SOLID principles.
Split into focused interfaces that clients can depend on selectively.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Protocol

from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.feature_types import (
    AnomalyResult,
    CompletenessReport,
    FeatureValidationResult,
    UnifiedFeatureSet,
)
from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary


class FeatureExtractorInterface(ABC):
    """
    Interface for feature extraction operations.

    Follows Single Responsibility: Only handles feature extraction.
    """

    @abstractmethod
    def extract_features_for_date(
        self,
        target_date: date,
        sleep_data: list[DailySleepSummary],
        activity_data: list[DailyActivitySummary],
        heart_data: list[DailyHeartSummary],
        lookback_days: int = 30,
    ) -> UnifiedFeatureSet:
        """
        Extract all features for a specific date.

        Args:
            target_date: Date to extract features for
            sleep_data: Historical sleep summaries
            activity_data: Historical activity summaries
            heart_data: Historical heart summaries
            lookback_days: Days of history to consider

        Returns:
            Complete feature set for the date
        """
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
        """
        Extract features for a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range (inclusive)
            sleep_data: Historical sleep summaries
            activity_data: Historical activity summaries
            heart_data: Historical heart summaries
            lookback_days: Days of history to consider

        Returns:
            List of feature sets for each date
        """
        pass


class FeatureValidatorInterface(ABC):
    """
    Interface for feature validation operations.

    Follows Single Responsibility: Only handles validation.
    """

    @abstractmethod
    def validate_features(self, features: UnifiedFeatureSet) -> FeatureValidationResult:
        """
        Validate feature quality and completeness.

        Args:
            features: Feature set to validate

        Returns:
            Validation result with quality metrics
        """
        pass

    @abstractmethod
    def generate_completeness_report(
        self,
        sleep_data: list[DailySleepSummary],
        activity_data: list[DailyActivitySummary],
        heart_data: list[DailyHeartSummary],
    ) -> CompletenessReport:
        """
        Generate data completeness report.

        Args:
            sleep_data: Sleep summaries
            activity_data: Activity summaries
            heart_data: Heart summaries

        Returns:
            Completeness report with coverage metrics
        """
        pass

    @abstractmethod
    def detect_anomalies(self, features: UnifiedFeatureSet) -> AnomalyResult:
        """
        Detect anomalies in feature set.

        Args:
            features: Feature set to analyze

        Returns:
            Anomaly detection result
        """
        pass


class FeatureExporterInterface(ABC):
    """
    Interface for feature export operations.

    Follows Single Responsibility: Only handles export/formatting.
    """

    @abstractmethod
    def export_features_to_dict(
        self, feature_sets: list[UnifiedFeatureSet]
    ) -> list[dict[str, Any]]:
        """
        Export feature sets to dictionary format for DataFrame conversion.

        Args:
            feature_sets: List of feature sets

        Returns:
            List of dictionaries with flattened features
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary of feature names to importance scores (0-1)
        """
        pass


class FeatureOrchestratorProtocol(Protocol):
    """
    Protocol combining all feature orchestration capabilities.

    This is for clients that need the full orchestrator functionality.
    Follows Liskov Substitution Principle: Any implementation can be used.
    """

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
        ...

    def validate_features(self, features: UnifiedFeatureSet) -> FeatureValidationResult:
        """Validate feature quality."""
        ...

    def generate_completeness_report(
        self,
        sleep_data: list[DailySleepSummary],
        activity_data: list[DailyActivitySummary],
        heart_data: list[DailyHeartSummary],
    ) -> CompletenessReport:
        """Generate completeness report."""
        ...

    def detect_anomalies(self, features: UnifiedFeatureSet) -> AnomalyResult:
        """Detect anomalies."""
        ...

    def export_features_to_dict(
        self, feature_sets: list[UnifiedFeatureSet]
    ) -> list[dict[str, Any]]:
        """Export to dictionary format."""
        ...

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance."""
        ...

    def clear_cache(self) -> None:
        """Clear feature cache."""
        ...
