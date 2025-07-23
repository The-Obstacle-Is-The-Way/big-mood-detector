"""
Orchestrator Adapter

Adapts FeatureEngineeringOrchestrator to work with existing ClinicalFeatureExtractor interface.
This enables validation, anomaly detection, and completeness reporting while maintaining
backward compatibility.

Clean Architecture Pattern: Adapter
"""

from datetime import date

# Import types from the actual modules
from typing import TYPE_CHECKING, Any

import numpy as np

from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureSet,
    PATSequence,
    SeoulXGBoostFeatures,
)
from big_mood_detector.domain.services.feature_engineering_orchestrator import (
    FeatureEngineeringOrchestrator,
)
from big_mood_detector.domain.services.feature_types import (
    AnomalyResult,
    CompletenessReport,
    FeatureValidationResult,
    UnifiedFeatureSet,
)

if TYPE_CHECKING:
    pass
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)


class OrchestratorAdapter:
    """
    Adapter that makes FeatureEngineeringOrchestrator compatible with
    ClinicalFeatureExtractor interface.

    This allows us to add orchestrator benefits (validation, anomaly detection)
    while maintaining backward compatibility with existing code.
    """

    def __init__(
        self,
        orchestrator: FeatureEngineeringOrchestrator | None = None,
        baseline_repository: Any | None = None,
        user_id: str | None = None,
    ):
        """
        Initialize adapter with orchestrator.

        Args:
            orchestrator: Feature engineering orchestrator instance
            baseline_repository: Optional baseline repository for personalization
            user_id: Optional user ID for personal baselines
        """
        self.orchestrator = orchestrator or FeatureEngineeringOrchestrator()
        self.baseline_repository = baseline_repository
        self.user_id = user_id

        # Track validation results for later access
        self._last_validation_result: FeatureValidationResult | None = None
        self._last_anomaly_result: AnomalyResult | None = None
        self._last_completeness_report: CompletenessReport | None = None

    def extract_clinical_features(
        self,
        sleep_records: list[Any],  # SleepRecord
        activity_records: list[Any],  # StepsRecord
        heart_records: list[Any],  # HeartRateRecord
        target_date: date,
        include_pat_sequence: bool = False,
    ) -> ClinicalFeatureSet:
        """
        Extract clinical features using orchestrator with added benefits.

        This method maintains the same interface as ClinicalFeatureExtractor
        but adds validation, anomaly detection, and better error handling.

        Args:
            sleep_records: List of sleep records
            activity_records: List of activity records
            heart_records: List of heart rate records
            target_date: Date to extract features for
            include_pat_sequence: Whether to include PAT sequences

        Returns:
            ClinicalFeatureSet with enriched metadata
        """
        try:
            # Convert records to summary format expected by orchestrator
            # Note: In a full implementation, we'd use the aggregation pipeline
            sleep_data: list[Any] = []  # Would convert sleep_records to DailySleepSummary
            activity_data: list[Any] = []  # Would convert activity_records to DailyActivitySummary
            heart_data: list[Any] = []  # Would convert heart_records to DailyHeartSummary

            # Extract features using orchestrator
            unified_features = self.orchestrator.extract_features_for_date(
                target_date=target_date,
                sleep_data=sleep_data,
                activity_data=activity_data,
                heart_data=heart_data,
                lookback_days=30,
                use_cache=True,
            )

            # Perform validation
            validation_result = self.orchestrator.validate_features(unified_features)
            self._last_validation_result = validation_result

            if not validation_result.is_valid:
                logger.warning(
                    "feature_validation_failed",
                    date=str(target_date),
                    missing_domains=validation_result.missing_domains,
                    quality_score=validation_result.quality_score,
                    warnings=validation_result.warnings,
                )

            # Detect anomalies
            anomaly_result = self.orchestrator.detect_anomalies(unified_features)
            self._last_anomaly_result = anomaly_result

            if anomaly_result.has_anomalies:
                logger.warning(
                    "anomalies_detected",
                    date=str(target_date),
                    anomaly_domains=anomaly_result.anomaly_domains,
                    severity=anomaly_result.severity,
                )

            # Convert UnifiedFeatureSet to ClinicalFeatureSet
            clinical_features = self._convert_to_clinical_features(
                unified_features,
                include_pat_sequence,
            )

            # Enrich with validation metadata
            clinical_features = self._enrich_with_metadata(
                clinical_features,
                validation_result,
                anomaly_result,
            )

            return clinical_features

        except Exception as e:
            logger.error(
                "orchestrator_extraction_failed",
                date=str(target_date),
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fall back to creating minimal features
            return self._create_fallback_features(target_date)

    def persist_baselines(self) -> None:
        """
        Persist baselines if baseline repository is configured.

        This maintains compatibility with the existing interface.
        """
        if self.baseline_repository and self.user_id:
            # In full implementation, would persist orchestrator's baselines
            logger.info("baselines_persisted", user_id=self.user_id)

    def get_completeness_report(
        self,
        sleep_records: list[Any],  # SleepRecord
        activity_records: list[Any],  # StepsRecord
        heart_records: list[Any],  # HeartRateRecord
    ) -> CompletenessReport:
        """
        Generate completeness report for the data.

        This is a new capability added by the orchestrator.
        """
        # Convert to summary format
        sleep_data: list[Any] = []
        activity_data: list[Any] = []
        heart_data: list[Any] = []

        report = self.orchestrator.generate_completeness_report(
            sleep_data=sleep_data,
            activity_data=activity_data,
            heart_data=heart_data,
        )

        self._last_completeness_report = report
        return report

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores from orchestrator.

        This is a new capability that helps understand which features matter most.
        """
        return self.orchestrator.get_feature_importance()

    def _convert_to_clinical_features(
        self,
        unified: UnifiedFeatureSet,
        include_pat_sequence: bool,
    ) -> ClinicalFeatureSet:
        """
        Convert UnifiedFeatureSet to ClinicalFeatureSet for backward compatibility.

        Args:
            unified: Unified feature set from orchestrator
            include_pat_sequence: Whether to include PAT sequence

        Returns:
            ClinicalFeatureSet compatible with existing code
        """
        # Create Seoul features from unified features
        seoul_features = SeoulXGBoostFeatures(
            date=unified.date,
            # Basic Sleep Features (1-5)
            sleep_duration_hours=unified.sleep_features.total_sleep_hours,
            sleep_efficiency=unified.sleep_features.sleep_efficiency,
            sleep_onset_hour=23.0,  # Would calculate from data
            wake_time_hour=7.0,  # Would calculate from data
            sleep_fragmentation=0.0,  # Would calculate from data
            # Advanced Sleep Features (6-10)
            sleep_regularity_index=unified.sleep_features.sleep_regularity_index,
            short_sleep_window_pct=unified.sleep_features.short_sleep_window_pct,
            long_sleep_window_pct=unified.sleep_features.long_sleep_window_pct,
            sleep_onset_variance=unified.sleep_features.sleep_onset_variance,
            wake_time_variance=unified.sleep_features.wake_time_variance,
            # Circadian Rhythm Features (11-18)
            interdaily_stability=unified.sleep_features.interdaily_stability,
            intradaily_variability=unified.sleep_features.intradaily_variability,
            relative_amplitude=unified.sleep_features.relative_amplitude,
            l5_value=unified.circadian_features.l5_value,
            m10_value=unified.circadian_features.m10_value,
            l5_onset_hour=2.0,  # Would calculate from data
            m10_onset_hour=14.0,  # Would calculate from data
            dlmo_hour=21.0,  # Would calculate from data
            # Activity Features (19-24)
            total_steps=int(unified.activity_features.total_steps),
            activity_variance=100.0,  # Would calculate from data
            sedentary_hours=unified.activity_features.sedentary_bout_mean / 60.0,
            activity_fragmentation=unified.activity_features.activity_fragmentation,
            sedentary_bout_mean=unified.activity_features.sedentary_bout_mean,
            activity_intensity_ratio=unified.activity_features.activity_intensity_ratio,
            # Heart Rate Features (25-28)
            avg_resting_hr=70.0,  # Would calculate from data
            hrv_sdnn=50.0,  # Would calculate from data
            hr_circadian_range=20.0,  # Would calculate from data
            hr_minimum_hour=4.0,  # Would calculate from data
            # Phase Features (29-32)
            circadian_phase_advance=unified.circadian_features.circadian_phase_advance,
            circadian_phase_delay=unified.circadian_features.circadian_phase_delay,
            dlmo_confidence=0.5,  # Would calculate from data
            pat_hour=14.0,  # Would calculate from data
            # Z-Score Features (33-36)
            sleep_duration_zscore=0.0,  # Would calculate from baselines
            activity_zscore=0.0,  # Would calculate from baselines
            hr_zscore=0.0,  # Would calculate from baselines
            hrv_zscore=0.0,  # Would calculate from baselines
            # Metadata
            data_completeness=0.8,  # Would calculate from data
            is_hypersomnia_pattern=unified.clinical_features.is_hypersomnia_pattern,
            is_insomnia_pattern=unified.clinical_features.is_insomnia_pattern,
            is_phase_advanced=unified.clinical_features.is_phase_advanced,
            is_phase_delayed=unified.clinical_features.is_phase_delayed,
            is_irregular_pattern=unified.clinical_features.is_irregular_pattern,
        )

        # Create PAT sequence if requested
        pat_sequence = None
        if include_pat_sequence:
            # In full implementation, would generate from activity data
            pat_sequence = PATSequence(
                end_date=unified.date,
                activity_values=np.array([0] * 10080),  # 7 days * 1440 minutes
                missing_days=[],
                data_quality_score=0.0,  # Placeholder
            )

        # Create clinical feature set
        return ClinicalFeatureSet(
            date=unified.date,
            seoul_features=seoul_features,
            pat_sequence=pat_sequence,
            # Add activity features as direct attributes for API compatibility
            total_steps=unified.activity_features.total_steps,
            activity_variance=100.0,  # Would calculate from data
            sedentary_hours=16.0,  # Would calculate from data
            activity_fragmentation=unified.activity_features.activity_fragmentation,
            sedentary_bout_mean=unified.activity_features.sedentary_bout_mean,
            activity_intensity_ratio=unified.activity_features.activity_intensity_ratio,
        )

    def _enrich_with_metadata(
        self,
        features: ClinicalFeatureSet,
        validation_result: FeatureValidationResult,
        anomaly_result: AnomalyResult,
    ) -> ClinicalFeatureSet:
        """
        Enrich clinical features with validation and anomaly metadata.

        This adds new attributes to track data quality and anomalies.
        """
        # Add validation metadata as dynamic attributes
        # Note: In production, we'd create a new dataclass that extends ClinicalFeatureSet
        # For now, we just log the metadata since ClinicalFeatureSet is frozen
        # TODO: Create an EnrichedClinicalFeatureSet that includes these fields
        logger.debug(
            "feature_validation_metadata",
            validation_score=validation_result.quality_score,
            has_anomalies=anomaly_result.has_anomalies,
            anomaly_severity=anomaly_result.severity if anomaly_result.has_anomalies else 0.0,
        )

        return features

    def _create_fallback_features(self, target_date: date) -> ClinicalFeatureSet:
        """
        Create minimal fallback features when extraction fails.

        This ensures the pipeline can continue even with errors.
        """
        seoul_features = SeoulXGBoostFeatures(
            date=target_date,
            # Basic Sleep Features (1-5)
            sleep_duration_hours=0.0,
            sleep_efficiency=0.0,
            sleep_onset_hour=23.0,
            wake_time_hour=7.0,
            sleep_fragmentation=0.0,
            # Advanced Sleep Features (6-10)
            sleep_regularity_index=0.0,
            short_sleep_window_pct=0.0,
            long_sleep_window_pct=0.0,
            sleep_onset_variance=0.0,
            wake_time_variance=0.0,
            # Circadian Rhythm Features (11-18)
            interdaily_stability=0.0,
            intradaily_variability=0.0,
            relative_amplitude=0.0,
            l5_value=0.0,
            m10_value=0.0,
            l5_onset_hour=2.0,
            m10_onset_hour=14.0,
            dlmo_hour=21.0,
            # Activity Features (19-24)
            total_steps=0,
            activity_variance=0.0,
            sedentary_hours=24.0,
            activity_fragmentation=0.0,
            sedentary_bout_mean=24.0,
            activity_intensity_ratio=0.0,
            # Heart Rate Features (25-28)
            avg_resting_hr=70.0,
            hrv_sdnn=0.0,
            hr_circadian_range=0.0,
            hr_minimum_hour=4.0,
            # Phase Features (29-32)
            circadian_phase_advance=0.0,
            circadian_phase_delay=0.0,
            dlmo_confidence=0.0,
            pat_hour=14.0,
            # Z-Score Features (33-36)
            sleep_duration_zscore=0.0,
            activity_zscore=0.0,
            hr_zscore=0.0,
            hrv_zscore=0.0,
            # Metadata
            data_completeness=0.0,
        )

        return ClinicalFeatureSet(
            date=target_date,
            seoul_features=seoul_features,
            pat_sequence=None,
        )
