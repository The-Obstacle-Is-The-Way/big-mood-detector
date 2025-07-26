"""
Test Feature Engineering Orchestrator Integration

Tests the integration of FeatureEngineeringOrchestrator into MoodPredictionPipeline
following TDD principles. Tests written BEFORE implementation.
"""

from datetime import date

import pandas as pd
import pytest

from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
)
from big_mood_detector.domain.services.feature_engineering_orchestrator import (
    FeatureEngineeringOrchestrator,
)
from big_mood_detector.domain.services.feature_types import (
    ActivityFeatureSet,
    CircadianFeatureSet,
    ClinicalFeatureSet,
    SleepFeatureSet,
    TemporalFeatureSet,
    UnifiedFeatureSet,
)


@pytest.fixture(autouse=True)
def clear_orchestrator_cache():
    """Clear orchestrator cache before each test to prevent cross-pollution."""
    # Clear before test
    orchestrator = FeatureEngineeringOrchestrator()
    orchestrator.clear_cache()

    yield

    # Clear after test
    orchestrator.clear_cache()


@pytest.mark.integration
class TestOrchestratorIntegration:
    """Test suite for orchestrator integration."""

    def test_orchestrator_converts_unified_to_clinical_features(self):
        """Test that orchestrator can convert UnifiedFeatureSet to ClinicalFeatureSet."""
        # GIVEN: An orchestrator instance
        orchestrator = FeatureEngineeringOrchestrator()

        # AND: Mock data for testing
        target_date = date(2025, 7, 1)
        sleep_data = []
        activity_data = []
        heart_data = []

        # WHEN: Extracting features
        unified_features = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=sleep_data,
            activity_data=activity_data,
            heart_data=heart_data,
        )

        # THEN: Should return UnifiedFeatureSet
        assert isinstance(unified_features, UnifiedFeatureSet)
        assert unified_features.date == target_date

        # AND: Should have all feature domains
        assert hasattr(unified_features, 'sleep_features')
        assert hasattr(unified_features, 'circadian_features')
        assert hasattr(unified_features, 'activity_features')
        assert hasattr(unified_features, 'temporal_features')
        assert hasattr(unified_features, 'clinical_features')

    def test_orchestrator_validation_detects_missing_data(self):
        """Test that validation correctly identifies missing data domains."""
        # GIVEN: An orchestrator with test features
        orchestrator = FeatureEngineeringOrchestrator()

        # Create features with missing sleep data
        features = UnifiedFeatureSet(
            date=date(2025, 7, 1),
            sleep_features=SleepFeatureSet(
                total_sleep_hours=0.0,  # No sleep data
                sleep_efficiency=0.0,
                sleep_regularity_index=0.0,
                interdaily_stability=0.0,
                intradaily_variability=0.0,
                relative_amplitude=0.0,
                short_sleep_window_pct=0.0,
                long_sleep_window_pct=0.0,
                sleep_onset_variance=0.0,
                wake_time_variance=0.0,
            ),
            circadian_features=CircadianFeatureSet(
                l5_value=100.0,
                m10_value=500.0,
                circadian_phase_advance=0.0,
                circadian_phase_delay=0.0,
                circadian_amplitude=0.5,
                phase_angle=2.0,
            ),
            activity_features=ActivityFeatureSet(
                total_steps=8000,
                activity_fragmentation=0.3,
                sedentary_bout_mean=45.0,
                sedentary_bout_max=120.0,
                activity_intensity_ratio=0.2,
                activity_rhythm_strength=0.7,
            ),
            temporal_features=TemporalFeatureSet(
                sleep_7day_mean=7.0,
                sleep_7day_std=1.0,
                activity_7day_mean=8000.0,
                activity_7day_std=2000.0,
                hr_7day_mean=65.0,
                hr_7day_std=5.0,
                sleep_trend_slope=-0.1,
                activity_trend_slope=50.0,
                sleep_momentum=0.0,
                activity_momentum=100.0,
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

        # WHEN: Validating features
        validation_result = orchestrator.validate_features(features)

        # THEN: Should detect missing sleep domain
        assert not validation_result.is_valid
        assert "sleep" in validation_result.missing_domains
        assert validation_result.quality_score < 1.0

    def test_orchestrator_anomaly_detection(self):
        """Test that orchestrator detects various anomalies."""
        # GIVEN: An orchestrator and features with anomalies
        orchestrator = FeatureEngineeringOrchestrator()

        # Create features with sleep anomaly (hypersomnia)
        features = UnifiedFeatureSet(
            date=date(2025, 7, 1),
            sleep_features=SleepFeatureSet(
                total_sleep_hours=14.0,  # Hypersomnia
                sleep_efficiency=0.95,
                sleep_regularity_index=80.0,
                interdaily_stability=0.8,
                intradaily_variability=0.4,
                relative_amplitude=0.7,
                short_sleep_window_pct=0.0,
                long_sleep_window_pct=60.0,  # Many long sleep windows
                sleep_onset_variance=0.5,
                wake_time_variance=0.5,
            ),
            circadian_features=CircadianFeatureSet(
                l5_value=50.0,
                m10_value=300.0,
                circadian_phase_advance=6.0,  # Large phase advance
                circadian_phase_delay=0.0,
                circadian_amplitude=0.4,
                phase_angle=3.0,
            ),
            activity_features=ActivityFeatureSet(
                total_steps=500,  # Very low activity
                activity_fragmentation=0.8,  # High fragmentation
                sedentary_bout_mean=180.0,  # Long sedentary periods
                sedentary_bout_max=360.0,
                activity_intensity_ratio=0.05,  # Very low intensity
                activity_rhythm_strength=0.2,
            ),
            temporal_features=TemporalFeatureSet(
                sleep_7day_mean=12.0,
                sleep_7day_std=2.0,
                activity_7day_mean=1000.0,
                activity_7day_std=500.0,
                hr_7day_mean=60.0,
                hr_7day_std=3.0,
                sleep_trend_slope=-0.5,
                activity_trend_slope=-100.0,
                sleep_momentum=-0.2,
                activity_momentum=-50.0,
            ),
            clinical_features=ClinicalFeatureSet(
                is_hypersomnia_pattern=True,
                is_insomnia_pattern=False,
                is_phase_advanced=True,
                is_phase_delayed=False,
                is_irregular_pattern=False,
                mood_risk_score=0.75,
            ),
        )

        # WHEN: Detecting anomalies
        anomaly_result = orchestrator.detect_anomalies(features)

        # THEN: Should detect multiple anomalies
        assert anomaly_result.has_anomalies
        assert "sleep" in anomaly_result.anomaly_domains
        assert "circadian" in anomaly_result.anomaly_domains
        assert "activity" in anomaly_result.anomaly_domains
        assert anomaly_result.severity >= 0.6

    def test_orchestrator_feature_importance(self):
        """Test that orchestrator provides feature importance scores."""
        # GIVEN: An orchestrator
        orchestrator = FeatureEngineeringOrchestrator()

        # WHEN: Getting feature importance
        importance = orchestrator.get_feature_importance()

        # THEN: Should have expected important features
        assert "sleep_regularity_index" in importance
        assert "circadian_phase_advance" in importance
        assert "mood_risk_score" in importance

        # AND: Values should be in valid range
        for _feature, score in importance.items():
            assert 0.0 <= score <= 1.0

        # AND: Sleep regularity should be most important
        assert importance["sleep_regularity_index"] >= 0.9

    def test_clinical_extractor_orchestrator_compatibility(self):
        """Test that we can adapt between ClinicalFeatureExtractor and Orchestrator."""
        # This test demonstrates how to integrate the orchestrator
        # while maintaining backward compatibility

        # GIVEN: Both extractors
        clinical_extractor = ClinicalFeatureExtractor()
        orchestrator = FeatureEngineeringOrchestrator()

        # Mock some data
        target_date = date(2025, 7, 1)
        sleep_records = []
        activity_records = []
        heart_records = []

        # WHEN: Using clinical extractor (current approach)
        clinical_features = clinical_extractor.extract_clinical_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=target_date,
        )

        # AND: Using orchestrator (new approach)
        unified_features = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=[],  # Note: different parameter names
            activity_data=[],
            heart_data=[],
        )

        # THEN: Both should produce features for the same date
        assert clinical_features.date == unified_features.date

        # AND: We should be able to convert unified to clinical format
        # This demonstrates the adapter pattern we'll implement
        assert hasattr(unified_features.clinical_features, 'mood_risk_score')

    def test_orchestrator_caching_performance(self):
        """Test that orchestrator caching improves performance."""
        # GIVEN: An orchestrator with caching
        orchestrator = FeatureEngineeringOrchestrator()

        # Clear cache to start fresh
        orchestrator.clear_cache()

        # Mock the internal feature engineer to track calls
        original_extract = orchestrator.feature_engineer.extract_advanced_features
        call_count = 0

        def counting_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_extract(*args, **kwargs)

        orchestrator.feature_engineer.extract_advanced_features = counting_extract

        # WHEN: Extracting features for same date twice
        target_date = date(2025, 7, 1)

        # First call
        features1 = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=[],
            activity_data=[],
            heart_data=[],
            lookback_days=30,
            use_cache=True,
        )

        # Second call (should use cache)
        features2 = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=[],
            activity_data=[],
            heart_data=[],
            lookback_days=30,
            use_cache=True,
        )

        # THEN: Extract should only be called once
        assert call_count == 1

        # AND: Results should be identical
        assert features1.date == features2.date
        assert features1.clinical_features.mood_risk_score == features2.clinical_features.mood_risk_score

    def test_orchestrator_export_features(self):
        """Test that orchestrator can export features to dict format."""
        # GIVEN: An orchestrator with some features
        orchestrator = FeatureEngineeringOrchestrator()

        # Extract features for multiple dates
        start_date = date(2025, 7, 1)
        end_date = date(2025, 7, 3)

        feature_sets = orchestrator.extract_features_batch(
            start_date=start_date,
            end_date=end_date,
            sleep_data=[],
            activity_data=[],
            heart_data=[],
        )

        # WHEN: Exporting to dict format
        exported = orchestrator.export_features_to_dict(feature_sets)

        # THEN: Should have correct structure
        assert len(exported) == 3  # 3 days

        # Check first day has all expected fields
        first_day = exported[0]
        assert first_day["date"] == start_date
        assert "sleep_duration_hours" in first_day
        assert "total_steps" in first_day
        assert "mood_risk_score" in first_day

        # Verify it's suitable for DataFrame conversion
        df = pd.DataFrame(exported)
        assert len(df) == 3
        assert "sleep_duration_hours" in df.columns
