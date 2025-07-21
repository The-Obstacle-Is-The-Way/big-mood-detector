"""
Test Feature Engineering Orchestrator

TDD approach for creating a high-level orchestrator that coordinates
all feature engineering components in a clean, modular way.
"""

from datetime import date, datetime, time, timedelta

import pytest

class TestFeatureEngineeringOrchestrator:
    """Test the feature engineering orchestration service."""

    @pytest.fixture
    def orchestrator(self):
        """Create FeatureEngineeringOrchestrator instance."""

        return FeatureEngineeringOrchestrator()

    @pytest.fixture
    def sample_sleep_data(self):
        """Create sample sleep data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(30):
            summary = DailySleepSummary(
                date=base_date + timedelta(days=i),
                total_time_in_bed_hours=8.5,
                total_sleep_hours=7.5 + (i % 3) * 0.2,
                sleep_efficiency=0.88,
                sleep_sessions=1,
                longest_sleep_hours=7.5,
                sleep_fragmentation_index=0.1,
                earliest_bedtime=time(23, 0),
                latest_wake_time=time(7, 0),
                mid_sleep_time=datetime.combine(
                    base_date + timedelta(days=i + 1), time(3, 0)
                ),
            )
            summaries.append(summary)

        return summaries

    @pytest.fixture
    def sample_activity_data(self):
        """Create sample activity data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(30):
            summary = DailyActivitySummary(
                date=base_date + timedelta(days=i),
                total_steps=8000.0 + (i % 7) * 500,
                total_active_energy=300.0,
                total_distance_km=6.0,
                flights_climbed=10.0,
                activity_sessions=3,
                peak_activity_hour=14,
                activity_variance=0.2,
                sedentary_hours=14.0,
                active_hours=3.0,
                earliest_activity=time(7, 0),
                latest_activity=time(21, 0),
            )
            summaries.append(summary)

        return summaries

    @pytest.fixture
    def sample_heart_data(self):
        """Create sample heart rate data."""
        summaries = []
        base_date = date(2024, 1, 1)

        for i in range(30):
            summary = DailyHeartSummary(
                date=base_date + timedelta(days=i),
                avg_resting_hr=65.0 + (i % 5),
                min_hr=50.0,
                max_hr=140.0,
                avg_hrv_sdnn=45.0 + (i % 3) * 2,
                min_hrv_sdnn=40.0,
                hr_measurements=100,
                hrv_measurements=20,
                high_hr_episodes=0,
                low_hr_episodes=0,
                circadian_hr_range=15.0,
                morning_hr=62.0,
                evening_hr=68.0,
            )
            summaries.append(summary)

        return summaries

    def test_extract_features_for_date(
        self, orchestrator, sample_sleep_data, sample_activity_data, sample_heart_data
    ):
        """Test feature extraction for a specific date."""
        target_date = date(2024, 1, 15)

        features = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=sample_sleep_data,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
            lookback_days=14,
        )

        assert features is not None
        assert features.date == target_date

        # Should have all feature categories
        assert hasattr(features, "sleep_features")
        assert hasattr(features, "circadian_features")
        assert hasattr(features, "activity_features")
        assert hasattr(features, "temporal_features")
        assert hasattr(features, "clinical_features")

        # Check some specific features
        assert features.sleep_features.total_sleep_hours > 0
        assert 0 <= features.sleep_features.sleep_efficiency <= 1
        assert features.activity_features.total_steps > 0

    def test_extract_features_batch(
        self, orchestrator, sample_sleep_data, sample_activity_data, sample_heart_data
    ):
        """Test batch feature extraction for multiple dates."""
        start_date = date(2024, 1, 10)
        end_date = date(2024, 1, 20)

        feature_set = orchestrator.extract_features_batch(
            start_date=start_date,
            end_date=end_date,
            sleep_data=sample_sleep_data,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
            lookback_days=7,
        )

        # Should have features for each day in range
        assert len(feature_set) == 11  # 10-20 inclusive

        # Check dates are correct
        dates = [f.date for f in feature_set]
        assert min(dates) == start_date
        assert max(dates) == end_date

    def test_extract_features_with_missing_domains(
        self, orchestrator, sample_sleep_data
    ):
        """Test feature extraction when some domains are missing."""
        target_date = date(2024, 1, 15)

        # Only provide sleep data
        features = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=sample_sleep_data,
            activity_data=[],
            heart_data=[],
            lookback_days=14,
        )

        assert features is not None
        # Sleep features should be calculated
        assert features.sleep_features.total_sleep_hours > 0
        # Activity features should have defaults
        assert features.activity_features.total_steps == 0
        # Should still calculate what's possible
        assert features.sleep_features.sleep_regularity_index >= 0

    def test_feature_validation(
        self, orchestrator, sample_sleep_data, sample_activity_data, sample_heart_data
    ):
        """Test feature validation and quality checks."""
        target_date = date(2024, 1, 15)

        features = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=sample_sleep_data,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
            lookback_days=14,
        )

        # Validate features
        validation_result = orchestrator.validate_features(features)

        assert hasattr(validation_result, "is_valid")
        assert hasattr(validation_result, "missing_domains")
        assert hasattr(validation_result, "quality_score")
        assert hasattr(validation_result, "warnings")

        # With complete data, should be valid
        assert validation_result.is_valid is True
        assert len(validation_result.missing_domains) == 0
        assert validation_result.quality_score > 0.8

    def test_feature_completeness_report(
        self, orchestrator, sample_sleep_data, sample_activity_data, sample_heart_data
    ):
        """Test feature completeness reporting."""
        report = orchestrator.generate_completeness_report(
            sleep_data=sample_sleep_data,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
        )

        assert hasattr(report, "total_days")
        assert hasattr(report, "sleep_coverage")
        assert hasattr(report, "activity_coverage")
        assert hasattr(report, "heart_coverage")
        assert hasattr(report, "full_coverage_days")
        assert hasattr(report, "gaps")

        # With our sample data, should have full coverage
        assert report.total_days == 30
        assert report.sleep_coverage == 1.0
        assert report.activity_coverage == 1.0
        assert report.heart_coverage == 1.0

    def test_configuration_injection(self):
        """Test configuration injection for all calculators."""
        from big_mood_detector.domain.services.feature_engineering_orchestrator import FeatureEngineeringOrchestrator

        config = {
            "sleep_windows": {"short_sleep_threshold": 5.0},
            "circadian": {"phase_threshold": 3.0},
            "activity": {"high_activity_threshold": 12000},
            "temporal": {"significance_level": 0.01},
        }

        orchestrator = FeatureEngineeringOrchestrator(config=config)

        # Should pass config to all calculators
        assert orchestrator.config == config
        # Verify calculators are configured (this would be more detailed in real impl)
        assert orchestrator is not None

    def test_feature_caching(
        self, orchestrator, sample_sleep_data, sample_activity_data, sample_heart_data
    ):
        """Test feature caching for performance."""
        target_date = date(2024, 1, 15)

        # First extraction
        features1 = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=sample_sleep_data,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
            lookback_days=14,
            use_cache=True,
        )

        # Second extraction (should use cache)
        features2 = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=sample_sleep_data,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
            lookback_days=14,
            use_cache=True,
        )

        # Should return same features
        assert (
            features1.sleep_features.total_sleep_hours
            == features2.sleep_features.total_sleep_hours
        )

        # Clear cache
        orchestrator.clear_cache()

        # Third extraction (cache cleared)
        features3 = orchestrator.extract_features_for_date(
            target_date=target_date,
            sleep_data=sample_sleep_data,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
            lookback_days=14,
            use_cache=False,
        )

        # Should still be same values (deterministic)
        assert (
            features1.sleep_features.total_sleep_hours
            == features3.sleep_features.total_sleep_hours
        )

    def test_get_feature_importance(self, orchestrator):
        """Test getting feature importance for model interpretability."""
        importance = orchestrator.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0

        # Should have importance for key features
        assert "sleep_regularity_index" in importance
        assert "circadian_phase_advance" in importance
        assert "activity_fragmentation" in importance

        # Importance values should be normalized
        for _feature, value in importance.items():
            assert 0 <= value <= 1

    def test_anomaly_detection_integration(
        from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary

        self, orchestrator, sample_sleep_data, sample_activity_data, sample_heart_data
    ):
        """Test integration with anomaly detection."""
        # Add an anomalous day
        anomalous_sleep = sample_sleep_data.copy()
        anomalous_sleep[15] = DailySleepSummary(
            date=anomalous_sleep[15].date,
            total_time_in_bed_hours=2.0,  # Very short sleep
            total_sleep_hours=1.5,
            sleep_efficiency=0.75,
            sleep_sessions=3,  # Fragmented
            longest_sleep_hours=0.5,
            sleep_fragmentation_index=0.8,
            earliest_bedtime=time(3, 0),
            latest_wake_time=time(5, 0),
            mid_sleep_time=datetime.combine(date(2024, 1, 16), time(4, 0)),
        )

        features = orchestrator.extract_features_for_date(
            target_date=date(2024, 1, 16),
            sleep_data=anomalous_sleep,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
            lookback_days=14,
        )

        # Should detect anomalies
        anomalies = orchestrator.detect_anomalies(features)

        assert hasattr(anomalies, "has_anomalies")
        assert hasattr(anomalies, "anomaly_domains")
        assert hasattr(anomalies, "severity")

        assert anomalies.has_anomalies is True
        assert "sleep" in anomalies.anomaly_domains

    def test_export_features_to_dataframe(
        self, orchestrator, sample_sleep_data, sample_activity_data, sample_heart_data
    ):
        """Test exporting features to pandas DataFrame format."""
        start_date = date(2024, 1, 10)
        end_date = date(2024, 1, 15)

        feature_set = orchestrator.extract_features_batch(
            start_date=start_date,
            end_date=end_date,
            sleep_data=sample_sleep_data,
            activity_data=sample_activity_data,
            heart_data=sample_heart_data,
            lookback_days=7,
        )

        # Export to dict format (for DataFrame conversion)
        df_data = orchestrator.export_features_to_dict(feature_set)

        assert isinstance(df_data, list)
        assert len(df_data) == 6  # 6 days

        # Each row should have all features flattened
        first_row = df_data[0]
        assert "date" in first_row
        assert "sleep_duration_hours" in first_row
        assert "sleep_regularity_index" in first_row
        assert "total_steps" in first_row
        assert "mood_risk_score" in first_row
