"""
Test Clinical Feature Extractor with Personal Calibration

Tests that ClinicalFeatureExtractor can use personal baselines
for Z-score normalization.
"""

from datetime import date, datetime, time, timedelta
from unittest.mock import Mock

import pytest

class TestClinicalFeatureExtractorWithCalibration:
    """Test personal calibration integration in feature extraction."""

    @pytest.fixture
    def mock_baseline_repository(self):
        """Create mock baseline repository."""
        mock_repo = Mock(spec=BaselineRepositoryInterface)

        # Mock baseline data that AdvancedFeatureEngineer expects
        mock_baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date(2024, 1, 1),
            sleep_mean=7.5,  # 7.5 hours average
            sleep_std=1.0,  # 1 hour standard deviation
            activity_mean=8000.0,  # 8000 steps average
            activity_std=2000.0,  # 2000 steps standard deviation
            circadian_phase=22.0,  # 10 PM phase
            last_updated=datetime(2024, 1, 1, 0, 0),
            data_points=30,
        )

        mock_repo.get_baseline.return_value = mock_baseline
        mock_repo.save_baseline.return_value = None

        return mock_repo

    @pytest.fixture
    def sample_records(self):
        """Create sample records for testing."""
        sleep_records = []
        activity_records = []
        heart_rate_records = []

        # Create 30 days of sample data
        for i in range(30):
            record_date = date(2024, 1, 1) + timedelta(days=i)
            record_datetime = datetime.combine(record_date, time(9, 0))

            # Sleep record
            sleep_records.append(
                SleepRecord(
                    source_name="com.apple.health",
                    start_date=datetime.combine(
                        record_date, time(23, 0)
                    ),  # Sleep starts at 11 PM
                    end_date=datetime.combine(
                        record_date + timedelta(days=1), time(7, 0)
                    ),  # Ends at 7 AM next day
                    state=SleepState.IN_BED,
                )
            )

            # Activity record
            activity_records.append(
                ActivityRecord(
                    source_name="com.apple.health",
                    start_date=record_datetime,
                    end_date=record_datetime + timedelta(hours=1),
                    activity_type=ActivityType.STEP_COUNT,
                    value=1000.0 + i * 100,  # Varying step counts
                    unit="count",
                )
            )

            # Heart rate record
            heart_rate_records.append(
                HeartRateRecord(
                    source_name="com.apple.health",
                    timestamp=record_datetime,
                    metric_type=HeartMetricType.HEART_RATE,
                    value=70.0 + i,  # Varying heart rates
                    unit="bpm",
                )
            )

        return sleep_records, activity_records, heart_rate_records

    def test_feature_extractor_accepts_calibration_params(self):
        """Test that ClinicalFeatureExtractor can be initialized with calibration."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import BaselineRepositoryInterface
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor

        mock_repo = Mock(spec=BaselineRepositoryInterface)

        # Should accept baseline_repository and user_id
        extractor = ClinicalFeatureExtractor(
            baseline_repository=mock_repo, user_id="test_user"
        )

        assert extractor.baseline_repository == mock_repo
        assert extractor.user_id == "test_user"

    def test_uses_personal_baselines_for_zscores(
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        self, mock_baseline_repository, sample_records
    ):
        """Test that personal baselines are used for Z-score calculation."""
        sleep_records, activity_records, heart_records = sample_records

        # Mock existing baseline
        existing_baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date(2024, 1, 1),
            sleep_mean=7.5,
            sleep_std=0.5,
            activity_mean=8500.0,
            activity_std=1200.0,
            circadian_phase=0.0,
            last_updated=datetime(2024, 1, 1, 12, 0),
            data_points=30,
        )
        mock_baseline_repository.get_baseline.return_value = existing_baseline

        # Create extractor with calibration
        extractor = ClinicalFeatureExtractor(
            baseline_repository=mock_baseline_repository, user_id="test_user"
        )

        # Extract features
        features = extractor.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=date(2024, 1, 15),
        )

        # Verify baseline was loaded
        mock_baseline_repository.get_baseline.assert_called_with("test_user")

        # Z-scores should be calculated (not just 0)
        assert features.sleep_duration_zscore != 0.0
        assert features.activity_zscore != 0.0

    def test_persists_updated_baselines(self, mock_baseline_repository, sample_records):
        """Test that baselines are persisted after feature extraction."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor

        sleep_records, activity_records, heart_records = sample_records

        # Create extractor with calibration
        extractor = ClinicalFeatureExtractor(
            baseline_repository=mock_baseline_repository, user_id="test_user"
        )

        # Extract features
        extractor.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=date(2024, 1, 15),
        )

        # Persist baselines
        extractor.persist_baselines()

        # Verify baseline was saved
        assert mock_baseline_repository.save_baseline.called

    def test_works_without_calibration(self, sample_records):
        """Test backward compatibility without calibration."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor

        sleep_records, activity_records, heart_records = sample_records

        # Create extractor without calibration params
        extractor = ClinicalFeatureExtractor()

        # Should work normally
        features = extractor.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=date(2024, 1, 15),
        )

        assert features is not None
        # Z-scores will be 0 without baselines
        assert features.sleep_duration_zscore == 0.0
