"""
Test Clinical Feature Extractor with Personal Calibration

Tests that ClinicalFeatureExtractor can use personal baselines
for Z-score normalization.
"""

from datetime import date, datetime, time, timedelta
from unittest.mock import Mock

import pytest

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
    UserBaseline,
)
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
)


class TestClinicalFeatureExtractorWithCalibration:
    """Test personal calibration integration in feature extraction."""

    @pytest.fixture
    def mock_baseline_repository(self):
        """Create mock baseline repository."""
        return Mock(spec=BaselineRepositoryInterface)

    @pytest.fixture
    def sample_records(self):
        """Create sample health records."""
        base_date = datetime(2024, 1, 15, 23, 0)
        
        sleep_records = []
        activity_records = []
        heart_records = []
        
        for i in range(14):  # 2 weeks of data
            record_date = base_date - timedelta(days=i)
            
            # Sleep record
            sleep_records.append(
                SleepRecord(
                    source_name="com.apple.health",
                    start_date=record_date,
                    end_date=record_date + timedelta(hours=8),
                    state=SleepState.IN_BED,
                )
            )
            
            # Activity record
            activity_records.append(
                ActivityRecord(
                    id=f"activity_{i}",
                    start_date=record_date.replace(hour=10),
                    end_date=record_date.replace(hour=10) + timedelta(minutes=30),
                    activity_type="Walking",
                    duration_minutes=30,
                    distance_meters=2000,
                    step_count=2500,
                    energy_burned=100,
                    source="com.apple.health",
                    metadata={},
                )
            )
            
            # Heart rate record
            heart_records.append(
                HeartRateRecord(
                    id=f"hr_{i}",
                    start_date=record_date.replace(hour=12),
                    end_date=record_date.replace(hour=12) + timedelta(minutes=1),
                    value=65,
                    unit="bpm",
                    source="com.apple.health",
                    metadata={},
                )
            )
        
        return sleep_records, activity_records, heart_records

    def test_feature_extractor_accepts_calibration_params(self):
        """Test that ClinicalFeatureExtractor can be initialized with calibration."""
        mock_repo = Mock(spec=BaselineRepositoryInterface)
        
        # Should accept baseline_repository and user_id
        extractor = ClinicalFeatureExtractor(
            baseline_repository=mock_repo,
            user_id="test_user"
        )
        
        assert extractor.baseline_repository == mock_repo
        assert extractor.user_id == "test_user"

    def test_uses_personal_baselines_for_zscores(
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
            baseline_repository=mock_baseline_repository,
            user_id="test_user"
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

    def test_persists_updated_baselines(
        self, mock_baseline_repository, sample_records
    ):
        """Test that baselines are persisted after feature extraction."""
        sleep_records, activity_records, heart_records = sample_records
        
        # Create extractor with calibration
        extractor = ClinicalFeatureExtractor(
            baseline_repository=mock_baseline_repository,
            user_id="test_user"
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