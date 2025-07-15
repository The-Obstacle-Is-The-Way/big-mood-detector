"""
Tests for Feature Extraction Service

Test-driven development for clinical feature extraction.
Following Clean Architecture and SOLID principles.
"""

from datetime import UTC, date, datetime

import pytest

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.feature_extraction_service import (
    ClinicalFeatures,
    FeatureExtractionService,
)


class TestClinicalFeatures:
    """Test suite for ClinicalFeatures value object."""

    def test_create_clinical_features(self):
        """Test creating clinical features."""
        # ARRANGE & ACT
        features = ClinicalFeatures(
            date=date(2024, 1, 1),
            # Sleep features
            sleep_duration_hours=7.5,
            sleep_efficiency=0.88,
            sleep_fragmentation=0.15,
            sleep_onset_hour=23.5,
            wake_time_hour=7.0,
            # Activity features
            total_steps=10000,
            activity_variance=0.25,
            sedentary_hours=8.0,
            peak_activity_hour=14,
            # Heart features
            avg_resting_hr=65.0,
            hrv_sdnn=45.0,
            hr_circadian_range=20.0,
            # Circadian features
            circadian_alignment_score=0.85,
            # Clinical indicators
            is_clinically_significant=False,
            clinical_notes=[],
        )

        # ASSERT
        assert features.date == date(2024, 1, 1)
        assert features.sleep_duration_hours == 7.5
        assert features.total_steps == 10000
        assert features.avg_resting_hr == 65.0
        assert features.circadian_alignment_score == 0.85

    def test_clinical_features_is_immutable(self):
        """Test that clinical features cannot be modified."""
        # ARRANGE
        features = ClinicalFeatures(
            date=date(2024, 1, 1),
            sleep_duration_hours=7.5,
        )

        # ACT & ASSERT
        with pytest.raises(AttributeError):
            features.sleep_duration_hours = 8.0

    def test_feature_vector_generation(self):
        """Test generation of ML-ready feature vector."""
        # ARRANGE
        features = ClinicalFeatures(
            date=date(2024, 1, 1),
            sleep_duration_hours=7.5,
            sleep_efficiency=0.88,
            sleep_fragmentation=0.15,
            sleep_onset_hour=23.5,
            wake_time_hour=7.0,
            total_steps=10000,
            activity_variance=0.25,
            sedentary_hours=8.0,
            peak_activity_hour=14,
            avg_resting_hr=65.0,
            hrv_sdnn=45.0,
            hr_circadian_range=20.0,
            circadian_alignment_score=0.85,
        )

        # ACT
        vector = features.to_feature_vector()

        # ASSERT
        assert isinstance(vector, list)
        assert len(vector) == 13  # Number of numeric features
        assert vector[0] == 7.5  # sleep_duration_hours
        assert vector[5] == 10000  # total_steps
        assert vector[9] == 65.0  # avg_resting_hr

    def test_clinical_significance_detection(self):
        """Test detection of clinically significant patterns."""
        # ARRANGE - Multiple concerning patterns
        features = ClinicalFeatures(
            date=date(2024, 1, 1),
            sleep_duration_hours=3.5,  # Very low
            sleep_efficiency=0.65,  # Poor
            total_steps=20000,  # Very high
            avg_resting_hr=95.0,  # Elevated
            hrv_sdnn=15.0,  # Low
            is_clinically_significant=True,
            clinical_notes=[
                "Severe sleep deprivation",
                "Hyperactivity pattern",
                "Elevated resting heart rate",
                "Low HRV indicates poor recovery",
            ],
        )

        # ASSERT
        assert features.is_clinically_significant
        assert len(features.clinical_notes) == 4
        assert "sleep deprivation" in features.clinical_notes[0]


class TestFeatureExtractionService:
    """Test suite for FeatureExtractionService."""

    @pytest.fixture
    def service(self):
        """Provide FeatureExtractionService instance."""
        return FeatureExtractionService()

    @pytest.fixture
    def sample_day_data(self):
        """Provide sample data for a single day."""
        day = datetime(2024, 1, 1, tzinfo=UTC)
        
        sleep_records = [
            SleepRecord(
                source_name="Apple Watch",
                start_date=day.replace(hour=23) - datetime.timedelta(days=1),
                end_date=day.replace(hour=7),
                state=SleepState.ASLEEP,
            )
        ]
        
        activity_records = [
            ActivityRecord(
                source_name="iPhone",
                start_date=day.replace(hour=10),
                end_date=day.replace(hour=11),
                activity_type=ActivityType.STEP_COUNT,
                value=5000.0,
                unit="count",
            ),
            ActivityRecord(
                source_name="iPhone",
                start_date=day.replace(hour=14),
                end_date=day.replace(hour=15),
                activity_type=ActivityType.STEP_COUNT,
                value=3000.0,
                unit="count",
            ),
        ]
        
        heart_records = [
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=day.replace(hour=7),
                metric_type=HeartMetricType.HEART_RATE,
                value=58.0,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            ),
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=day.replace(hour=20),
                metric_type=HeartMetricType.HEART_RATE,
                value=72.0,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            ),
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=day.replace(hour=12),
                metric_type=HeartMetricType.HRV_SDNN,
                value=45.0,
                unit="ms",
            ),
        ]
        
        return sleep_records, activity_records, heart_records

    def test_extract_features_empty_data(self, service):
        """Test feature extraction with no data."""
        # ACT
        features = service.extract_features(
            sleep_records=[],
            activity_records=[],
            heart_records=[],
        )

        # ASSERT
        assert features == {}

    def test_extract_features_single_day(self, service, sample_day_data):
        """Test feature extraction for a single day."""
        # ARRANGE
        sleep_records, activity_records, heart_records = sample_day_data

        # ACT
        features = service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
        )

        # ASSERT
        assert len(features) == 1
        assert date(2024, 1, 1) in features
        
        day_features = features[date(2024, 1, 1)]
        assert day_features.sleep_duration_hours == 8.0
        assert day_features.total_steps == 8000  # 5000 + 3000
        assert day_features.avg_resting_hr == 65.0  # (58 + 72) / 2
        assert day_features.hrv_sdnn == 45.0

    def test_missing_data_handling(self, service):
        """Test handling of partial data (e.g., no heart rate data)."""
        # ARRANGE
        sleep_records = [
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2023, 12, 31, 23, tzinfo=UTC),
                end_date=datetime(2024, 1, 1, 7, tzinfo=UTC),
                state=SleepState.ASLEEP,
            )
        ]
        
        activity_records = [
            ActivityRecord(
                source_name="iPhone",
                start_date=datetime(2024, 1, 1, 10, tzinfo=UTC),
                end_date=datetime(2024, 1, 1, 11, tzinfo=UTC),
                activity_type=ActivityType.STEP_COUNT,
                value=5000.0,
                unit="count",
            )
        ]

        # ACT
        features = service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],  # No heart data
        )

        # ASSERT
        assert len(features) == 1
        day_features = features[date(2024, 1, 1)]
        assert day_features.sleep_duration_hours == 8.0
        assert day_features.total_steps == 5000
        assert day_features.avg_resting_hr == 0.0  # Default when no data
        assert day_features.hrv_sdnn == 0.0

    def test_circadian_alignment_calculation(self, service):
        """Test calculation of circadian alignment score."""
        # ARRANGE - Well-aligned circadian rhythm
        day = datetime(2024, 1, 1, tzinfo=UTC)
        
        sleep_records = [
            SleepRecord(
                source_name="Apple Watch",
                start_date=day.replace(hour=22) - datetime.timedelta(days=1),
                end_date=day.replace(hour=6),
                state=SleepState.ASLEEP,
            )
        ]
        
        activity_records = [
            # Morning activity
            ActivityRecord(
                source_name="iPhone",
                start_date=day.replace(hour=7),
                end_date=day.replace(hour=8),
                activity_type=ActivityType.STEP_COUNT,
                value=2000.0,
                unit="count",
            ),
            # Evening low activity
            ActivityRecord(
                source_name="iPhone",
                start_date=day.replace(hour=21),
                end_date=day.replace(hour=22),
                activity_type=ActivityType.STEP_COUNT,
                value=500.0,
                unit="count",
            ),
        ]
        
        heart_records = [
            # Low morning HR
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=day.replace(hour=6),
                metric_type=HeartMetricType.HEART_RATE,
                value=55.0,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            ),
            # Higher evening HR
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=day.replace(hour=20),
                metric_type=HeartMetricType.HEART_RATE,
                value=70.0,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            ),
        ]

        # ACT
        features = service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
        )

        # ASSERT
        day_features = features[date(2024, 1, 1)]
        assert day_features.circadian_alignment_score > 0.7  # Good alignment

    def test_clinical_significance_flagging(self, service):
        """Test flagging of clinically significant patterns."""
        # ARRANGE - Multiple concerning patterns
        day = datetime(2024, 1, 1, tzinfo=UTC)
        
        sleep_records = [
            # Very short sleep
            SleepRecord(
                source_name="Apple Watch",
                start_date=day.replace(hour=3),
                end_date=day.replace(hour=6),
                state=SleepState.ASLEEP,
            )
        ]
        
        activity_records = [
            # Extremely high activity
            ActivityRecord(
                source_name="iPhone",
                start_date=day.replace(hour=0),
                end_date=day.replace(hour=23),
                activity_type=ActivityType.STEP_COUNT,
                value=25000.0,
                unit="count",
            )
        ]
        
        heart_records = [
            # High resting HR
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=day.replace(hour=10),
                metric_type=HeartMetricType.HEART_RATE,
                value=105.0,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            ),
            # Low HRV
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=day.replace(hour=7),
                metric_type=HeartMetricType.HRV_SDNN,
                value=15.0,
                unit="ms",
            ),
        ]

        # ACT
        features = service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
        )

        # ASSERT
        day_features = features[date(2024, 1, 1)]
        assert day_features.is_clinically_significant
        assert len(day_features.clinical_notes) > 0
        assert any("sleep" in note.lower() for note in day_features.clinical_notes)
        assert any("activity" in note.lower() for note in day_features.clinical_notes)

    def test_multiple_days_feature_extraction(self, service):
        """Test feature extraction across multiple days."""
        # ARRANGE
        records = []
        for day_offset in range(3):
            day = datetime(2024, 1, day_offset + 1, tzinfo=UTC)
            records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=day.replace(hour=23) - datetime.timedelta(days=1),
                    end_date=day.replace(hour=7),
                    state=SleepState.ASLEEP,
                )
            )

        # ACT
        features = service.extract_features(
            sleep_records=records,
            activity_records=[],
            heart_records=[],
        )

        # ASSERT
        assert len(features) == 3
        for day_num in range(1, 4):
            assert date(2024, 1, day_num) in features
            assert features[date(2024, 1, day_num)].sleep_duration_hours == 8.0