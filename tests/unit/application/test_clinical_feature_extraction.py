"""
Unit tests for clinical feature extraction with activity data.

Tests that ClinicalFeatureExtractor properly extracts activity features
from health data, ensuring they're available for API exposure.
"""

import pytest
from datetime import date, datetime, timedelta

from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
    ClinicalFeatureSet,
)


class TestClinicalFeatureExtractorActivity:
    """Test activity feature extraction in ClinicalFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a ClinicalFeatureExtractor instance."""
        return ClinicalFeatureExtractor()

    @pytest.fixture
    def sample_activity_records(self):
        """Generate sample activity records for testing."""
        records = []
        base_date = date.today() - timedelta(days=7)
        
        # Generate 7 days of step count data
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            
            # Morning walk
            records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=8),
                    end_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=9),
                    activity_type=ActivityType.STEP_COUNT,
                    value=3000.0,
                    unit="count",
                )
            )
            
            # Afternoon activity
            records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=14),
                    end_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=16),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000.0,
                    unit="count",
                )
            )
            
            # Evening walk
            records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=19),
                    end_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=20),
                    activity_type=ActivityType.STEP_COUNT,
                    value=2000.0,
                    unit="count",
                )
            )
            
        return records

    @pytest.fixture
    def sample_sleep_records(self):
        """Generate sample sleep records."""
        records = []
        base_date = date.today() - timedelta(days=7)
        
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=23),
                    end_date=datetime.combine(current_date + timedelta(days=1), datetime.min.time()) + timedelta(hours=7),
                    state=SleepState.ASLEEP,
                )
            )
            
        return records

    def test_extract_activity_features(self, extractor, sample_activity_records, sample_sleep_records):
        """Test that activity features are extracted and non-null."""
        # Extract features for the last date in our sample data
        target_date = date.today() - timedelta(days=1)  # Yesterday since our data goes up to yesterday
        feature_set = extractor.extract_clinical_features(
            sleep_records=sample_sleep_records,
            activity_records=sample_activity_records,
            heart_records=[],
            target_date=target_date,
        )
        
        # Verify feature set is returned
        assert isinstance(feature_set, ClinicalFeatureSet)
        
        # Verify activity features are present and non-null in seoul_features
        assert feature_set.seoul_features.total_steps is not None
        assert feature_set.seoul_features.total_steps > 0
        assert feature_set.seoul_features.activity_variance is not None
        assert feature_set.seoul_features.activity_variance >= 0
        assert feature_set.seoul_features.sedentary_hours is not None
        assert feature_set.seoul_features.sedentary_hours >= 0
        assert feature_set.seoul_features.activity_fragmentation is not None
        assert feature_set.seoul_features.sedentary_bout_mean is not None
        assert feature_set.seoul_features.activity_intensity_ratio is not None

    def test_activity_features_calculation(self, extractor, sample_activity_records, sample_sleep_records):
        """Test that activity features are calculated correctly."""
        # Extract features for the last date in our sample data
        target_date = date.today() - timedelta(days=1)
        feature_set = extractor.extract_clinical_features(
            sleep_records=sample_sleep_records,
            activity_records=sample_activity_records,
            heart_records=[],
            target_date=target_date,
        )
        
        # Expected daily steps = 3000 + 5000 + 2000 = 10000
        assert feature_set.seoul_features.total_steps == 10000
        
        # Activity variance should be > 0 given the varied activity pattern
        assert feature_set.seoul_features.activity_variance > 0
        
        # Sedentary hours should be reasonable (not all 24 hours)
        assert 0 <= feature_set.seoul_features.sedentary_hours < 24

    def test_missing_activity_data_handling(self, extractor, sample_sleep_records):
        """Test feature extraction with missing activity data."""
        # Extract features without activity records
        target_date = date.today() - timedelta(days=1)
        feature_set = extractor.extract_clinical_features(
            sleep_records=sample_sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=target_date,
        )
        
        # Activity features should have sensible defaults
        assert feature_set.seoul_features.total_steps == 0
        assert feature_set.seoul_features.activity_variance == 0
        assert feature_set.seoul_features.sedentary_hours == 24.0  # Assume full sedentary if no data
        assert feature_set.seoul_features.activity_fragmentation == 0
        assert feature_set.seoul_features.sedentary_bout_mean == 1440  # 24 hours in minutes
        assert feature_set.seoul_features.activity_intensity_ratio == 0

    def test_activity_features_in_xgboost_vector(self, extractor, sample_activity_records, sample_sleep_records):
        """Test that activity features are included in XGBoost feature vector."""
        target_date = date.today() - timedelta(days=1)
        feature_set = extractor.extract_clinical_features(
            sleep_records=sample_sleep_records,
            activity_records=sample_activity_records,
            heart_records=[],
            target_date=target_date,
        )
        
        # Get XGBoost feature vector from seoul_features
        feature_vector = feature_set.seoul_features.to_xgboost_features()
        
        # Should have 36 features
        assert len(feature_vector) == 36
        
        # Activity features are at indices 18-23 in the Seoul study format
        # Check that they're not all zeros
        activity_features = feature_vector[18:24]
        assert not all(f == 0 for f in activity_features)
        
        # Specifically check total_steps (index 18)
        assert feature_vector[18] == 10000.0

    def test_circadian_features_with_activity(self, extractor, sample_activity_records, sample_sleep_records):
        """Test that circadian rhythm features use activity data."""
        target_date = date.today() - timedelta(days=1)
        feature_set = extractor.extract_clinical_features(
            sleep_records=sample_sleep_records,
            activity_records=sample_activity_records,
            heart_records=[],
            target_date=target_date,
        )
        
        # Circadian features should be calculated from activity
        assert feature_set.seoul_features.interdaily_stability is not None
        assert 0 <= feature_set.seoul_features.interdaily_stability <= 1
        
        assert feature_set.seoul_features.intradaily_variability is not None
        assert feature_set.seoul_features.intradaily_variability >= 0
        
        assert feature_set.seoul_features.relative_amplitude is not None
        assert 0 <= feature_set.seoul_features.relative_amplitude <= 1
        
        # PAT (Principal Activity Time) should be calculated
        assert feature_set.seoul_features.pat_hour is not None
        assert 0 <= feature_set.seoul_features.pat_hour < 24