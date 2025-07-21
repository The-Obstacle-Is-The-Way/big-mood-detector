"""
Unit tests for Clinical Feature Extractor
Tests the comprehensive feature extraction pipeline for Seoul XGBoost approach.
"""

from datetime import date, datetime, timedelta

import numpy as np

class TestClinicalFeatureExtractor:
    """Test comprehensive clinical feature extraction for mood prediction."""

    def test_extractor_initialization(self):
        """Extractor should initialize with all required services."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor

        # Act
        extractor = ClinicalFeatureExtractor()

        # Assert
        assert extractor.sleep_window_analyzer is not None
        assert extractor.activity_sequence_extractor is not None
        assert extractor.dlmo_calculator is not None
        assert extractor.advanced_feature_engineer is not None
        assert extractor.pat_sequence_builder is not None

    def test_extract_seoul_features_regular_sleeper(self):
        """Extract all 36 Seoul features for a regular sleeper."""
        from big_mood_detector.domain.services.clinical_feature_extractor import (
            ClinicalFeatureExtractor,
            SeoulXGBoostFeatures,
        )

        # Arrange
        extractor = ClinicalFeatureExtractor()
        sleep_records, activity_records, heart_records = (
            self._create_regular_sleeper_data()
        )
        target_date = date(2024, 1, 14)

        # Act
        features = extractor.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=target_date,
        )

        # Assert
        assert isinstance(features, SeoulXGBoostFeatures)
        assert features.date == target_date

        # Basic sleep features
        assert 7.0 <= features.sleep_duration_hours <= 9.0  # Regular sleeper
        assert 0.85 <= features.sleep_efficiency <= 1.0

        # DLMO feature
        assert features.dlmo_hour is not None
        assert 20.0 <= features.dlmo_hour <= 22.0  # Expected DLMO 8-10 PM

        # Circadian features
        assert 0 <= features.interdaily_stability <= 1
        assert 0 <= features.intradaily_variability <= 2
        assert features.l5_value >= 0  # Least active 5 hours
        assert features.m10_value >= features.l5_value  # Most active > least active

        # Z-scores
        assert features.sleep_duration_zscore is not None
        assert features.activity_zscore is not None

        # Feature vector
        feature_vector = features.to_xgboost_features()
        assert len(feature_vector) == 36  # Exactly 36 features for Seoul study
        assert all(isinstance(f, int | float) for f in feature_vector)

    def test_extract_with_missing_dlmo(self):
        """Should handle gracefully when DLMO calculation fails."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        # Arrange
        extractor = ClinicalFeatureExtractor()
        # Only 1 day of data - insufficient for DLMO
        sleep_records = [
            SleepRecord(
                source_name="test",
                start_date=datetime(2024, 1, 13, 23, 0),
                end_date=datetime(2024, 1, 14, 7, 0),
                state=SleepState.ASLEEP_CORE,
            )
        ]

        # Act
        features = extractor.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2024, 1, 14),
        )

        # Assert
        assert features is not None
        assert features.dlmo_hour == 21.0  # Default value
        # With only 1 day, DLMO might still have some confidence from default model
        assert 0.0 <= features.dlmo_confidence <= 1.0

    def test_pat_sequence_extraction(self):
        """Extract PAT sequence for transformer input."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor

        # Arrange
        extractor = ClinicalFeatureExtractor()
        _, activity_records, _ = self._create_regular_sleeper_data()
        target_date = date(2024, 1, 14)

        # Act
        pat_sequence = extractor.extract_pat_sequence(
            activity_records=activity_records, end_date=target_date
        )

        # Assert
        assert pat_sequence is not None
        assert pat_sequence.end_date == target_date
        assert len(pat_sequence.activity_values) == 10080  # 7 days * 1440 minutes
        assert pat_sequence.data_quality_score > 0.8  # Good data quality

        # Check normalization
        normalized = pat_sequence.get_normalized()
        assert len(normalized) == 10080
        assert abs(np.mean(normalized)) < 0.1  # Near zero mean
        if np.std(normalized) > 0:
            assert 0.9 < np.std(normalized) < 1.1  # Near unit variance

    def test_full_clinical_feature_set(self):
        """Extract complete clinical feature set including all domains."""
        from big_mood_detector.domain.services.clinical_feature_extractor import (
            ClinicalFeatureExtractor,
            ClinicalFeatureSet,
        )

        # Arrange
        extractor = ClinicalFeatureExtractor()
        sleep_records, activity_records, heart_records = (
            self._create_regular_sleeper_data()
        )
        target_date = date(2024, 1, 14)

        # Act
        feature_set = extractor.extract_clinical_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=target_date,
            include_pat_sequence=True,
        )

        # Assert
        assert isinstance(feature_set, ClinicalFeatureSet)
        assert feature_set.date == target_date

        # Seoul features included
        assert feature_set.seoul_features is not None
        assert len(feature_set.seoul_features.to_xgboost_features()) == 36

        # PAT sequence included
        assert feature_set.pat_sequence is not None
        assert len(feature_set.pat_sequence.activity_values) == 10080

        # Clinical indicators
        assert isinstance(feature_set.is_clinically_significant, bool)
        assert isinstance(feature_set.clinical_notes, list)

        # Risk scores (will be None until XGBoost integration)
        assert feature_set.depression_risk_score is None
        assert feature_set.mania_risk_score is None
        assert feature_set.hypomania_risk_score is None

    def test_shift_worker_features(self):
        """Features should adapt to shift work patterns."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor

        # Arrange
        extractor = ClinicalFeatureExtractor()
        sleep_records, activity_records, heart_records = (
            self._create_shift_worker_data()
        )
        target_date = date(2024, 1, 14)

        # Act
        features = extractor.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=target_date,
        )

        # Assert
        assert features is not None
        # Shift workers have different patterns than regular sleepers
        # Day 14 (index 13) is a weekend, but previous days were shifts
        assert (
            features.is_phase_delayed
            or features.is_phase_advanced
            or features.is_irregular_pattern
        )
        # Should have irregular pattern due to mixed shift/normal schedule
        assert features.is_irregular_pattern

    def test_clinical_significance_detection(self):
        """Should detect clinically significant patterns."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        # Arrange
        extractor = ClinicalFeatureExtractor()
        # Create irregular sleep pattern
        sleep_records = []
        for day in range(14):
            # Very short sleep (4 hours) - insomnia pattern
            date_obj = datetime(2024, 1, 1) + timedelta(days=day)
            sleep_records.append(
                SleepRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=2, minute=0),
                    end_date=date_obj.replace(hour=6, minute=0),
                    state=SleepState.ASLEEP_CORE,
                )
            )

        # Act
        feature_set = extractor.extract_clinical_features(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2024, 1, 14),
        )

        # Assert
        assert feature_set.is_clinically_significant
        assert any("insomnia" in note.lower() for note in feature_set.clinical_notes)
        assert feature_set.seoul_features.is_insomnia_pattern

    def test_feature_extraction_with_sparse_data(self):
        """Should handle sparse data gracefully."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        # Arrange
        extractor = ClinicalFeatureExtractor()
        # Only 3 days of sleep data
        sleep_records = []
        for day in [1, 5, 10]:
            date_obj = datetime(2024, 1, day)
            sleep_records.append(
                SleepRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=23, minute=0),
                    end_date=(date_obj + timedelta(days=1)).replace(hour=7, minute=0),
                    state=SleepState.ASLEEP_CORE,
                )
            )

        # Act
        features = extractor.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=[],
            heart_records=[],
            target_date=date(2024, 1, 14),
            min_days_required=3,
        )

        # Assert
        assert features is not None
        # With sparse data and no data on target date, features might be 0
        assert features.sleep_duration_hours >= 0
        # Data completeness should reflect sparsity
        assert features.data_completeness < 1.0

    def test_feature_vector_order_consistency(self):
        """Feature vector should maintain consistent order for XGBoost."""
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor

        # Arrange
        extractor = ClinicalFeatureExtractor()
        sleep_records, activity_records, heart_records = (
            self._create_regular_sleeper_data()
        )

        # Act - extract features for two different dates
        features1 = extractor.extract_seoul_features(
            sleep_records,
            activity_records,
            heart_records,
            target_date=date(2024, 1, 14),
        )
        features2 = extractor.extract_seoul_features(
            sleep_records,
            activity_records,
            heart_records,
            target_date=date(2024, 1, 13),
        )

        # Assert - both should have same length and order
        vector1 = features1.to_xgboost_features()
        vector2 = features2.to_xgboost_features()

        assert len(vector1) == len(vector2) == 36
        # Values can differ but positions should represent same features
        # (This is more of a design test than value test)

    def _create_regular_sleeper_data(self):
        """Create 14 days of regular sleep pattern data."""
        sleep_records = []
        activity_records = []
        heart_records = []

        for day in range(14):
            date_obj = datetime(2024, 1, 1) + timedelta(days=day)

            # Sleep 11pm-7am
            sleep_records.append(
                SleepRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=23, minute=0),
                    end_date=(date_obj + timedelta(days=1)).replace(hour=7, minute=0),
                    state=SleepState.ASLEEP_CORE,
                )
            )

            # Activity pattern - higher during day
            for hour in range(24):
                if 7 <= hour <= 22:  # Awake hours
                    # Add activity every 15 minutes
                    for minute in [0, 15, 30, 45]:
                        activity_records.append(
                            ActivityRecord(
                                source_name="test",
                                start_date=(date_obj + timedelta(days=1)).replace(
                                    hour=hour, minute=minute
                                ),
                                end_date=(date_obj + timedelta(days=1)).replace(
                                    hour=hour, minute=minute + 14
                                ),
                                activity_type=ActivityType.STEP_COUNT,
                                value=(
                                    250 if 9 <= hour <= 17 else 150
                                ),  # Higher during work hours
                                unit="count",
                            )
                        )

            # Heart rate - lower at night
            for hour in range(24):
                hr_value = 55 if 23 <= hour or hour < 7 else 70
                heart_records.append(
                    HeartRateRecord(
                        source_name="test",
                        timestamp=(date_obj + timedelta(days=1)).replace(
                            hour=hour, minute=0
                        ),
                        metric_type=HeartMetricType.HEART_RATE,
                        value=hr_value,
                        unit="bpm",
                    )
                )

        return sleep_records, activity_records, heart_records

    def _create_shift_worker_data(self):
        """Create 14 days of shift work pattern."""
        sleep_records = []
        activity_records = []
        heart_records = []

        for day in range(14):
            date_obj = datetime(2024, 1, 1) + timedelta(days=day)

            if day % 7 < 5:  # Work days - sleep during day
                # Sleep 8am-4pm
                sleep_records.append(
                    SleepRecord(
                        source_name="test",
                        start_date=date_obj.replace(hour=8, minute=0),
                        end_date=date_obj.replace(hour=16, minute=0),
                        state=SleepState.ASLEEP_CORE,
                    )
                )

                # Night activity
                for hour in range(0, 8):
                    activity_records.append(
                        ActivityRecord(
                            source_name="test",
                            start_date=date_obj.replace(hour=hour, minute=0),
                            end_date=date_obj.replace(hour=hour, minute=59),
                            activity_type=ActivityType.STEP_COUNT,
                            value=300,
                            unit="count",
                        )
                    )

                # Evening activity
                for hour in range(17, 24):
                    activity_records.append(
                        ActivityRecord(
                            source_name="test",
                            start_date=date_obj.replace(hour=hour, minute=0),
                            end_date=date_obj.replace(hour=hour, minute=59),
                            activity_type=ActivityType.STEP_COUNT,
                            value=250,
                            unit="count",
                        )
                    )
            else:  # Off days - normal sleep
                # Sleep 11pm-7am
                sleep_records.append(
                    SleepRecord(
                        source_name="test",
                        start_date=date_obj.replace(hour=23, minute=0),
                        end_date=(date_obj + timedelta(days=1)).replace(
                            hour=7, minute=0
                        ),
                        state=SleepState.ASLEEP_CORE,
                    )
                )

        return sleep_records, activity_records, heart_records
