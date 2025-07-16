"""
Test Suite for Sparse Data Handler

Tests the infrastructure for handling sparse temporal health data,
including alignment, interpolation, and quality assessment.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from big_mood_detector.domain.services.sparse_data_handler import (
    AlignmentStrategy,
    DataDensity,
    InterpolationMethod,
    SparseDataHandler,
)


class TestDataQualityAssessment:
    """Test data quality and density assessment."""

    def test_identifies_dense_data(self):
        """Dense data should be identified correctly."""
        # Given: 7 consecutive days of data
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(7)]

        handler = SparseDataHandler()

        # When: Assessing density
        density = handler.assess_density(dates)

        # Then: Should identify as dense
        assert density.density_class == DataDensity.DENSE
        assert density.coverage_ratio == 1.0
        assert density.max_gap_days == 0
        assert density.consecutive_days >= 7

    def test_identifies_sparse_data(self):
        """Sparse data with gaps should be identified."""
        # Given: Data with gaps
        dates = [
            date(2025, 1, 1),
            date(2025, 1, 3),  # 1 day gap
            date(2025, 1, 7),  # 3 day gap
            date(2025, 1, 10),  # 2 day gap
        ]

        handler = SparseDataHandler()

        # When: Assessing density
        density = handler.assess_density(dates)

        # Then: Should identify as sparse
        assert density.density_class == DataDensity.SPARSE
        assert density.coverage_ratio < 0.5
        assert density.max_gap_days == 3
        assert density.consecutive_days < 3

    def test_identifies_very_sparse_data(self):
        """Very sparse data should trigger warnings."""
        # Given: Very sparse data
        dates = [
            date(2025, 1, 1),
            date(2025, 1, 15),  # 13 day gap
            date(2025, 2, 1),  # 16 day gap
        ]

        handler = SparseDataHandler()

        # When: Assessing density
        density = handler.assess_density(dates)

        # Then: Should identify as very sparse
        assert density.density_class == DataDensity.VERY_SPARSE
        assert density.coverage_ratio < 0.1
        assert density.max_gap_days > 7
        assert density.requires_special_handling is True


class TestDataAlignment:
    """Test multi-sensor data alignment strategies."""

    def test_aligns_overlapping_sensors(self):
        """Should align data from sensors with different coverage."""
        # Given: Sleep and activity data with partial overlap
        sleep_dates = pd.date_range("2025-01-01", "2025-01-10", freq="D")
        activity_dates = pd.date_range("2025-01-05", "2025-01-15", freq="D")

        sleep_data = pd.DataFrame(
            {"date": sleep_dates, "value": np.random.randn(len(sleep_dates))}
        )

        activity_data = pd.DataFrame(
            {"date": activity_dates, "value": np.random.randn(len(activity_dates))}
        )

        handler = SparseDataHandler()

        # When: Aligning data
        aligned = handler.align_sensors(
            {"sleep": sleep_data, "activity": activity_data},
            strategy=AlignmentStrategy.INTERSECTION,
        )

        # Then: Should only include overlapping dates
        assert len(aligned) == 6  # Jan 5-10
        assert aligned.index[0] == pd.Timestamp("2025-01-05")
        assert aligned.index[-1] == pd.Timestamp("2025-01-10")
        assert "sleep_value" in aligned.columns
        assert "activity_value" in aligned.columns

    def test_handles_misaligned_timestamps(self):
        """Should handle data with different time resolutions."""
        # Given: Hourly sleep data and daily activity data
        sleep_times = pd.date_range("2025-01-01", "2025-01-02", freq="h")
        activity_dates = pd.date_range("2025-01-01", "2025-01-05", freq="D")

        sleep_data = pd.DataFrame(
            {"timestamp": sleep_times, "value": np.random.randn(len(sleep_times))}
        )

        activity_data = pd.DataFrame(
            {
                "date": activity_dates,
                "steps": np.random.randint(1000, 20000, len(activity_dates)),
            }
        )

        handler = SparseDataHandler()

        # When: Aligning with aggregation
        aligned = handler.align_sensors(
            {"sleep": sleep_data, "activity": activity_data},
            strategy=AlignmentStrategy.AGGREGATE_TO_DAILY,
        )

        # Then: Should aggregate to daily level
        assert len(aligned) == 2  # Two days with both sensors
        assert isinstance(aligned.index, pd.DatetimeIndex)
        assert "sleep_value_mean" in aligned.columns
        assert "activity_steps" in aligned.columns

    def test_union_alignment_combines_all_timestamps(self):
        """UNION strategy should include all timestamps with NaN for missing values."""
        # Given: Sleep and activity data with partial overlap
        sleep_dates = pd.date_range("2025-01-01", "2025-01-05", freq="D")
        activity_dates = pd.date_range("2025-01-03", "2025-01-08", freq="D")

        sleep_data = pd.DataFrame(
            {"date": sleep_dates, "sleep_hours": [8.0, 7.5, 8.2, 6.8, 7.9]}
        )

        activity_data = pd.DataFrame(
            {"date": activity_dates, "steps": [8000, 12000, 9500, 11000, 7500, 10200]}
        )

        handler = SparseDataHandler()

        # When: Aligning with UNION strategy
        aligned = handler.align_sensors(
            {"sleep": sleep_data, "activity": activity_data},
            strategy=AlignmentStrategy.UNION,
        )

        # Then: Should include all dates from both sensors
        assert len(aligned) == 8  # Jan 1-8 (union of all dates)
        assert aligned.index[0] == pd.Timestamp("2025-01-01")
        assert aligned.index[-1] == pd.Timestamp("2025-01-08")

        # Should have NaN for non-overlapping periods
        assert pd.isna(aligned.loc["2025-01-01", "activity_steps"])  # Activity missing for Jan 1-2
        assert pd.isna(aligned.loc["2025-01-02", "activity_steps"])
        assert pd.isna(aligned.loc["2025-01-07", "sleep_sleep_hours"])  # Sleep missing for Jan 7-8
        assert pd.isna(aligned.loc["2025-01-08", "sleep_sleep_hours"])

        # Should have values for overlapping periods
        assert not pd.isna(aligned.loc["2025-01-03", "sleep_sleep_hours"])
        assert not pd.isna(aligned.loc["2025-01-03", "activity_steps"])

    def test_union_alignment_handles_empty_sensors(self):
        """UNION strategy should handle empty sensor data gracefully."""
        # Given: One sensor with data, one empty
        sleep_dates = pd.date_range("2025-01-01", "2025-01-03", freq="D")
        sleep_data = pd.DataFrame(
            {"date": sleep_dates, "sleep_hours": [8.0, 7.5, 8.2]}
        )

        empty_data = pd.DataFrame(columns=["date", "value"])

        handler = SparseDataHandler()

        # When: Aligning with UNION strategy
        aligned = handler.align_sensors(
            {"sleep": sleep_data, "empty": empty_data},
            strategy=AlignmentStrategy.UNION,
        )

        # Then: Should include all sleep dates with NaN for empty sensor
        assert len(aligned) == 3
        assert "sleep_sleep_hours" in aligned.columns
        assert all(pd.isna(aligned["empty_value"]))

    def test_union_alignment_preserves_column_names(self):
        """UNION strategy should preserve original column names with sensor prefixes."""
        # Given: Multiple sensors with different column names
        sleep_data = pd.DataFrame({
            "date": pd.date_range("2025-01-01", "2025-01-02", freq="D"),
            "duration": [8.0, 7.5],
            "efficiency": [0.85, 0.90]
        })

        heart_data = pd.DataFrame({
            "date": pd.date_range("2025-01-02", "2025-01-03", freq="D"),
            "bpm": [65, 68],
            "hrv": [45, 42]
        })

        handler = SparseDataHandler()

        # When: Aligning with UNION strategy
        aligned = handler.align_sensors(
            {"sleep": sleep_data, "heart": heart_data},
            strategy=AlignmentStrategy.UNION,
        )

        # Then: Should preserve all columns with proper prefixes
        expected_columns = ["sleep_duration", "sleep_efficiency", "heart_bpm", "heart_hrv"]
        for col in expected_columns:
            assert col in aligned.columns

        # Should span all dates
        assert len(aligned) == 3  # Jan 1-3

    def test_interpolate_align_creates_common_grid(self):
        """INTERPOLATE_ALIGN should interpolate all data to a common time grid."""
        # Given: Sensors with different time resolutions
        # 6-hourly sleep data (2 days = 5 points: 00, 06, 12, 18 on day 1, 00 on day 2)
        sleep_times = pd.date_range("2025-01-01", "2025-01-02", freq="6h")
        sleep_data = pd.DataFrame({
            "timestamp": sleep_times,
            "sleep_state": [0, 1, 1, 0, 0]  # Binary sleep state
        })

        # Daily activity data
        activity_dates = pd.date_range("2025-01-01", "2025-01-03", freq="D")
        activity_data = pd.DataFrame({
            "date": activity_dates,
            "steps": [8000, 12000, 9500]
        })

        handler = SparseDataHandler()

        # When: Aligning with INTERPOLATE_ALIGN strategy
        aligned = handler.align_sensors(
            {"sleep": sleep_data, "activity": activity_data},
            strategy=AlignmentStrategy.INTERPOLATE_ALIGN,
        )

        # Then: Should create a common grid (e.g., daily) with interpolated values
        assert len(aligned) == 3  # Should cover all days
        assert isinstance(aligned.index, pd.DatetimeIndex)

        # Should have data for all sensors interpolated to the common grid
        assert "sleep_sleep_state" in aligned.columns
        assert "activity_steps" in aligned.columns

        # Sleep data should be interpolated/aggregated to daily
        assert not aligned["sleep_sleep_state"].isna().all()

        # Activity data should be preserved
        assert aligned.loc["2025-01-01", "activity_steps"] == 8000
        assert aligned.loc["2025-01-02", "activity_steps"] == 12000

    def test_interpolate_align_handles_mixed_resolutions(self):
        """INTERPOLATE_ALIGN should handle sensors with very different time resolutions."""
        # Given: High-frequency heart rate and low-frequency sleep
        hr_times = pd.date_range("2025-01-01 00:00", "2025-01-01 06:00", freq="h")
        hr_data = pd.DataFrame({
            "timestamp": hr_times,
            "bpm": [65, 62, 60, 58, 62, 68, 70]  # Hourly HR data
        })

        sleep_dates = pd.date_range("2025-01-01", "2025-01-02", freq="D")
        sleep_data = pd.DataFrame({
            "date": sleep_dates,
            "duration": [7.5, 8.0]
        })

        handler = SparseDataHandler()

        # When: Aligning with interpolation
        aligned = handler.align_sensors(
            {"heart": hr_data, "sleep": sleep_data},
            strategy=AlignmentStrategy.INTERPOLATE_ALIGN,
        )

        # Then: Should create aligned data at daily resolution
        assert len(aligned) == 2  # 2 days
        assert "heart_bpm" in aligned.columns
        assert "sleep_duration" in aligned.columns

        # Heart rate should be aggregated to daily (mean of hourly values)
        expected_hr_day1 = (65 + 62 + 60 + 58 + 62 + 68 + 70) / 7
        assert abs(aligned.loc["2025-01-01", "heart_bpm"] - expected_hr_day1) < 0.1

    def test_interpolate_align_with_gaps(self):
        """INTERPOLATE_ALIGN should handle gaps in sensor data appropriately."""
        # Given: Sparse sensor data with gaps
        temp_times = pd.to_datetime([
            "2025-01-01 00:00", "2025-01-01 12:00",  # Day 1
            # Gap on Day 2
            "2025-01-03 06:00", "2025-01-03 18:00"   # Day 3
        ])
        temp_data = pd.DataFrame({
            "timestamp": temp_times,
            "temperature": [36.5, 37.0, 36.8, 36.9]
        })

        steps_data = pd.DataFrame({
            "date": pd.date_range("2025-01-01", "2025-01-03", freq="D"),
            "steps": [8000, 10000, 7500]
        })

        handler = SparseDataHandler()

        # When: Aligning with interpolation
        aligned = handler.align_sensors(
            {"temp": temp_data, "steps": steps_data},
            strategy=AlignmentStrategy.INTERPOLATE_ALIGN,
        )

        # Then: Should handle gaps gracefully
        assert len(aligned) == 3  # All 3 days
        assert "temp_temperature" in aligned.columns
        assert "steps_steps" in aligned.columns

        # Day 2 temperature should be interpolated or NaN
        day2_temp = aligned.loc["2025-01-02", "temp_temperature"]
        assert pd.isna(day2_temp) or isinstance(day2_temp, int | float)


class TestInterpolationStrategies:
    """Test different interpolation methods for sparse data."""

    def test_no_interpolation_for_large_gaps(self):
        """Should not interpolate gaps larger than threshold."""
        # Given: Data with a large gap
        dates = [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 10)]
        values = [100, 110, 200]

        df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values}).set_index(
            "date"
        )

        handler = SparseDataHandler(max_interpolation_gap_hours=24)

        # When: Attempting interpolation
        interpolated = handler.interpolate(df, method=InterpolationMethod.LINEAR)

        # Then: Large gap should remain as NaN
        assert len(interpolated) == 10  # Full date range
        assert pd.isna(interpolated.loc["2025-01-05", "value"])
        assert not pd.isna(interpolated.loc["2025-01-01", "value"])

    def test_circadian_aware_interpolation(self):
        """Should use circadian patterns for activity interpolation."""
        # Given: Activity data with small gaps
        times = pd.date_range("2025-01-01", periods=48, freq="h")
        # Create circadian pattern with gaps
        values = 100 + 50 * np.sin(2 * np.pi * np.arange(48) / 24)
        values[[12, 13, 36, 37]] = np.nan  # Add gaps

        df = pd.DataFrame({"activity": values}, index=times)

        handler = SparseDataHandler()

        # When: Using circadian interpolation
        interpolated = handler.interpolate(
            df, method=InterpolationMethod.CIRCADIAN_SPLINE
        )

        # Then: Should preserve circadian pattern
        assert not pd.isna(interpolated.iloc[12]["activity"])
        assert not pd.isna(interpolated.iloc[36]["activity"])
        # Check if interpolated values follow circadian pattern
        assert 50 < interpolated.iloc[12]["activity"] < 150  # Daytime range
        assert 50 < interpolated.iloc[36]["activity"] < 150  # Nighttime range

    def test_forward_fill_for_sleep_states(self):
        """Should forward-fill categorical sleep states."""
        # Given: Sleep state data with gaps
        times = pd.date_range("2025-01-01 22:00", periods=10, freq="h")
        states = [
            "AWAKE",
            "LIGHT",
            "DEEP",
            np.nan,
            np.nan,
            "REM",
            np.nan,
            "LIGHT",
            "AWAKE",
            "AWAKE",
        ]

        df = pd.DataFrame({"sleep_state": states}, index=times)

        handler = SparseDataHandler()

        # When: Using forward fill for categorical data
        interpolated = handler.interpolate(
            df, method=InterpolationMethod.FORWARD_FILL, max_gap_hours=3
        )

        # Then: Should fill small gaps
        assert interpolated.iloc[3]["sleep_state"] == "DEEP"
        assert interpolated.iloc[4]["sleep_state"] == "DEEP"
        assert interpolated.iloc[6]["sleep_state"] == "REM"


class TestMissingDataFeatures:
    """Test extraction of missingness patterns as features."""

    def test_extracts_missingness_features(self):
        """Should extract features from missing data patterns."""
        # Given: Data with known missing patterns
        dates = pd.date_range("2025-01-01", "2025-01-30", freq="D")
        values = np.random.randn(len(dates))
        # Create weekend gaps
        weekend_mask = dates.weekday.isin([5, 6])
        values[weekend_mask] = np.nan

        df = pd.DataFrame({"value": values}, index=dates)

        handler = SparseDataHandler()

        # When: Extracting missingness features
        features = handler.extract_missingness_features(df)

        # Then: Should capture patterns
        assert "missing_ratio" in features
        assert "max_consecutive_missing" in features
        assert "missing_weekend_ratio" in features
        assert features["missing_weekend_ratio"] > 0.8  # Most weekends missing
        assert features["missing_weekday_ratio"] < 0.2  # Few weekdays missing

    def test_confidence_scores_based_on_density(self):
        """Should assign confidence scores based on data density."""
        # Given: Variable density data
        dense_dates = pd.date_range("2025-01-01", "2025-01-07", freq="D")
        sparse_dates = pd.date_range("2025-01-15", "2025-01-30", freq="3D")

        handler = SparseDataHandler()

        # When: Computing confidence scores
        dense_confidence = handler.compute_confidence(dense_dates)
        sparse_confidence = handler.compute_confidence(sparse_dates)

        # Then: Dense data should have higher confidence
        assert dense_confidence > 0.9
        assert sparse_confidence < 0.5
        assert 0 <= dense_confidence <= 1
        assert 0 <= sparse_confidence <= 1


class TestHybridPipeline:
    """Test adaptive pipeline that switches strategies based on data density."""

    def test_selects_appropriate_algorithm(self):
        """Should select algorithm based on data characteristics."""
        # Given: Mixed density dataset
        data_windows = {
            "dense_period": {
                "dates": pd.date_range("2025-01-01", "2025-01-14", freq="D"),
                "values": np.random.randn(14),
            },
            "sparse_period": {
                "dates": pd.to_datetime(
                    ["2025-02-01", "2025-02-05", "2025-02-10", "2025-02-20"]
                ),
                "values": np.random.randn(4),
            },
        }

        handler = SparseDataHandler()

        # When: Processing windows
        results = {}
        for period, data in data_windows.items():
            strategy = handler.select_processing_strategy(data["dates"])
            results[period] = strategy

        # Then: Should select appropriate strategies
        assert results["dense_period"].algorithm == "full_circadian_analysis"
        assert results["sparse_period"].algorithm == "interpolation_with_uncertainty"
        assert results["dense_period"].confidence > results["sparse_period"].confidence

    def test_graceful_degradation(self):
        """Should gracefully degrade functionality with less data."""
        # Given: Progressively sparser data
        handler = SparseDataHandler()

        datasets = {
            "excellent": pd.date_range("2025-01-01", "2025-01-28", freq="D"),
            "good": pd.date_range("2025-01-01", "2025-01-14", freq="D"),
            "fair": pd.date_range("2025-01-01", "2025-01-07", freq="2D"),
            "poor": pd.to_datetime(["2025-01-01", "2025-01-05", "2025-01-15"]),
        }

        # When: Determining available features
        feature_sets = {}
        for quality, dates in datasets.items():
            features = handler.get_available_features(dates)
            feature_sets[quality] = features

        # Then: Should have decreasing feature availability
        assert feature_sets["excellent"]["circadian_phase"] is True
        assert feature_sets["good"]["weekly_patterns"] is True
        assert feature_sets["fair"]["circadian_phase"] is False
        # Count available features
        excellent_count = sum(1 for v in feature_sets["excellent"].values() if v)
        poor_count = sum(1 for v in feature_sets["poor"].values() if v)
        assert poor_count < excellent_count
        assert all("confidence" in fs for fs in feature_sets.values())


class TestRealWorldScenarios:
    """Test with patterns from actual user data."""

    def test_handles_january_to_april_gap(self):
        """Should handle the actual gap in user's data."""
        # Given: User's actual data pattern
        sleep_dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
        activity_dates = pd.date_range("2025-04-15", "2025-07-14", freq="D")

        handler = SparseDataHandler()

        # When: Finding overlapping windows
        windows = handler.find_analysis_windows(
            sleep_dates=sleep_dates.tolist(),
            activity_dates=activity_dates.tolist(),
            min_overlap_days=3,
        )

        # Then: Should recognize no overlap
        assert len(windows) == 0
        assert handler.get_recommendation(windows) == "Use sensor-specific analysis"

    def test_handles_may_sparse_overlap(self):
        """Should handle sparse May overlap from user data."""
        # Given: Actual May overlap pattern
        overlap_dates = pd.to_datetime(
            [
                "2025-05-02",
                "2025-05-04",
                "2025-05-06",
                "2025-05-07",
                "2025-05-13",
                "2025-05-15",
                "2025-05-16",
                "2025-05-17",
            ]
        )

        handler = SparseDataHandler()

        # When: Finding usable windows
        windows = handler.find_consecutive_windows(
            dates=overlap_dates.tolist(), min_consecutive_days=3
        )

        # Then: Should find the May 15-17 window
        assert len(windows) >= 1
        assert any(len(w) >= 3 for w in windows)
        best_window = max(windows, key=len)
        assert pd.Timestamp("2025-05-15") in best_window
