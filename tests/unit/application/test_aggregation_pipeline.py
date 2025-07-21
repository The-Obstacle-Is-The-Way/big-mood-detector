"""
Test Aggregation Pipeline

TDD approach for extracting feature aggregation logic from MoodPredictionPipeline.
The AggregationPipeline will handle all feature calculation and statistical aggregation.
"""

from datetime import date, datetime, timedelta
from unittest.mock import Mock

import pytest

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestAggregationPipeline:
    """Test the feature aggregation pipeline extraction."""

    @pytest.fixture
    def aggregation_pipeline(self):
        """Create AggregationPipeline instance."""
        from big_mood_detector.application.services.aggregation_pipeline import (
            AggregationPipeline,
        )

        return AggregationPipeline()

    @pytest.fixture
    def sample_sleep_records(self):
        """Create sample sleep records."""
        records = []
        base_date = datetime(2024, 1, 1, 23, 0)

        for i in range(30):
            start = base_date + timedelta(days=i)
            end = start + timedelta(hours=8)
            record = SleepRecord(
                source_name="test",
                start_date=start,
                end_date=end,
                state=SleepState.ASLEEP,
            )
            records.append(record)

        return records

    @pytest.fixture
    def sample_activity_records(self):
        """Create sample activity records."""
        records = []
        base_date = datetime(2024, 1, 1, 8, 0)

        for i in range(30):
            record_date = base_date + timedelta(days=i)
            record = ActivityRecord(
                source_name="test",
                start_date=record_date,
                end_date=record_date + timedelta(hours=1),
                activity_type=ActivityType.STEP_COUNT,
                value=8000.0 + (i * 500),
                unit="count",
            )
            records.append(record)

        return records

    @pytest.fixture
    def sample_sleep_windows(self):
        """Create sample sleep window analysis results."""
        from big_mood_detector.domain.services.sleep_window_analyzer import (
            SleepWindow,
        )

        # Mock sleep window with proper attributes
        window = Mock(spec=SleepWindow)
        window.total_duration_hours = 7.5
        window.gap_hours = [0.5, 0.2]  # Wake periods

        return [window]

    def test_aggregate_daily_features(
        self, aggregation_pipeline, sample_sleep_records, sample_activity_records
    ):
        """Test aggregating features for a date range."""
        start_date = date(2024, 1, 15)
        end_date = date(2024, 1, 20)

        features = aggregation_pipeline.aggregate_daily_features(
            sleep_records=sample_sleep_records,
            activity_records=sample_activity_records,
            heart_records=[],
            start_date=start_date,
            end_date=end_date,
        )

        # We get 4 features because the first 2 days don't have enough history (min_window_size=3)
        assert len(features) == 4  # Days 17-20 have enough history
        assert all(hasattr(f, "date") for f in features)
        assert all(hasattr(f, "to_dict") for f in features)

    def test_calculate_daily_metrics(self, aggregation_pipeline, sample_sleep_windows):
        """Test calculating metrics for a single day."""
        # Mock activity sequence
        activity_sequence = Mock()
        activity_sequence.total_steps = 8000

        # Mock circadian metrics
        circadian_metrics = Mock()
        circadian_metrics.relative_amplitude = 0.8

        # Mock DLMO result
        dlmo_result = Mock()
        dlmo_result.dlmo_hour = 21.5

        metrics = aggregation_pipeline.calculate_daily_metrics(
            sleep_windows=sample_sleep_windows,
            activity_sequence=activity_sequence,
            circadian_metrics=circadian_metrics,
            dlmo_result=dlmo_result,
        )

        assert "sleep" in metrics
        assert "circadian" in metrics
        assert "sleep_percentage" in metrics["sleep"]
        assert "sleep_amplitude" in metrics["sleep"]
        assert metrics["circadian"]["amplitude"] == 0.8
        assert metrics["circadian"]["phase"] == 21.5

    def test_calculate_sleep_metrics(self, aggregation_pipeline, sample_sleep_windows):
        """Test sleep metrics calculation."""
        metrics = aggregation_pipeline.calculate_sleep_metrics(sample_sleep_windows)

        assert "sleep_percentage" in metrics
        assert "sleep_amplitude" in metrics
        assert "long_num" in metrics
        assert "short_num" in metrics

        # Check calculations
        assert 0 <= metrics["sleep_percentage"] <= 1
        assert metrics["long_num"] >= 0
        assert metrics["short_num"] >= 0

    def test_calculate_statistics(self, aggregation_pipeline):
        """Test statistical calculations (mean, std, z-score)."""
        # Create a window of daily metrics
        metrics_window = [
            {"sleep_percentage": 0.31, "sleep_amplitude": 0.1},
            {"sleep_percentage": 0.32, "sleep_amplitude": 0.12},
            {"sleep_percentage": 0.30, "sleep_amplitude": 0.11},
            {"sleep_percentage": 0.33, "sleep_amplitude": 0.13},
            {"sleep_percentage": 0.29, "sleep_amplitude": 0.09},
        ]

        current_metrics = {"sleep_percentage": 0.35, "sleep_amplitude": 0.15}

        stats = aggregation_pipeline.calculate_statistics(
            metric_name="sleep_percentage",
            window_values=[m["sleep_percentage"] for m in metrics_window],
            current_value=current_metrics["sleep_percentage"],
        )

        assert "mean" in stats
        assert "std" in stats
        assert "zscore" in stats

        # Z-score should be positive since current value is above mean
        assert stats["zscore"] > 0

    def test_rolling_window_management(self, aggregation_pipeline):
        """Test rolling window updates."""
        window = aggregation_pipeline.create_rolling_window(size=7)

        # Add metrics for 10 days
        for i in range(10):
            metrics = {"day": i, "value": i * 10}
            aggregation_pipeline.update_rolling_window(window, metrics, size=7)

        # Window should only contain last 7 entries
        assert len(window) == 7
        assert window[0]["day"] == 3  # First 3 were removed
        assert window[-1]["day"] == 9  # Last one added

    def test_feature_extraction_with_sparse_data(self, aggregation_pipeline):
        """Test handling of sparse/missing data."""
        # Only 3 days of sleep data
        sparse_sleep = [
            SleepRecord(
                source_name="test",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 8),
                state=SleepState.ASLEEP,
            ),
            SleepRecord(
                source_name="test",
                start_date=datetime(2024, 1, 5),
                end_date=datetime(2024, 1, 5, 8),
                state=SleepState.ASLEEP,
            ),
            SleepRecord(
                source_name="test",
                start_date=datetime(2024, 1, 10),
                end_date=datetime(2024, 1, 10, 8),
                state=SleepState.ASLEEP,
            ),
        ]

        features = aggregation_pipeline.aggregate_daily_features(
            sleep_records=sparse_sleep,
            activity_records=[],
            heart_records=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            min_window_size=3,
        )

        # Should handle gaps appropriately
        assert len(features) > 0
        # Features should only be generated when enough data is available

    def test_circadian_window_aggregation(self, aggregation_pipeline):
        """Test aggregation of circadian rhythm data over multiple days."""
        # Mock activity sequences for a week
        activity_sequences = []
        for i in range(7):
            seq = Mock()
            seq.date = date(2024, 1, 1) + timedelta(days=i)
            seq.hourly_steps = [100] * 24  # Simplified
            activity_sequences.append(seq)

        circadian_window = aggregation_pipeline.aggregate_circadian_window(
            activity_sequences=activity_sequences, lookback_days=7
        )

        assert len(circadian_window) == 7
        assert all(hasattr(seq, "date") for seq in circadian_window)

    def test_dlmo_window_aggregation(self, aggregation_pipeline):
        """Test DLMO calculation window aggregation."""
        sleep_records = []
        base_date = datetime(2024, 1, 1, 23, 0)

        # Create 14 days of sleep records
        for i in range(14):
            start = base_date + timedelta(days=i)
            sleep_records.append(
                Mock(start_date=start, end_date=start + timedelta(hours=8))
            )

        dlmo_window = aggregation_pipeline.aggregate_dlmo_window(
            sleep_records=sleep_records, target_date=date(2024, 1, 14), lookback_days=14
        )

        assert len(dlmo_window) <= 14
        assert all(r.start_date.date() <= date(2024, 1, 14) for r in dlmo_window)

    def test_feature_normalization(self, aggregation_pipeline):
        """Test feature normalization across the 36 features."""
        raw_features = {
            "sleep_percentage": 0.33,
            "sleep_amplitude": 0.12,
            "long_num": 2,
            "short_num": 1,
            # ... other features
        }

        normalized = aggregation_pipeline.normalize_features(
            features=raw_features, normalization_params=None  # Use defaults
        )

        # All normalized values should be in reasonable ranges
        for key, value in normalized.items():
            if "zscore" in key:
                assert -5 <= value <= 5  # Z-scores typically in this range

    def test_export_to_dataframe_format(self, aggregation_pipeline):
        """Test exporting aggregated features to DataFrame-ready format."""
        # Create sample daily features
        features = []
        for i in range(5):
            feature = Mock()
            feature.date = date(2024, 1, 1) + timedelta(days=i)
            # Use default parameters to capture values by value, not reference
            feature.to_dict = lambda f=feature, idx=i: {
                "date": f.date,
                "sleep_percentage_mean": 0.31 + idx * 0.01,
                "sleep_percentage_std": 0.02,
                "sleep_percentage_zscore": 0.5,
                # ... other features
            }
            features.append(feature)

        df_data = aggregation_pipeline.export_to_dataframe(features)

        assert isinstance(df_data, list)
        assert len(df_data) == 5
        assert all("date" in row for row in df_data)
        assert all("sleep_percentage_mean" in row for row in df_data)

    def test_parallel_processing_capability(self, aggregation_pipeline):
        """Test that aggregation can process days in parallel."""
        # This tests the design - actual parallel implementation is optional
        assert hasattr(aggregation_pipeline, "supports_parallel_processing")

        if aggregation_pipeline.supports_parallel_processing:
            # Test parallel aggregation
            features = aggregation_pipeline.aggregate_daily_features(
                sleep_records=[],
                activity_records=[],
                heart_records=[],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 30),
                parallel=True,
            )
            assert features is not None
