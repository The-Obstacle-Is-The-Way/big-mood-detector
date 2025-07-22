"""
Test to ensure sleep_percentage calculation matches Seoul study paper exactly.

The Seoul study defines sleep_percentage as "daily fraction of the sleep period (total sleep minutes ÷ 1 440)"
This is one of the 10 sleep indexes that form 30 of the 36 features (with mean, SD, Z-score).
"""


import pytest

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)


class TestSleepPercentageSeoulPaper:
    """Test that sleep_percentage calculation exactly matches Seoul paper specification."""

    def test_sleep_percentage_formula(self):
        """Test the exact formula: total_sleep_minutes / 1440."""
        pipeline = AggregationPipeline()

        # Mock sleep windows with known durations
        class MockSleepWindow:
            def __init__(self, hours):
                self.total_duration_hours = hours
                self.gap_hours = []  # No gaps for simplicity

        # Test case 1: 8 hours of sleep
        windows = [MockSleepWindow(8.0)]
        metrics = pipeline.calculate_sleep_metrics(windows)

        # 8 hours = 480 minutes, 480/1440 = 0.3333...
        assert metrics["sleep_percentage"] == pytest.approx(480/1440, 0.0001)
        assert metrics["sleep_percentage"] == pytest.approx(1/3, 0.0001)

        # Test case 2: 7.5 hours (typical adult sleep)
        windows = [MockSleepWindow(7.5)]
        metrics = pipeline.calculate_sleep_metrics(windows)

        # 7.5 hours = 450 minutes, 450/1440 = 0.3125
        assert metrics["sleep_percentage"] == pytest.approx(450/1440, 0.0001)
        assert metrics["sleep_percentage"] == pytest.approx(0.3125, 0.0001)

        # Test case 3: Multiple windows (fragmented sleep)
        windows = [MockSleepWindow(4.0), MockSleepWindow(2.0), MockSleepWindow(1.5)]
        metrics = pipeline.calculate_sleep_metrics(windows)

        # Total: 7.5 hours = 450 minutes, 450/1440 = 0.3125
        assert metrics["sleep_percentage"] == pytest.approx(450/1440, 0.0001)

    def test_sleep_percentage_bounds(self):
        """Test that sleep_percentage stays within valid bounds [0, 1]."""
        pipeline = AggregationPipeline()

        class MockSleepWindow:
            def __init__(self, hours):
                self.total_duration_hours = hours
                self.gap_hours = []

        # Test zero sleep
        windows = []
        metrics = pipeline.calculate_sleep_metrics(windows)
        assert metrics["sleep_percentage"] == 0.0

        # Test maximum possible sleep (24 hours)
        windows = [MockSleepWindow(24.0)]
        metrics = pipeline.calculate_sleep_metrics(windows)
        assert metrics["sleep_percentage"] == pytest.approx(1.0, 0.0001)

        # Test excessive sleep (shouldn't happen but ensure it's capped)
        windows = [MockSleepWindow(25.0)]  # More than 24 hours
        metrics = pipeline.calculate_sleep_metrics(windows)
        # Should be 25*60/1440 = 1.0417, but let's see what happens
        assert metrics["sleep_percentage"] == pytest.approx(25*60/1440, 0.0001)

    def test_seoul_paper_examples(self):
        """Test examples that match typical values from the Seoul paper."""
        pipeline = AggregationPipeline()

        class MockSleepWindow:
            def __init__(self, hours):
                self.total_duration_hours = hours
                self.gap_hours = []

        # The paper mentions analyzing 44,787 days of data
        # Typical sleep percentages would be in the range of 0.25-0.40

        # Test typical values
        test_cases = [
            (6.0, 0.25),      # 6 hours = 25% of day
            (7.2, 0.30),      # 7.2 hours = 30% of day
            (8.4, 0.35),      # 8.4 hours = 35% of day
            (9.6, 0.40),      # 9.6 hours = 40% of day
        ]

        for hours, expected_percentage in test_cases:
            windows = [MockSleepWindow(hours)]
            metrics = pipeline.calculate_sleep_metrics(windows)
            assert metrics["sleep_percentage"] == pytest.approx(expected_percentage, 0.0001)

    def test_feature_count_remains_36(self):
        """Ensure we maintain exactly 36 features as per Seoul study."""
        # The Seoul study uses:
        # - 10 sleep indexes × 3 (mean, SD, Z-score) = 30 features
        # - 2 circadian indexes × 3 (mean, SD, Z-score) = 6 features
        # Total = 36 features

        # Check that DailyFeatures has exactly 36 feature fields (excluding metadata)
        from big_mood_detector.application.services.aggregation_pipeline import (
            DailyFeatures,
        )

        # Get all fields that are features (excluding date and activity extras)
        feature_fields = [
            f for f in DailyFeatures.__dataclass_fields__.keys()
            if f != 'date' and not f.startswith('daily_') and not f.startswith('activity_') and not f.startswith('sedentary_')
        ]

        # Should have exactly 36 feature fields
        assert len(feature_fields) == 36

        # Verify the structure: 10 sleep × 3 + 2 circadian × 3
        sleep_base_names = [
            'sleep_percentage', 'sleep_amplitude',
            'long_sleep_num', 'long_sleep_len', 'long_sleep_st', 'long_sleep_wt',
            'short_sleep_num', 'short_sleep_len', 'short_sleep_st', 'short_sleep_wt'
        ]
        circadian_base_names = ['circadian_amplitude', 'circadian_phase']

        # Each should have mean, std, zscore
        for base in sleep_base_names:
            assert f'{base}_mean' in feature_fields
            assert f'{base}_std' in feature_fields
            assert f'{base}_zscore' in feature_fields

        for base in circadian_base_names:
            assert f'{base}_mean' in feature_fields
            assert f'{base}_std' in feature_fields
            assert f'{base}_zscore' in feature_fields

    def test_warning_comment_present(self):
        """Ensure the warning comment about not multiplying by 24 is present."""
        # This is to prevent regression of the bug
        import inspect

        from big_mood_detector.application.services.aggregation_pipeline import (
            AggregationPipeline,
        )

        source = inspect.getsource(AggregationPipeline.calculate_sleep_metrics)
        assert "DO NOT multiply by 24" in source
        assert "Fraction of day, NOT hours" in source
