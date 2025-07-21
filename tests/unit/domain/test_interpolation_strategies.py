"""
Tests for InterpolationStrategy implementations

Following TDD principles - write tests first, then implement strategies.
"""

import pandas as pd

class TestInterpolationStrategy:
    """Test the InterpolationStrategy interface."""

    def test_interpolation_strategy_interface(self):
        """Test that InterpolationStrategy defines the correct interface."""
        from big_mood_detector.domain.services.interpolation_strategies import LinearInterpolationStrategy

        # This test will fail initially - that's the point of TDD
        strategy = LinearInterpolationStrategy(max_gap_hours=24)

        # Create test data with missing values
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({"values": [1.0, None, None, 4.0, 5.0]}, index=dates)

        # Should return interpolated DataFrame
        result = strategy.interpolate(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == data.shape

class TestLinearInterpolationStrategy:
    """Test linear interpolation strategy."""

    def test_linear_interpolation_fills_gaps(self):
        """Test that linear interpolation fills gaps correctly."""
        from big_mood_detector.domain.services.interpolation_strategies import LinearInterpolationStrategy

        strategy = LinearInterpolationStrategy(max_gap_hours=48)  # 2 days

        # Create data with 1-day gap
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame({"sleep_hours": [8.0, None, None, 6.0]}, index=dates)

        result = strategy.interpolate(data)

        # Should interpolate missing values
        assert not result["sleep_hours"].isna().any()
        assert (
            abs(result["sleep_hours"].iloc[1] - 7.33) < 0.01
        )  # Approximately (8+6)/2 trending down

    def test_linear_interpolation_respects_max_gap(self):
        """Test that linear interpolation respects max gap limit."""
        from big_mood_detector.domain.services.interpolation_strategies import LinearInterpolationStrategy

        strategy = LinearInterpolationStrategy(
            max_gap_hours=0
        )  # No interpolation allowed

        # Create data with missing values
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame({"steps": [1000, None, None, 2000]}, index=dates)

        result = strategy.interpolate(data)

        # Should NOT interpolate any values due to limit = 0
        assert result["steps"].isna().sum() == 2  # 2 missing values should remain

class TestForwardFillInterpolationStrategy:
    """Test forward fill interpolation strategy."""

    def test_forward_fill_categorical_data(self):
        """Test forward fill for categorical data."""
        from big_mood_detector.domain.services.interpolation_strategies import ForwardFillInterpolationStrategy

        strategy = ForwardFillInterpolationStrategy(max_gap_hours=48)

        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame({"sleep_stage": ["deep", None, None, "light"]}, index=dates)

        result = strategy.interpolate(data)

        # Should forward fill 'deep' for missing values
        assert result["sleep_stage"].iloc[1] == "deep"
        assert result["sleep_stage"].iloc[2] == "deep"

class TestCircadianSplineInterpolationStrategy:
    """Test circadian-aware spline interpolation."""

    def test_circadian_interpolation_preserves_rhythm(self):
        """Test that circadian interpolation preserves daily rhythms."""
        from big_mood_detector.domain.services.interpolation_strategies import CircadianSplineInterpolationStrategy

        strategy = CircadianSplineInterpolationStrategy(max_gap_hours=48)

        # Create hourly activity data with missing values
        dates = pd.date_range("2024-01-01", periods=48, freq="h")  # 2 days
        activity_pattern = [100 if 6 <= h <= 22 else 20 for h in range(24)] * 2
        activity_pattern[12:36] = [None] * 24  # Remove 1 full day

        data = pd.DataFrame({"activity_level": activity_pattern}, index=dates)

        result = strategy.interpolate(data)

        # Should preserve circadian pattern in interpolated values
        assert not result["activity_level"].isna().any()
        # Check that interpolated values are reasonable (not too far from expected range)
        interpolated_section = result["activity_level"].iloc[
            12:36
        ]  # The missing section
        assert interpolated_section.min() >= 10  # Should not go too low
        assert interpolated_section.max() <= 110  # Should not go too high
        assert len(interpolated_section) == 24  # Should interpolate all missing values

class TestNoInterpolationStrategy:
    """Test no interpolation strategy."""

    def test_no_interpolation_preserves_missing_values(self):
        """Test that no interpolation preserves missing values."""
        from big_mood_detector.domain.services.interpolation_strategies import NoInterpolationStrategy

        strategy = NoInterpolationStrategy()

        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame({"heart_rate": [70, None, None, 75]}, index=dates)

        result = strategy.interpolate(data)

        # Should preserve missing values
        assert result["heart_rate"].isna().sum() == 2
        assert result["heart_rate"].iloc[0] == 70
        assert result["heart_rate"].iloc[3] == 75

class TestInterpolationStrategyFactory:
    """Test factory for creating interpolation strategies."""

    def test_factory_creates_correct_strategy(self):
        """Test that factory creates the correct strategy type."""
        from big_mood_detector.domain.services.interpolation_strategies import (
            CircadianSplineInterpolationStrategy,
            ForwardFillInterpolationStrategy,
            InterpolationMethod,
            InterpolationStrategyFactory,
            LinearInterpolationStrategy,
            NoInterpolationStrategy,
        )

        # Test linear strategy creation
        linear_strategy = InterpolationStrategyFactory.create(
            InterpolationMethod.LINEAR, max_gap_hours=24
        )
        assert isinstance(linear_strategy, LinearInterpolationStrategy)

        # Test forward fill strategy creation
        ff_strategy = InterpolationStrategyFactory.create(
            InterpolationMethod.FORWARD_FILL, max_gap_hours=24
        )
        assert isinstance(ff_strategy, ForwardFillInterpolationStrategy)

        # Test circadian strategy creation
        circadian_strategy = InterpolationStrategyFactory.create(
            InterpolationMethod.CIRCADIAN_SPLINE, max_gap_hours=24
        )
        assert isinstance(circadian_strategy, CircadianSplineInterpolationStrategy)

        # Test no interpolation strategy creation
        none_strategy = InterpolationStrategyFactory.create(InterpolationMethod.NONE)
        assert isinstance(none_strategy, NoInterpolationStrategy)
