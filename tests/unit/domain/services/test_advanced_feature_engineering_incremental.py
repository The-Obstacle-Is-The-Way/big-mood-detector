"""
Tests for incremental statistics in AdvancedFeatureEngineering.

Verifies the incremental baseline update calculations are mathematically correct.
"""
import pytest
import numpy as np
from datetime import date
from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineering


class TestAdvancedFeatureEngineeringIncremental:
    """Test incremental statistics calculations in feature engineering."""
    
    def test_incremental_mean_calculation(self):
        """Test that incremental mean matches numpy calculation."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        # Add values incrementally
        values = [7.5, 8.0, 7.0, 8.5, 6.5, 7.5, 8.0]
        for value in values:
            engineer._update_individual_baseline("sleep", value)
        
        # Get the calculated mean
        baseline = engineer.individual_baselines["sleep"]
        incremental_mean = baseline["mean"]
        
        # Compare with numpy
        numpy_mean = np.mean(values)
        assert abs(incremental_mean - numpy_mean) < 1e-10, (
            f"Incremental mean {incremental_mean} != numpy mean {numpy_mean}"
        )
    
    def test_incremental_std_calculation(self):
        """Test that incremental std matches numpy calculation."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        # Add values incrementally
        values = [70.0, 65.0, 75.0, 72.0, 68.0, 71.0, 69.0]
        for value in values:
            engineer._update_individual_baseline("hr", value)
        
        # Get the calculated std
        baseline = engineer.individual_baselines["hr"]
        incremental_std = baseline["std"]
        
        # Compare with numpy (population std since we use variance formula)
        numpy_std = np.std(values, ddof=0)  # Population std
        assert abs(incremental_std - numpy_std) < 1e-8, (
            f"Incremental std {incremental_std} != numpy std {numpy_std}"
        )
    
    def test_single_value_baseline(self):
        """Test baseline with single value."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        engineer._update_individual_baseline("activity", 5000.0)
        
        baseline = engineer.individual_baselines["activity"]
        assert baseline["mean"] == 5000.0
        assert baseline["std"] == 0.0
        assert baseline["count"] == 1
    
    def test_zscore_calculation(self):
        """Test z-score calculation."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        # Build baseline
        values = [50.0, 52.0, 48.0, 51.0, 49.0]  # mean=50, std≈1.414
        for value in values:
            engineer._update_individual_baseline("hrv", value)
        
        # Test z-scores
        baseline = engineer.individual_baselines["hrv"]
        mean = baseline["mean"]
        std = baseline["std"]
        
        # Test various values
        test_cases = [
            (50.0, 0.0),  # At mean
            (mean + std, 1.0),  # One std above
            (mean - std, -1.0),  # One std below
            (mean + 2*std, 2.0),  # Two std above
        ]
        
        for test_value, expected_z in test_cases:
            actual_z = engineer._calculate_zscore("hrv", test_value)
            assert abs(actual_z - expected_z) < 0.1, (
                f"Z-score for {test_value} should be {expected_z}, got {actual_z}"
            )
    
    def test_incremental_update_preserves_count(self):
        """Test that count is properly tracked."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        for i in range(1, 11):
            engineer._update_individual_baseline("test", float(i))
            baseline = engineer.individual_baselines["test"]
            assert baseline["count"] == i
    
    def test_baseline_values_list_limited(self):
        """Test that values list is limited to 30 entries."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        # Add 50 values
        for i in range(50):
            engineer._update_individual_baseline("test", float(i))
        
        baseline = engineer.individual_baselines["test"]
        assert len(baseline["values"]) == 30
        # Should keep the last 30 values
        assert baseline["values"][0] == 20.0
        assert baseline["values"][-1] == 49.0
    
    def test_numerical_stability_large_values(self):
        """Test numerical stability with large values."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        # Large values that could cause numerical issues
        values = [1e6, 1e6 + 1, 1e6 + 2, 1e6 - 1, 1e6 - 2]
        
        for value in values:
            engineer._update_individual_baseline("large", value)
        
        baseline = engineer.individual_baselines["large"]
        
        # Mean should be close to 1e6
        assert abs(baseline["mean"] - 1e6) < 1.0
        
        # Std should be small relative to mean
        assert baseline["std"] < 10.0
    
    def test_variance_calculation_accuracy(self):
        """Test variance calculation using the incremental formula."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        # Known dataset with exact variance
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        
        for value in values:
            engineer._update_individual_baseline("variance_test", value)
        
        baseline = engineer.individual_baselines["variance_test"]
        
        # Calculate expected variance and std
        numpy_var = np.var(values, ddof=0)  # Population variance
        numpy_std = np.sqrt(numpy_var)
        
        assert abs(baseline["std"] - numpy_std) < 1e-10
    
    def test_incremental_with_loaded_baseline(self):
        """Test that loaded baseline statistics are preserved."""
        engineer = AdvancedFeatureEngineering(user_id="test_user")
        
        # Simulate loading a baseline
        engineer.individual_baselines["preloaded"] = {
            "values": [],
            "mean": 75.0,
            "std": 5.0,
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0
        }
        
        # Simulate having a loaded baseline
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline
        engineer._loaded_baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date.today(),
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=5000.0,
            activity_std=1000.0,
            circadian_phase=0.0,
            data_points=14
        )
        
        # Update with new value
        engineer._update_individual_baseline("preloaded", 76.0)
        
        baseline = engineer.individual_baselines["preloaded"]
        
        # Should have initialized from loaded baseline
        assert baseline["count"] == 15  # 14 + 1
        
        # Mean should have shifted slightly toward new value
        assert 75.0 < baseline["mean"] < 76.0
        
        # Std should still be reasonable
        assert baseline["std"] > 0