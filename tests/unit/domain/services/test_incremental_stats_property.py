"""
Property-based tests for incremental statistics calculations.

Uses hypothesis to verify the incremental baseline update implementation
in AdvancedFeatureEngineer is mathematically correct.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from datetime import date
from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer


class TestIncrementalStatsProperty:
    """Property-based tests for incremental statistics using hypothesis."""
    
    @given(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=1))
    def test_mean_matches_numpy(self, values):
        """
        Test that incremental mean calculation matches numpy.
        
        Property: For any list of values, incremental mean == numpy mean
        """
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        for value in values:
            engineer._update_individual_baseline("test_metric", value)
        
        # Compare with numpy
        baseline = engineer.individual_baselines["test_metric"]
        np_mean = np.mean(values)
        
        assert abs(baseline["mean"] - np_mean) < 1e-9, (
            f"Incremental mean {baseline['mean']} != numpy mean {np_mean}"
        )
    
    @given(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=2))
    def test_std_matches_numpy(self, values):
        """
        Test that incremental std calculation matches numpy.
        
        Property: For any list of values with n>=2, incremental std == numpy std
        """
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        for value in values:
            engineer._update_individual_baseline("std_test", value)
        
        # Compare with numpy (using ddof=0 for population std as per implementation)
        baseline = engineer.individual_baselines["std_test"]
        np_std = np.std(values, ddof=0)
        
        assert abs(baseline["std"] - np_std) < 1e-9, (
            f"Incremental std {baseline['std']} != numpy std {np_std}"
        )
    
    @given(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=1))
    def test_count_is_correct(self, values):
        """
        Test that count tracks number of values.
        
        Property: count == len(values)
        """
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        for value in values:
            engineer._update_individual_baseline("count_test", value)
        
        baseline = engineer.individual_baselines["count_test"]
        assert baseline["count"] == len(values)
    
    @given(st.floats(min_value=-1000, max_value=1000, allow_nan=False))
    def test_single_value_properties(self, value):
        """
        Test properties when only one value is added.
        
        Properties:
        - mean == value
        - std == 0
        - count == 1
        """
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        engineer._update_individual_baseline("single", value)
        
        baseline = engineer.individual_baselines["single"]
        assert baseline["mean"] == value
        assert baseline["std"] == 0.0
        assert baseline["count"] == 1
    
    @given(
        st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=1),
        st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=1)
    )
    def test_order_independence(self, values1, values2):
        """
        Test that order of updates doesn't matter for final result.
        
        Property: stats(values1 + values2) == stats(values2 + values1)
        """
        # Forward order
        engineer1 = AdvancedFeatureEngineer(user_id="test_user")
        
        for v in values1 + values2:
            engineer1._update_individual_baseline("order_test", v)
        
        # Reverse order
        engineer2 = AdvancedFeatureEngineer(user_id="test_user")
        
        for v in values2 + values1:
            engineer2._update_individual_baseline("order_test", v)
        
        baseline1 = engineer1.individual_baselines["order_test"]
        baseline2 = engineer2.individual_baselines["order_test"]
        
        assert abs(baseline1["mean"] - baseline2["mean"]) < 1e-9
        assert abs(baseline1["std"] - baseline2["std"]) < 1e-9
        assert baseline1["count"] == baseline2["count"]
    
    @given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=10, max_size=50))
    @settings(max_examples=50)
    def test_incremental_vs_batch(self, values):
        """
        Test incremental updates vs batch calculation.
        
        Property: Processing values one-by-one == processing all at once
        """
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        # Incremental
        for value in values:
            engineer._update_individual_baseline("batch_test", value)
        
        baseline = engineer.individual_baselines["batch_test"]
        
        # Batch (simulate with numpy)
        batch_mean = np.mean(values)
        batch_std = np.std(values, ddof=0)  # Population std
        
        assert abs(baseline["mean"] - batch_mean) < 1e-9
        assert abs(baseline["std"] - batch_std) < 1e-9
    
    @given(st.floats(min_value=0.1, max_value=100))
    def test_constant_values(self, constant):
        """
        Test with all values being the same.
        
        Properties:
        - mean == constant
        - std == 0
        """
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        # Add same value 10 times
        for _ in range(10):
            engineer._update_individual_baseline("constant", constant)
        
        baseline = engineer.individual_baselines["constant"]
        assert abs(baseline["mean"] - constant) < 1e-9
        # Allow tiny numerical error in std for constant values
        assert baseline["std"] < 1e-5, f"Std should be ~0 for constant values, got {baseline['std']}'"
        assert baseline["count"] == 10
    
    @given(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=3))
    def test_zscore_calculation(self, values):
        """
        Test z-score calculation is correct.
        
        Property: zscore = (value - mean) / std
        """
        if len(values) < 2:
            return  # Need at least 2 values for std
        
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        # Add all but last value
        for value in values[:-1]:
            engineer._update_individual_baseline("zscore", value)
        
        # Calculate z-score for last value
        last_value = values[-1]
        baseline = engineer.individual_baselines["zscore"]
        if baseline["std"] > 0:
            expected_zscore = (last_value - baseline["mean"]) / baseline["std"]
            actual_zscore = engineer._calculate_zscore("zscore", last_value)
            
            assert abs(actual_zscore - expected_zscore) < 1e-9
    
    @given(st.lists(st.floats(min_value=1e6 - 10, max_value=1e6 + 10, allow_nan=False), min_size=5))
    def test_numerical_stability_large_numbers(self, values):
        """
        Test numerical stability with large numbers.
        
        Property: Algorithm should handle large numbers without overflow
        """
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        for value in values:
            engineer._update_individual_baseline("large", value)
        
        baseline = engineer.individual_baselines["large"]
        
        # Mean should be close to 1e6
        assert abs(baseline["mean"] - 1e6) < 20.0
        
        # Std should be reasonable (not exploded due to numerical errors)
        assert baseline["std"] < 100.0
    
    @given(st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False), min_size=5))
    def test_sum_and_sum_sq_tracking(self, values):
        """
        Test that sum and sum_sq are tracked correctly.
        
        Properties:
        - sum == np.sum(values)
        - sum_sq == np.sum(values**2)
        """
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        for value in values:
            engineer._update_individual_baseline("sum_test", value)
        
        baseline = engineer.individual_baselines["sum_test"]
        
        expected_sum = np.sum(values)
        expected_sum_sq = np.sum(np.array(values) ** 2)
        
        assert abs(baseline["sum"] - expected_sum) < 1e-9
        assert abs(baseline["sum_sq"] - expected_sum_sq) < 1e-9
    
    def test_empty_baseline_behavior(self):
        """Test behavior with no values added."""
        engineer = AdvancedFeatureEngineer(user_id="test_user")
        
        # Calculate z-score without any baseline
        zscore = engineer._calculate_zscore("nonexistent", 5.0)
        assert zscore == 0.0  # Should handle gracefully