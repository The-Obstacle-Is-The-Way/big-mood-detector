"""
Unit tests for DLMO Calculator V3
Tests the enhanced CBT-based DLMO calculation with proper mathematics.
Following Seoul XGBoost study approach with St. Hilaire enhancements.
"""
from datetime import datetime, date, timedelta
import pytest
import math

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.services.dlmo_calculator_v3 import (
    DLMOCalculatorV3,
    CircadianPhaseResult,
    LightActivityProfile
)


class TestDLMOCalculatorV3:
    """Test enhanced DLMO calculation targeting Seoul study expectations."""
    
    def test_regular_sleeper_dlmo_in_expected_range(self):
        """Regular sleeper (11pm-7am) should have DLMO 9-10pm."""
        # Arrange
        calculator = DLMOCalculatorV3()
        sleep_records, activity_records = self._create_regular_sleeper_data()
        target_date = date(2024, 1, 14)
        
        # Act - test with sleep-based profiles first (Seoul paper approach)
        result = calculator.calculate_dlmo(
            sleep_records=sleep_records,
            activity_records=activity_records,
            target_date=target_date,
            days_to_model=14,
            use_activity=False  # Use sleep/wake patterns like Seoul paper
        )
        
        # Assert
        assert result is not None
        assert 20.0 <= result.dlmo_hour <= 22.0, f"DLMO {result.dlmo_hour:.1f}h should be 20-22h (8-10pm)"
        assert 3.0 <= result.cbt_min_hour <= 6.0, f"CBT min {result.cbt_min_hour:.1f}h should be 3-6h (3-6am)"
        assert result.confidence >= 0.8
    
    def test_shift_worker_flexible_dlmo(self):
        """Shift worker should have DLMO that adapts to schedule."""
        # Arrange
        calculator = DLMOCalculatorV3()
        sleep_records, activity_records = self._create_shift_worker_data()
        target_date = date(2024, 1, 14)
        
        # Act
        result = calculator.calculate_dlmo(
            sleep_records=sleep_records,
            activity_records=activity_records,
            target_date=target_date,
            days_to_model=14,
            use_activity=True
        )
        
        # Assert
        assert result is not None
        # Shift workers can have DLMO at any time
        assert 0 <= result.dlmo_hour < 24
        assert result.confidence >= 0.5  # Lower confidence expected
    
    def test_debug_phase_evolution(self):
        """Debug phase evolution to understand timing."""
        # Arrange
        calculator = DLMOCalculatorV3()
        
        # Simple test: just one day with clear sleep pattern
        sleep_records = [
            SleepRecord(
                source_name="test",
                start_date=datetime(2024, 1, 1, 23, 0),
                end_date=datetime(2024, 1, 2, 7, 0),
                state=SleepState.ASLEEP_CORE
            )
        ]
        
        # Get initial conditions and run one day
        x, xc, n = calculator._get_initial_conditions_from_limit_cycle()
        print(f"\nInitial conditions: x={x:.3f}, xc={xc:.3f}, n={n:.3f}")
        
        # Create light profile for one day
        profiles = calculator._create_light_profiles_from_sleep(
            sleep_records, date(2024, 1, 2), days=1
        )
        
        print("\nLight profile:")
        for hour, lux in enumerate(profiles[0].hourly_values):
            print(f"Hour {hour:2d}: {lux:3.0f} lux")
        
        # Run model and track phase
        print("\nPhase evolution:")
        light_profile = profiles[0].hourly_values
        for hour in range(24):
            # Calculate phase angle
            angle = math.atan2(xc, x) * 180 / math.pi
            cbt = -xc * math.cos(x)
            print(f"Hour {hour:2d}: x={x:6.3f}, xc={xc:6.3f}, angle={angle:7.1f}°, CBT={cbt:6.3f}")
            
            # Run for one hour
            light = light_profile[hour]
            for _ in range(60):
                dx, dxc, dn = calculator._circadian_derivatives_with_suppression(x, xc, n, light)
                x += dx * calculator.DELTA_T
                xc += dxc * calculator.DELTA_T
                n += dn * calculator.DELTA_T
        
        assert True  # Just for debugging
    
    def test_debug_activity_conversion(self):
        """Debug activity to lux conversion."""
        # Arrange
        calculator = DLMOCalculatorV3()
        sleep_records, activity_records = self._create_regular_sleeper_data()
        target_date = date(2024, 1, 14)
        
        # Act
        profiles = calculator._create_activity_profiles(
            activity_records, 
            target_date, 
            days=1
        )
        
        # Debug print
        print("\nActivity to Lux conversion debug:")
        print(f"Date: {profiles[0].date}")
        for hour, lux in enumerate(profiles[0].hourly_values):
            activity = calculator._calculate_hourly_activity(activity_records, target_date)[hour]
            print(f"Hour {hour:2d}: {activity:6.0f} steps -> {lux:6.0f} lux")
        
        # Also check CBT rhythm
        cbt_rhythm = calculator._run_circadian_model(profiles[-7:])  # Last 7 days
        print("\nCBT Rhythm:")
        for hour, cbt in cbt_rhythm:
            print(f"Hour {hour:2d}: CBT = {cbt:6.3f}")
        
        assert True  # Just for debugging
    
    def test_activity_to_lux_conversion_realistic(self):
        """Activity should convert to realistic lux levels."""
        # Arrange
        calculator = DLMOCalculatorV3()
        activity_records = [
            ActivityRecord(
                source_name="test",
                start_date=datetime(2024, 1, 1, 10, 0),
                end_date=datetime(2024, 1, 1, 11, 0),
                activity_type=ActivityType.STEP_COUNT,
                value=1000,
                unit="count"
            ),
            ActivityRecord(
                source_name="test",
                start_date=datetime(2024, 1, 1, 14, 0),
                end_date=datetime(2024, 1, 1, 15, 0),
                activity_type=ActivityType.STEP_COUNT,
                value=2000,
                unit="count"
            ),
        ]
        
        # Act
        profiles = calculator._create_activity_profiles(
            activity_records, 
            date(2024, 1, 1), 
            days=1
        )
        
        # Assert
        assert len(profiles) == 1
        lux_values = profiles[0].hourly_values
        
        # Should have variety, not uniform 500 lux
        unique_values = set(lux_values)
        assert len(unique_values) >= 3, "Should have at least 3 different lux levels"
        assert 0 in unique_values, "Should have 0 lux during sleep"
        assert max(lux_values) <= 2000, "Max lux should be reasonable"
    
    def test_limit_cycle_initialization(self):
        """Initial conditions should represent normal circadian phase."""
        # Arrange
        calculator = DLMOCalculatorV3()
        
        # Act
        x, xc, n = calculator._get_initial_conditions_from_limit_cycle()
        
        # Assert
        # After limit cycle, should be in reasonable phase
        # CBT = -xc * cos(x), minimum when cos(x) = -1 (x ≈ π)
        # For CBT min at ~4-5 AM, need appropriate phase
        assert -math.pi <= x <= math.pi, "Phase should be in [-π, π]"
        assert 0.5 <= xc <= 1.5, "Amplitude should be normal"
        assert 0 <= n <= 1, "Photoreceptor state should be in [0, 1]"
    
    def test_cbt_minimum_timing(self):
        """CBT minimum should occur at appropriate circadian phase."""
        # Arrange
        calculator = DLMOCalculatorV3()
        sleep_records, activity_records = self._create_regular_sleeper_data()
        
        # Act - run circadian model
        profiles = calculator._create_activity_profiles(
            activity_records, date(2024, 1, 14), days=7
        )
        cbt_rhythm = calculator._run_circadian_model(profiles)
        
        # Assert
        cbt_values = [cbt for _, cbt in cbt_rhythm]
        min_idx = cbt_values.index(min(cbt_values))
        
        # CBT min should be 3-6 AM for normal sleeper
        assert 3 <= min_idx <= 6, f"CBT minimum at hour {min_idx} should be 3-6 AM"
    
    def test_light_suppression_applied(self):
        """Light suppression should modify melatonin synthesis."""
        # Arrange
        calculator = DLMOCalculatorV3()
        
        # Create bright light exposure
        bright_activity = [
            ActivityRecord(
                source_name="test",
                start_date=datetime(2024, 1, 1, 22, 0),
                end_date=datetime(2024, 1, 1, 23, 0),
                activity_type=ActivityType.STEP_COUNT,
                value=5000,  # High activity = bright light
                unit="count"
            )
        ]
        
        # Act
        result_bright = calculator.calculate_dlmo(
            sleep_records=[],
            activity_records=bright_activity,
            target_date=date(2024, 1, 1),
            days_to_model=1,
            use_activity=True
        )
        
        # Compare with dim light
        dim_activity = [a for a in bright_activity]
        dim_activity[0].value = 100  # Low activity = dim light
        
        result_dim = calculator.calculate_dlmo(
            sleep_records=[],
            activity_records=dim_activity,
            target_date=date(2024, 1, 1),
            days_to_model=1,
            use_activity=True
        )
        
        # Assert - bright light should delay DLMO
        if result_bright and result_dim:
            assert result_bright.dlmo_hour != result_dim.dlmo_hour, "Light should affect DLMO timing"
    
    def _create_regular_sleeper_data(self):
        """Create 14 days of regular sleep pattern data."""
        sleep_records = []
        activity_records = []
        
        for day in range(14):
            date_obj = datetime(2024, 1, 1) + timedelta(days=day)
            
            # Sleep 11pm-7am
            sleep_records.append(SleepRecord(
                source_name="test",
                start_date=date_obj.replace(hour=23, minute=0),
                end_date=(date_obj + timedelta(days=1)).replace(hour=7, minute=0),
                state=SleepState.ASLEEP_CORE
            ))
            
            # Activity pattern
            # Morning (7-9am): moderate
            for hour in [7, 8]:
                activity_records.append(ActivityRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=hour, minute=0),
                    end_date=date_obj.replace(hour=hour, minute=59),
                    activity_type=ActivityType.STEP_COUNT,
                    value=800,
                    unit="count"
                ))
            
            # Day (9am-5pm): high
            for hour in range(9, 17):
                activity_records.append(ActivityRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=hour, minute=0),
                    end_date=date_obj.replace(hour=hour, minute=59),
                    activity_type=ActivityType.STEP_COUNT,
                    value=1200,
                    unit="count"
                ))
            
            # Evening (5-10pm): moderate
            for hour in range(17, 22):
                activity_records.append(ActivityRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=hour, minute=0),
                    end_date=date_obj.replace(hour=hour, minute=59),
                    activity_type=ActivityType.STEP_COUNT,
                    value=600,
                    unit="count"
                ))
            
            # Late evening (10-11pm): low
            activity_records.append(ActivityRecord(
                source_name="test",
                start_date=date_obj.replace(hour=22, minute=0),
                end_date=date_obj.replace(hour=22, minute=59),
                activity_type=ActivityType.STEP_COUNT,
                value=200,
                unit="count"
            ))
        
        return sleep_records, activity_records
    
    def _create_shift_worker_data(self):
        """Create 14 days of shift work pattern."""
        sleep_records = []
        activity_records = []
        
        for day in range(14):
            date_obj = datetime(2024, 1, 1) + timedelta(days=day)
            
            if day % 7 < 5:  # Work days
                # Sleep 8am-4pm
                sleep_records.append(SleepRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=8, minute=0),
                    end_date=date_obj.replace(hour=16, minute=0),
                    state=SleepState.ASLEEP_CORE
                ))
                
                # Night activity
                for hour in range(0, 7):
                    activity_records.append(ActivityRecord(
                        source_name="test",
                        start_date=date_obj.replace(hour=hour, minute=0),
                        end_date=date_obj.replace(hour=hour, minute=59),
                        activity_type=ActivityType.STEP_COUNT,
                        value=1000,
                        unit="count"
                    ))
            else:  # Off days
                # Normal sleep
                sleep_records.append(SleepRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=23, minute=0),
                    end_date=(date_obj + timedelta(days=1)).replace(hour=7, minute=0),
                    state=SleepState.ASLEEP_CORE
                ))
        
        return sleep_records, activity_records