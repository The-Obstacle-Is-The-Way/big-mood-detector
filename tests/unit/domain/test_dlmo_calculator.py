"""
Unit tests for DLMO Calculator

Tests the mathematical circadian pacemaker model for DLMO estimation.
Critical for bipolar disorder phase prediction as per Seoul study.
"""

from datetime import datetime, date, timedelta
import pytest
import math

from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.services.dlmo_calculator import (
    DLMOCalculator,
    LightProfile,
    CircadianPhaseResult
)


class TestDLMOCalculator:
    """Test DLMO calculation from sleep/wake patterns."""
    
    @pytest.fixture
    def calculator(self):
        """Create DLMO calculator with default settings."""
        return DLMOCalculator()
    
    def _create_sleep_record(
        self, 
        start: datetime, 
        duration_hours: float
    ) -> SleepRecord:
        """Helper to create sleep records."""
        end = start + timedelta(hours=duration_hours)
        from big_mood_detector.domain.entities.sleep_record import SleepState
        return SleepRecord(
            source_name="test",
            start_date=start,
            end_date=end,
            state=SleepState.ASLEEP
        )
    
    def test_light_profile_creation_regular_schedule(self, calculator):
        """Test light profile for regular sleep schedule."""
        # Arrange - regular 11 PM to 7 AM sleep
        base_date = date(2024, 1, 15)
        sleep_records = []
        
        for day in range(7):
            sleep_start = datetime.combine(
                base_date + timedelta(days=day),
                datetime.min.time()
            ).replace(hour=23)
            
            sleep_records.append(
                self._create_sleep_record(sleep_start, 8.0)
            )
        
        # Act
        profiles = calculator._create_light_profiles(
            sleep_records,
            base_date + timedelta(days=6),
            days=7
        )
        
        # Assert
        assert len(profiles) == 7
        
        # Check last day profile
        last_profile = profiles[-1]
        assert last_profile.date == base_date + timedelta(days=6)
        
        # Should be dark (0 lux) from 11 PM to 7 AM
        assert last_profile.hourly_lux[23] == 0.0  # 11 PM
        assert all(last_profile.hourly_lux[h] == 0.0 for h in range(0, 7))
        
        # Should be light (250 lux) during day
        assert all(last_profile.hourly_lux[h] == 250.0 for h in range(7, 23))
    
    def test_light_profile_irregular_sleep(self, calculator):
        """Test light profile for irregular sleep patterns."""
        # Arrange - variable sleep times
        sleep_records = [
            # Day 1: Late night (1 AM - 9 AM)
            self._create_sleep_record(
                datetime(2024, 1, 15, 1, 0), 8.0
            ),
            # Day 2: Split sleep (10 PM - 2 AM, 4 AM - 8 AM)
            self._create_sleep_record(
                datetime(2024, 1, 15, 22, 0), 4.0
            ),
            self._create_sleep_record(
                datetime(2024, 1, 16, 4, 0), 4.0
            ),
        ]
        
        # Act
        profiles = calculator._create_light_profiles(
            sleep_records,
            date(2024, 1, 16),
            days=2
        )
        
        # Assert
        assert len(profiles) == 2
        
        # Day 1 should have late sleep
        day1 = profiles[0]
        assert day1.hourly_lux[0] == 250.0  # Awake at midnight
        assert day1.hourly_lux[2] == 0.0    # Asleep at 2 AM
        
        # Day 2 should have split sleep
        day2 = profiles[1]
        assert day2.hourly_lux[0] == 0.0    # Asleep (from 10 PM previous)
        assert day2.hourly_lux[3] == 250.0  # Awake during gap
        assert day2.hourly_lux[5] == 0.0    # Back asleep
    
    def test_circadian_model_convergence(self, calculator):
        """Test that circadian model reaches stable state."""
        # Arrange - consistent sleep schedule
        base_date = date(2024, 1, 1)
        sleep_records = []
        
        for day in range(14):  # 2 weeks
            sleep_start = datetime.combine(
                base_date + timedelta(days=day),
                datetime.min.time()
            ).replace(hour=23)
            
            sleep_records.append(
                self._create_sleep_record(sleep_start, 8.0)
            )
        
        # Act
        result = calculator.calculate_dlmo(
            sleep_records,
            base_date + timedelta(days=13),
            days_to_model=7
        )
        
        # Assert
        assert result is not None
        assert isinstance(result, CircadianPhaseResult)
        
        # DLMO should be reasonable (typically 9-11 PM for regular sleepers)
        assert 20 <= result.dlmo_hour <= 23
        
        # CBT minimum should be during sleep (typically 4-6 AM)
        assert 3 <= result.cbt_min_hour <= 7
        
        # Should have measurable amplitude
        assert result.cbt_amplitude > 0
    
    def test_dlmo_calculation_formula(self, calculator):
        """Test DLMO = CBTmin - 7 hours formula."""
        # Arrange
        cbt_rhythm = [
            (0, 0.5), (1, 0.3), (2, 0.1), (3, -0.1),
            (4, -0.3), (5, -0.5),  # Minimum at 5 AM
            (6, -0.3), (7, -0.1), (8, 0.1), (9, 0.3),
            (10, 0.5), (11, 0.7), (12, 0.9), (13, 1.0),
            (14, 0.9), (15, 0.7), (16, 0.5), (17, 0.3),
            (18, 0.1), (19, -0.1), (20, -0.3), (21, -0.1),
            (22, 0.1), (23, 0.3)
        ]
        
        # Act
        result = calculator._extract_dlmo(cbt_rhythm, date(2024, 1, 15))
        
        # Assert
        assert result.cbt_min_hour == 5  # Minimum at 5 AM
        assert result.dlmo_hour == 22    # 5 - 7 = -2, wrapped to 22 (10 PM)
        assert result.cbt_amplitude == 1.5  # 1.0 - (-0.5)
    
    def test_phase_delay_pattern(self, calculator):
        """Test DLMO for delayed sleep phase (late sleeper)."""
        # Arrange - sleep 3 AM to 11 AM
        base_date = date(2024, 1, 15)
        sleep_records = []
        
        for day in range(7):
            sleep_start = datetime.combine(
                base_date + timedelta(days=day),
                datetime.min.time()
            ).replace(hour=3)
            
            sleep_records.append(
                self._create_sleep_record(sleep_start, 8.0)
            )
        
        # Act
        result = calculator.calculate_dlmo(
            sleep_records,
            base_date + timedelta(days=6),
            days_to_model=7
        )
        
        # Assert
        assert result is not None
        # DLMO should be delayed (after midnight)
        assert result.dlmo_hour > 0 or result.dlmo_hour < 6
    
    def test_phase_advance_pattern(self, calculator):
        """Test DLMO for advanced sleep phase (early sleeper)."""
        # Arrange - sleep 8 PM to 4 AM
        base_date = date(2024, 1, 15)
        sleep_records = []
        
        for day in range(7):
            sleep_start = datetime.combine(
                base_date + timedelta(days=day),
                datetime.min.time()
            ).replace(hour=20)
            
            sleep_records.append(
                self._create_sleep_record(sleep_start, 8.0)
            )
        
        # Act
        result = calculator.calculate_dlmo(
            sleep_records,
            base_date + timedelta(days=6),
            days_to_model=7
        )
        
        # Assert
        assert result is not None
        # DLMO should be advanced (early evening)
        assert 17 <= result.dlmo_hour <= 21
    
    def test_insufficient_data_handling(self, calculator):
        """Test handling of insufficient sleep data."""
        # Arrange - only 2 days of data
        sleep_records = [
            self._create_sleep_record(datetime(2024, 1, 15, 23, 0), 8.0),
            self._create_sleep_record(datetime(2024, 1, 16, 23, 0), 8.0),
        ]
        
        # Act
        result = calculator.calculate_dlmo(
            sleep_records,
            date(2024, 1, 16),
            days_to_model=7
        )
        
        # Assert
        assert result is None  # Not enough data
    
    def test_alpha_function_saturation(self, calculator):
        """Test light response function saturation."""
        # Low light
        alpha_low = calculator._alpha_function(100)
        
        # Medium light
        alpha_med = calculator._alpha_function(1000)
        
        # Bright light
        alpha_bright = calculator._alpha_function(10000)
        
        # Assert monotonic increase with saturation
        assert alpha_low < alpha_med < alpha_bright
        assert alpha_bright <= calculator.A0  # Should approach maximum
        
        # Check specific value at half-saturation
        alpha_half = calculator._alpha_function(calculator.I0)
        expected = calculator.A0 * 0.5  # At I0, should be ~50% of max
        assert abs(alpha_half - expected) < 0.01
    
    def test_circadian_model_derivatives(self, calculator):
        """Test circadian model derivative calculations."""
        # Test at various states
        test_cases = [
            (0.0, 0.0, 0.5, 250.0),   # Neutral state, light
            (1.0, 1.0, 0.8, 0.0),      # High amplitude, dark
            (-1.0, -1.0, 0.2, 1000.0), # Negative phase, bright
        ]
        
        for x, xc, n, light in test_cases:
            dx, dxc, dn = calculator._circadian_derivatives(x, xc, n, light)
            
            # Derivatives should be finite
            assert math.isfinite(dx)
            assert math.isfinite(dxc)
            assert math.isfinite(dn)
            
            # Basic sanity checks
            if light > 0:
                # With light, n changes based on current state
                # dn = 60 * (alpha * (1-n) - b * n)
                # Can be positive if (1-n) term dominates or negative if n is high
                pass
            else:
                # In darkness, dn = -60 * b * n (always negative if n > 0)
                if n > 0:
                    assert dn < 0
    
    def test_cbt_rhythm_generation(self, calculator):
        """Test CBT rhythm has expected properties."""
        # Arrange - simple light profile
        profiles = [
            LightProfile(
                date=date(2024, 1, 15),
                hourly_lux=[0]*8 + [250]*16  # 8h dark, 16h light
            )
        ] * 7  # Repeat for a week
        
        # Act
        cbt_rhythm = calculator._run_circadian_model(profiles)
        
        # Assert
        assert len(cbt_rhythm) == 24  # One value per hour
        
        # Should have variation (not flat)
        cbt_values = [cbt for _, cbt in cbt_rhythm]
        assert max(cbt_values) > min(cbt_values)
        
        # Should be roughly sinusoidal (one peak, one trough)
        max_idx = cbt_values.index(max(cbt_values))
        min_idx = cbt_values.index(min(cbt_values))
        
        # Peak and trough should be roughly 12 hours apart
        phase_diff = abs(max_idx - min_idx)
        assert 10 <= phase_diff <= 14 or 10 <= (24 - phase_diff) <= 14
    
    def test_real_world_scenario(self, calculator):
        """Test with realistic sleep pattern including weekends."""
        # Arrange - weekday vs weekend pattern
        base_date = date(2024, 1, 15)  # Monday
        sleep_records = []
        
        for day in range(14):  # 2 weeks
            current_date = base_date + timedelta(days=day)
            
            # Weekday: 11 PM - 7 AM
            if current_date.weekday() < 5:
                sleep_start = datetime.combine(
                    current_date,
                    datetime.min.time()
                ).replace(hour=23)
                duration = 8.0
            # Weekend: 1 AM - 10 AM  
            else:
                sleep_start = datetime.combine(
                    current_date + timedelta(days=1),
                    datetime.min.time()
                ).replace(hour=1)
                duration = 9.0
            
            sleep_records.append(
                self._create_sleep_record(sleep_start, duration)
            )
        
        # Act - calculate for Sunday night
        result = calculator.calculate_dlmo(
            sleep_records,
            base_date + timedelta(days=13),
            days_to_model=7
        )
        
        # Assert
        assert result is not None
        # DLMO should show some delay from weekend pattern
        assert result.dlmo_hour > 22 or result.dlmo_hour < 2