"""
DLMO (Dim Light Melatonin Onset) Calculator Service V3

Enhanced CBT-based DLMO calculation following Seoul XGBoost study approach
with mathematical improvements from St. Hilaire and Cheng validation papers.

Key improvements:
1. Better initial conditions from limit cycle analysis
2. Realistic activity-to-light conversion with dynamic thresholds
3. Light suppression for more accurate circadian modeling
4. Targets Seoul study expectations: DLMO ~21-22h for normal sleepers

Design Principles:
- Pure domain logic (no external dependencies except numpy/scipy)
- Immutable value objects
- Single Responsibility: DLMO calculation
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Union
import math
import numpy as np
from scipy.signal import find_peaks

from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.entities.activity_record import ActivityRecord


@dataclass(frozen=True)
class LightActivityProfile:
    """
    Immutable light/activity exposure profile for circadian modeling.
    
    Contains hourly values for a 24-hour period.
    """
    date: date
    hourly_values: List[float]  # 24 values, one per hour
    data_type: str  # 'light', 'activity', or 'combined'
    
    @property
    def average_exposure(self) -> float:
        """Average exposure across the day."""
        return sum(self.hourly_values) / len(self.hourly_values)


@dataclass(frozen=True)
class CircadianPhaseResult:
    """
    Immutable result of circadian phase calculation.
    
    Contains DLMO timing and related circadian metrics.
    """
    date: date
    dlmo_hour: float  # DLMO time in hours (0-23.99)
    cbt_min_hour: float  # CBT minimum time
    cbt_amplitude: float  # Circadian amplitude (strength)
    phase_angle: float  # Phase angle between DLMO and sleep
    confidence: float  # Confidence in prediction (0-1)
    
    @property
    def dlmo_time(self) -> str:
        """DLMO as HH:MM format."""
        hours = int(self.dlmo_hour)
        minutes = int((self.dlmo_hour - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"


class DLMOCalculatorV3:
    """
    Enhanced DLMO calculator using Seoul study approach with better mathematics.
    
    Core: CBT minimum - 7 hours (validated approximation)
    Enhanced with: proper initial conditions, realistic light modeling, suppression
    """
    
    # Model constants
    AWAKE_LUX = 250.0  # Light level when awake
    ASLEEP_LUX = 0.0   # Light level when asleep
    DELTA_T = 1/60.0   # Time step (1 minute in hours)
    CBT_TO_DLMO_OFFSET = 7.0  # Hours from CBT min to DLMO (Seoul study validated)
    
    # Circadian model parameters (from St. Hilaire/Kronauer)
    TAU = 24.2  # Intrinsic period
    G = 33.75   # Coupling strength
    K = 0.55    # Stiffness
    MU = 0.23   # Amplitude recovery rate (original value)
    B = 0.0075  # Photoreceptor decay rate
    
    # Light response parameters
    I0 = 9500.0  # Half-saturation constant
    P = 0.5      # Hill coefficient (adjusted)
    A0 = 0.05    # Maximum drive
    LIGHT_SUPPRESSION_M = 3.0  # Light suppression factor (moderate suppression)
    
    # Activity conversion parameters (Cheng validation)
    # Dynamic thresholds as multipliers of (max_activity/2)
    # Adjusted for better variation with step count data
    ACTIVITY_THRESHOLD_MULTIPLIERS = [0, 0.1, 0.25, 0.4, 0.7]
    ACTIVITY_LUX_LEVELS = [0, 50, 150, 300, 500, 1000]
    
    def calculate_dlmo(
        self, 
        sleep_records: Optional[List[SleepRecord]] = None,
        activity_records: Optional[List[ActivityRecord]] = None,
        target_date: Optional[date] = None,
        days_to_model: int = 14,
        use_activity: bool = True,
        day_length_hours: Optional[float] = None
    ) -> Optional[CircadianPhaseResult]:
        """
        Calculate DLMO using enhanced CBT-based approach.
        
        Args:
            sleep_records: Sleep/wake data
            activity_records: Activity data (preferred)
            target_date: Date to calculate DLMO for
            days_to_model: Days of data to model (14+ recommended)
            use_activity: Use activity-based prediction (more accurate)
            day_length_hours: Photoperiod for seasonal adjustment
            
        Returns:
            DLMO timing and circadian metrics, or None if insufficient data
        """
        if not target_date:
            target_date = date.today()
        
        # Create light/activity profiles
        if use_activity and activity_records:
            profiles = self._create_activity_profiles(
                activity_records, target_date, days_to_model
            )
        elif sleep_records:
            profiles = self._create_light_profiles_from_sleep(
                sleep_records, target_date, days_to_model
            )
        else:
            return None
        
        if not profiles:
            return None
        
        # Run circadian model with proper initial conditions
        cbt_rhythm = self._run_circadian_model(profiles)
        
        # Extract DLMO using CBT minimum - 7 hours
        return self._extract_dlmo_from_cbt(cbt_rhythm, target_date)
    
    def _create_activity_profiles(
        self,
        activity_records: List[ActivityRecord],
        target_date: date,
        days: int
    ) -> List[LightActivityProfile]:
        """
        Create activity-based light profiles with dynamic thresholds.
        """
        profiles = []
        
        # Find max activity for dynamic thresholds
        all_activity_values = [r.value for r in activity_records if r.value > 0]
        max_activity = max(all_activity_values) if all_activity_values else 1000
        
        for day_offset in range(days):
            current_date = target_date - timedelta(days=days-1-day_offset)
            
            # Get hourly activity
            hourly_activity = self._calculate_hourly_activity(
                activity_records, current_date
            )
            
            # Convert to lux with dynamic thresholds
            hourly_lux = []
            for activity in hourly_activity:
                lux = self._activity_to_lux_dynamic(activity, max_activity)
                hourly_lux.append(lux)
            
            profiles.append(LightActivityProfile(
                date=current_date,
                hourly_values=hourly_lux,
                data_type='activity'
            ))
        
        return profiles
    
    def _calculate_hourly_activity(
        self,
        activity_records: List[ActivityRecord],
        target_date: date
    ) -> List[float]:
        """Calculate total activity for each hour of the day."""
        hourly_totals = [0.0] * 24
        
        for record in activity_records:
            if record.start_date.date() == target_date:
                hour = record.start_date.hour
                if 0 <= hour < 24:
                    hourly_totals[hour] += record.value
        
        return hourly_totals
    
    def _activity_to_lux_dynamic(self, activity: float, max_activity: float) -> float:
        """
        Convert activity to lux using dynamic thresholds.
        
        Based on Cheng validation: thresholds at fractions of (max_activity/2).
        """
        # Calculate dynamic thresholds
        half_max = max_activity / 2.0
        
        # Find appropriate lux level
        for i in range(len(self.ACTIVITY_THRESHOLD_MULTIPLIERS) - 1, -1, -1):
            threshold = self.ACTIVITY_THRESHOLD_MULTIPLIERS[i] * half_max
            if activity >= threshold:
                return self.ACTIVITY_LUX_LEVELS[i]
        
        return 0.0
    
    def _create_light_profiles_from_sleep(
        self,
        sleep_records: List[SleepRecord],
        target_date: date,
        days: int
    ) -> List[LightActivityProfile]:
        """
        Create light profiles from sleep/wake patterns (fallback method).
        """
        profiles = []
        
        for day_offset in range(days):
            current_date = target_date - timedelta(days=days-1-day_offset)
            
            # Initialize with awake (250 lux)
            hourly_lux = [self.AWAKE_LUX] * 24
            
            # Set sleep periods to 0 lux
            for sleep in sleep_records:
                if sleep.start_date.date() <= current_date <= sleep.end_date.date():
                    # Mark sleep hours
                    for hour in range(24):
                        hour_start = datetime.combine(
                            current_date, 
                            datetime.min.time()
                        ) + timedelta(hours=hour)
                        hour_end = hour_start + timedelta(hours=1)
                        
                        # If this hour overlaps with sleep, set to 0 lux
                        if (hour_start < sleep.end_date and 
                            hour_end > sleep.start_date):
                            hourly_lux[hour] = self.ASLEEP_LUX
            
            profiles.append(LightActivityProfile(
                date=current_date,
                hourly_values=hourly_lux,
                data_type='sleep'
            ))
        
        return profiles
    
    def _run_circadian_model(
        self, 
        profiles: List[LightActivityProfile]
    ) -> List[Tuple[float, float]]:
        """
        Run circadian pacemaker model with proper initial conditions.
        
        Returns hourly CBT values for the last day.
        """
        # Get initial conditions from limit cycle
        x, xc, n = self._get_initial_conditions_from_limit_cycle()
        
        # Store states for the last day
        last_day_states = []
        
        # Run model for all days
        for profile_idx, profile in enumerate(profiles):
            is_last_day = (profile_idx == len(profiles) - 1)
            
            # Simulate each hour
            for hour in range(24):
                light = profile.hourly_values[hour]
                
                # Store state at start of hour for last day
                if is_last_day:
                    last_day_states.append((hour, x, xc, n))
                
                # Run model for this hour (60 steps of 1 minute)
                for _ in range(60):
                    # Calculate derivatives with light suppression
                    dx, dxc, dn = self._circadian_derivatives_with_suppression(
                        x, xc, n, light
                    )
                    
                    # Update state (Euler integration)
                    x += dx * self.DELTA_T
                    xc += dxc * self.DELTA_T
                    n += dn * self.DELTA_T
        
        # Generate CBT rhythm from states
        cbt_rhythm = []
        for hour, state_x, state_xc, _ in last_day_states:
            # CBT is proportional to circadian state
            # CBT minimum when cos(x) = -1, which is at x = π
            cbt = -state_xc * math.cos(state_x)
            cbt_rhythm.append((hour, cbt))
        
        return cbt_rhythm
    
    def _get_initial_conditions_from_limit_cycle(self) -> Tuple[float, float, float]:
        """
        Get initial conditions from limit cycle with standard sleep schedule.
        
        Simulates regular sleep pattern to find steady state for normal phase.
        Target: CBT minimum at 4-5 AM for DLMO at 9-10 PM.
        """
        # Standard schedule: wake at 7am, bed at 11pm
        wake_hour = 7
        bed_hour = 23
        
        # Create standard light pattern
        hourly_lux = []
        for hour in range(24):
            if wake_hour <= hour < bed_hour:
                hourly_lux.append(self.AWAKE_LUX)
            else:
                hourly_lux.append(self.ASLEEP_LUX)
        
        # Initialize for proper CBT minimum timing
        # For 11pm-7am sleeper, CBT min should be at 4-5 AM
        # Phase evolution: starts at wake (7am), minimum 21 hours later
        # Phase advances ~π/12 per hour during wake, slower during sleep
        # Empirically calibrated for this light schedule
        x = -2.5    # Initial phase for proper evolution
        xc = 1.0    # Normal amplitude
        n = 0.1     # Low photoreceptor state
        
        # Run for standard duration to establish rhythm
        for day in range(14):
            for hour in range(24):
                light = hourly_lux[hour]
                # Run for one hour
                for _ in range(60):
                    dx, dxc, dn = self._circadian_derivatives_with_suppression(
                        x, xc, n, light
                    )
                    x += dx * self.DELTA_T
                    xc += dxc * self.DELTA_T
                    n += dn * self.DELTA_T
                    
                    # Keep phase in [-π, π]
                    if x > math.pi:
                        x -= 2 * math.pi
                    elif x < -math.pi:
                        x += 2 * math.pi
        
        return x, xc, n
    
    def _circadian_derivatives_with_suppression(
        self, 
        x: float, 
        xc: float, 
        n: float, 
        light: float
    ) -> Tuple[float, float, float]:
        """
        Calculate derivatives with light suppression effect.
        
        Includes A'(t) = A(t)(1 - m*B̂) suppression from St. Hilaire.
        """
        # Light response
        alpha = self._alpha_function(light)
        
        # Process B (light input) with suppression
        B_hat = self.G * (1 - n) * alpha
        
        # Apply light suppression (St. Hilaire equation 12)
        suppression_factor = 1 - self.LIGHT_SUPPRESSION_M * B_hat
        suppression_factor = max(0, suppression_factor)  # Keep non-negative
        
        # Modified light drive
        B = B_hat * (1 - 0.4 * x) * (1 - 0.4 * xc) * suppression_factor
        
        # Derivatives (equations 7-8 from St. Hilaire)
        dx = math.pi / 12.0 * (xc + B)
        dxc = math.pi / 12.0 * (
            self.MU * (xc - 4.0 * xc**3 / 3.0) - 
            x * ((24.0 / (0.99669 * self.TAU))**2 + self.K * B)
        )
        dn = 60.0 * (alpha * (1.0 - n) - self.B * n)
        
        return dx, dxc, dn
    
    def _alpha_function(self, light: float) -> float:
        """
        Photic sensitivity function.
        
        Converts light intensity to circadian drive.
        """
        # Hill equation: alpha = a0 * (I^p / (I^p + I0^p))
        return self.A0 * (light**self.P / (light**self.P + self.I0**self.P))
    
    def _extract_dlmo_from_cbt(
        self, 
        cbt_rhythm: List[Tuple[float, float]],
        target_date: date
    ) -> CircadianPhaseResult:
        """
        Extract DLMO from CBT rhythm using Seoul study approach.
        
        DLMO = CBT minimum - 7 hours
        """
        # Extract CBT values
        hours = [h for h, _ in cbt_rhythm]
        cbt_values = np.array([cbt for _, cbt in cbt_rhythm])
        
        # Find CBT minimum
        min_idx = np.argmin(cbt_values)
        cbt_min_hour = hours[min_idx]
        cbt_min_value = cbt_values[min_idx]
        
        # Calculate amplitude (max - min)
        cbt_amplitude = np.max(cbt_values) - np.min(cbt_values)
        
        # DLMO = CBT min - 7 hours (Seoul study)
        dlmo_hour = (cbt_min_hour - self.CBT_TO_DLMO_OFFSET) % 24
        
        # Calculate phase angle (DLMO to sleep onset)
        # Assuming typical sleep at 23:00 for confidence calculation
        phase_angle = (23.0 - dlmo_hour) % 24
        
        # Confidence based on amplitude and reasonable phase angle
        confidence = min(1.0, cbt_amplitude / 1.5)  # Normal amplitude ~1.5
        if not (1.5 <= phase_angle <= 3.5):  # DLMO should be 1.5-3.5h before sleep
            confidence *= 0.8
        
        return CircadianPhaseResult(
            date=target_date,
            dlmo_hour=dlmo_hour,
            cbt_min_hour=cbt_min_hour,
            cbt_amplitude=cbt_amplitude,
            phase_angle=phase_angle,
            confidence=confidence
        )