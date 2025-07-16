"""
DLMO (Dim Light Melatonin Onset) Calculator Service V2

Enhanced implementation based on the validated approach from:
"Predicting circadian misalignment with wearable technology" (Cheng et al., 2021)

Key improvements:
1. Activity-based prediction (concordance 0.72 vs 0.63 for light-only)
2. Proper CBT minimum detection using local minima with prominence
3. Seasonal adjustment for winter months
4. Validated on 45 night shift workers with extreme circadian disruption

Design Principles:
- Pure domain logic (no external dependencies)
- Immutable value objects
- Single Responsibility: Only DLMO calculation
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
    Can use light (lux), activity counts, or combined approach.
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


class DLMOCalculatorV2:
    """
    Enhanced DLMO calculator using validated methods from clinical research.
    
    Implements both light-based and activity-based approaches,
    with preference for activity data which shows better performance.
    """
    
    # Model constants
    AWAKE_LUX = 250.0  # Light level when awake
    ASLEEP_LUX = 0.0   # Light level when asleep
    DELTA_T = 1/60.0   # Time step (1 minute in hours)
    CBT_TO_DLMO_OFFSET = 7.1  # Hours from CBT min to DLMO (validated)
    
    # Circadian model parameters (from St. Hilaire 2007)
    TAU = 24.2  # Intrinsic period
    G = 33.75   # Coupling strength
    K = 0.55    # Stiffness
    MU = 0.23   # Amplitude recovery rate
    B = 0.0075  # Photoreceptor decay rate
    
    # Light response parameters
    I0 = 9500.0  # Half-saturation constant
    P = 0.6      # Hill coefficient
    A0 = 0.05    # Maximum drive
    
    # Activity conversion parameters (from MATLAB code)
    ACTIVITY_THRESHOLDS = [0, 0.1, 0.25, 0.4]  # Relative to max
    ACTIVITY_LUX_LEVELS = [0, 100, 200, 500, 2000]  # Corresponding lux
    
    def calculate_dlmo(
        self, 
        sleep_records: Optional[List[SleepRecord]] = None,
        activity_records: Optional[List[ActivityRecord]] = None,
        target_date: Optional[date] = None,
        days_to_model: int = 7,
        use_activity: bool = True,
        day_length_hours: Optional[float] = None
    ) -> Optional[CircadianPhaseResult]:
        """
        Calculate DLMO using activity and/or sleep data.
        
        Args:
            sleep_records: Historical sleep records
            activity_records: Historical activity records (preferred)
            target_date: Date to calculate DLMO for
            days_to_model: Days of history to use (default 7)
            use_activity: Whether to use activity-based approach
            day_length_hours: Hours of daylight for seasonal adjustment
            
        Returns:
            CircadianPhaseResult with DLMO estimate
        """
        if not target_date:
            target_date = date.today()
        
        # Prefer activity-based approach if available
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
        
        # Apply seasonal adjustment if needed
        if day_length_hours and day_length_hours < 12:
            profiles = self._apply_seasonal_adjustment(profiles)
        
        # Run circadian model
        cbt_rhythm = self._run_circadian_model(profiles)
        
        # Extract DLMO with proper minimum detection
        return self._extract_dlmo_enhanced(cbt_rhythm, target_date)
    
    def _create_activity_profiles(
        self,
        activity_records: List[ActivityRecord],
        target_date: date,
        days: int
    ) -> List[LightActivityProfile]:
        """
        Create activity-based light profiles.
        
        Converts activity counts to equivalent light exposure
        using validated thresholds from the MATLAB implementation.
        """
        profiles = []
        
        # Find max activity for normalization
        all_activity_values = [r.value for r in activity_records if r.value > 0]
        max_activity = max(all_activity_values) if all_activity_values else 1000
        
        for day_offset in range(days):
            current_date = target_date - timedelta(days=days-1-day_offset)
            
            # Get hourly activity
            hourly_activity = self._calculate_hourly_activity(
                activity_records, current_date
            )
            
            # Convert activity to light equivalent
            hourly_lux = []
            for activity in hourly_activity:
                # Normalize activity
                normalized = activity / (max_activity / 2)  # Match MATLAB scaling
                
                # Map to lux levels using thresholds
                lux = self._activity_to_lux(normalized)
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
    
    def _activity_to_lux(self, normalized_activity: float) -> float:
        """
        Convert normalized activity to lux equivalent.
        
        Based on the basicsteps function from MATLAB code.
        """
        # Find appropriate threshold
        for i in range(len(self.ACTIVITY_THRESHOLDS) - 1, -1, -1):
            if normalized_activity >= self.ACTIVITY_THRESHOLDS[i]:
                return self.ACTIVITY_LUX_LEVELS[i]
        
        return 0.0
    
    def _create_light_profiles_from_sleep(
        self,
        sleep_records: List[SleepRecord],
        target_date: date,
        days: int
    ) -> List[LightActivityProfile]:
        """
        Create light profiles from sleep/wake patterns.
        
        Fallback method when activity data is not available.
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
                        hour_start = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour)
                        hour_end = hour_start + timedelta(hours=1)
                        
                        # Check overlap with sleep period
                        if self._periods_overlap(
                            sleep.start_date, sleep.end_date,
                            hour_start, hour_end
                        ):
                            hourly_lux[hour] = self.ASLEEP_LUX
            
            profiles.append(LightActivityProfile(
                date=current_date,
                hourly_values=hourly_lux,
                data_type='light'
            ))
        
        return profiles
    
    def _apply_seasonal_adjustment(
        self,
        profiles: List[LightActivityProfile]
    ) -> List[LightActivityProfile]:
        """
        Apply seasonal adjustment for winter months.
        
        Doubles light sensitivity when day length < 12 hours.
        """
        adjusted = []
        
        for profile in profiles:
            # Double all non-zero values
            adjusted_values = [
                v * 2 if v > 0 else 0
                for v in profile.hourly_values
            ]
            
            adjusted.append(LightActivityProfile(
                date=profile.date,
                hourly_values=adjusted_values,
                data_type=profile.data_type
            ))
        
        return adjusted
    
    def _run_circadian_model(
        self, 
        profiles: List[LightActivityProfile]
    ) -> List[Tuple[float, float]]:
        """
        Run Forger circadian pacemaker model.
        
        Returns hourly CBT values for the last day.
        """
        # Initial conditions (standard DLMO at 9pm = 21:00)
        x = 0.0   # Circadian phase
        xc = 0.0  # Circadian amplitude  
        n = 0.5   # Photoreceptor state
        
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
                    # Calculate derivatives
                    dx, dxc, dn = self._circadian_derivatives(x, xc, n, light)
                    
                    # Update state (Euler integration)
                    x += dx * self.DELTA_T
                    xc += dxc * self.DELTA_T
                    n += dn * self.DELTA_T
        
        # Generate CBT rhythm from states
        cbt_rhythm = []
        for hour, state_x, state_xc, _ in last_day_states:
            # CBT is proportional to circadian state
            cbt = -state_xc * math.cos(state_x)
            cbt_rhythm.append((hour, cbt))
        
        return cbt_rhythm
    
    def _circadian_derivatives(
        self, 
        x: float, 
        xc: float, 
        n: float, 
        light: float
    ) -> Tuple[float, float, float]:
        """
        Calculate derivatives for circadian model.
        
        Based on Forger simplified model equations.
        """
        # Light response
        alpha = self._alpha_function(light)
        
        # Process B (light input)
        Bh = self.G * (1 - n) * alpha
        B = Bh * (1 - 0.4 * x) * (1 - 0.4 * xc)
        
        # Derivatives
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
    
    def _extract_dlmo_enhanced(
        self, 
        cbt_rhythm: List[Tuple[float, float]],
        target_date: date
    ) -> CircadianPhaseResult:
        """
        Extract DLMO from CBT rhythm using proper local minimum detection.
        
        Based on MATLAB implementation using islocalmin with MinProminence.
        """
        # Extract CBT values
        hours = [h for h, _ in cbt_rhythm]
        cbt_values = np.array([cbt for _, cbt in cbt_rhythm])
        
        # Find local minima with prominence threshold
        # Using negative values for find_peaks to find minima
        minima_indices, properties = find_peaks(
            -cbt_values,
            prominence=0.05,  # MinProminence from MATLAB
            distance=12  # At least 12 hours between minima
        )
        
        if len(minima_indices) == 0:
            # Fallback to simple minimum if no prominent minima found
            min_idx = np.argmin(cbt_values)
            min_hour = hours[min_idx]
            min_cbt = cbt_values[min_idx]
            confidence = 0.5  # Lower confidence
        else:
            # Use the most prominent minimum
            prominences = properties['prominences']
            best_idx = minima_indices[np.argmax(prominences)]
            min_hour = hours[best_idx]
            min_cbt = cbt_values[best_idx]
            confidence = min(1.0, prominences[np.argmax(prominences)] / 0.5)
        
        # Calculate CBT amplitude
        max_cbt = np.max(cbt_values)
        amplitude = max_cbt - min_cbt
        
        # DLMO is 7.1 hours before CBT minimum
        dlmo_hour = (min_hour - self.CBT_TO_DLMO_OFFSET) % 24
        
        # Calculate phase angle if we have sleep onset
        # (This would need actual sleep onset time)
        phase_angle = 0.0  # Placeholder
        
        return CircadianPhaseResult(
            date=target_date,
            dlmo_hour=dlmo_hour,
            cbt_min_hour=min_hour,
            cbt_amplitude=amplitude,
            phase_angle=phase_angle,
            confidence=confidence
        )
    
    def _periods_overlap(
        self,
        start1: datetime,
        end1: datetime,
        start2: datetime,
        end2: datetime
    ) -> bool:
        """Check if two time periods overlap."""
        return start1 < end2 and start2 < end1