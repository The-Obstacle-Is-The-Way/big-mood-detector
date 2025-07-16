"""
DLMO (Dim Light Melatonin Onset) Calculator Service

Implements the circadian pacemaker model to estimate DLMO from sleep/wake patterns.
Based on the St. Hilaire/Forger model as used in the Seoul study.

Key steps:
1. Convert sleep/wake to light profile (250 lux awake, 0 lux asleep)
2. Run circadian pacemaker model to simulate Core Body Temperature (CBT)
3. Find CBT minimum and subtract 7 hours to estimate DLMO

Design Principles:
- Pure domain logic (no external dependencies)
- Immutable value objects
- Single Responsibility: Only DLMO calculation
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional
import math

from big_mood_detector.domain.entities.sleep_record import SleepRecord


@dataclass(frozen=True)
class LightProfile:
    """
    Immutable light exposure profile for circadian modeling.
    
    Contains hourly light values (lux) for a 24-hour period.
    """
    date: date
    hourly_lux: List[float]  # 24 values, one per hour
    
    @property
    def average_light_exposure(self) -> float:
        """Average light exposure across the day."""
        return sum(self.hourly_lux) / len(self.hourly_lux)


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
    
    @property
    def dlmo_time(self) -> str:
        """DLMO as HH:MM format."""
        hours = int(self.dlmo_hour)
        minutes = int((self.dlmo_hour - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"


class DLMOCalculator:
    """
    Calculates DLMO using mathematical circadian pacemaker model.
    
    Implements the simplified Forger model equations from the
    St. Hilaire framework, as validated for bipolar disorder detection.
    """
    
    # Model constants
    AWAKE_LUX = 250.0  # Light level when awake
    ASLEEP_LUX = 0.0   # Light level when asleep
    DELTA_T = 1/60.0   # Time step (1 minute in hours)
    CBT_TO_DLMO_OFFSET = 7.1  # Hours from CBT min to DLMO (from MATLAB reference)
    
    # Circadian model parameters (from MATLAB implementation)
    TAU = 24.2  # Intrinsic period
    G = 33.75   # Coupling strength (changed from 19.875)
    K = 0.55    # Stiffness
    MU = 0.23   # Amplitude recovery rate
    B = 0.0075  # Photoreceptor decay rate (changed from 0.013)
    
    # Light response parameters
    I0 = 9500.0  # Half-saturation constant
    P = 0.6      # Hill coefficient
    A0 = 0.05    # Maximum drive
    
    def calculate_dlmo(
        self, 
        sleep_records: List[SleepRecord],
        target_date: date,
        days_to_model: int = 7
    ) -> Optional[CircadianPhaseResult]:
        """
        Calculate DLMO for target date using sleep history.
        
        Args:
            sleep_records: Historical sleep records
            target_date: Date to calculate DLMO for
            days_to_model: Days of history to use (default 7)
            
        Returns:
            DLMO calculation result or None if insufficient data
        """
        # Create light profiles from sleep data
        light_profiles = self._create_light_profiles(
            sleep_records, 
            target_date, 
            days_to_model
        )
        
        if len(light_profiles) < 3:
            return None  # Need at least 3 days
        
        # Run circadian model
        cbt_rhythm = self._run_circadian_model(light_profiles)
        
        # Extract DLMO from CBT rhythm
        return self._extract_dlmo(cbt_rhythm, target_date)
    
    def _create_light_profiles(
        self,
        sleep_records: List[SleepRecord],
        target_date: date,
        days: int
    ) -> List[LightProfile]:
        """
        Convert sleep/wake patterns to light exposure profiles.
        
        Assumes 250 lux when awake, 0 lux when asleep.
        """
        profiles = []
        start_date = target_date - timedelta(days=days-1)
        
        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)
            
            # Initialize with awake (250 lux)
            hourly_lux = [self.AWAKE_LUX] * 24
            
            # Set sleep periods to 0 lux
            day_sleep = [
                s for s in sleep_records 
                if s.start_date.date() <= current_date <= s.end_date.date()
            ]
            
            for sleep in day_sleep:
                # Mark sleep hours as dark
                for hour in range(24):
                    hour_start = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour)
                    hour_end = hour_start + timedelta(hours=1)
                    
                    # Check overlap with sleep period
                    if self._periods_overlap(
                        sleep.start_date, sleep.end_date,
                        hour_start, hour_end
                    ):
                        hourly_lux[hour] = self.ASLEEP_LUX
            
            profiles.append(LightProfile(
                date=current_date,
                hourly_lux=hourly_lux
            ))
        
        return profiles
    
    def _run_circadian_model(
        self, 
        light_profiles: List[LightProfile]
    ) -> List[Tuple[float, float]]:
        """
        Run Forger circadian pacemaker model.
        
        Returns hourly CBT values for the last day.
        """
        # Initial conditions
        x = 0.0   # Circadian phase
        xc = 0.0  # Circadian amplitude  
        n = 0.5   # Photoreceptor state
        
        # Run model for all days to allow entrainment
        for profile_idx, profile in enumerate(light_profiles):
            # Simulate each hour
            for hour in range(24):
                light = profile.hourly_lux[hour]
                
                # Run model for this hour (60 steps of 1 minute)
                for _ in range(60):
                    # Calculate derivatives
                    dx, dxc, dn = self._circadian_derivatives(x, xc, n, light)
                    
                    # Update state (Euler integration)
                    x += dx * self.DELTA_T
                    xc += dxc * self.DELTA_T
                    n += dn * self.DELTA_T
        
        # Generate CBT rhythm for last day by continuing simulation
        cbt_rhythm = []
        last_profile = light_profiles[-1]
        
        # Store hourly CBT values while simulating the last day
        for hour in range(24):
            light = last_profile.hourly_lux[hour]
            
            # Store CBT at the start of each hour
            # CBT is proportional to circadian state x
            cbt = -xc * math.cos(x)
            cbt_rhythm.append((hour, cbt))
            
            # Continue simulation for this hour
            for _ in range(60):
                dx, dxc, dn = self._circadian_derivatives(x, xc, n, light)
                x += dx * self.DELTA_T
                xc += dxc * self.DELTA_T
                n += dn * self.DELTA_T
        
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
    
    def _extract_dlmo(
        self, 
        cbt_rhythm: List[Tuple[float, float]],
        target_date: date
    ) -> CircadianPhaseResult:
        """
        Extract DLMO from CBT rhythm.
        
        DLMO = CBT minimum - 7 hours (Seoul paper methodology)
        """
        # Find CBT minimum
        min_hour = 0
        min_cbt = float('inf')
        
        for hour, cbt in cbt_rhythm:
            if cbt < min_cbt:
                min_cbt = cbt
                min_hour = hour
        
        # Calculate CBT amplitude (max - min)
        max_cbt = max(cbt for _, cbt in cbt_rhythm)
        amplitude = max_cbt - min_cbt
        
        # DLMO is 7.1 hours before CBT minimum (validated offset)
        dlmo_hour = (min_hour - self.CBT_TO_DLMO_OFFSET) % 24
        
        # Calculate phase angle (DLMO to sleep onset)
        # This would need actual sleep onset time
        phase_angle = 0.0  # Placeholder
        
        return CircadianPhaseResult(
            date=target_date,
            dlmo_hour=dlmo_hour,
            cbt_min_hour=min_hour,
            cbt_amplitude=amplitude,
            phase_angle=phase_angle
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