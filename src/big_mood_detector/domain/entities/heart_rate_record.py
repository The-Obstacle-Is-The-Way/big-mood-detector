"""
Heart Rate Record Entity

Domain entity representing heart rate and HRV data with clinical significance.
Following Domain-Driven Design and Clean Architecture principles.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class HeartMetricType(Enum):
    """Types of heart metrics from Apple HealthKit."""

    HEART_RATE = "HKQuantityTypeIdentifierHeartRate"
    HRV_SDNN = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
    RESTING_HEART_RATE = "HKQuantityTypeIdentifierRestingHeartRate"
    WALKING_HEART_RATE_AVG = "HKQuantityTypeIdentifierWalkingHeartRateAverage"
    HEART_RATE_RECOVERY = "HKQuantityTypeIdentifierHeartRateRecoveryOneMinute"

    @classmethod
    def from_healthkit_identifier(cls, identifier: str) -> "HeartMetricType":
        """Convert HealthKit identifier to HeartMetricType enum."""
        for metric_type in cls:
            if metric_type.value == identifier:
                return metric_type
        raise ValueError(f"Unknown heart metric type: {identifier}")

    def is_hrv_metric(self) -> bool:
        """Check if this is an HRV-related metric."""
        return self == HeartMetricType.HRV_SDNN


class MotionContext(Enum):
    """Motion context for heart rate measurements."""

    SEDENTARY = "sedentary"
    ACTIVE = "active"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | None) -> "MotionContext":
        """Create MotionContext from string value."""
        if not value:
            return cls.UNKNOWN

        for context in cls:
            if context.value == value.lower():
                return context
        return cls.UNKNOWN


@dataclass(frozen=True)
class HeartRateRecord:
    """
    Immutable heart rate record entity.

    Represents a single heart rate or HRV measurement.
    """

    source_name: str
    timestamp: datetime
    metric_type: HeartMetricType
    value: float
    unit: str
    motion_context: MotionContext = MotionContext.UNKNOWN

    def __post_init__(self) -> None:
        """Validate business rules."""
        if not self.source_name:
            raise ValueError("Source name is required")

        if self.value < 0:
            raise ValueError("Heart metric value cannot be negative")

        if not self.unit:
            raise ValueError("Unit is required")

    @property
    def is_instantaneous(self) -> bool:
        """Heart rate measurements are always instantaneous."""
        return True

    @property
    def is_high_heart_rate(self) -> bool:
        """
        Detect abnormally high heart rate.

        Clinical thresholds:
        - Resting: >100 bpm (tachycardia)
        - Active: Not flagged as high
        """
        if self.metric_type != HeartMetricType.HEART_RATE:
            return False

        # Don't flag high HR during activity
        if self.motion_context == MotionContext.ACTIVE:
            return False

        return self.value > 100

    @property
    def is_low_heart_rate(self) -> bool:
        """
        Detect abnormally low heart rate.

        Clinical threshold: <50 bpm (bradycardia)
        """
        if self.metric_type not in [
            HeartMetricType.HEART_RATE,
            HeartMetricType.RESTING_HEART_RATE,
        ]:
            return False

        return self.value < 50

    @property
    def is_low_hrv(self) -> bool:
        """
        Detect low heart rate variability.

        Clinical significance:
        - SDNN <20ms indicates poor autonomic function
        - Associated with stress, poor recovery, mood disorders
        """
        if self.metric_type != HeartMetricType.HRV_SDNN:
            return False

        return self.value < 20

    @property
    def is_clinically_significant(self) -> bool:
        """
        Determine if this measurement warrants clinical attention.

        Significant patterns:
        - High resting heart rate (manic/anxiety indicator)
        - Low heart rate (depression/medication effects)
        - Low HRV (autonomic dysfunction, stress)
        """
        return self.is_high_heart_rate or self.is_low_heart_rate or self.is_low_hrv

    def is_same_type(self, other: "HeartRateRecord") -> bool:
        """Check if two records are of the same metric type."""
        return self.metric_type == other.metric_type

