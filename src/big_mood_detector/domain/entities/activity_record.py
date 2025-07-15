"""
Activity Record Entity

Domain entity representing physical activity data with clinical significance.
Following Domain-Driven Design and Clean Architecture principles.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ActivityType(Enum):
    """Types of physical activity from Apple HealthKit."""

    STEP_COUNT = "HKQuantityTypeIdentifierStepCount"
    DISTANCE_WALKING = "HKQuantityTypeIdentifierDistanceWalkingRunning"
    FLIGHTS_CLIMBED = "HKQuantityTypeIdentifierFlightsClimbed"
    ACTIVE_ENERGY = "HKQuantityTypeIdentifierActiveEnergyBurned"
    BASAL_ENERGY = "HKQuantityTypeIdentifierBasalEnergyBurned"
    EXERCISE_TIME = "HKQuantityTypeIdentifierAppleExerciseTime"
    STAND_TIME = "HKQuantityTypeIdentifierAppleStandTime"

    @classmethod
    def from_healthkit_identifier(cls, identifier: str) -> "ActivityType":
        """Convert HealthKit identifier to ActivityType enum."""
        for activity_type in cls:
            if activity_type.value == identifier:
                return activity_type
        raise ValueError(f"Unknown activity type: {identifier}")


@dataclass(frozen=True)
class ActivityRecord:
    """
    Immutable activity record entity.
    
    Represents a single activity measurement from health tracking.
    """

    source_name: str
    start_date: datetime
    end_date: datetime
    activity_type: ActivityType
    value: float
    unit: str

    def __post_init__(self) -> None:
        """Validate business rules."""
        if self.end_date < self.start_date:
            raise ValueError("End date must be after or equal to start date")

        if not self.source_name:
            raise ValueError("Source name is required")

        if self.value < 0:
            raise ValueError("Activity value cannot be negative")

        if not self.unit:
            raise ValueError("Unit is required")

    @property
    def duration_hours(self) -> float:
        """Calculate activity duration in hours."""
        delta = self.end_date - self.start_date
        return delta.total_seconds() / 3600

    @property
    def is_instantaneous(self) -> bool:
        """Check if this is an instantaneous measurement."""
        return self.start_date == self.end_date

    @property
    def intensity_per_hour(self) -> float:
        """Calculate intensity (value per hour) for rate-based metrics."""
        if self.is_instantaneous:
            return self.value

        duration = self.duration_hours
        return self.value / duration if duration > 0 else 0.0

    @property
    def is_high_activity(self) -> bool:
        """
        Determine if this represents high activity level.
        
        Clinical significance for mood episodes:
        - Manic: Very high step counts (>15000/day)
        - Depressive: Very low step counts (<2000/day)
        """
        if self.activity_type == ActivityType.STEP_COUNT:
            # Normalize to daily rate
            daily_rate = self.intensity_per_hour * 24
            return daily_rate > 15000
        elif self.activity_type == ActivityType.ACTIVE_ENERGY:
            # High energy burn rate (>500 cal/day active)
            daily_rate = self.intensity_per_hour * 24
            return daily_rate > 500
        return False

    @property
    def is_low_activity(self) -> bool:
        """Determine if this represents low activity level."""
        if self.activity_type == ActivityType.STEP_COUNT:
            daily_rate = self.intensity_per_hour * 24
            return daily_rate < 2000
        elif self.activity_type == ActivityType.ACTIVE_ENERGY:
            daily_rate = self.intensity_per_hour * 24
            return daily_rate < 100
        return False

    def is_same_type(self, other: "ActivityRecord") -> bool:
        """Check if two records are of the same activity type."""
        return self.activity_type == other.activity_type

    def can_aggregate_with(self, other: "ActivityRecord") -> bool:
        """
        Check if this record can be aggregated with another.
        
        Records can be aggregated if they:
        - Are the same type
        - Have the same unit
        - Are from the same source
        """
        return (
            self.is_same_type(other)
            and self.unit == other.unit
            and self.source_name == other.source_name
        )
