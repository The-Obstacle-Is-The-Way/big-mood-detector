"""
Sleep Record Entity

Domain entity representing a sleep period with clinical significance.
Following Domain-Driven Design principles.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SleepState(Enum):
    """Clinical sleep states from Apple HealthKit."""

    IN_BED = "HKCategoryValueSleepAnalysisInBed"
    ASLEEP = "HKCategoryValueSleepAnalysisAsleep"
    AWAKE = "HKCategoryValueSleepAnalysisAwake"
    REM = "HKCategoryValueSleepAnalysisREM"
    CORE = "HKCategoryValueSleepAnalysisCore"
    DEEP = "HKCategoryValueSleepAnalysisDeep"
    # Newer Apple Health values
    ASLEEP_CORE = "HKCategoryValueSleepAnalysisAsleepCore"
    ASLEEP_DEEP = "HKCategoryValueSleepAnalysisAsleepDeep"
    ASLEEP_REM = "HKCategoryValueSleepAnalysisAsleepREM"
    ASLEEP_UNSPECIFIED = "HKCategoryValueSleepAnalysisAsleepUnspecified"

    @classmethod
    def from_healthkit_value(cls, value: str) -> "SleepState":
        """Convert HealthKit string to SleepState enum."""
        for state in cls:
            if state.value == value:
                return state
        raise ValueError(f"Unknown sleep state: {value}")


@dataclass(frozen=True)
class SleepRecord:
    """
    Immutable sleep record entity.

    Represents a single continuous period of sleep or rest.
    """

    source_name: str
    start_date: datetime
    end_date: datetime
    state: SleepState

    def __post_init__(self) -> None:
        """Validate business rules."""
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")

        if not self.source_name:
            raise ValueError("Source name is required")

    @property
    def duration_hours(self) -> float:
        """Calculate sleep duration in hours."""
        delta = self.end_date - self.start_date
        return delta.total_seconds() / 3600

    @property
    def is_actual_sleep(self) -> bool:
        """Check if this represents actual sleep (not just in bed)."""
        return self.state in {
            SleepState.ASLEEP,
            SleepState.REM,
            SleepState.CORE,
            SleepState.DEEP,
            SleepState.ASLEEP_CORE,
            SleepState.ASLEEP_DEEP,
            SleepState.ASLEEP_REM,
            SleepState.ASLEEP_UNSPECIFIED,
        }

    @property
    def sleep_quality_indicator(self) -> str | None:
        """Get sleep quality based on state."""
        quality_map = {
            SleepState.DEEP: "restorative",
            SleepState.ASLEEP_DEEP: "restorative",
            SleepState.REM: "rem",
            SleepState.ASLEEP_REM: "rem",
            SleepState.CORE: "light",
            SleepState.ASLEEP_CORE: "light",
            SleepState.ASLEEP: "general",
            SleepState.ASLEEP_UNSPECIFIED: "general",
            SleepState.IN_BED: "resting",
            SleepState.AWAKE: "disrupted",
        }
        return quality_map.get(self.state)
