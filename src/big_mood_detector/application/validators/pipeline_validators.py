"""
Pipeline validators for PAT and XGBoost models.

These validators assess whether sufficient data is available
for each model to make predictions, following their specific requirements.
"""

from dataclasses import dataclass
from datetime import date

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord


@dataclass
class ValidationResult:
    """Result of pipeline validation."""

    is_valid: bool
    days_available: int
    consecutive_days: int
    missing_data: list[str]
    can_run: bool
    message: str


class PATValidator:
    """
    Validates data sufficiency for PAT model.

    PAT (Pretrained Actigraphy Transformer) requires exactly 7 consecutive days
    of activity data to create a 10,080-minute sequence for depression screening.
    """

    REQUIRED_CONSECUTIVE_DAYS = 7

    def validate(
        self,
        activity_records: list[ActivityRecord],
        start_date: date,
        end_date: date,
    ) -> ValidationResult:
        """
        Validate if PAT can run with the available data.

        Args:
            activity_records: Available activity records
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            ValidationResult with detailed information
        """
        if not activity_records:
            return ValidationResult(
                is_valid=False,
                days_available=0,
                consecutive_days=0,
                missing_data=["No activity data available"],
                can_run=False,
                message="PAT requires 7 consecutive days, found 0",
            )

        # Extract unique dates with activity data
        dates_with_data = {record.start_date.date() for record in activity_records}
        sorted_dates = sorted(dates_with_data)

        # Find longest consecutive sequence
        consecutive_days = self._find_max_consecutive_days(sorted_dates)

        # Check if we have sufficient consecutive days
        is_valid = consecutive_days >= self.REQUIRED_CONSECUTIVE_DAYS

        missing_data = []
        if not is_valid:
            days_needed = self.REQUIRED_CONSECUTIVE_DAYS - consecutive_days
            missing_data.append(f"Need {days_needed} more consecutive days")

        message = (
            f"PAT requires {self.REQUIRED_CONSECUTIVE_DAYS} consecutive days, "
            f"found {consecutive_days}"
        )

        return ValidationResult(
            is_valid=is_valid,
            days_available=len(dates_with_data),
            consecutive_days=consecutive_days,
            missing_data=missing_data,
            can_run=is_valid,
            message=message,
        )

    def _find_max_consecutive_days(self, sorted_dates: list[date]) -> int:
        """Find the longest sequence of consecutive days."""
        if not sorted_dates:
            return 0

        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return max_consecutive


class XGBoostValidator:
    """
    Validates data sufficiency for XGBoost model.

    XGBoost requires 30-60 days of data for optimal circadian rhythm analysis,
    but can work with sparse data (gaps are acceptable).
    """

    MINIMUM_DAYS = 30
    OPTIMAL_DAYS = 60

    def validate(
        self,
        sleep_records: list[SleepRecord],
        activity_records: list[ActivityRecord],
        start_date: date,
        end_date: date,
    ) -> ValidationResult:
        """
        Validate if XGBoost can run with the available data.

        Args:
            sleep_records: Available sleep records
            activity_records: Available activity records
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            ValidationResult with detailed information
        """
        # Collect all unique dates with any data
        dates_with_data = set()

        # Add dates from sleep records
        for sleep_record in sleep_records:
            dates_with_data.add(sleep_record.start_date.date())

        # Add dates from activity records
        for activity_record in activity_records:
            dates_with_data.add(activity_record.start_date.date())

        days_available = len(dates_with_data)

        # Check if we have sufficient days (sparse is OK)
        is_valid = days_available >= self.MINIMUM_DAYS

        missing_data = []
        if not is_valid:
            days_needed = self.MINIMUM_DAYS - days_available
            missing_data.append(f"Need {days_needed} more days")

        message = (
            f"XGBoost needs {self.MINIMUM_DAYS}+ days (any distribution), "
            f"found {days_available}"
        )

        return ValidationResult(
            is_valid=is_valid,
            days_available=days_available,
            consecutive_days=0,  # Not required for XGBoost
            missing_data=missing_data,
            can_run=is_valid,
            message=message,
        )
