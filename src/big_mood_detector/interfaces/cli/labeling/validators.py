"""
Clinical Validators for Labeling

Validates labels against DSM-5 criteria and detects conflicts.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Tuple


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    warning: Optional[str] = None
    suggestion: Optional[str] = None


class ClinicalValidator:
    """Validate labels against DSM-5 criteria."""
    
    DSM5_MIN_DURATION = {
        "depressive": 14,  # Major depressive episode
        "manic": 7,        # Manic episode  
        "hypomanic": 4,    # Hypomanic episode
        "mixed": 7,        # Mixed features
    }
    
    def validate_episode_duration(
        self,
        episode_type: str,
        start_date: date,
        end_date: date,
    ) -> ValidationResult:
        """Check if episode meets minimum duration."""
        if not episode_type or episode_type == "baseline":
            return ValidationResult(valid=True)
            
        duration = (end_date - start_date).days + 1
        min_duration = self.DSM5_MIN_DURATION.get(episode_type, 1)
        
        if duration < min_duration:
            return ValidationResult(
                valid=False,
                warning=f"{episode_type.title()} episodes require ≥{min_duration} days (DSM-5). "
                       f"You entered {duration} days.",
                suggestion="Consider if this is part of a longer episode or use a different mood type."
            )
        return ValidationResult(valid=True)


def parse_date_range(date_range: str) -> Tuple[date, date]:
    """Parse date range string YYYY-MM-DD:YYYY-MM-DD."""
    parts = date_range.split(":")
    if len(parts) != 2:
        raise ValueError("Date range must be in format YYYY-MM-DD:YYYY-MM-DD")
    
    start = datetime.strptime(parts[0].strip(), "%Y-%m-%d").date()
    end = datetime.strptime(parts[1].strip(), "%Y-%m-%d").date()
    
    if end < start:
        raise ValueError("End date cannot be before start date")
    
    return start, end