"""
Parser utilities for handling common parsing tasks.
Ensures consistency across JSON and XML parsers.
"""

from datetime import datetime
from typing import Optional


def parse_apple_health_date(date_string: str) -> datetime:
    """
    Parse Apple Health date strings with timezone handling.
    
    Handles formats:
    - "2025-01-14 23:29:06 -0500" (with timezone)
    - "2025-01-14 23:29:06" (without timezone)
    
    Returns timezone-naive datetime for consistency.
    """
    # Extract just the datetime part (first 19 chars)
    datetime_part = date_string[:19]
    
    # Parse as naive datetime
    dt = datetime.strptime(datetime_part, "%Y-%m-%d %H:%M:%S")
    
    # If timezone info is present, we could parse it but for now
    # we'll keep everything naive for consistency
    # This avoids the comparison issues between naive and aware datetimes
    
    return dt


def parse_date_only(date_string: str) -> datetime:
    """
    Parse date-only strings.
    
    Handles formats:
    - "2025-01-14"
    - "2025-01-14 00:00:00"
    """
    if len(date_string) >= 10:
        return datetime.strptime(date_string[:10], "%Y-%m-%d")
    raise ValueError(f"Invalid date string: {date_string}")


def safe_float(value: any, default: float = 0.0) -> float:
    """Safely convert value to float with default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: any, default: int = 0) -> int:
    """Safely convert value to int with default."""
    try:
        return int(float(value))  # Handle "123.0" -> 123
    except (TypeError, ValueError):
        return default