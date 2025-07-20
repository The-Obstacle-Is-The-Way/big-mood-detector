"""Type definitions for repository infrastructure.

This module provides proper type hints for ORM models and other
infrastructure types to avoid type: ignore comments.
"""

from datetime import date, datetime
from typing import Protocol


class BaselineAggregateRecordProtocol(Protocol):
    """Protocol for BaselineAggregateRecord to provide proper type hints."""

    user_id: str
    feature_name: str
    window: str
    as_of: date
    mean: float
    std: float
    n: int
    created_at: datetime


class BaselineRawRecordProtocol(Protocol):
    """Protocol for BaselineRawRecord to provide proper type hints."""

    user_id: str
    metric: str
    ts: datetime
    effective_ts: datetime
    value: float
    window_days: int
    source: str
