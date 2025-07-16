"""
Mathematical helper functions for domain calculations.

Pure functions with no side effects.
"""


def clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp a value between lower and upper bounds.

    Args:
        value: The value to clamp
        lo: Lower bound (inclusive)
        hi: Upper bound (inclusive)

    Returns:
        Value clamped to [lo, hi] range

    Examples:
        >>> clamp(150.0, 0.0, 100.0)
        100.0
        >>> clamp(-10.0, 0.0, 100.0)
        0.0
        >>> clamp(50.0, 0.0, 100.0)
        50.0
    """
    return max(lo, min(hi, value))


def safe_std(values: list[float], default: float = 0.0) -> float:
    """
    Calculate standard deviation safely, handling edge cases.

    Args:
        values: List of numeric values
        default: Value to return for edge cases

    Returns:
        Standard deviation or default if < 2 values
    """
    if len(values) < 2:
        return default

    import numpy as np
    return float(np.std(values))


def safe_var(values: list[float], default: float = 0.0) -> float:
    """
    Calculate variance safely, handling edge cases.

    Args:
        values: List of numeric values
        default: Value to return for edge cases

    Returns:
        Variance or default if < 2 values
    """
    if len(values) < 2:
        return default

    import numpy as np
    return float(np.var(values))
