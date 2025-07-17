"""
Minimal conftest.py for Big Mood Detector test suite.

Provides basic fixtures for TDD workflow.
"""

import os

# Set thread limits BEFORE any imports to prevent segfaults
# This must happen before NumPy, XGBoost, or PyTorch are imported
for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "XGBOOST_NUM_THREADS",
):
    os.environ.setdefault(var, "1")

# Disable Rich formatting in tests for speed
os.environ["RICH_DISABLE"] = "True"

import pytest  # noqa: E402


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "ml: Machine learning model tests")
    config.addinivalue_line("markers", "clinical: Clinical validation tests")
    config.addinivalue_line("markers", "slow: Slow tests (can be skipped in CI)")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on directory structure
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "ml/" in str(item.fspath):
            item.add_marker(pytest.mark.ml)


@pytest.fixture(scope="session")
def sample_sleep_xml():
    """Sample Apple HealthKit XML data for sleep testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis"
            sourceName="Apple Watch"
            startDate="2024-01-01 23:30:00 -0800"
            endDate="2024-01-02 07:30:00 -0800"
            value="HKCategoryValueSleepAnalysisAsleep"/>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis"
            sourceName="Apple Watch"
            startDate="2024-01-02 23:00:00 -0800"
            endDate="2024-01-03 06:45:00 -0800"
            value="HKCategoryValueSleepAnalysisInBed"/>
</HealthData>"""


@pytest.fixture(scope="session")
def expected_sleep_data():
    """Expected parsed sleep data structure."""
    return [
        {
            "sourceName": "Apple Watch",
            "startDate": "2024-01-01 23:30:00 -0800",
            "endDate": "2024-01-02 07:30:00 -0800",
            "value": "HKCategoryValueSleepAnalysisAsleep",
        },
        {
            "sourceName": "Apple Watch",
            "startDate": "2024-01-02 23:00:00 -0800",
            "endDate": "2024-01-03 06:45:00 -0800",
            "value": "HKCategoryValueSleepAnalysisInBed",
        },
    ]
