"""
Root conftest.py for Big Mood Detector test suite.

Provides global fixtures, parallel testing configuration, and clinical data setup.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Import our application modules
from big_mood_detector.infrastructure.config.settings import Settings
from big_mood_detector.infrastructure.database.connection import DatabaseManager
from big_mood_detector.application.services.mood_detection_service import MoodDetectionService


# ============================================================================
# PARALLEL TESTING CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest for parallel execution and clinical testing."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "ml: Machine learning model tests")
    config.addinivalue_line("markers", "clinical: Clinical validation tests")
    config.addinivalue_line("markers", "slow: Slow tests (can be skipped in CI)")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "database: Database integration tests")


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
        
        # Mark slow tests
        if "clinical_benchmarks" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.clinical)


# ============================================================================
# ASYNC EVENT LOOP SETUP
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test-specific application settings."""
    return Settings(
        environment="test",
        database_url="sqlite:///:memory:",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        enable_cors=True,
        ml_model_path="tests/fixtures/models/",
    )


@pytest.fixture(scope="function")
async def database_manager(test_settings: Settings) -> AsyncGenerator[DatabaseManager, None]:
    """Database manager for tests with automatic cleanup."""
    manager = DatabaseManager(test_settings.database_url)
    await manager.initialize()
    yield manager
    await manager.close()


# ============================================================================
# CLINICAL DATA FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def sample_healthkit_data() -> dict:
    """Sample Apple HealthKit JSON data for testing."""
    return {
        "data": {
            "metrics": {
                "step_count": [
                    {"date": "2024-01-01", "value": 8500},
                    {"date": "2024-01-02", "value": 6200},
                    {"date": "2024-01-03", "value": 12000},
                ],
                "heart_rate": [
                    {"timestamp": "2024-01-01T08:00:00Z", "value": 72},
                    {"timestamp": "2024-01-01T12:00:00Z", "value": 85},
                    {"timestamp": "2024-01-01T20:00:00Z", "value": 65},
                ],
                "sleep_analysis": [
                    {
                        "start_date": "2024-01-01T23:00:00Z",
                        "end_date": "2024-01-02T07:30:00Z",
                        "value": "InBed",
                    }
                ],
            }
        }
    }


@pytest.fixture(scope="session")
def clinical_test_dataset() -> pd.DataFrame:
    """Clinical-grade test dataset with known mood outcomes."""
    dates = pd.date_range(start="2024-01-01", end="2024-01-30", freq="D")
    
    # Simulate clinical data patterns
    data = []
    for i, date in enumerate(dates):
        # Simulate bipolar patterns: manic episodes (high activity) vs depressive (low activity)
        if i < 10:  # Baseline period
            mood_state = "stable"
            activity_multiplier = 1.0
        elif i < 15:  # Manic episode
            mood_state = "manic" 
            activity_multiplier = 1.8
        elif i < 25:  # Depressive episode
            mood_state = "depressive"
            activity_multiplier = 0.4
        else:  # Recovery
            mood_state = "stable"
            activity_multiplier = 1.0
            
        data.append({
            "date": date,
            "step_count": int(8000 * activity_multiplier + np.random.normal(0, 1000)),
            "sleep_duration": 7.5 + (0.5 if mood_state == "manic" else -1.0 if mood_state == "depressive" else 0),
            "heart_rate_avg": 70 + (15 if mood_state == "manic" else -5 if mood_state == "depressive" else 0),
            "mood_state": mood_state,
            "clinical_score": 0 if mood_state == "stable" else (1 if mood_state == "manic" else -1),
        })
    
    return pd.DataFrame(data)


@pytest.fixture(scope="function") 
def mock_xgboost_model():
    """Mock XGBoost model for unit testing."""
    mock_model = Mock()
    mock_model.predict.return_value = [0.75]  # Mock prediction
    mock_model.predict_proba.return_value = [[0.25, 0.75]]  # Mock probabilities
    return mock_model


@pytest.fixture(scope="function")
def mock_tsfresh_features():
    """Mock tsfresh feature extraction output."""
    return pd.DataFrame({
        "step_count__mean": [8500.0],
        "step_count__std": [2000.0],
        "heart_rate__mean": [72.0],
        "heart_rate__variance": [120.0],
        "sleep_duration__mean": [7.5],
        "circadian_rhythm_score": [0.85],
    })


# ============================================================================
# SERVICE LAYER FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
async def mood_detection_service(
    database_manager: DatabaseManager,
    test_settings: Settings
) -> MoodDetectionService:
    """Mood detection service with test configuration."""
    return MoodDetectionService(
        database_manager=database_manager,
        settings=test_settings
    )


# ============================================================================
# API CLIENT FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
async def test_client():
    """FastAPI test client for API testing."""
    from fastapi.testclient import TestClient
    from big_mood_detector.interfaces.api.main import create_app
    
    app = create_app()
    return TestClient(app)


# ============================================================================
# PERFORMANCE & LOAD TESTING
# ============================================================================

@pytest.fixture(scope="session")
def performance_test_data():
    """Large dataset for performance testing."""
    # Generate 1000 days of data for stress testing
    dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")
    return {
        "step_count": np.random.randint(5000, 15000, len(dates)),
        "heart_rate": np.random.randint(60, 100, len(dates)),
        "sleep_duration": np.random.uniform(6.0, 9.0, len(dates)),
    }


# ============================================================================
# CLEANUP UTILITIES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically clean up test artifacts after each test."""
    yield  # Run the test
    
    # Cleanup logic
    test_files = Path("tests/temp/")
    if test_files.exists():
        import shutil
        shutil.rmtree(test_files)


# ============================================================================
# PARALLEL EXECUTION HELPERS
# ============================================================================

import numpy as np  # Add this import

def pytest_xdist_setupnodes(config, specs):
    """Configure parallel test execution."""
    # Ensure each worker gets isolated test data
    pass


@pytest.fixture(scope="session")
def worker_id(request):
    """Get the worker ID for parallel test isolation."""
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master" 