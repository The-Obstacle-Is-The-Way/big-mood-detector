"""
Test input validation for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from big_mood_detector.interfaces.api.main import app

client = TestClient(app)


def test_hrv_rmssd_validation():
    """Test that HRV RMSSD values outside valid range return 422."""
    # Valid request
    valid_payload = {
        "sleep_duration": 8.0,
        "sleep_efficiency": 0.85,
        "sleep_timing_variance": 1.5,
        "daily_steps": 10000,
        "activity_variance": 2.0,
        "sedentary_hours": 6.0,
        "hrv_rmssd": 50.0,  # Valid value
    }
    
    response = client.post("/api/v1/predictions/predict", json=valid_payload)
    assert response.status_code == 200
    
    # Test HRV RMSSD too high
    invalid_payload = valid_payload.copy()
    invalid_payload["hrv_rmssd"] = 350.0  # Over 300 limit
    
    response = client.post("/api/v1/predictions/predict", json=invalid_payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert "hrv_rmssd" in error_detail["loc"]
    assert "less than or equal to 300" in error_detail["msg"]
    
    # Test HRV RMSSD negative
    invalid_payload["hrv_rmssd"] = -10.0
    
    response = client.post("/api/v1/predictions/predict", json=invalid_payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert "hrv_rmssd" in error_detail["loc"]
    assert "greater than or equal to 0" in error_detail["msg"]


def test_sleep_duration_validation():
    """Test sleep duration validation (0-24 hours)."""
    valid_payload = {
        "sleep_duration": 8.0,
        "sleep_efficiency": 0.85,
        "sleep_timing_variance": 1.5,
        "daily_steps": 10000,
        "activity_variance": 2.0,
        "sedentary_hours": 6.0,
    }
    
    # Test sleep duration too high
    invalid_payload = valid_payload.copy()
    invalid_payload["sleep_duration"] = 25.0  # Over 24 hour limit
    
    response = client.post("/api/v1/predictions/predict", json=invalid_payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert "sleep_duration" in error_detail["loc"]
    assert "less than or equal to 24" in error_detail["msg"]


def test_heart_rate_validation():
    """Test resting heart rate validation (30-200 bpm)."""
    valid_payload = {
        "sleep_duration": 8.0,
        "sleep_efficiency": 0.85,
        "sleep_timing_variance": 1.5,
        "daily_steps": 10000,
        "activity_variance": 2.0,
        "sedentary_hours": 6.0,
        "resting_hr": 65.0,  # Valid value
    }
    
    response = client.post("/api/v1/predictions/predict", json=valid_payload)
    assert response.status_code == 200
    
    # Test heart rate too low
    invalid_payload = valid_payload.copy()
    invalid_payload["resting_hr"] = 25.0  # Below 30 bpm limit
    
    response = client.post("/api/v1/predictions/predict", json=invalid_payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert "resting_hr" in error_detail["loc"]
    assert "greater than or equal to 30" in error_detail["msg"]