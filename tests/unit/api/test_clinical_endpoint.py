"""
Test Clinical Interpretation Endpoint

Verifies the new /predictions/clinical endpoint integrates correctly.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class TestClinicalEndpoint:
    """Test the clinical interpretation endpoint."""

    @pytest.fixture
    def mock_predictor(self):
        """Mock mood predictor."""
        predictor = MagicMock()
        predictor.predict.return_value = MoodPrediction(
            depression_risk=0.75,
            hypomanic_risk=0.15,
            manic_risk=0.10,
            confidence=0.85,
        )
        return predictor

    @pytest.fixture
    def app_client(self, mock_predictor):
        """Create test client with mocked dependencies."""
        # Need to mock dependencies before importing
        from big_mood_detector.interfaces.api.dependencies import (
            get_ensemble_orchestrator,
            get_mood_predictor,
        )
        from big_mood_detector.interfaces.api.main import app

        # Override dependency
        app.dependency_overrides[get_mood_predictor] = lambda: mock_predictor
        app.dependency_overrides[get_ensemble_orchestrator] = lambda: None

        yield TestClient(app)

        # Clean up
        app.dependency_overrides.clear()

    def test_clinical_endpoint_success(self, app_client):
        """Test successful clinical interpretation."""
        # Arrange
        payload = {
            "sleep_duration": 6.5,
            "sleep_efficiency": 0.78,
            "sleep_timing_variance": 1.2,
            "daily_steps": 8500,
            "activity_variance": 0.3,
            "sedentary_hours": 8.0,
        }

        # Act
        response = app_client.post("/api/v1/predictions/clinical", json=payload)

        # Assert
        if response.status_code != status.HTTP_200_OK:
            print(f"Response: {response.json()}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify structure
        assert "primary_diagnosis" in data
        assert "risk_level" in data
        assert "confidence" in data
        assert "clinical_notes" in data
        assert "recommendations" in data
        assert "secondary_risks" in data
        assert "monitoring_frequency" in data
        assert "ml_predictions" in data

        # Verify values
        assert data["risk_level"] in ["low", "moderate", "high", "critical"]
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["clinical_notes"], list)
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) > 0

        # Verify ML predictions match our mock
        assert data["ml_predictions"]["depression"] == 0.75
        assert data["ml_predictions"]["mania"] == 0.10
        assert data["ml_predictions"]["hypomania"] == 0.15

        # Verify DSM-5 compliance
        assert data["dsm5_compliant"] is True

    def test_clinical_endpoint_with_optional_features(self, app_client):
        """Test endpoint with optional features included."""
        # Arrange
        payload = {
            "sleep_duration": 7.0,
            "sleep_efficiency": 0.85,
            "sleep_timing_variance": 0.5,
            "daily_steps": 10000,
            "activity_variance": 0.2,
            "sedentary_hours": 6.0,
            "interdaily_stability": 0.8,
            "intradaily_variability": 0.3,
            "relative_amplitude": 0.7,
            "resting_hr": 65.0,
            "hrv_rmssd": 45.0,
        }

        # Act
        response = app_client.post("/api/v1/predictions/clinical", json=payload)

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "primary_diagnosis" in data
        assert "monitoring_frequency" in data

    def test_clinical_endpoint_validation(self, app_client):
        """Test input validation."""
        # Arrange - invalid sleep duration
        payload = {
            "sleep_duration": 25.0,  # Invalid: > 24 hours
            "sleep_efficiency": 0.85,
            "sleep_timing_variance": 0.5,
            "daily_steps": 10000,
            "activity_variance": 0.2,
            "sedentary_hours": 6.0,
        }

        # Act
        response = app_client.post("/api/v1/predictions/clinical", json=payload)

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
