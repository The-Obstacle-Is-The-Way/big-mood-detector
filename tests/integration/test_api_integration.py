"""
Integration tests for FastAPI endpoints.

Tests the full API surface with TestClient to ensure proper wiring.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from big_mood_detector.interfaces.api.main import app


class TestPredictionsAPI:
    """Test predictions endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_features(self) -> dict[str, Any]:
        """Create sample feature payload."""
        return {
            "sleep_duration": 7.5,
            "sleep_efficiency": 0.85,
            "sleep_timing_variance": 1.2,
            "daily_steps": 8000,
            "activity_variance": 2500.0,
            "sedentary_hours": 14.5,
            "interdaily_stability": 0.85,
            "intradaily_variability": 0.65,
            "relative_amplitude": 0.78,
            "resting_hr": 65.0,
            "hrv_rmssd": 45.0,
        }

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_predict_xgboost_success(
        self, client: TestClient, sample_features: dict
    ) -> None:
        """Test successful XGBoost prediction."""
        response = client.post("/api/v1/predictions/predict", json=sample_features)
        assert response.status_code == 200

        data = response.json()
        assert "depression_risk" in data
        assert "hypomanic_risk" in data
        assert "manic_risk" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "interpretation" in data

        # Validate prediction values
        assert data["risk_level"] in ["low", "moderate", "high", "critical"]
        assert 0 <= data["depression_risk"] <= 1
        assert 0 <= data["hypomanic_risk"] <= 1
        assert 0 <= data["manic_risk"] <= 1
        assert 0 <= data["confidence"] <= 1

    def test_predict_xgboost_missing_features(self, client: TestClient) -> None:
        """Test XGBoost prediction with missing features."""
        incomplete_features = {
            "sleep_duration": 7.5,
            "sleep_efficiency": 0.85,
            # Missing required features: sleep_timing_variance, daily_steps, activity_variance, sedentary_hours
        }
        response = client.post("/api/v1/predictions/predict", json=incomplete_features)
        assert response.status_code == 422

    def test_predict_xgboost_invalid_values(
        self, client: TestClient, sample_features: dict
    ) -> None:
        """Test XGBoost prediction with invalid feature values."""
        invalid_features = sample_features.copy()
        invalid_features["sleep_efficiency"] = 1.5  # Invalid: > 1.0

        response = client.post("/api/v1/predictions/predict", json=invalid_features)
        assert response.status_code == 422


class TestLabelsAPI:
    """Test labels/episodes endpoints."""

    @pytest.fixture(autouse=True)
    def cleanup_database(self) -> Generator[None, None, None]:
        """Clean up test database before and after each test."""
        import os
        db_path = "labels.db"
        # Clean before test
        if os.path.exists(db_path):
            os.remove(db_path)
        yield
        # Clean after test
        if os.path.exists(db_path):
            os.remove(db_path)

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_episode(self) -> dict[str, Any]:
        """Create sample episode payload."""
        now = datetime.utcnow()
        return {
            "start_date": (now - timedelta(days=7)).date().isoformat(),
            "end_date": (now - timedelta(days=3)).date().isoformat(),
            "episode_type": "depressive",
            "severity": 6,  # Integer 1-10
            "notes": "Test episode for integration testing",
            "rater_id": "test_user",
        }

    def test_create_episode_success(
        self, client: TestClient, sample_episode: dict
    ) -> None:
        """Test successful episode creation."""
        response = client.post("/api/v1/labels/episodes", json=sample_episode)
        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert data["start_date"] == sample_episode["start_date"]
        assert data["episode_type"] == sample_episode["episode_type"]
        assert data["severity"] == sample_episode["severity"]

    def test_list_episodes_empty(self, client: TestClient) -> None:
        """Test listing episodes when none exist."""
        response = client.get("/api/v1/labels/episodes")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_and_list_episodes(
        self, client: TestClient, sample_episode: dict
    ) -> None:
        """Test creating and then listing episodes."""
        # Create episode
        create_response = client.post("/api/v1/labels/episodes", json=sample_episode)
        assert create_response.status_code == 201
        created_id = create_response.json()["id"]

        # List episodes
        list_response = client.get("/api/v1/labels/episodes")
        assert list_response.status_code == 200
        episodes = list_response.json()
        assert len(episodes) >= 1
        assert any(ep["id"] == created_id for ep in episodes)

    def test_get_nonexistent_episode_returns_empty(self, client: TestClient) -> None:
        """Test that listing episodes returns empty when none exist."""
        response = client.get("/api/v1/labels/episodes")
        assert response.status_code == 200
        assert response.json() == []

    def test_delete_episode(self, client: TestClient, sample_episode: dict) -> None:
        """Test deleting an episode."""
        # Create episode
        create_response = client.post("/api/v1/labels/episodes", json=sample_episode)
        assert create_response.status_code == 201
        created_id = create_response.json()["id"]

        # Delete episode
        delete_response = client.delete(f"/api/v1/labels/episodes/{created_id}")
        assert delete_response.status_code == 204

        # Verify it's gone by listing all episodes
        list_response = client.get("/api/v1/labels/episodes")
        assert list_response.status_code == 200
        assert list_response.json() == []  # Should be empty after deletion

    def test_invalid_episode_type(
        self, client: TestClient, sample_episode: dict
    ) -> None:
        """Test creating episode with invalid type."""
        invalid_episode = sample_episode.copy()
        invalid_episode["episode_type"] = "invalid_type"

        response = client.post("/api/v1/labels/episodes", json=invalid_episode)
        assert response.status_code == 422

    def test_invalid_date_range(self, client: TestClient, sample_episode: dict) -> None:
        """Test creating episode with end before start."""
        invalid_episode = sample_episode.copy()
        invalid_episode["start_date"], invalid_episode["end_date"] = (
            invalid_episode["end_date"],
            invalid_episode["start_date"],
        )

        response = client.post("/api/v1/labels/episodes", json=invalid_episode)
        assert response.status_code == 422


class TestFeatureExtraction:
    """Test feature extraction endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_health_data(self, tmp_path: Path) -> Path:
        """Create sample Apple Health XML export file with realistic data."""
        # Create XML with enough data for aggregation pipeline (7 days)
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!ELEMENT HealthData (Record+)>
<!ATTLIST HealthData locale CDATA #IMPLIED>
<!ELEMENT Record EMPTY>
<!ATTLIST Record
  type CDATA #REQUIRED
  sourceName CDATA #REQUIRED
  startDate CDATA #REQUIRED
  endDate CDATA #REQUIRED
  value CDATA #IMPLIED>
]>
<HealthData locale="en_US">\n"""
        
        # Add 7 days of sleep data
        for day in range(7):
            date_str = f"2024-01-{day+1:02d}"
            xml_content += f"""  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" 
          startDate="{date_str} 23:00:00 -0500" endDate="2024-01-{day+2:02d} 07:00:00 -0500" 
          value="HKCategoryValueSleepAnalysisAsleepCore"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" 
          startDate="2024-01-{day+2:02d} 03:00:00 -0500" endDate="2024-01-{day+2:02d} 03:30:00 -0500" 
          value="HKCategoryValueSleepAnalysisAwake"/>
"""
        
        # Add heart rate data
        for day in range(7):
            for hour in [6, 12, 18]:
                xml_content += f"""  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" 
          startDate="2024-01-{day+1:02d} {hour:02d}:00:00 -0500" endDate="2024-01-{day+1:02d} {hour:02d}:00:00 -0500" 
          value="{65 + hour}"/>
"""
        
        # Add activity data
        for day in range(7):
            xml_content += f"""  <Record type="HKQuantityTypeIdentifierActiveEnergyBurned" sourceName="Apple Watch" 
          startDate="2024-01-{day+1:02d} 12:00:00 -0500" endDate="2024-01-{day+1:02d} 12:00:00 -0500" 
          value="{250 + day * 50}"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Apple Watch" 
          startDate="2024-01-{day+1:02d} 00:00:00 -0500" endDate="2024-01-{day+1:02d} 23:59:59 -0500" 
          value="{8000 + day * 1000}"/>
"""
        
        xml_content += "</HealthData>"
        
        file_path = tmp_path / "export.xml"
        with open(file_path, "w") as f:
            f.write(xml_content)

        return file_path

    def test_extract_features_from_file(
        self, client: TestClient, sample_health_data: Path
    ) -> None:
        """Test extracting features from uploaded file."""
        with open(sample_health_data, "rb") as f:
            response = client.post(
                "/api/v1/features/extract",
                files={"file": ("export.xml", f, "application/xml")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert isinstance(data["features"], dict)
        assert len(data["features"]) > 0
