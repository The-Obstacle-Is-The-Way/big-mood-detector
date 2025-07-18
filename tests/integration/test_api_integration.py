"""
Integration tests for FastAPI endpoints.

Tests the full API surface with TestClient to ensure proper wiring.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
            "hrv_rmssd": 45.0
        }

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_predict_xgboost_success(self, client: TestClient, sample_features: dict) -> None:
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
            "sleep_efficiency": 0.85
            # Missing required features: sleep_timing_variance, daily_steps, activity_variance, sedentary_hours
        }
        response = client.post("/api/v1/predictions/predict", json=incomplete_features)
        assert response.status_code == 422

    def test_predict_xgboost_invalid_values(self, client: TestClient, sample_features: dict) -> None:
        """Test XGBoost prediction with invalid feature values."""
        invalid_features = sample_features.copy()
        invalid_features["sleep_efficiency"] = 1.5  # Invalid: > 1.0
        
        response = client.post("/api/v1/predictions/predict", json=invalid_features)
        assert response.status_code == 422


class TestLabelsAPI:
    """Test labels/episodes endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_episode(self) -> dict[str, Any]:
        """Create sample episode payload."""
        now = datetime.utcnow()
        return {
            "start_date": (now - timedelta(days=7)).isoformat(),
            "end_date": (now - timedelta(days=3)).isoformat(),
            "episode_type": "depressive",
            "severity": "moderate",
            "confidence": 0.8,
            "notes": "Test episode for integration testing"
        }

    def test_create_episode_success(self, client: TestClient, sample_episode: dict) -> None:
        """Test successful episode creation."""
        response = client.post("/api/v1/labels/episodes", json=sample_episode)
        assert response.status_code == 201
        
        data = response.json()
        assert "id" in data
        assert data["start_date"] == sample_episode["start_date"]
        assert data["episode_type"] == sample_episode["episode_type"]

    def test_list_episodes_empty(self, client: TestClient) -> None:
        """Test listing episodes when none exist."""
        response = client.get("/api/v1/labels/episodes")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_and_list_episodes(self, client: TestClient, sample_episode: dict) -> None:
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

    def test_get_episode_by_id(self, client: TestClient, sample_episode: dict) -> None:
        """Test getting a specific episode by ID."""
        # Create episode
        create_response = client.post("/api/v1/labels/episodes", json=sample_episode)
        assert create_response.status_code == 201
        created_id = create_response.json()["id"]
        
        # Get episode
        get_response = client.get(f"/api/v1/labels/episodes/{created_id}")
        assert get_response.status_code == 200
        episode = get_response.json()
        assert episode["id"] == created_id
        assert episode["episode_type"] == sample_episode["episode_type"]

    def test_get_nonexistent_episode(self, client: TestClient) -> None:
        """Test getting an episode that doesn't exist."""
        response = client.get("/api/v1/labels/episodes/nonexistent-id")
        assert response.status_code == 404

    def test_delete_episode(self, client: TestClient, sample_episode: dict) -> None:
        """Test deleting an episode."""
        # Create episode
        create_response = client.post("/api/v1/labels/episodes", json=sample_episode)
        assert create_response.status_code == 201
        created_id = create_response.json()["id"]
        
        # Delete episode
        delete_response = client.delete(f"/api/v1/labels/episodes/{created_id}")
        assert delete_response.status_code == 204
        
        # Verify it's gone
        get_response = client.get(f"/api/v1/labels/episodes/{created_id}")
        assert get_response.status_code == 404

    def test_invalid_episode_type(self, client: TestClient, sample_episode: dict) -> None:
        """Test creating episode with invalid type."""
        invalid_episode = sample_episode.copy()
        invalid_episode["episode_type"] = "invalid_type"
        
        response = client.post("/labels/episodes", json=invalid_episode)
        assert response.status_code == 422

    def test_invalid_date_range(self, client: TestClient, sample_episode: dict) -> None:
        """Test creating episode with end before start."""
        invalid_episode = sample_episode.copy()
        invalid_episode["start_date"], invalid_episode["end_date"] = (
            invalid_episode["end_date"], 
            invalid_episode["start_date"]
        )
        
        response = client.post("/labels/episodes", json=invalid_episode)
        assert response.status_code == 422


class TestFeatureExtraction:
    """Test feature extraction endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_health_data(self, tmp_path: Path) -> Path:
        """Create sample health data file."""
        # Create minimal valid JSON health data
        data = {
            "sleep_records": [
                {
                    "start_date": "2024-01-01T23:00:00",
                    "end_date": "2024-01-02T07:00:00",
                    "value": "InBed"
                }
            ],
            "heart_rate_records": [
                {
                    "date": "2024-01-02T06:00:00",
                    "value": 65
                }
            ],
            "activity_records": [
                {
                    "date": "2024-01-02T12:00:00",
                    "active_calories": 250,
                    "step_count": 5000
                }
            ]
        }
        
        file_path = tmp_path / "health_data.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        
        return file_path

    @pytest.mark.skip(reason="Feature extraction endpoint not yet implemented")
    def test_extract_features_from_file(self, client: TestClient, sample_health_data: Path) -> None:
        """Test extracting features from uploaded file."""
        with open(sample_health_data, "rb") as f:
            response = client.post(
                "/features/extract",
                files={"file": ("health_data.json", f, "application/json")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert isinstance(data["features"], dict)
        assert len(data["features"]) > 0