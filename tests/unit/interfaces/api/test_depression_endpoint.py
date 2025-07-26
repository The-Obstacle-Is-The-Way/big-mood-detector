"""
Test Depression Prediction API Endpoint

Following TDD principles - writing tests first for the depression prediction endpoint.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from big_mood_detector.domain.services.pat_predictor import PATBinaryPredictions


class TestDepressionPredictionEndpoint:
    """Test the /predictions/depression endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from big_mood_detector.interfaces.api.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_pat_predictor(self):
        """Mock PAT predictor for testing."""
        mock = MagicMock()
        mock.is_loaded = True
        mock.predict_depression.return_value = 0.75
        mock.predict_from_embeddings.return_value = PATBinaryPredictions(
            depression_probability=0.75,
            benzodiazepine_probability=0.0,
            confidence=0.85
        )
        return mock

    def test_depression_prediction_endpoint_exists(self, client, mock_pat_predictor):
        """Endpoint should exist at /predictions/depression."""
        # Mock the DI container to return our mock predictor
        with patch('big_mood_detector.interfaces.api.routes.depression.get_container') as mock_get_container:
            mock_container = MagicMock()
            mock_container.resolve.return_value = mock_pat_predictor
            mock_get_container.return_value = mock_container
            
            # This will fail initially (RED)
            response = client.post("/predictions/depression")
            assert response.status_code != 404

    def test_depression_prediction_with_activity_sequence(self, client, mock_pat_predictor):
        """Should predict depression from 7-day activity sequence."""
        # Mock the DI container to return our mock predictor
        with patch('big_mood_detector.interfaces.api.routes.depression.get_container') as mock_get_container:
            mock_container = MagicMock()
            mock_container.resolve.return_value = mock_pat_predictor
            mock_get_container.return_value = mock_container

            # Prepare test data - 7 days of activity
            activity_data = {
                "activity_sequence": [float(i % 100) for i in range(10080)]  # 7 days
            }

        with patch("big_mood_detector.interfaces.api.routes.depression.get_pat_predictor") as mock_get:
            mock_get.return_value = mock_pat_predictor

            response = client.post(
                "/predictions/depression",
                json=activity_data
            )

        assert response.status_code == 200
        result = response.json()
        assert "depression_probability" in result
        assert "confidence" in result
        assert 0 <= result["depression_probability"] <= 1
        assert 0 <= result["confidence"] <= 1

    def test_depression_prediction_validates_sequence_length(self, client, mock_pat_predictor):
        """Should validate that activity sequence is exactly 10,080 timesteps."""
        # Wrong length sequence
        activity_data = {
            "activity_sequence": [0.0] * 5000  # Wrong length
        }

        with patch("big_mood_detector.interfaces.api.routes.depression.get_pat_predictor") as mock_get:
            mock_get.return_value = mock_pat_predictor

            response = client.post(
                "/predictions/depression",
                json=activity_data
            )

        assert response.status_code == 422  # Validation error
        error_msg = response.json()["detail"][0]["msg"]
        assert "10,080" in error_msg or "10080" in error_msg

    def test_depression_prediction_with_embeddings(self, client, mock_pat_predictor):
        """Should predict depression from pre-computed embeddings."""
        # 96-dimensional embeddings
        embeddings_data = {
            "embeddings": [0.1] * 96
        }

        with patch("big_mood_detector.interfaces.api.routes.depression.get_pat_predictor") as mock_get:
            mock_get.return_value = mock_pat_predictor

            response = client.post(
                "/predictions/depression/from-embeddings",
                json=embeddings_data
            )

        assert response.status_code == 200
        result = response.json()
        assert "depression_probability" in result
        assert "confidence" in result
        assert "benzodiazepine_probability" in result

    def test_returns_error_when_model_not_loaded(self):
        """Should return 503 when model is not loaded."""
        from big_mood_detector.interfaces.api.main import app
        from big_mood_detector.interfaces.api.routes.depression import get_pat_predictor

        # Create a mock predictor that's not loaded
        mock_predictor = MagicMock()
        mock_predictor.is_loaded = False

        # Override the dependency
        app.dependency_overrides[get_pat_predictor] = lambda: mock_predictor

        # Create test client with overridden dependency
        client = TestClient(app)

        try:
            response = client.post(
                "/predictions/depression",
                json={"activity_sequence": [0.0] * 10080}
            )

            assert response.status_code == 503
            assert "not loaded" in response.json()["detail"].lower()
        finally:
            # Clean up override
            app.dependency_overrides.clear()

    def test_handles_prediction_errors_gracefully(self):
        """Should return 500 with helpful message on prediction error."""
        from big_mood_detector.interfaces.api.main import app
        from big_mood_detector.interfaces.api.routes.depression import get_pat_predictor

        # Create a mock predictor that raises an error
        mock_predictor = MagicMock()
        mock_predictor.is_loaded = True
        mock_predictor.predict_depression.side_effect = RuntimeError("CUDA out of memory")

        # Override the dependency
        app.dependency_overrides[get_pat_predictor] = lambda: mock_predictor

        # Create test client with overridden dependency
        client = TestClient(app)

        try:
            response = client.post(
                "/predictions/depression",
                json={"activity_sequence": [0.0] * 10080}
            )

            assert response.status_code == 500
            assert "prediction failed" in response.json()["detail"].lower()
        finally:
            # Clean up override
            app.dependency_overrides.clear()

    def test_response_includes_metadata(self, client, mock_pat_predictor):
        """Response should include helpful metadata."""
        activity_data = {
            "activity_sequence": [0.0] * 10080
        }

        with patch("big_mood_detector.interfaces.api.routes.depression.get_pat_predictor") as mock_get:
            mock_get.return_value = mock_pat_predictor

            response = client.post(
                "/predictions/depression",
                json=activity_data
            )

        assert response.status_code == 200
        result = response.json()

        # Should include metadata
        assert "model_version" in result
        assert "prediction_timestamp" in result
        assert result["model_version"] == "pat_conv_l_v0.5929"
