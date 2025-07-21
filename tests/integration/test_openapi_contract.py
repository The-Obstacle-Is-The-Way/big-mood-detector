"""OpenAPI contract tests to ensure routes don't disappear."""

import pytest
from fastapi.testclient import TestClient

from big_mood_detector.interfaces.api.main import app


class TestOpenAPIContract:
    """Test that critical API endpoints remain in the OpenAPI spec."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_openapi_spec_available(self, client: TestClient):
        """Test that OpenAPI spec is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert "paths" in spec
        assert "info" in spec
        assert spec["info"]["title"] == "Big Mood Detector"

    def test_critical_endpoints_present(self, client: TestClient):
        """Test that all critical endpoints are in the spec."""
        response = client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Core endpoints that must always exist
        critical_endpoints = [
            "/health",
            "/healthz",
            "/api/v1/features/extract",
            "/api/v1/predictions/predict",
            "/api/v1/clinical/thresholds",
            "/api/v1/labels/episodes",
        ]

        for endpoint in critical_endpoints:
            assert (
                endpoint in paths
            ), f"Critical endpoint {endpoint} missing from OpenAPI spec"

    def test_upload_endpoints_conditional(self, client: TestClient):
        """Test that upload endpoints are only present when enabled."""
        import os

        response = client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        upload_endpoints = [
            "/api/v1/upload/file",
            "/api/v1/upload/batch",
            "/api/v1/upload/status/{upload_id}",
            "/api/v1/upload/result/{upload_id}",
        ]

        # Check based on environment variable
        if os.environ.get("ENABLE_ASYNC_UPLOAD", "false").lower() == "true":
            for endpoint in upload_endpoints:
                assert (
                    endpoint in paths
                ), f"Upload endpoint {endpoint} should be present when enabled"
        else:
            for endpoint in upload_endpoints:
                assert (
                    endpoint not in paths
                ), f"Upload endpoint {endpoint} should not be present when disabled"

    def test_endpoint_methods_correct(self, client: TestClient):
        """Test that endpoints have the correct HTTP methods."""
        response = client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # Check specific methods
        assert "get" in paths.get("/health", {}), "Health check should support GET"
        assert "post" in paths.get(
            "/api/v1/features/extract", {}
        ), "Feature extract should support POST"
        assert "post" in paths.get(
            "/api/v1/predictions/predict", {}
        ), "Predictions should support POST"
        assert "get" in paths.get(
            "/api/v1/labels/episodes", {}
        ), "Label listing should support GET"
        assert "post" in paths.get(
            "/api/v1/labels/episodes", {}
        ), "Label creation should support POST"

    def test_response_schemas_present(self, client: TestClient):
        """Test that response schemas are properly defined."""
        response = client.get("/openapi.json")
        spec = response.json()

        # Check that components/schemas exist
        assert "components" in spec
        assert "schemas" in spec["components"]

        # Check for some key schemas
        schemas = spec["components"]["schemas"]
        important_schemas = [
            "FeatureExtractionResponse",
            "PredictionResponse",
            "EpisodeResponse",
        ]

        for schema_name in important_schemas:
            assert (
                schema_name in schemas
            ), f"Schema {schema_name} missing from OpenAPI spec"

        # Check for ClinicalInterpretationResponse with flexible naming
        clinical_schema_found = any(
            "ClinicalInterpretationResponse" in key
            for key in schemas.keys()
        )
        assert clinical_schema_found, "ClinicalInterpretationResponse schema missing from OpenAPI spec"

    def test_api_version_consistency(self, client: TestClient):
        """Test that all API endpoints use consistent versioning."""
        response = client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]

        # All API endpoints should start with /api/v1/
        api_paths = [p for p in paths if p.startswith("/api/")]
        for path in api_paths:
            assert path.startswith(
                "/api/v1/"
            ), f"API endpoint {path} doesn't use v1 versioning"
