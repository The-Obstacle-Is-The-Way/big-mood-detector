"""
Test Upload Endpoints

TDD for file upload functionality in the API.
"""

import io
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TestUploadEndpoints:
    """Test file upload endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from big_mood_detector.interfaces.api.main import app

        return TestClient(app)

    def test_upload_endpoint_exists(self, client):
        """Test that upload endpoint is registered."""
        # Should return 422 (missing file) not 404
        response = client.post("/api/v1/upload/file")
        assert response.status_code != 404

    def test_upload_single_file(self, client):
        """Test uploading a single health data file."""
        # Create test file
        test_content = b"test health data"
        test_file = io.BytesIO(test_content)

        # Upload file
        files = {"file": ("test.json", test_file, "application/json")}
        response = client.post("/api/v1/upload/file", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert "filename" in data
        assert data["filename"] == "test.json"
        assert "status" in data
        assert data["status"] == "queued"

    def test_upload_xml_file(self, client):
        """Test uploading XML file."""
        xml_content = b'<?xml version="1.0"?><HealthData></HealthData>'
        test_file = io.BytesIO(xml_content)

        files = {"file": ("export.xml", test_file, "application/xml")}
        response = client.post("/api/v1/upload/file", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "export.xml"

    def test_upload_invalid_file_type(self, client):
        """Test uploading unsupported file type."""
        test_file = io.BytesIO(b"not health data")

        files = {"file": ("test.txt", test_file, "text/plain")}
        response = client.post("/api/v1/upload/file", files=files)

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_empty_file(self, client):
        """Test uploading empty file."""
        test_file = io.BytesIO(b"")

        files = {"file": ("empty.json", test_file, "application/json")}
        response = client.post("/api/v1/upload/file", files=files)

        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_upload_large_file(self, client):
        """Test uploading file exceeding size limit."""
        # Create 11MB file (assuming 10MB limit)
        large_content = b"x" * (11 * 1024 * 1024)
        test_file = io.BytesIO(large_content)

        files = {"file": ("large.xml", test_file, "application/xml")}
        response = client.post("/api/v1/upload/file", files=files)

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    @patch("big_mood_detector.interfaces.api.routes.upload.process_upload")
    def test_upload_status_endpoint(self, mock_process, client):
        """Test checking upload status."""
        # Mock background processing to prevent actual execution
        mock_process.return_value = None

        # First upload a file to get a valid upload_id
        test_file = io.BytesIO(b'{"data": []}')
        files = {"file": ("test.json", test_file, "application/json")}
        upload_response = client.post("/api/v1/upload/file", files=files)
        upload_id = upload_response.json()["upload_id"]

        # Now check its status
        response = client.get(f"/api/v1/upload/status/{upload_id}")

        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert "status" in data
        assert data["upload_id"] == upload_id
        assert data["status"] in ["queued", "processing", "completed", "failed"]

    def test_upload_status_not_found(self, client):
        """Test checking status for non-existent upload."""
        response = client.get("/api/v1/upload/status/non-existent-id")

        assert response.status_code == 404
        assert "Upload not found" in response.json()["detail"]

    @patch("big_mood_detector.interfaces.api.routes.upload.process_upload")
    def test_upload_triggers_processing(self, mock_process, client):
        """Test that upload triggers background processing."""
        test_file = io.BytesIO(b'{"data": []}')

        files = {"file": ("test.json", test_file, "application/json")}
        response = client.post("/api/v1/upload/file", files=files)

        assert response.status_code == 200
        # Verify background task was queued
        assert mock_process.called

    def test_batch_upload_endpoint(self, client):
        """Test uploading multiple files."""
        file1 = io.BytesIO(b'{"sleep": []}')
        file2 = io.BytesIO(b'{"activity": []}')

        files = [
            ("files", ("sleep.json", file1, "application/json")),
            ("files", ("activity.json", file2, "application/json")),
        ]

        response = client.post("/api/v1/upload/batch", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert "files" in data
        assert len(data["files"]) == 2

    def test_upload_result_retrieval(self, client):
        """Test retrieving processing results after upload."""
        from big_mood_detector.interfaces.api.routes.upload import upload_status_store

        # Create a completed upload
        upload_id = "completed-upload-123"
        upload_status_store[upload_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Processing complete",
            "result": {
                "depression_risk": 0.4,
                "hypomanic_risk": 0.05,
                "manic_risk": 0.01,
                "confidence": 0.85,
                "days_analyzed": 7,
            },
        }

        response = client.get(f"/api/v1/upload/result/{upload_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "result" in data
        assert data["result"]["depression_risk"] == 0.4

    def test_download_processed_file(self, client):
        """Test downloading processed results as CSV."""
        from big_mood_detector.interfaces.api.routes.upload import UPLOAD_DIR

        upload_id = "completed-upload-123"

        # Create the upload directory and CSV file
        upload_path = UPLOAD_DIR / upload_id
        upload_path.mkdir(exist_ok=True, parents=True)
        csv_path = upload_path / f"{upload_id}_results.csv"

        # Write test CSV content
        csv_content = (
            "date,depression_risk,hypomanic_risk,manic_risk\n2024-01-01,0.4,0.05,0.01"
        )
        csv_path.write_text(csv_content)

        response = client.get(f"/api/v1/upload/download/{upload_id}")

        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        assert response.text == csv_content

        # Cleanup
        csv_path.unlink()
        upload_path.rmdir()

    def test_concurrent_uploads(self, client):
        """Test handling multiple concurrent uploads."""
        import concurrent.futures

        def upload_file(filename):
            test_file = io.BytesIO(f'{{"file": "{filename}"}}'.encode())
            files = {"file": (filename, test_file, "application/json")}
            return client.post("/api/v1/upload/file", files=files)

        # Upload 5 files concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload_file, f"test{i}.json") for i in range(5)]
            responses = [f.result() for f in futures]

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert "upload_id" in response.json()

    def test_upload_with_metadata(self, client):
        """Test uploading file with additional metadata."""
        test_file = io.BytesIO(b'{"data": []}')

        # Include metadata in form data
        files = {"file": ("test.json", test_file, "application/json")}
        data = {
            "patient_id": "test-patient-123",
            "date_range": "2024-01-01,2024-01-31",
            "processing_options": json.dumps({"include_pat": True}),
        }

        response = client.post("/api/v1/upload/file", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert "metadata" in result
        assert result["metadata"]["patient_id"] == "test-patient-123"
