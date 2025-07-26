"""
Test API Server Startup Integration

Ensures API server can start with existing model files.
"""

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.e2e
from pathlib import Path

import pytest
import requests


class TestAPIStartup:
    """Test API server startup with real model files."""

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_api_server_starts_successfully(self):
        """Test that API server can start with existing model files."""
        # Find an available port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]

        # Start server in background
        env = os.environ.copy()
        env["DISABLE_RATE_LIMIT"] = "1"

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "big_mood_detector.interfaces.api.main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--workers", "1"
        ]

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Give server time to start
            max_wait = 30
            interval = 1
            waited = 0

            while waited < max_wait:
                # Check if process died
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    pytest.fail(f"Server died. stdout:\n{stdout}\nstderr:\n{stderr}")

                # Try to connect
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                    if response.status_code == 200:
                        # Success!
                        break
                except Exception:
                    pass

                time.sleep(interval)
                waited += interval
            else:
                pytest.fail(f"Server didn't start in {max_wait}s")

            # Test the health endpoint
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert "version" in data

        finally:
            # Clean shutdown
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    def test_model_files_exist_with_correct_names(self):
        """Test that expected model files exist."""
        # This test defines what the fix should accomplish

        model_dir = Path("model_weights/xgboost/converted")

        # These are the files that should exist and be loadable
        expected_files = [
            "XGBoost_DE.json",  # Depression
            "XGBoost_HME.json",  # Hypomanic Episode
            "XGBoost_ME.json",  # Manic Episode
        ]

        for filename in expected_files:
            file_path = model_dir / filename
            assert file_path.exists(), f"Model file {filename} should exist"
            assert (
                file_path.stat().st_size > 0
            ), f"Model file {filename} should not be empty"
