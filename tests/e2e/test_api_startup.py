"""
Test API Server Startup Integration

Ensures API server can start with existing model files.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest
import requests


class TestAPIStartup:
    """Test API server startup with real model files."""

    def test_api_server_starts_successfully(self):
        """Test that API server can start with existing model files."""

        # Start server in background
        env = os.environ.copy()
        env["DISABLE_RATE_LIMIT"] = "1"

        process = subprocess.Popen(
            ["python", "src/big_mood_detector/main.py", "serve", "--port", "8002"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Capture both stdout and stderr
            text=True,
            bufsize=1,
        )

        try:
            # Wait for server to start with better error checking
            max_wait_time = 10  # seconds
            wait_interval = 0.5
            waited = 0.0

            while waited < max_wait_time:
                # Check if process is still running
                if process.poll() is not None:
                    # Process died, get the output
                    stdout, _ = process.communicate()
                    pytest.fail(f"Server failed to start. Output:\n{stdout}")

                # Try to connect
                try:
                    response = requests.get("http://127.0.0.1:8002/health", timeout=2)
                    if response.status_code == 200:
                        break  # Success!
                except requests.exceptions.ConnectionError:
                    pass  # Still starting up
                except requests.exceptions.Timeout:
                    pass  # Still starting up

                time.sleep(wait_interval)
                waited += wait_interval
            else:
                # Timeout reached
                stdout, _ = process.communicate(timeout=2)
                pytest.fail(f"Server did not start within {max_wait_time}s. Output:\n{stdout}")

            # Server is responding, now test endpoints
            response = requests.get("http://127.0.0.1:8002/health", timeout=5)
            assert response.status_code == 200

            # Verify models are loaded via health check
            health_data = response.json()
            assert "status" in health_data
            assert health_data["status"] == "healthy"

        finally:
            # Clean shutdown
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def test_model_files_exist_with_correct_names(self):
        """Test that expected model files exist."""
        # This test defines what the fix should accomplish

        model_dir = Path("model_weights/xgboost/converted")

        # These are the files that should exist and be loadable
        expected_files = [
            "XGBoost_DE.json",    # Depression
            "XGBoost_HME.json",   # Hypomanic Episode
            "XGBoost_ME.json",    # Manic Episode
        ]

        for filename in expected_files:
            file_path = model_dir / filename
            assert file_path.exists(), f"Model file {filename} should exist"
            assert file_path.stat().st_size > 0, f"Model file {filename} should not be empty"
