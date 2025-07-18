"""
Test API Server Startup Integration

Ensures API server can start with existing model files.
"""

import pytest
import requests
import time
import subprocess
import signal
import os
from pathlib import Path


class TestAPIStartup:
    """Test API server startup with real model files."""

    def test_api_server_starts_successfully(self):
        """Test that API server can start with existing model files."""
        # This test should FAIL initially due to model path mismatch
        
        # Start server in background
        env = os.environ.copy()
        env["DISABLE_RATE_LIMIT"] = "1"
        
        process = subprocess.Popen(
            ["python", "src/big_mood_detector/main.py", "serve", "--port", "8002"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        try:
            # Wait for server to start
            time.sleep(3)
            
            # Check if server is responding
            response = requests.get("http://127.0.0.1:8002/health", timeout=5)
            assert response.status_code == 200
            
            # Verify model status endpoint
            response = requests.get("http://127.0.0.1:8002/api/v1/models/status", timeout=5)
            assert response.status_code == 200
            
            data = response.json()
            assert "models_loaded" in data
            assert data["models_loaded"] >= 1  # At least XGBoost should load
            
        finally:
            # Clean shutdown
            process.terminate()
            process.wait(timeout=5)

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