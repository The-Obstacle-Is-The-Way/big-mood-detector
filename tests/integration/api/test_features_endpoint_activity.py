"""
Integration tests for activity features in the feature extraction API endpoint.

Tests that the /api/v1/features/extract endpoint returns activity-derived features
alongside sleep and circadian features.
"""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from big_mood_detector.interfaces.api.main import app


class TestFeatureExtractionAPIActivity:
    """Test activity feature extraction via API endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)

    @pytest.fixture
    def sample_health_export_json(self, tmp_path):
        """Create a sample Health Auto Export JSON file with activity data."""
        # Create sample Step Count.json
        step_data = {
            "data": [
                {
                    "sourceName": "Apple Watch",
                    "sourceVersion": "10.1",
                    "device": "Apple Watch",
                    "type": "stepCount",
                    "unit": "count",
                    "value": 3500,
                    "startDate": "2024-01-15 08:00:00",
                    "endDate": "2024-01-15 12:00:00"
                },
                {
                    "sourceName": "Apple Watch",
                    "sourceVersion": "10.1",
                    "device": "Apple Watch",
                    "type": "stepCount",
                    "unit": "count",
                    "value": 5200,
                    "startDate": "2024-01-15 12:00:00",
                    "endDate": "2024-01-15 18:00:00"
                },
                {
                    "sourceName": "Apple Watch",
                    "sourceVersion": "10.1",
                    "device": "Apple Watch",
                    "type": "stepCount",
                    "unit": "count",
                    "value": 1300,
                    "startDate": "2024-01-15 18:00:00",
                    "endDate": "2024-01-15 22:00:00"
                }
            ]
        }
        
        # Create sample Sleep Analysis.json
        sleep_data = {
            "data": [
                {
                    "sourceName": "Apple Watch",
                    "sourceVersion": "10.1",
                    "device": "Apple Watch",
                    "type": "sleepAnalysis",
                    "value": "AsleepCore",
                    "startDate": "2024-01-14 23:00:00",
                    "endDate": "2024-01-15 07:00:00"
                }
            ]
        }
        
        # Create a directory structure
        health_dir = tmp_path / "health_export"
        health_dir.mkdir()
        
        # Write the JSON files
        with open(health_dir / "Step Count.json", "w") as f:
            json.dump(step_data, f)
            
        with open(health_dir / "Sleep Analysis.json", "w") as f:
            json.dump(sleep_data, f)
            
        # Create a zip file
        import zipfile
        zip_path = tmp_path / "health_export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(health_dir / "Step Count.json", "Step Count.json")
            zf.write(health_dir / "Sleep Analysis.json", "Sleep Analysis.json")
            
        return zip_path

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_extract_features_includes_activity(self, client, sample_health_export_json):
        """Test that feature extraction returns activity features."""
        # Read the zip file
        with open(sample_health_export_json, "rb") as f:
            file_content = f.read()
        
        # Post to the API
        response = client.post(
            "/api/v1/features/extract",
            files={"file": ("health_export.zip", file_content, "application/zip")},
        )
        
        # Should succeed
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "features" in data
        assert "metadata" in data
        
        features = data["features"]
        
        # Verify activity features are present
        assert "daily_steps" in features
        assert "activity_variance" in features  
        assert "sedentary_hours" in features
        assert "activity_fragmentation" in features
        assert "sedentary_bout_mean" in features
        assert "activity_intensity_ratio" in features
        
        # Verify activity features have reasonable values
        assert features["daily_steps"] == 10000  # 3500 + 5200 + 1300
        assert features["activity_variance"] > 0
        assert 0 <= features["sedentary_hours"] <= 24

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_extract_features_activity_from_xml(self, client, tmp_path):
        """Test activity feature extraction from Apple Health XML export."""
        # Create a minimal XML export with activity data
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!ELEMENT HealthData (Record*)>
<!ATTLIST HealthData locale CDATA #REQUIRED>
<!ELEMENT Record EMPTY>
<!ATTLIST Record type CDATA #REQUIRED
                  sourceName CDATA #REQUIRED
                  sourceVersion CDATA #IMPLIED
                  device CDATA #IMPLIED
                  unit CDATA #IMPLIED
                  creationDate CDATA #IMPLIED
                  startDate CDATA #REQUIRED
                  endDate CDATA #REQUIRED
                  value CDATA #IMPLIED>
]>
<HealthData locale="en_US">
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Apple Watch" unit="count" 
          startDate="2024-01-15 08:00:00 -0800" endDate="2024-01-15 12:00:00 -0800" value="4500"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Apple Watch" unit="count"
          startDate="2024-01-15 13:00:00 -0800" endDate="2024-01-15 17:00:00 -0800" value="6000"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="Apple Watch" unit="count"
          startDate="2024-01-15 18:00:00 -0800" endDate="2024-01-15 20:00:00 -0800" value="2000"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" 
          startDate="2024-01-14 23:00:00 -0800" endDate="2024-01-15 07:00:00 -0800" value="HKCategoryValueSleepAnalysisAsleepCore"/>
</HealthData>'''
        
        # Save XML file
        xml_path = tmp_path / "export.xml"
        xml_path.write_text(xml_content)
        
        # Post to API
        with open(xml_path, "rb") as f:
            response = client.post(
                "/api/v1/features/extract",
                files={"file": ("export.xml", f, "text/xml")},
            )
        
        assert response.status_code == 200
        data = response.json()
        
        features = data["features"]
        
        # Check activity features from XML
        assert "daily_steps" in features
        assert features["daily_steps"] == 12500  # 4500 + 6000 + 2000
        assert "activity_variance" in features
        assert features["activity_variance"] > 0

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_feature_schema_validation(self, client, sample_health_export_json):
        """Test that the API response schema includes all expected fields."""
        with open(sample_health_export_json, "rb") as f:
            response = client.post(
                "/api/v1/features/extract",
                files={"file": ("health_export.zip", f.read(), "application/zip")},
            )
        
        assert response.status_code == 200
        data = response.json()
        
        features = data["features"]
        
        # Expected feature groups
        sleep_features = [
            "sleep_percentage_mean", "sleep_percentage_std", "sleep_percentage_zscore",
            "sleep_amplitude_mean", "sleep_amplitude_std", "sleep_amplitude_zscore"
        ]
        
        circadian_features = [
            "circadian_amplitude_mean", "circadian_amplitude_std", "circadian_amplitude_zscore",
            "circadian_phase_mean", "circadian_phase_std", "circadian_phase_zscore"
        ]
        
        activity_features = [
            "daily_steps", "activity_variance", "sedentary_hours",
            "activity_fragmentation", "sedentary_bout_mean", "activity_intensity_ratio"
        ]
        
        # All features should be present
        all_expected = sleep_features + circadian_features + activity_features
        for feature in all_expected:
            assert feature in features, f"Missing feature: {feature}"
        
        # Feature count should include activity features
        assert data["feature_count"] >= 42  # 36 Seoul + 6 activity

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_missing_activity_data_defaults(self, client, tmp_path):
        """Test API response when activity data is missing."""
        # Create export with only sleep data
        sleep_only_data = {
            "data": [
                {
                    "sourceName": "Apple Watch",
                    "type": "sleepAnalysis",
                    "value": "AsleepCore",
                    "startDate": "2024-01-14 23:00:00",
                    "endDate": "2024-01-15 07:00:00"
                }
            ]
        }
        
        json_path = tmp_path / "Sleep Analysis.json"
        with open(json_path, "w") as f:
            json.dump(sleep_only_data, f)
        
        with open(json_path, "rb") as f:
            response = client.post(
                "/api/v1/features/extract",
                files={"file": ("Sleep Analysis.json", f, "application/json")},
            )
        
        assert response.status_code == 200
        features = response.json()["features"]
        
        # Activity features should have sensible defaults
        assert features["daily_steps"] == 0
        assert features["activity_variance"] == 0
        assert features["sedentary_hours"] == 24.0
        assert features["activity_fragmentation"] == 0

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_backwards_compatibility(self, client, sample_health_export_json):
        """Test that existing features remain unchanged with activity additions."""
        with open(sample_health_export_json, "rb") as f:
            response = client.post(
                "/api/v1/features/extract",
                files={"file": ("health_export.zip", f.read(), "application/zip")},
            )
        
        features = response.json()["features"]
        
        # Original 36 features should still be present
        original_features = [
            "sleep_percentage_mean", "long_sleep_num_mean", "short_sleep_num_mean",
            "circadian_amplitude_mean", "circadian_phase_mean"
        ]
        
        for feature in original_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
        
        # New activity features should be additional
        assert len(features) > 36