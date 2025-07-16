"""
Integration Test for Health Data Parsing and Feature Extraction
Tests the complete flow from XML parsing to clinical feature extraction.
Following TDD and Clean Architecture principles.
"""

import xml.etree.ElementTree as ET
from datetime import date
from io import StringIO

import pytest

from big_mood_detector.domain.services.feature_extraction_service import (
    FeatureExtractionService,
)
from big_mood_detector.infrastructure.parsers.xml import (
    ActivityParser,
    HeartRateParser,
    SleepParser,
)


class TestHealthDataIntegration:
    """Integration tests for the complete health data pipeline."""

    @pytest.fixture
    def sample_health_xml(self):
        """Create a realistic Apple Health export XML sample."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!ELEMENT HealthData (ExportDate?, Record*)>
<!ATTLIST ExportDate value CDATA #REQUIRED>
<!ELEMENT Record EMPTY>
<!ATTLIST Record
    type CDATA #REQUIRED
    sourceName CDATA #REQUIRED
    value CDATA #IMPLIED
    startDate CDATA #REQUIRED
    endDate CDATA #REQUIRED
    creationDate CDATA #IMPLIED
    unit CDATA #IMPLIED
>
]>
<HealthData locale="en_US">
  <ExportDate value="2024-01-05 10:00:00 -0800"/>

  <!-- Sleep data for multiple nights -->
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-01 23:00:00 -0800" endDate="2024-01-02 03:00:00 -0800" creationDate="2024-01-02 03:00:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepREM" startDate="2024-01-02 03:00:00 -0800" endDate="2024-01-02 05:00:00 -0800" creationDate="2024-01-02 05:00:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepDeep" startDate="2024-01-02 05:00:00 -0800" endDate="2024-01-02 07:00:00 -0800" creationDate="2024-01-02 07:00:00 -0800"/>

  <!-- Activity data for multiple days -->
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="5000" startDate="2024-01-02 08:00:00 -0800" endDate="2024-01-02 12:00:00 -0800" creationDate="2024-01-02 12:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="3000" startDate="2024-01-02 14:00:00 -0800" endDate="2024-01-02 18:00:00 -0800" creationDate="2024-01-02 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierFlightsClimbed" sourceName="iPhone" value="10" startDate="2024-01-02 10:00:00 -0800" endDate="2024-01-02 10:30:00 -0800" creationDate="2024-01-02 10:30:00 -0800" unit="count"/>

  <!-- Heart rate data including resting and active -->
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="62" startDate="2024-01-02 06:00:00 -0800" endDate="2024-01-02 06:01:00 -0800" creationDate="2024-01-02 06:01:00 -0800" unit="count/min">
    <MetadataEntry key="HKMetadataKeyHeartRateMotionContext" value="HKHeartRateMotionContextSedentary"/>
  </Record>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="120" startDate="2024-01-02 10:00:00 -0800" endDate="2024-01-02 10:01:00 -0800" creationDate="2024-01-02 10:01:00 -0800" unit="count/min">
    <MetadataEntry key="HKMetadataKeyHeartRateMotionContext" value="HKHeartRateMotionContextActive"/>
  </Record>
  <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" sourceName="Apple Watch" value="45" startDate="2024-01-02 06:00:00 -0800" endDate="2024-01-02 06:05:00 -0800" creationDate="2024-01-02 06:05:00 -0800" unit="ms"/>

  <!-- Edge case: Very short sleep (nap) -->
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="iPhone" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-02 14:00:00 -0800" endDate="2024-01-02 14:30:00 -0800" creationDate="2024-01-02 14:30:00 -0800"/>

  <!-- Edge case: Very high heart rate (potential anxiety/mania indicator) -->
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="110" startDate="2024-01-02 02:00:00 -0800" endDate="2024-01-02 02:01:00 -0800" creationDate="2024-01-02 02:01:00 -0800" unit="count/min">
    <MetadataEntry key="HKMetadataKeyHeartRateMotionContext" value="HKHeartRateMotionContextSedentary"/>
  </Record>

  <!-- Edge case: Very low HRV (stress indicator) -->
  <Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" sourceName="Apple Watch" value="15" startDate="2024-01-02 14:00:00 -0800" endDate="2024-01-02 14:05:00 -0800" creationDate="2024-01-02 14:05:00 -0800" unit="ms"/>
</HealthData>
"""
        return xml_content

    def test_full_pipeline_integration(self, sample_health_xml):
        """Test the complete flow from XML parsing to feature extraction."""
        # ARRANGE
        sleep_parser = SleepParser()
        activity_parser = ActivityParser()
        heart_parser = HeartRateParser()
        feature_service = FeatureExtractionService()

        # Parse the XML
        root = ET.parse(StringIO(sample_health_xml)).getroot()

        # ACT
        # Parse all data types
        sleep_records = sleep_parser.parse(root)
        activity_records = activity_parser.parse(root)
        heart_records = heart_parser.parse(root)

        # Extract clinical features
        features = feature_service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
        )

        # ASSERT
        # Verify parsing worked correctly
        assert len(sleep_records) == 4  # 3 night segments + 1 nap
        assert len(activity_records) == 3  # 2 step counts + 1 flights
        assert len(heart_records) == 4  # 3 heart rates + 1 HRV

        # Verify feature extraction
        assert len(features) >= 1
        jan2_features = features.get(date(2024, 1, 2))
        assert jan2_features is not None

        # Check aggregated values
        assert jan2_features.sleep_duration_hours == 8.5  # 8h night + 0.5h nap
        assert jan2_features.total_steps == 8000  # 5000 + 3000
        assert jan2_features.flights_climbed == 10

        # Check clinical significance flags
        assert jan2_features.is_clinically_significant
        assert any(
            "heart rate" in note.lower() for note in jan2_features.clinical_notes
        )
        assert any("hrv" in note.lower() for note in jan2_features.clinical_notes)

    def test_missing_data_types_handling(self):
        """Test handling when some data types are missing."""
        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
  <ExportDate value="2024-01-05 10:00:00 -0800"/>
  <!-- Only sleep data, no activity or heart rate -->
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-01 23:00:00 -0800" endDate="2024-01-02 07:00:00 -0800" creationDate="2024-01-02 07:00:00 -0800"/>
</HealthData>
"""
        sleep_parser = SleepParser()
        activity_parser = ActivityParser()
        heart_parser = HeartRateParser()
        feature_service = FeatureExtractionService()

        root = ET.parse(StringIO(xml_content)).getroot()

        # ACT
        sleep_records = sleep_parser.parse(root)
        activity_records = activity_parser.parse(root)
        heart_records = heart_parser.parse(root)

        features = feature_service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
        )

        # ASSERT
        assert len(sleep_records) == 1
        assert len(activity_records) == 0
        assert len(heart_records) == 0

        # Should still generate features with defaults for missing data
        assert len(features) >= 1
        day_features = list(features.values())[0]
        assert day_features.sleep_duration_hours == 8.0
        assert day_features.total_steps == 0  # Default
        assert day_features.avg_resting_hr == 0.0  # Default

    def test_clinical_edge_cases(self, sample_health_xml):
        """Test detection of clinical edge cases in integrated data."""
        # ARRANGE
        parsers = {
            "sleep": SleepParser(),
            "activity": ActivityParser(),
            "heart": HeartRateParser(),
        }
        feature_service = FeatureExtractionService()

        root = ET.parse(StringIO(sample_health_xml)).getroot()

        # ACT
        records = {
            "sleep": parsers["sleep"].parse(root),
            "activity": parsers["activity"].parse(root),
            "heart": parsers["heart"].parse(root),
        }

        features = feature_service.extract_features(
            sleep_records=records["sleep"],
            activity_records=records["activity"],
            heart_records=records["heart"],
        )

        # ASSERT - Verify clinical indicators are detected
        jan2_features = features.get(date(2024, 1, 2))
        assert jan2_features is not None

        # Should detect high resting heart rate during sleep (110 bpm at 2 AM)
        assert jan2_features.is_clinically_significant

        # Should have low HRV warning (15ms is below threshold)
        clinical_notes_text = " ".join(jan2_features.clinical_notes).lower()
        assert "hrv" in clinical_notes_text

    def test_data_quality_and_completeness(self):
        """Test that the integration handles incomplete or poor quality data."""
        # ARRANGE
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData locale="en_US">
  <ExportDate value="2024-01-05 10:00:00 -0800"/>
  <!-- Duplicate records that should be deduplicated -->
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="1000" startDate="2024-01-02 10:00:00 -0800" endDate="2024-01-02 11:00:00 -0800" creationDate="2024-01-02 11:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="1000" startDate="2024-01-02 10:00:00 -0800" endDate="2024-01-02 11:00:00 -0800" creationDate="2024-01-02 11:00:00 -0800" unit="count"/>

  <!-- Overlapping sleep records -->
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-01 23:00:00 -0800" endDate="2024-01-02 03:00:00 -0800" creationDate="2024-01-02 03:00:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="iPhone" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-02 02:00:00 -0800" endDate="2024-01-02 04:00:00 -0800" creationDate="2024-01-02 04:00:00 -0800"/>
</HealthData>
"""
        # Setup parsers and service
        activity_parser = ActivityParser()
        sleep_parser = SleepParser()
        feature_service = FeatureExtractionService()

        root = ET.parse(StringIO(xml_content)).getroot()

        # ACT
        activity_records = activity_parser.parse(root)
        sleep_records = sleep_parser.parse(root)

        features = feature_service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
        )

        # ASSERT
        # Parser should handle duplicates (though our simple parser doesn't deduplicate yet)
        assert len(activity_records) == 2  # Both records parsed
        assert len(sleep_records) == 2  # Both sleep records parsed

        # Feature extraction should still work
        assert len(features) >= 1
