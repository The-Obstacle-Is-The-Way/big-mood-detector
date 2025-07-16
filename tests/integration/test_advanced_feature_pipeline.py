"""
Integration test for advanced feature engineering pipeline

Demonstrates end-to-end feature extraction for mood prediction.
"""

import pytest
from pathlib import Path
import tempfile

from big_mood_detector.domain.services.feature_extraction_service import (
    FeatureExtractionService,
)
from big_mood_detector.infrastructure.parsers.parser_factory import (
    UnifiedHealthDataParser,
)


class TestAdvancedFeaturePipeline:
    """Test complete pipeline with advanced features."""

    @pytest.fixture
    def sample_xml_file(self):
        """Create a sample XML file with test data."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData>
<HealthData locale="en_US">
  <ExportDate value="2024-01-20 10:00:00 -0800"/>
  
  <!-- Multiple days of sleep data for temporal features (need at least 7 days history) -->
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-01 23:00:00 -0800" endDate="2024-01-02 03:00:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepREM" startDate="2024-01-02 03:00:00 -0800" endDate="2024-01-02 05:00:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepDeep" startDate="2024-01-02 05:00:00 -0800" endDate="2024-01-02 07:00:00 -0800"/>
  
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-02 22:30:00 -0800" endDate="2024-01-03 02:30:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepREM" startDate="2024-01-03 02:30:00 -0800" endDate="2024-01-03 06:30:00 -0800"/>
  
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-03 23:15:00 -0800" endDate="2024-01-04 03:15:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepDeep" startDate="2024-01-04 03:15:00 -0800" endDate="2024-01-04 06:45:00 -0800"/>
  
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-04 23:00:00 -0800" endDate="2024-01-05 07:00:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-05 22:45:00 -0800" endDate="2024-01-06 06:45:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-06 23:30:00 -0800" endDate="2024-01-07 07:30:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-07 23:00:00 -0800" endDate="2024-01-08 07:00:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-08 22:30:00 -0800" endDate="2024-01-09 06:30:00 -0800"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch" value="HKCategoryValueSleepAnalysisAsleepCore" startDate="2024-01-09 23:00:00 -0800" endDate="2024-01-10 07:00:00 -0800"/>
  
  <!-- Activity data -->
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="8000" startDate="2024-01-02 08:00:00 -0800" endDate="2024-01-02 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="12000" startDate="2024-01-03 08:00:00 -0800" endDate="2024-01-03 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="6000" startDate="2024-01-04 08:00:00 -0800" endDate="2024-01-04 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="9000" startDate="2024-01-05 08:00:00 -0800" endDate="2024-01-05 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="7500" startDate="2024-01-06 08:00:00 -0800" endDate="2024-01-06 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="10000" startDate="2024-01-07 08:00:00 -0800" endDate="2024-01-07 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="8500" startDate="2024-01-08 08:00:00 -0800" endDate="2024-01-08 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="11000" startDate="2024-01-09 08:00:00 -0800" endDate="2024-01-09 18:00:00 -0800" unit="count"/>
  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" value="9500" startDate="2024-01-10 08:00:00 -0800" endDate="2024-01-10 18:00:00 -0800" unit="count"/>
  
  <!-- Heart rate data -->
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="65" startDate="2024-01-02 06:00:00 -0800" endDate="2024-01-02 06:01:00 -0800" unit="count/min"/>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="70" startDate="2024-01-03 06:00:00 -0800" endDate="2024-01-03 06:01:00 -0800" unit="count/min"/>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="68" startDate="2024-01-04 06:00:00 -0800" endDate="2024-01-04 06:01:00 -0800" unit="count/min"/>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="66" startDate="2024-01-05 06:00:00 -0800" endDate="2024-01-05 06:01:00 -0800" unit="count/min"/>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="64" startDate="2024-01-06 06:00:00 -0800" endDate="2024-01-06 06:01:00 -0800" unit="count/min"/>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="67" startDate="2024-01-07 06:00:00 -0800" endDate="2024-01-07 06:01:00 -0800" unit="count/min"/>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="69" startDate="2024-01-08 06:00:00 -0800" endDate="2024-01-08 06:01:00 -0800" unit="count/min"/>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="65" startDate="2024-01-09 06:00:00 -0800" endDate="2024-01-09 06:01:00 -0800" unit="count/min"/>
  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Apple Watch" value="66" startDate="2024-01-10 06:00:00 -0800" endDate="2024-01-10 06:01:00 -0800" unit="count/min"/>
</HealthData>"""
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_extract_advanced_features_from_xml(self, sample_xml_file):
        """Test extracting 36 research-based features from XML data."""
        # Parse data
        parser = UnifiedHealthDataParser()
        parser.add_xml_export(sample_xml_file)

        all_records = parser.get_all_records()
        sleep_records = all_records['sleep']
        activity_records = all_records['activity']
        heart_records = all_records['heart_rate']

        # Extract advanced features
        service = FeatureExtractionService()
        advanced_features = service.extract_advanced_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            lookback_days=30,
        )

        # Verify we have features
        assert len(advanced_features) > 0

        # Check a recent date
        recent_dates = sorted(advanced_features.keys(), reverse=True)
        if recent_dates:
            latest_date = recent_dates[0]
            features = advanced_features[latest_date]

            # Verify feature structure
            assert features.date == latest_date

            # Check ML features
            ml_features = features.to_ml_features()
            assert len(ml_features) == 36  # Seoul study requirement

            # Print sample features for demonstration
            print(f"\n📊 Advanced Features for {latest_date}:")
            print(f"  Sleep Duration: {features.sleep_duration_hours:.1f} hours")
            print(f"  Sleep Regularity Index: {features.sleep_regularity_index:.1f}")
            print(f"  Circadian Phase Delay: {features.circadian_phase_delay:.1f} hours")
            print(f"  Activity Level: {features.total_steps} steps")
            print(f"  Mood Risk Score: {features.mood_risk_score:.2f}")

            # Check clinical indicators
            if features.is_hypersomnia_pattern:
                print("  ⚠️ Hypersomnia pattern detected")
            if features.is_insomnia_pattern:
                print("  ⚠️ Insomnia pattern detected")
            if features.is_phase_delayed:
                print("  ⚠️ Circadian phase delay detected")
            if features.is_irregular_pattern:
                print("  ⚠️ Irregular sleep pattern detected")

    def test_advanced_features_clinical_relevance(self, sample_xml_file):
        """Test that advanced features capture clinically relevant patterns."""
        # Parse data
        parser = UnifiedHealthDataParser()
        parser.add_xml_export(sample_xml_file)

        all_records = parser.get_all_records()
        sleep_records = all_records['sleep']
        activity_records = all_records['activity']
        heart_records = all_records['heart_rate']

        # Extract features
        service = FeatureExtractionService()

        # Get both basic and advanced features
        basic_features = service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
        )

        advanced_features = service.extract_advanced_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            lookback_days=30,
        )

        # Compare feature richness
        if basic_features and advanced_features:
            basic_date = max(basic_features.keys())
            adv_date = max(advanced_features.keys())

            if basic_date in basic_features and adv_date in advanced_features:
                basic = basic_features[basic_date]
                advanced = advanced_features[adv_date]

                print("\n📊 Feature Comparison:")
                print(f"Basic features: {len(basic.to_feature_vector())} features")
                print(f"Advanced features: {len(advanced.to_ml_features())} features")

                # Advanced features provide much richer information
                assert len(advanced.to_ml_features()) > len(basic.to_feature_vector())

    def test_temporal_feature_evolution(self, sample_xml_file):
        """Test how features evolve over time."""
        # Parse data
        parser = UnifiedHealthDataParser()
        parser.add_xml_export(sample_xml_file)

        all_records = parser.get_all_records()
        sleep_records = all_records['sleep']
        activity_records = all_records['activity']
        heart_records = all_records['heart_rate']

        # Extract features
        service = FeatureExtractionService()
        advanced_features = service.extract_advanced_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            lookback_days=30,
        )

        # Look at feature evolution over a week
        if len(advanced_features) >= 7:
            sorted_dates = sorted(advanced_features.keys())[-7:]

            print("\n📈 Weekly Feature Evolution:")
            print("Date       | Sleep(hrs) | Regularity | Risk Score")
            print("-" * 50)

            for d in sorted_dates:
                f = advanced_features[d]
                print(f"{d} | {f.sleep_duration_hours:10.1f} | "
                      f"{f.sleep_regularity_index:10.1f} | "
                      f"{f.mood_risk_score:10.2f}")

            # Check that temporal features capture variability
            risk_scores = [advanced_features[d].mood_risk_score for d in sorted_dates]
            assert max(risk_scores) != min(risk_scores), "Risk scores should vary over time"
