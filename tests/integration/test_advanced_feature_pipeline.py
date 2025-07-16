"""
Integration test for advanced feature engineering pipeline

Demonstrates end-to-end feature extraction for mood prediction.
"""

import pytest
from datetime import date, timedelta

from big_mood_detector.domain.services.feature_extraction_service import (
    FeatureExtractionService,
)
from big_mood_detector.infrastructure.parsers.parser_factory import (
    UnifiedHealthDataParser,
)


class TestAdvancedFeaturePipeline:
    """Test complete pipeline with advanced features."""

    def test_extract_advanced_features_from_xml(self, sample_xml_file):
        """Test extracting 36 research-based features from XML data."""
        # Parse data
        parser = UnifiedHealthDataParser()
        parser.add_xml_export(sample_xml_file)
        
        sleep_records = parser.get_all_sleep_records()
        activity_records = parser.get_all_activity_records()
        heart_records = parser.get_all_heart_rate_records()
        
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
        
        sleep_records = parser.get_all_sleep_records()
        activity_records = parser.get_all_activity_records()
        heart_records = parser.get_all_heart_rate_records()
        
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
                
                print(f"\n📊 Feature Comparison:")
                print(f"Basic features: {len(basic.to_feature_vector())} features")
                print(f"Advanced features: {len(advanced.to_ml_features())} features")
                
                # Advanced features provide much richer information
                assert len(advanced.to_ml_features()) > len(basic.to_feature_vector())

    def test_temporal_feature_evolution(self, sample_xml_file):
        """Test how features evolve over time."""
        # Parse data
        parser = UnifiedHealthDataParser()
        parser.add_xml_export(sample_xml_file)
        
        sleep_records = parser.get_all_sleep_records()
        activity_records = parser.get_all_activity_records()
        heart_records = parser.get_all_heart_rate_records()
        
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
            
            print(f"\n📈 Weekly Feature Evolution:")
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