"""
Test Data Structure Consolidation

Tests the elimination of manual copying between DailyFeatures and ClinicalFeatureSet.
This addresses the critical dual data structure problem identified in the audit.
"""

import pytest
from datetime import date, timedelta
import numpy as np

from big_mood_detector.application.services.aggregation_pipeline import AggregationPipeline
from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureExtractor
from tests.factories.health_data_factory import create_sample_records


class TestDataStructureConsolidation:
    """Test that aggregation pipeline and clinical extractor use unified data structures."""

    @pytest.fixture
    def sample_data(self):
        """Create sample health data for testing."""
        return create_sample_records()

    def test_aggregation_pipeline_returns_clinical_feature_set(self, sample_data):
        """Test that aggregation pipeline returns ClinicalFeatureSet instead of DailyFeatures."""
        # This test should FAIL initially because aggregation_pipeline returns DailyFeatures
        
        pipeline = AggregationPipeline()
        target_date = date.today() - timedelta(days=1)
        
        # Process sample data
        result = pipeline.extract_daily_features(
            sleep_records=sample_data["sleep_records"],
            activity_records=sample_data["activity_records"],
            heart_rate_records=sample_data["heart_rate_records"],
            target_date=target_date,
        )
        
        # Should return ClinicalFeatureSet, not DailyFeatures
        from big_mood_detector.domain.services.clinical_feature_extractor import ClinicalFeatureSet
        assert isinstance(result, ClinicalFeatureSet), f"Expected ClinicalFeatureSet, got {type(result)}"

    def test_no_manual_copying_between_structures(self, sample_data):
        """Test that there's no manual field copying between data structures."""
        # This will fail initially due to manual copying in aggregation_pipeline.py
        
        # Read the aggregation pipeline source to check for manual copying
        import inspect
        from big_mood_detector.application.services.aggregation_pipeline import AggregationPipeline
        
        source = inspect.getsource(AggregationPipeline)
        
        # Should not contain manual field assignments
        manual_copy_indicators = [
            "DailyFeatures(",
            ".get(",  # dict.get() calls for manual copying
            "activity_metrics.get",
            "total_steps=",
            "activity_variance=",
        ]
        
        for indicator in manual_copy_indicators:
            assert indicator not in source, f"Found manual copying indicator: {indicator}"

    def test_clinical_feature_extractor_consistency(self, sample_data):
        """Test that clinical feature extractor returns consistent structure."""
        extractor = ClinicalFeatureExtractor()
        target_date = date.today() - timedelta(days=1)
        
        result = extractor.extract_clinical_features(
            sleep_records=sample_data["sleep_records"],
            activity_records=sample_data["activity_records"],
            heart_records=sample_data["heart_rate_records"],
            target_date=target_date,
        )
        
        # Activity features should be at top level, not nested in seoul_features
        assert hasattr(result, 'total_steps'), "total_steps should be a direct attribute"
        assert hasattr(result, 'activity_variance'), "activity_variance should be a direct attribute"
        assert hasattr(result, 'sedentary_hours'), "sedentary_hours should be a direct attribute"
        
        # Should not need nested access
        assert result.total_steps is not None, "total_steps should be accessible directly" 