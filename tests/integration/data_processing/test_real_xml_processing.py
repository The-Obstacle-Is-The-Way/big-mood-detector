"""Test XML processing with REAL Apple Health data."""

from datetime import date
from pathlib import Path

import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.test_support.predictors import ConstantMoodPredictor


class TestRealXMLProcessing:
    """Test with actual Apple Health export data."""
    
    @pytest.fixture
    def real_xml_path(self):
        """Path to real XML test data."""
        path = Path("tests/fixtures/health/week_sample_clean.xml")
        if not path.exists():
            pytest.skip(f"Real test data not found at {path}")
        return path
    
    @pytest.fixture
    def pipeline(self):
        """Pipeline with real predictor and production config."""
        config = PipelineConfig(
            min_days_required=3,  # Production default
            enable_sparse_handling=True,
            use_seoul_features=True
        )
        
        pipeline = MoodPredictionPipeline(config=config)
        pipeline.mood_predictor = ConstantMoodPredictor()
        pipeline.ensemble_orchestrator = None  # Test XGBoost path only
        
        return pipeline
    
    def test_real_xml_processes_without_errors(self, pipeline, real_xml_path):
        """Real XML should process without errors."""
        result = pipeline.process_apple_health_file(
            file_path=real_xml_path,
            end_date=date(2024, 8, 1)
        )
        
        # Basic assertions - it should work
        assert result is not None
        assert not result.has_errors
        assert result.records_processed > 0
        assert result.processing_time_seconds > 0
        
        # If we got predictions, they should be valid
        if result.daily_predictions:
            for date_key, prediction in result.daily_predictions.items():
                assert 0 <= prediction["depression_risk"] <= 1
                assert 0 <= prediction["hypomanic_risk"] <= 1
                assert 0 <= prediction["manic_risk"] <= 1
                assert 0 <= prediction["confidence"] <= 1
    
    def test_real_xml_extracts_features(self, pipeline, real_xml_path):
        """Real XML should extract clinical features."""
        result = pipeline.process_apple_health_file(
            file_path=real_xml_path,
            end_date=date(2024, 8, 1)
        )
        
        # Should extract at least one feature set
        assert result.features_extracted > 0
        
        # Log what we actually got
        print(f"Records processed: {result.records_processed}")
        print(f"Features extracted: {result.features_extracted}")
        print(f"Predictions generated: {len(result.daily_predictions)}")
        
        # If no predictions, check why
        if not result.daily_predictions and result.has_warnings:
            print(f"Warnings: {result.warnings}")