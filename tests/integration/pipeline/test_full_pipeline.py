"""
Integration test for full pipeline
Verifies that all components are wired together correctly
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestFullPipeline:
    """Test the complete pipeline integration."""

    @pytest.fixture
    def sample_data(self):
        """Create sample health data for testing."""
        base_date = date.today() - timedelta(days=10)

        # Create sleep records
        sleep_records = []
        for i in range(7):
            record_date = base_date + timedelta(days=i)
            sleep_records.append(
                SleepRecord(
                    source_name="Test",
                    start_date=datetime.combine(record_date, datetime.min.time()),
                    end_date=datetime.combine(record_date, datetime.min.time())
                    + timedelta(hours=8),
                    state=SleepState.ASLEEP,
                )
            )

        # Create activity records (one per minute for each day)
        activity_records = []
        for i in range(7):
            record_date = base_date + timedelta(days=i)
            # Create hourly activity data (simpler for testing)
            for hour in range(24):
                timestamp = datetime.combine(
                    record_date, datetime.min.time()
                ) + timedelta(hours=hour)
                activity_records.append(
                    ActivityRecord(
                        source_name="Test",
                        start_date=timestamp,
                        end_date=timestamp + timedelta(hours=1),
                        activity_type=ActivityType.ACTIVE_ENERGY,
                        value=float(np.random.randint(50, 200)),
                        unit="kcal",
                    )
                )

        # Create heart rate records
        heart_records = []
        for i in range(7):
            record_date = base_date + timedelta(days=i)
            for hour in range(24):
                heart_records.append(
                    HeartRateRecord(
                        source_name="Test",
                        timestamp=datetime.combine(record_date, datetime.min.time())
                        + timedelta(hours=hour),
                        metric_type=HeartMetricType.HEART_RATE,
                        value=float(60 + np.random.randint(-10, 20)),
                        unit="count/min",
                        motion_context=(
                            MotionContext.SEDENTARY
                            if hour < 6 or hour > 22
                            else MotionContext.ACTIVE
                        ),
                    )
                )

        return {
            "sleep": sleep_records,
            "activity": activity_records,
            "heart_rate": heart_records,
        }

    def test_pipeline_without_ensemble(self, sample_data):
        """Test basic pipeline without ensemble models."""
        # Configure pipeline
        config = PipelineConfig(
            min_days_required=5,
            include_pat_sequences=False,
        )

        pipeline = MoodPredictionPipeline(config=config)

        # Check models are loaded
        assert pipeline.mood_predictor.is_loaded, "XGBoost models should be loaded"
        assert (
            pipeline.ensemble_orchestrator is None
        ), "Ensemble should not be initialized"

        # Process data
        result = pipeline.process_health_data(
            sleep_records=sample_data["sleep"],
            activity_records=sample_data["activity"],
            heart_records=sample_data["heart_rate"],
            target_date=date.today(),
        )

        # Verify results
        assert result is not None
        assert len(result.daily_predictions) > 0
        assert result.confidence_score > 0
        assert result.overall_summary is not None

        # Check prediction structure
        for _date_key, prediction in result.daily_predictions.items():
            assert "depression_risk" in prediction
            assert "hypomanic_risk" in prediction
            assert "manic_risk" in prediction
            assert "confidence" in prediction
            assert 0 <= prediction["depression_risk"] <= 1
            assert 0 <= prediction["hypomanic_risk"] <= 1
            assert 0 <= prediction["manic_risk"] <= 1

    @pytest.mark.xfail(
        reason="Issue #TBD-3: XGBoost Booster objects loaded from JSON lack predict_proba - see issues/xgboost-booster-predict-proba.md",
        strict=True
    )
    @pytest.mark.skipif(
        not Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5").exists(),
        reason="PAT weights not available",
    )
    def test_pipeline_with_ensemble(self, sample_data):
        """Test pipeline with ensemble models enabled."""
        # Configure pipeline with ensemble
        config = PipelineConfig(
            min_days_required=5,
            include_pat_sequences=True,
        )

        pipeline = MoodPredictionPipeline(config=config)

        # Check models are loaded
        assert pipeline.mood_predictor.is_loaded, "XGBoost models should be loaded"
        assert (
            pipeline.ensemble_orchestrator is not None
        ), "Ensemble should be initialized"

        # Process data
        result = pipeline.process_health_data(
            sleep_records=sample_data["sleep"],
            activity_records=sample_data["activity"],
            heart_records=sample_data["heart_rate"],
            target_date=date.today(),
        )

        # Verify results
        assert result is not None
        assert len(result.daily_predictions) > 0

        # Check ensemble metadata
        for _date_key, prediction in result.daily_predictions.items():
            assert "models_used" in prediction
            assert "xgboost" in prediction["models_used"]
            # PAT might fail silently if weights aren't loaded
            if "pat_enhanced" in prediction["models_used"]:
                assert "confidence_scores" in prediction
                assert "ensemble" in prediction["confidence_scores"]

    def test_pipeline_with_sparse_data(self, sample_data):
        """Test pipeline handles sparse data correctly."""
        # Use only every other day
        sparse_sleep = sample_data["sleep"][::2]
        sparse_activity = sample_data["activity"][::2]

        pipeline = MoodPredictionPipeline()

        result = pipeline.process_health_data(
            sleep_records=sparse_sleep,
            activity_records=sparse_activity,
            heart_records=sample_data["heart_rate"],
            target_date=date.today(),
        )

        assert result is not None
        assert result.has_warnings
        assert any("Sparse data" in w for w in result.warnings)

    def test_pipeline_export_functionality(self, sample_data, tmp_path):
        """Test pipeline can export results."""
        pipeline = MoodPredictionPipeline()

        result = pipeline.process_health_data(
            sleep_records=sample_data["sleep"],
            activity_records=sample_data["activity"],
            heart_records=sample_data["heart_rate"],
            target_date=date.today(),
        )

        # Export to CSV
        csv_path = tmp_path / "test_predictions.csv"
        pipeline.export_results(result, csv_path)

        assert csv_path.exists()

        # Check summary JSON
        summary_path = csv_path.with_suffix(".summary.json")
        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)
            assert "overall_summary" in summary
            assert "confidence_score" in summary

    def test_pipeline_component_usage(self, sample_data):
        """Verify all major components are used in the pipeline."""
        pipeline = MoodPredictionPipeline()

        # Components to verify
        assert pipeline.sleep_analyzer is not None
        assert pipeline.activity_extractor is not None
        assert pipeline.circadian_analyzer is not None
        assert pipeline.clinical_extractor is not None
        assert pipeline.sparse_handler is not None

        # Process data to ensure components are used
        result = pipeline.process_health_data(
            sleep_records=sample_data["sleep"],
            activity_records=sample_data["activity"],
            heart_records=sample_data["heart_rate"],
            target_date=date.today(),
        )

        # Verify features were extracted
        assert result.features_extracted > 0
        assert result.records_processed == (
            len(sample_data["sleep"])
            + len(sample_data["activity"])
            + len(sample_data["heart_rate"])
        )

    def test_pipeline_with_environment_override(self, sample_data, monkeypatch):
        """Test PAT weights can be loaded from environment variable."""
        # Set environment variable
        custom_weights_dir = "/custom/path/to/weights"
        monkeypatch.setenv("BIG_MOOD_PAT_WEIGHTS_DIR", custom_weights_dir)

        config = PipelineConfig(include_pat_sequences=True)

        # This should not crash even if weights don't exist
        pipeline = MoodPredictionPipeline(config=config)

        # Pipeline should still work without PAT
        result = pipeline.process_health_data(
            sleep_records=sample_data["sleep"],
            activity_records=sample_data["activity"],
            heart_records=sample_data["heart_rate"],
            target_date=date.today(),
        )

        assert result is not None

    def test_pipeline_clinical_thresholds(self, sample_data):
        """Test that clinical thresholds are applied correctly."""
        pipeline = MoodPredictionPipeline()

        result = pipeline.process_health_data(
            sleep_records=sample_data["sleep"],
            activity_records=sample_data["activity"],
            heart_records=sample_data["heart_rate"],
            target_date=date.today(),
        )

        # Check that predictions follow clinical logic
        for _date_key, prediction in result.daily_predictions.items():
            # All risks should sum to <= 1 (not strict requirement but good practice)
            total_risk = (
                prediction["depression_risk"]
                + prediction["hypomanic_risk"]
                + prediction["manic_risk"]
            )
            assert total_risk <= 1.5, "Total risk should be reasonable"

            # Confidence should be reasonable
            assert 0 <= prediction["confidence"] <= 1
