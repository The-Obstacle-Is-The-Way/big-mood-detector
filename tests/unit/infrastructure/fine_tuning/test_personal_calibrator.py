"""
Test Personal Calibrator

TDD for user-level adaptation and baseline extraction.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


class TestPersonalCalibrator:
    """Test personal calibration pipeline."""

    def test_calibrator_can_be_imported(self):
        """Test that calibrator can be imported."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            PersonalCalibrator,
            BaselineExtractor,
            EpisodeLabeler,
        )

        assert PersonalCalibrator is not None
        assert BaselineExtractor is not None
        assert EpisodeLabeler is not None

    def test_baseline_extractor_initialization(self):
        """Test baseline extractor initialization."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            BaselineExtractor,
        )

        extractor = BaselineExtractor(
            baseline_window_days=30,
            min_data_days=14,
        )

        assert extractor.baseline_window_days == 30
        assert extractor.min_data_days == 14

    def test_extract_sleep_baseline(self):
        """Test extracting sleep baseline from health data."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            BaselineExtractor,
        )

        # Create sample sleep data
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        sleep_data = pd.DataFrame({
            "date": dates,
            "sleep_duration": np.random.normal(420, 30, 30),  # 7h ± 30min
            "sleep_efficiency": np.random.uniform(0.8, 0.95, 30),
            "sleep_onset": np.random.normal(23, 1, 30),  # 11pm ± 1h
        })

        extractor = BaselineExtractor()
        baseline = extractor.extract_sleep_baseline(sleep_data)

        assert "mean_sleep_duration" in baseline
        assert "std_sleep_duration" in baseline
        assert "mean_sleep_efficiency" in baseline
        assert "mean_sleep_onset" in baseline
        
        # Check values are reasonable
        assert 300 < baseline["mean_sleep_duration"] < 600  # 5-10 hours
        assert 0 < baseline["std_sleep_duration"] < 60  # < 1 hour std

    def test_extract_activity_baseline(self):
        """Test extracting activity baseline patterns."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            BaselineExtractor,
        )

        # Create sample activity data (minute-level)
        n_days = 30
        minutes_per_day = 1440
        activity_data = []
        
        for day in range(n_days):
            # Simulate daily pattern
            daily_activity = np.zeros(minutes_per_day)
            # Active hours (8am-10pm)
            active_start = 8 * 60
            active_end = 22 * 60
            daily_activity[active_start:active_end] = np.random.exponential(100, active_end - active_start)
            
            for minute in range(minutes_per_day):
                activity_data.append({
                    "date": pd.Timestamp("2024-01-01") + timedelta(days=day, minutes=minute),
                    "activity": daily_activity[minute],
                })
        
        activity_df = pd.DataFrame(activity_data)
        
        extractor = BaselineExtractor()
        baseline = extractor.extract_activity_baseline(activity_df)

        assert "mean_daily_activity" in baseline
        assert "activity_rhythm" in baseline
        assert "peak_activity_time" in baseline
        assert "activity_amplitude" in baseline

    def test_calculate_circadian_baseline(self):
        """Test calculating circadian rhythm baseline."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            BaselineExtractor,
        )

        # Create activity with circadian pattern
        dates = pd.date_range("2024-01-01", periods=14, freq="D")
        circadian_data = []
        
        for date in dates:
            # 24-hour pattern
            for hour in range(24):
                # Simulate circadian rhythm (low at night, high during day)
                if 8 <= hour <= 20:
                    activity = np.random.normal(500, 100)
                else:
                    activity = np.random.normal(50, 20)
                
                circadian_data.append({
                    "timestamp": date + timedelta(hours=hour),
                    "activity": activity,
                })
        
        circadian_df = pd.DataFrame(circadian_data)
        
        extractor = BaselineExtractor()
        baseline = extractor.calculate_circadian_baseline(circadian_df)

        assert "circadian_phase" in baseline
        assert "circadian_amplitude" in baseline
        assert "circadian_stability" in baseline

    def test_episode_labeler_initialization(self):
        """Test episode labeler initialization."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            EpisodeLabeler,
        )

        labeler = EpisodeLabeler()
        
        assert labeler.episodes == []
        assert labeler.baseline_periods == []

    def test_add_episode_label(self):
        """Test adding episode labels."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            EpisodeLabeler,
        )

        labeler = EpisodeLabeler()
        
        # Add single day episode
        labeler.add_episode(
            date="2024-03-15",
            episode_type="hypomanic",
            severity=3,
            notes="Decreased sleep, increased energy",
        )
        
        assert len(labeler.episodes) == 1
        assert labeler.episodes[0]["episode_type"] == "hypomanic"
        assert labeler.episodes[0]["severity"] == 3
        
        # Add date range episode
        labeler.add_episode(
            start_date="2024-03-20",
            end_date="2024-03-25",
            episode_type="depressive",
            severity=4,
        )
        
        assert len(labeler.episodes) == 2
        assert labeler.episodes[1]["duration_days"] == 6

    def test_add_baseline_period(self):
        """Test marking baseline (stable) periods."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            EpisodeLabeler,
        )

        labeler = EpisodeLabeler()
        labeler.add_baseline(
            start_date="2024-01-01",
            end_date="2024-02-01",
            notes="Stable period, no episodes",
        )
        
        assert len(labeler.baseline_periods) == 1
        assert labeler.baseline_periods[0]["duration_days"] == 32

    def test_export_labels_to_dataframe(self):
        """Test exporting labels to DataFrame."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            EpisodeLabeler,
        )

        labeler = EpisodeLabeler()
        
        # Add various labels
        labeler.add_episode("2024-03-15", "hypomanic", 3)
        labeler.add_episode("2024-03-20", "2024-03-25", "depressive", 4)
        labeler.add_baseline("2024-01-01", "2024-02-01")
        
        # Export to DataFrame
        labels_df = labeler.to_dataframe()
        
        assert len(labels_df) == 39  # 1 + 6 + 32 days
        assert "date" in labels_df.columns
        assert "label" in labels_df.columns
        assert "severity" in labels_df.columns
        
        # Check specific labels
        hypomanic_days = labels_df[labels_df["label"] == "hypomanic"]
        assert len(hypomanic_days) == 1
        
        baseline_days = labels_df[labels_df["label"] == "baseline"]
        assert len(baseline_days) == 32

    def test_personal_calibrator_initialization(self):
        """Test personal calibrator initialization."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            PersonalCalibrator,
        )

        calibrator = PersonalCalibrator(
            user_id="test_user",
            model_type="pat",
            base_model_path="models/population/pat_depression.pt",
        )

        assert calibrator.user_id == "test_user"
        assert calibrator.model_type == "pat"
        assert calibrator.base_model_path == "models/population/pat_depression.pt"

    @patch("big_mood_detector.infrastructure.fine_tuning.personal_calibrator.load_population_model")
    def test_pat_personal_calibration(self, mock_load):
        """Test PAT model personal calibration with LoRA."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            PersonalCalibrator,
        )

        # Mock population model
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.random.rand(100, 768))
        mock_load.return_value = mock_model

        calibrator = PersonalCalibrator(model_type="pat")
        
        # Create sample personal data
        sequences = np.random.rand(100, 60)  # 100 sequences of 60 minutes
        labels = np.array([0] * 50 + [1] * 50)  # 50 baseline, 50 episode
        
        # Calibrate
        metrics = calibrator.calibrate(
            sequences=sequences,
            labels=labels,
            epochs=2,
        )

        assert "accuracy" in metrics
        assert "personal_improvement" in metrics
        assert calibrator.adapter is not None

    @patch("joblib.load")
    def test_xgboost_personal_calibration(self, mock_load):
        """Test XGBoost incremental calibration."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            PersonalCalibrator,
        )
        import xgboost as xgb

        # Mock pre-trained XGBoost
        base_model = xgb.XGBClassifier(n_estimators=10)
        # Pre-train on dummy data
        X_dummy = np.random.rand(50, 36)
        y_dummy = np.random.randint(0, 2, 50)
        base_model.fit(X_dummy, y_dummy)
        mock_load.return_value = base_model

        calibrator = PersonalCalibrator(
            model_type="xgboost",
            base_model_path="models/population/xgboost_depression.pkl",
        )
        
        # Personal features
        features = pd.DataFrame({
            f"feature_{i}": np.random.rand(100)
            for i in range(36)
        })
        labels = np.array([0] * 70 + [1] * 30)
        
        # Calibrate with higher weight on personal data
        metrics = calibrator.calibrate(
            features=features,
            labels=labels,
            sample_weight=2.0,
        )

        assert "accuracy" in metrics
        assert "n_trees_added" in metrics

    def test_save_personal_model(self, tmp_path):
        """Test saving calibrated personal model."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            PersonalCalibrator,
        )

        calibrator = PersonalCalibrator(
            user_id="test_user",
            output_dir=tmp_path,
        )
        
        # Mock calibrated components
        calibrator.adapter = Mock()
        calibrator.baseline = {"mean_sleep_duration": 420}
        
        # Save
        save_path = calibrator.save_model(
            metrics={"accuracy": 0.88},
        )

        assert save_path.exists()
        assert "test_user" in save_path.name
        
        # Check metadata saved
        metadata_path = tmp_path / "users" / "test_user" / "metadata.json"
        assert metadata_path.exists()

    def test_load_personal_model(self, tmp_path):
        """Test loading saved personal model."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            PersonalCalibrator,
        )

        # Create mock saved model
        user_dir = tmp_path / "users" / "test_user"
        user_dir.mkdir(parents=True)
        
        import json
        metadata = {
            "user_id": "test_user",
            "model_type": "pat",
            "baseline": {"mean_sleep_duration": 420},
            "calibration_date": "2024-03-15",
        }
        
        with open(user_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Load
        calibrator = PersonalCalibrator.load(
            user_id="test_user",
            model_dir=tmp_path,
        )

        assert calibrator.user_id == "test_user"
        assert calibrator.baseline["mean_sleep_duration"] == 420

    def test_deviation_features(self):
        """Test calculating deviation from personal baseline."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            PersonalCalibrator,
        )

        calibrator = PersonalCalibrator()
        calibrator.baseline = {
            "mean_sleep_duration": 420,
            "std_sleep_duration": 30,
            "mean_daily_activity": 50000,
        }
        
        # Current features
        current = {
            "sleep_duration": 360,  # 1 hour less than baseline
            "daily_activity": 60000,  # 20% more active
        }
        
        deviations = calibrator.calculate_deviations(current)
        
        assert deviations["sleep_duration_z_score"] == -2.0  # (360-420)/30
        assert deviations["activity_percent_change"] == 20.0

    def test_confidence_calibration(self):
        """Test probability calibration for personal models."""
        from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
            PersonalCalibrator,
        )

        calibrator = PersonalCalibrator()
        
        # Raw probabilities (tend to be overconfident)
        raw_probs = np.array([0.9, 0.95, 0.1, 0.05, 0.7, 0.3])
        true_labels = np.array([1, 0, 0, 0, 1, 0])  # Some wrong
        
        # Calibrate
        calibrator.fit_calibration(raw_probs, true_labels)
        calibrated_probs = calibrator.calibrate_probabilities(raw_probs)
        
        # Should be less extreme
        assert calibrated_probs[0] < raw_probs[0]  # Was too confident
        assert calibrated_probs[3] > raw_probs[3]  # Was too confident