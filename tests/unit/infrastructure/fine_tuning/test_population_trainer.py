"""
Test Population Trainer

TDD for training task-specific heads on NHANES cohorts.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Kill parallelism in ML libraries to prevent segfaults
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


class TestPopulationTrainer:
    """Test population-level fine-tuning for PAT and XGBoost."""

    def test_trainer_can_be_imported(self):
        """Test that trainer can be imported."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            PATPopulationTrainer,
            PopulationTrainer,
            XGBoostPopulationTrainer,
        )

        assert PopulationTrainer is not None
        assert PATPopulationTrainer is not None
        assert XGBoostPopulationTrainer is not None

    def test_pat_trainer_initialization(self):
        """Test PAT trainer initialization with base model."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            PATPopulationTrainer,
        )

        trainer = PATPopulationTrainer(
            base_model_path="weights/PAT-S_29k_weights.h5",
            task_name="depression",
            output_dir=Path("models/population"),
        )

        assert trainer.base_model_path == "weights/PAT-S_29k_weights.h5"
        assert trainer.task_name == "depression"
        assert trainer.output_dir == Path("models/population")

    @patch("big_mood_detector.infrastructure.fine_tuning.population_trainer.load_pat_model")
    def test_pat_model_loading(self, mock_load_model):
        """Test loading pre-trained PAT weights."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            PATPopulationTrainer,
        )

        # Mock model
        mock_model = Mock()
        mock_model.output_dim = 768
        mock_load_model.return_value = mock_model

        trainer = PATPopulationTrainer()
        model = trainer.load_base_model()

        assert model is not None
        assert model.output_dim == 768
        mock_load_model.assert_called_once()

    def test_pat_task_head_creation(self):
        """Test creating task-specific head for PAT."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            PATPopulationTrainer,
        )

        trainer = PATPopulationTrainer()

        # Create task head
        task_head = trainer.create_task_head(
            input_dim=768,
            num_classes=2,  # Binary classification
            dropout=0.2,
        )

        # Check architecture
        assert hasattr(task_head, "layers")
        assert len(task_head.layers) >= 2  # At least hidden + output
        assert task_head.output_dim == 2

    @pytest.mark.slow
    @patch("torch.save")
    def test_pat_training_pipeline(self, mock_save):
        """Test full PAT fine-tuning pipeline."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            PATPopulationTrainer,
        )

        # Create sample data
        n_samples = 1000
        sequences = np.random.rand(n_samples, 60)  # 60-minute windows
        labels = np.random.randint(0, 2, n_samples)  # Binary labels

        trainer = PATPopulationTrainer()

        # Mock the base model
        with patch.object(trainer, "load_base_model") as mock_load:
            mock_model = Mock()
            mock_model.encode = Mock(return_value=np.random.rand(n_samples, 768))
            mock_load.return_value = mock_model

            # Train
            metrics = trainer.fine_tune(
                sequences=sequences,
                labels=labels,
                epochs=2,
                batch_size=32,
                learning_rate=1e-4,
            )

            # Check training completed
            assert "final_loss" in metrics
            assert "final_accuracy" in metrics
            assert metrics["epochs_completed"] == 2

    def test_pat_model_saving(self, tmp_path):
        """Test saving fine-tuned PAT model."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            PATPopulationTrainer,
        )

        trainer = PATPopulationTrainer(output_dir=tmp_path)

        # Mock model components
        mock_encoder = Mock()
        mock_task_head = Mock()

        # Save model
        save_path = trainer.save_model(
            encoder=mock_encoder,
            task_head=mock_task_head,
            task_name="depression",
            metrics={"accuracy": 0.85},
        )

        assert save_path.exists()
        assert save_path.name == "pat_depression.pt"

        # Check metadata saved
        metadata_path = tmp_path / "pat_depression_metadata.json"
        assert metadata_path.exists()

    def test_xgboost_trainer_initialization(self):
        """Test XGBoost trainer initialization."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            XGBoostPopulationTrainer,
        )

        trainer = XGBoostPopulationTrainer(
            base_model_path="mood_ml/XGBoost_DE.pkl",
            task_name="depression",
        )

        assert trainer.base_model_path == "mood_ml/XGBoost_DE.pkl"
        assert trainer.task_name == "depression"

    @patch("joblib.load")
    def test_xgboost_model_loading(self, mock_load):
        """Test loading pre-trained XGBoost model."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            XGBoostPopulationTrainer,
        )

        # Mock XGBoost model
        mock_model = Mock()
        mock_model.n_estimators = 100
        mock_load.return_value = mock_model

        trainer = XGBoostPopulationTrainer(
            base_model_path="mood_ml/XGBoost_DE.pkl"
        )
        model = trainer.load_base_model()

        assert model is not None
        assert model.n_estimators == 100
        mock_load.assert_called_once_with("mood_ml/XGBoost_DE.pkl")

    def test_xgboost_feature_validation(self):
        """Test validating features match mood_ml expectations."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            XGBoostPopulationTrainer,
        )

        trainer = XGBoostPopulationTrainer()

        # Valid features (36 from mood_ml)
        valid_features = pd.DataFrame({
            "mean_sleep_duration": [420],
            "std_sleep_duration": [30],
            "mean_sleep_efficiency": [0.85],
            "IS": [0.7],
            "IV": [0.3],
            "RA": [0.8],
            "L5": [50],
            "M10": [500],
            # ... (would have all 36 in real implementation)
        })

        # Should not raise
        trainer.validate_features(valid_features)

        # Missing features should raise
        invalid_features = pd.DataFrame({"wrong_feature": [1]})
        with pytest.raises(ValueError, match="Missing required features"):
            trainer.validate_features(invalid_features)

    @pytest.mark.slow_finetune
    @patch("joblib.dump")
    def test_xgboost_incremental_training(self, mock_dump):
        """Test incremental training on XGBoost."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            XGBoostPopulationTrainer,
        )

        # Create sample data
        features = pd.DataFrame({
            f"feature_{i}": np.random.rand(100)
            for i in range(36)
        })
        labels = np.random.randint(0, 2, 100)

        trainer = XGBoostPopulationTrainer()

        # Mock base model
        with patch.object(trainer, "load_base_model") as mock_load:
            import xgboost as xgb
            mock_model = xgb.XGBClassifier(n_estimators=10)
            mock_model.fit(features, labels)  # Pre-train
            mock_load.return_value = mock_model

            # Incremental training
            metrics = trainer.incremental_train(
                features=features,
                labels=labels,
                num_boost_round=5,
            )

            assert "accuracy" in metrics
            assert "auc" in metrics
            assert metrics["total_estimators"] > 10  # Added trees

    def test_population_trainer_factory(self):
        """Test factory pattern for creating trainers."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            create_population_trainer,
        )

        # PAT trainer
        pat_trainer = create_population_trainer(
            model_type="pat",
            task_name="depression",
        )
        assert pat_trainer.__class__.__name__ == "PATPopulationTrainer"

        # XGBoost trainer
        xgb_trainer = create_population_trainer(
            model_type="xgboost",
            task_name="depression",
        )
        assert xgb_trainer.__class__.__name__ == "XGBoostPopulationTrainer"

        # Invalid type
        with pytest.raises(ValueError, match="Unknown model type"):
            create_population_trainer(model_type="invalid")

    def test_cross_validation_split(self):
        """Test time-series aware cross-validation."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            PATPopulationTrainer,
        )

        # Create time-indexed data
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "value": np.random.rand(365),
            "label": np.random.randint(0, 2, 365),
        })

        # Use concrete implementation instead of abstract class
        trainer = PATPopulationTrainer()
        splits = trainer.create_time_series_splits(
            data,
            n_splits=3,
            test_size=0.2,
        )

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            # Test always after train (time-series respect)
            assert data.iloc[train_idx]["date"].max() < data.iloc[test_idx]["date"].min()

    def test_model_evaluation_metrics(self):
        """Test comprehensive evaluation metrics."""
        from big_mood_detector.infrastructure.fine_tuning.population_trainer import (
            PATPopulationTrainer,
        )

        # Use concrete implementation instead of abstract class
        trainer = PATPopulationTrainer()

        # Mock predictions
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3, 0.7, 0.9])

        metrics = trainer.evaluate(y_true, y_pred, y_prob)

        # Check all metrics present
        assert "accuracy" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # Check values reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc"] <= 1
