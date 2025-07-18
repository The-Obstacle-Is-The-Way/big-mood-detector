import sys
import types
import numpy as np
import pandas as pd
from click.testing import CliRunner
from unittest.mock import patch

# Stub out heavy PAT dependencies before importing the CLI
pat_stub = types.ModuleType("pat_model")
pat_stub.PATModel = object
pat_stub.PATFeatureExtractor = object
sys.modules["big_mood_detector.infrastructure.ml_models.pat_model"] = pat_stub

from big_mood_detector.main_cli import cli  # type: ignore


def _create_csv(path, col_name, data):
    pd.DataFrame({col_name: data}).to_csv(path, index=False)


@patch("big_mood_detector.infrastructure.fine_tuning.personal_calibrator.PersonalCalibrator")
def test_train_command_xgboost(mock_calibrator, tmp_path):
    features_file = tmp_path / "features.csv"
    labels_file = tmp_path / "labels.csv"
    _create_csv(features_file, "f1", [1, 2])
    _create_csv(labels_file, "label", [0, 1])

    instance = mock_calibrator.return_value

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train",
            "--model-type",
            "xgboost",
            "--user-id",
            "user1",
            "--data",
            str(features_file),
            "--labels",
            str(labels_file),
        ],
    )
    assert result.exit_code == 0
    mock_calibrator.assert_called_once_with(user_id="user1", model_type="xgboost")
    instance.calibrate.assert_called_once()
    instance.save_model.assert_called_once()


@patch("big_mood_detector.infrastructure.fine_tuning.personal_calibrator.PersonalCalibrator")
def test_train_command_pat(mock_calibrator, tmp_path):
    seq_file = tmp_path / "seq.npy"
    labels_file = tmp_path / "labels.npy"
    np.save(seq_file, np.random.rand(2, 60))
    np.save(labels_file, np.array([0, 1]))

    instance = mock_calibrator.return_value

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train",
            "--model-type",
            "pat",
            "--user-id",
            "user2",
            "--data",
            str(seq_file),
            "--labels",
            str(labels_file),
        ],
    )
    assert result.exit_code == 0
    mock_calibrator.assert_called_once_with(user_id="user2", model_type="pat")
    instance.calibrate.assert_called_once()
    instance.save_model.assert_called_once()
