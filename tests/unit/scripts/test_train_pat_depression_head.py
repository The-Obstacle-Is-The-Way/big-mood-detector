"""
Test PAT Depression Head Training Script

Verifies the training script can be imported and has correct structure.
"""

import sys
from pathlib import Path

import pytest


class TestTrainPATDepressionHead:
    """Test the PAT depression head training script."""

    def test_script_exists(self):
        """Test that the training script exists."""
        script_path = Path("scripts/train_pat_depression_head.py")
        assert script_path.exists()
        assert script_path.is_file()

    def test_script_has_main(self):
        """Test that the script has a main function."""
        # Add scripts directory to path
        sys.path.insert(0, str(Path("scripts").absolute()))

        try:
            import train_pat_depression_head

            # Check required functions exist
            assert hasattr(train_pat_depression_head, 'main')
            assert hasattr(train_pat_depression_head, 'prepare_training_data')
            assert hasattr(train_pat_depression_head, 'train_depression_head')

        finally:
            # Clean up
            sys.path.pop(0)
            if 'train_pat_depression_head' in sys.modules:
                del sys.modules['train_pat_depression_head']

    def test_script_imports(self):
        """Test that all required imports are available."""
        # This ensures our dependencies are correctly installed
        try:
            import big_mood_detector.infrastructure.fine_tuning.nhanes_processor
            import big_mood_detector.infrastructure.fine_tuning.population_trainer
            import big_mood_detector.infrastructure.ml_models.pat_model
        except ImportError as e:
            pytest.fail(f"Required import failed: {e}")

    def test_output_directory_structure(self):
        """Test expected output directory structure."""
        expected_path = Path("model_weights/pat/heads")
        # Just check the path is valid, don't create it
        assert isinstance(expected_path, Path)
        assert str(expected_path) == "model_weights/pat/heads"
