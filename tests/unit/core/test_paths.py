"""
Test core path configuration.

Ensures paths are correctly resolved regardless of where tests are run from.
"""


class TestPaths:
    """Test path resolution."""

    def test_project_root_exists(self):
        """Test that PROJECT_ROOT points to actual project root."""
        from big_mood_detector.core.paths import PROJECT_ROOT
        
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

        # Should contain key project markers
        assert (PROJECT_ROOT / "pyproject.toml").exists()
        assert (PROJECT_ROOT / "src").exists()
        assert (PROJECT_ROOT / "tests").exists()

    def test_model_weights_dir_exists(self):
        """Test that model weights directory exists."""
        from big_mood_detector.core.paths import MODEL_WEIGHTS_DIR
        
        assert MODEL_WEIGHTS_DIR.exists()
        assert MODEL_WEIGHTS_DIR.is_dir()

        # Should contain model subdirectories
        assert (MODEL_WEIGHTS_DIR / "xgboost").exists()
        assert (MODEL_WEIGHTS_DIR / "xgboost" / "converted").exists()

    def test_data_dir_created(self):
        """Test that data directory is created if missing."""
        from big_mood_detector.core.paths import DATA_DIR
        
        assert DATA_DIR.exists()
        assert DATA_DIR.is_dir()

    def test_logs_dir_created(self):
        """Test that logs directory is created if missing."""
        from big_mood_detector.core.paths import LOGS_DIR
        
        assert LOGS_DIR.exists()
        assert LOGS_DIR.is_dir()

    def test_paths_are_absolute(self):
        """Test that all paths are absolute."""
        from big_mood_detector.core.paths import (
            DATA_DIR,
            LOGS_DIR,
            MODEL_WEIGHTS_DIR,
            PROJECT_ROOT,
        )
        
        assert PROJECT_ROOT.is_absolute()
        assert MODEL_WEIGHTS_DIR.is_absolute()
        assert DATA_DIR.is_absolute()
        assert LOGS_DIR.is_absolute()

    def test_paths_resolve_correctly_from_different_locations(self):
        """Test that paths work when imported from different locations."""
        from big_mood_detector.core.paths import (
            DATA_DIR,
            LOGS_DIR,
            MODEL_WEIGHTS_DIR,
            PROJECT_ROOT,
        )
        
        # This is implicitly tested by running tests from different directories
        # but we can verify the parent relationships
        assert MODEL_WEIGHTS_DIR.parent == PROJECT_ROOT
        assert DATA_DIR.parent == PROJECT_ROOT
        assert LOGS_DIR.parent == PROJECT_ROOT
