"""
Test Clinical Thresholds Configuration

Tests the loading and validation of clinical thresholds from YAML configuration.
"""

from pathlib import Path

import pytest
import yaml

class TestClinicalThresholdsConfig:
    """Test clinical thresholds configuration loading and validation."""

    @pytest.fixture
    def sample_config_path(self, tmp_path):
        """Create a sample configuration file."""
        config_data = {
            "depression": {
                "phq_cutoffs": {
                    "none": {"min": 0, "max": 4},
                    "mild": {"min": 5, "max": 9},
                    "moderate": {"min": 10, "max": 14},
                    "moderately_severe": {"min": 15, "max": 19},
                    "severe": {"min": 20, "max": 27},
                },
                "sleep_hours": {
                    "hypersomnia_threshold": 12,
                    "normal_min": 6,
                    "normal_max": 9,
                },
                "activity_steps": {
                    "severe_reduction": 2000,
                    "moderate_reduction": 4000,
                    "normal_min": 5000,
                },
            },
            "mania": {
                "asrm_cutoffs": {
                    "none": {"min": 0, "max": 5},
                    "hypomanic": {"min": 6, "max": 10},
                    "manic_moderate": {"min": 11, "max": 15},
                    "manic_severe": {"min": 16, "max": 20},
                },
                "sleep_hours": {"critical_threshold": 3, "reduced_threshold": 5},
                "activity_steps": {
                    "elevated_threshold": 15000,
                    "extreme_threshold": 20000,
                },
            },
            "biomarkers": {
                "circadian": {
                    "phase_advance_threshold": 2.0,
                    "interdaily_stability_low": 0.5,
                    "intradaily_variability_high": 1.0,
                },
                "sleep": {
                    "efficiency_threshold": 0.85,
                    "timing_variance_threshold": 2.0,
                },
            },
            "mixed_features": {
                "minimum_opposite_symptoms": 3,
                "depression_with_mixed": {
                    "required_manic_symptoms": [
                        "racing_thoughts",
                        "increased_energy",
                        "decreased_sleep",
                    ]
                },
                "mania_with_mixed": {
                    "required_depressive_symptoms": [
                        "depressed_mood",
                        "anhedonia",
                        "guilt",
                    ]
                },
            },
            "dsm5_duration": {
                "manic_days": 7,
                "hypomanic_days": 4,
                "depressive_days": 14,
            },
        }

        config_file = tmp_path / "clinical_thresholds.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        return config_file

    def test_load_valid_config(self, sample_config_path):
        """Test loading a valid configuration file."""
        from big_mood_detector.domain.services.clinical_thresholds import (
            BiomarkerThresholds,
            ClinicalThresholdsConfig,
            DepressionThresholds,
            ManiaThresholds,
            load_clinical_thresholds,
        )

        config = load_clinical_thresholds(sample_config_path)

        assert isinstance(config, ClinicalThresholdsConfig)
        assert isinstance(config.depression, DepressionThresholds)
        assert isinstance(config.mania, ManiaThresholds)
        assert isinstance(config.biomarkers, BiomarkerThresholds)

    def test_depression_thresholds(self, sample_config_path):
        """Test depression threshold values are loaded correctly."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        config = load_clinical_thresholds(sample_config_path)

        # PHQ cutoffs
        assert config.depression.phq_cutoffs.moderate.min == 10
        assert config.depression.phq_cutoffs.moderate.max == 14
        assert config.depression.phq_cutoffs.severe.min == 20

        # Sleep thresholds
        assert config.depression.sleep_hours.hypersomnia_threshold == 12
        assert config.depression.sleep_hours.normal_min == 6

        # Activity thresholds
        assert config.depression.activity_steps.severe_reduction == 2000
        assert config.depression.activity_steps.moderate_reduction == 4000

    def test_mania_thresholds(self, sample_config_path):
        """Test mania threshold values are loaded correctly."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        config = load_clinical_thresholds(sample_config_path)

        # ASRM cutoffs
        assert config.mania.asrm_cutoffs.hypomanic.min == 6
        assert config.mania.asrm_cutoffs.hypomanic.max == 10
        assert config.mania.asrm_cutoffs.manic_severe.min == 16

        # Sleep thresholds
        assert config.mania.sleep_hours.critical_threshold == 3
        assert config.mania.sleep_hours.reduced_threshold == 5

        # Activity thresholds
        assert config.mania.activity_steps.elevated_threshold == 15000
        assert config.mania.activity_steps.extreme_threshold == 20000

    def test_biomarker_thresholds(self, sample_config_path):
        """Test biomarker threshold values are loaded correctly."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        config = load_clinical_thresholds(sample_config_path)

        # Circadian thresholds
        assert config.biomarkers.circadian.phase_advance_threshold == 2.0
        assert config.biomarkers.circadian.interdaily_stability_low == 0.5
        assert config.biomarkers.circadian.intradaily_variability_high == 1.0

        # Sleep thresholds
        assert config.biomarkers.sleep.efficiency_threshold == 0.85
        assert config.biomarkers.sleep.timing_variance_threshold == 2.0

    def test_mixed_features_config(self, sample_config_path):
        """Test mixed features configuration is loaded correctly."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        config = load_clinical_thresholds(sample_config_path)

        assert config.mixed_features.minimum_opposite_symptoms == 3
        assert (
            "racing_thoughts"
            in config.mixed_features.depression_with_mixed.required_manic_symptoms
        )
        assert (
            "depressed_mood"
            in config.mixed_features.mania_with_mixed.required_depressive_symptoms
        )

    def test_dsm5_duration_config(self, sample_config_path):
        """Test DSM-5 duration requirements are loaded correctly."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        config = load_clinical_thresholds(sample_config_path)

        assert config.dsm5_duration.manic_days == 7
        assert config.dsm5_duration.hypomanic_days == 4
        assert config.dsm5_duration.depressive_days == 14

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        with pytest.raises(FileNotFoundError):
            load_clinical_thresholds(Path("nonexistent.yaml"))

    def test_invalid_config_structure(self, tmp_path):
        """Test handling of invalid configuration structure."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        invalid_config = tmp_path / "invalid.yaml"
        with open(invalid_config, "w") as f:
            yaml.dump({"invalid": "structure"}, f)

        with pytest.raises(ValueError, match="Missing required configuration section"):
            load_clinical_thresholds(invalid_config)

    def test_missing_required_fields(self, tmp_path):
        """Test handling of missing required fields."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        incomplete_config = tmp_path / "incomplete.yaml"
        with open(incomplete_config, "w") as f:
            yaml.dump({"depression": {"phq_cutoffs": {}}}, f)

        with pytest.raises(ValueError, match="Missing required"):
            load_clinical_thresholds(incomplete_config)

    def test_invalid_threshold_values(self, tmp_path, sample_config_path):
        """Test validation of threshold values."""
        from big_mood_detector.domain.services.clinical_thresholds import load_clinical_thresholds

        invalid_config = tmp_path / "invalid_values.yaml"
        # First create a complete valid config
        with open(sample_config_path) as f:
            config_data = yaml.safe_load(f)
        # Then make one threshold invalid
        config_data["depression"]["phq_cutoffs"]["moderate"] = {
            "min": 15,
            "max": 10,
        }  # Invalid: min > max

        with open(invalid_config, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="Invalid threshold range"):
            load_clinical_thresholds(invalid_config)

    def test_default_config_exists(self):
        """Test that default configuration file exists and is valid."""
        from big_mood_detector.domain.services.clinical_thresholds import (
            ClinicalThresholdsConfig,
            load_clinical_thresholds,
        )

        default_path = Path("config/clinical_thresholds.yaml")
        if default_path.exists():
            config = load_clinical_thresholds(default_path)
            assert isinstance(config, ClinicalThresholdsConfig)
