"""
Clinical Thresholds Configuration

Defines the data structures and loading logic for clinical thresholds
from YAML configuration files. This separates clinical parameters from
code implementation, making them easier to maintain and validate.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class ThresholdRange:
    """Represents a min-max threshold range."""

    min: float
    max: float

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(
                f"Invalid threshold range: min ({self.min}) > max ({self.max})"
            )


@dataclass
class PHQCutoffs:
    """PHQ-8/9 score cutoffs for depression severity."""

    none: ThresholdRange
    mild: ThresholdRange
    moderate: ThresholdRange
    moderately_severe: ThresholdRange
    severe: ThresholdRange


@dataclass
class DepressionSleepThresholds:
    """Sleep-related thresholds for depression."""

    hypersomnia_threshold: float
    normal_min: float
    normal_max: float


@dataclass
class DepressionActivityThresholds:
    """Activity-related thresholds for depression."""

    severe_reduction: int
    moderate_reduction: int
    normal_min: int


@dataclass
class DepressionThresholds:
    """All depression-related thresholds."""

    phq_cutoffs: PHQCutoffs
    sleep_hours: DepressionSleepThresholds
    activity_steps: DepressionActivityThresholds


@dataclass
class ASRMCutoffs:
    """ASRM score cutoffs for mania/hypomania severity."""

    none: ThresholdRange
    hypomanic: ThresholdRange
    manic_moderate: ThresholdRange
    manic_severe: ThresholdRange


@dataclass
class ManiaSleepThresholds:
    """Sleep-related thresholds for mania."""

    critical_threshold: float
    reduced_threshold: float


@dataclass
class ManiaActivityThresholds:
    """Activity-related thresholds for mania."""

    elevated_threshold: int
    extreme_threshold: int


@dataclass
class ManiaThresholds:
    """All mania-related thresholds."""

    asrm_cutoffs: ASRMCutoffs
    sleep_hours: ManiaSleepThresholds
    activity_steps: ManiaActivityThresholds


@dataclass
class CircadianThresholds:
    """Circadian rhythm-related thresholds."""

    phase_advance_threshold: float
    interdaily_stability_low: float
    intradaily_variability_high: float


@dataclass
class SleepBiomarkerThresholds:
    """Sleep biomarker thresholds."""

    efficiency_threshold: float
    timing_variance_threshold: float


@dataclass
class BiomarkerThresholds:
    """All biomarker-related thresholds."""

    circadian: CircadianThresholds
    sleep: SleepBiomarkerThresholds


@dataclass
class MixedFeaturesRequirements:
    """Requirements for mixed features diagnosis."""

    required_manic_symptoms: list[str]


@dataclass
class MixedDepressiveRequirements:
    """Requirements for mixed depressive features."""

    required_depressive_symptoms: list[str]


@dataclass
class MixedFeaturesConfig:
    """Configuration for mixed features detection."""

    minimum_opposite_symptoms: int
    depression_with_mixed: MixedFeaturesRequirements
    mania_with_mixed: MixedDepressiveRequirements


@dataclass
class DSM5DurationConfig:
    """DSM-5 episode duration requirements."""

    manic_days: int
    hypomanic_days: int
    depressive_days: int


@dataclass
class ClinicalThresholdsConfig:
    """Complete clinical thresholds configuration."""

    depression: DepressionThresholds
    mania: ManiaThresholds
    biomarkers: BiomarkerThresholds
    mixed_features: MixedFeaturesConfig
    dsm5_duration: DSM5DurationConfig


def _parse_threshold_range(data: dict[str, Any]) -> ThresholdRange:
    """Parse a threshold range from configuration data."""
    if "min" not in data or "max" not in data:
        raise ValueError("Threshold range must have 'min' and 'max' values")
    return ThresholdRange(min=float(data["min"]), max=float(data["max"]))


def _parse_phq_cutoffs(data: dict[str, Any]) -> PHQCutoffs:
    """Parse PHQ cutoffs from configuration data."""
    required_keys = ["none", "mild", "moderate", "moderately_severe", "severe"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required PHQ cutoff: {key}")

    return PHQCutoffs(
        none=_parse_threshold_range(data["none"]),
        mild=_parse_threshold_range(data["mild"]),
        moderate=_parse_threshold_range(data["moderate"]),
        moderately_severe=_parse_threshold_range(data["moderately_severe"]),
        severe=_parse_threshold_range(data["severe"]),
    )


def _parse_asrm_cutoffs(data: dict[str, Any]) -> ASRMCutoffs:
    """Parse ASRM cutoffs from configuration data."""
    required_keys = ["none", "hypomanic", "manic_moderate", "manic_severe"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required ASRM cutoff: {key}")

    return ASRMCutoffs(
        none=_parse_threshold_range(data["none"]),
        hypomanic=_parse_threshold_range(data["hypomanic"]),
        manic_moderate=_parse_threshold_range(data["manic_moderate"]),
        manic_severe=_parse_threshold_range(data["manic_severe"]),
    )


def load_clinical_thresholds(config_path: Path) -> ClinicalThresholdsConfig:
    """
    Load clinical thresholds from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        ClinicalThresholdsConfig object with all thresholds

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration is invalid or missing required fields
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid configuration: expected a dictionary")

    # Validate top-level structure
    required_sections = [
        "depression",
        "mania",
        "biomarkers",
        "mixed_features",
        "dsm5_duration",
    ]
    for section in required_sections:
        if section not in data:
            raise ValueError(f"Missing required configuration section: {section}")

    try:
        # Parse depression thresholds
        dep_data = data["depression"]
        depression = DepressionThresholds(
            phq_cutoffs=_parse_phq_cutoffs(dep_data.get("phq_cutoffs", {})),
            sleep_hours=DepressionSleepThresholds(
                hypersomnia_threshold=float(
                    dep_data["sleep_hours"]["hypersomnia_threshold"]
                ),
                normal_min=float(dep_data["sleep_hours"]["normal_min"]),
                normal_max=float(dep_data["sleep_hours"]["normal_max"]),
            ),
            activity_steps=DepressionActivityThresholds(
                severe_reduction=int(dep_data["activity_steps"]["severe_reduction"]),
                moderate_reduction=int(
                    dep_data["activity_steps"]["moderate_reduction"]
                ),
                normal_min=int(dep_data["activity_steps"]["normal_min"]),
            ),
        )

        # Parse mania thresholds
        mania_data = data["mania"]
        mania = ManiaThresholds(
            asrm_cutoffs=_parse_asrm_cutoffs(mania_data.get("asrm_cutoffs", {})),
            sleep_hours=ManiaSleepThresholds(
                critical_threshold=float(
                    mania_data["sleep_hours"]["critical_threshold"]
                ),
                reduced_threshold=float(mania_data["sleep_hours"]["reduced_threshold"]),
            ),
            activity_steps=ManiaActivityThresholds(
                elevated_threshold=int(
                    mania_data["activity_steps"]["elevated_threshold"]
                ),
                extreme_threshold=int(
                    mania_data["activity_steps"]["extreme_threshold"]
                ),
            ),
        )

        # Parse biomarker thresholds
        bio_data = data["biomarkers"]
        biomarkers = BiomarkerThresholds(
            circadian=CircadianThresholds(
                phase_advance_threshold=float(
                    bio_data["circadian"]["phase_advance_threshold"]
                ),
                interdaily_stability_low=float(
                    bio_data["circadian"]["interdaily_stability_low"]
                ),
                intradaily_variability_high=float(
                    bio_data["circadian"]["intradaily_variability_high"]
                ),
            ),
            sleep=SleepBiomarkerThresholds(
                efficiency_threshold=float(bio_data["sleep"]["efficiency_threshold"]),
                timing_variance_threshold=float(
                    bio_data["sleep"]["timing_variance_threshold"]
                ),
            ),
        )

        # Parse mixed features config
        mixed_data = data["mixed_features"]
        mixed_features = MixedFeaturesConfig(
            minimum_opposite_symptoms=int(mixed_data["minimum_opposite_symptoms"]),
            depression_with_mixed=MixedFeaturesRequirements(
                required_manic_symptoms=list(
                    mixed_data["depression_with_mixed"]["required_manic_symptoms"]
                )
            ),
            mania_with_mixed=MixedDepressiveRequirements(
                required_depressive_symptoms=list(
                    mixed_data["mania_with_mixed"]["required_depressive_symptoms"]
                )
            ),
        )

        # Parse DSM-5 duration config
        dsm5_data = data["dsm5_duration"]
        dsm5_duration = DSM5DurationConfig(
            manic_days=int(dsm5_data["manic_days"]),
            hypomanic_days=int(dsm5_data["hypomanic_days"]),
            depressive_days=int(dsm5_data["depressive_days"]),
        )

        return ClinicalThresholdsConfig(
            depression=depression,
            mania=mania,
            biomarkers=biomarkers,
            mixed_features=mixed_features,
            dsm5_duration=dsm5_duration,
        )

    except KeyError as e:
        raise ValueError(f"Missing required field in configuration: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid configuration value: {e}") from e
