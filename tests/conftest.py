"""
Minimal conftest.py for Big Mood Detector test suite.

Provides basic fixtures for TDD workflow.
"""

import os

# Set thread limits BEFORE any imports to prevent segfaults
# This must happen before NumPy, XGBoost, or PyTorch are imported
for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "XGBOOST_NUM_THREADS",
):
    os.environ.setdefault(var, "1")

# Disable Rich formatting in tests for speed
os.environ["RICH_DISABLE"] = "True"

# The TESTING=1 guards are now in the individual modules themselves

from copy import deepcopy  # noqa: E402
from pathlib import Path  # noqa: E402
from tempfile import NamedTemporaryFile  # noqa: E402

import pytest  # noqa: E402
import yaml  # noqa: E402

# Check if model weights are available
XGBOOST_WEIGHTS_AVAILABLE = (
    Path(__file__).parent.parent / "model_weights" / "xgboost" / "XGBoost_DE.json"
).exists()
PAT_WEIGHTS_AVAILABLE = (
    Path(__file__).parent.parent / "model_weights" / "pat" / "pretrained" / "PAT-M_29k_weights.h5"
).exists()


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require database"
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (performance, heavy computation)"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "ml: Machine learning model tests")
    config.addinivalue_line("markers", "clinical: Clinical validation tests")
    config.addinivalue_line("markers", "slow: Slow tests (can be skipped in CI)")
    config.addinivalue_line(
        "markers", "heavy: Tests that load real model weights or large data"
    )
    config.addinivalue_line(
        "markers", "requires_weights: Tests that require model weights to be present"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and handle slow test skipping."""
    # Handle slow test skipping
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow to execute")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Handle tests requiring model weights
    skip_no_weights = pytest.mark.skip(reason="Model weights not available - run 'make download-weights' locally")
    for item in items:
        if "requires_weights" in item.keywords:
            if not XGBOOST_WEIGHTS_AVAILABLE and not PAT_WEIGHTS_AVAILABLE:
                item.add_marker(skip_no_weights)

    # Mark tests based on directory structure
    for item in items:
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "ml/" in str(item.fspath):
            item.add_marker(pytest.mark.ml)


@pytest.fixture(scope="session")
def sample_sleep_xml():
    """Sample Apple HealthKit XML data for sleep testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis"
            sourceName="Apple Watch"
            startDate="2024-01-01 23:30:00 -0800"
            endDate="2024-01-02 07:30:00 -0800"
            value="HKCategoryValueSleepAnalysisAsleep"/>
    <Record type="HKCategoryTypeIdentifierSleepAnalysis"
            sourceName="Apple Watch"
            startDate="2024-01-02 23:00:00 -0800"
            endDate="2024-01-03 06:45:00 -0800"
            value="HKCategoryValueSleepAnalysisInBed"/>
</HealthData>"""


@pytest.fixture(scope="session")
def expected_sleep_data():
    """Expected parsed sleep data structure."""
    return [
        {
            "sourceName": "Apple Watch",
            "startDate": "2024-01-01 23:30:00 -0800",
            "endDate": "2024-01-02 07:30:00 -0800",
            "value": "HKCategoryValueSleepAnalysisAsleep",
        },
        {
            "sourceName": "Apple Watch",
            "startDate": "2024-01-02 23:00:00 -0800",
            "endDate": "2024-01-03 06:45:00 -0800",
            "value": "HKCategoryValueSleepAnalysisInBed",
        },
    ]


class DummyBooster:
    """Lightweight XGBoost Booster mock for unit tests.

    Following Eugene Yan's advice: mock at the API boundary,
    not at the algorithm boundary.
    """

    def __init__(self, probability=0.42):
        self.probability = probability

    def predict(self, X, **kwargs):
        """Mock predict method returning consistent probabilities."""
        import numpy as np

        # Return single probability value per sample
        if len(X.shape) == 1:
            return np.array([self.probability])
        else:
            return np.full((X.shape[0],), self.probability)

    def predict_proba(self, X, **kwargs):
        """Mock predict_proba for sklearn compatibility."""
        import numpy as np

        # Return probability array with shape (n_samples, 2)
        # First column is 1-probability, second is probability
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        neg_prob = 1 - self.probability
        return np.array([[neg_prob, self.probability]] * n_samples)


@pytest.fixture
def dummy_booster():
    """Fixture providing a DummyBooster instance."""
    return DummyBooster()


@pytest.fixture
def dummy_xgboost_models():
    """Fixture providing pre-configured XGBoost model mocks."""
    return {
        "depression": DummyBooster(probability=0.7),
        "hypomanic": DummyBooster(probability=0.2),
        "manic": DummyBooster(probability=0.1),
    }


@pytest.fixture(autouse=False if os.getenv("TESTING", "0") == "1" else True)
def _patch_pat_model(monkeypatch):
    """Provide a lightweight PATModel stub scoped to each test."""
    import sys
    import types

    pat_stub = types.ModuleType("pat_model")

    class MockPATModel:
        def __init__(self, model_size="medium", **kwargs):
            # Validate model size
            valid_sizes = {"small", "medium", "large"}
            if model_size not in valid_sizes:
                raise ValueError(
                    f"Invalid model size: {model_size}. Must be one of {valid_sizes}"
                )

            self.model_size = model_size
            self.embed_dim = 96
            self.depth = 6
            self.num_heads = 4
            self.is_loaded = False

            # Model size specific configs
            if model_size == "small":
                self.patch_size = 18
                self.encoder_num_heads = 6
                self.encoder_num_layers = 1
            elif model_size == "medium":
                self.patch_size = 18
                self.encoder_num_heads = 12
                self.encoder_num_layers = 2
            elif model_size == "large":
                self.patch_size = 9
                self.encoder_num_heads = 12
                self.encoder_num_layers = 4

        def load_pretrained_weights(self, weights_path=None, *args, **kwargs):
            # Simulate file not found for non-existent paths
            if weights_path and "nonexistent" in str(weights_path):
                return False
            self.is_loaded = True
            return True

        def extract_features(self, sequence):
            # Check if model is loaded
            if not self.is_loaded:
                raise RuntimeError("Model not loaded")
            # Mock feature extraction - return a numpy array with expected shape
            import numpy as np

            return np.random.rand(96)  # 96-dim feature vector for medium model

        def extract_features_batch(self, sequences):
            # Check if model is loaded
            if not self.is_loaded:
                raise RuntimeError("Model not loaded")
            import numpy as np

            return np.random.rand(len(sequences), 96)

        def _prepare_input(self, sequence):
            import numpy as np

            # Normalize data (mock normalization)
            data = np.array(sequence.activity_values).reshape(1, -1)
            return (data - np.mean(data)) / (np.std(data) + 1e-8)

        def get_attention_weights(self):
            pass

        def get_model_info(self):
            num_patches = 10080 // self.patch_size
            params = {"small": 285000, "medium": 1300000, "large": 7600000}
            return {
                "model_size": self.model_size,
                "patch_size": self.patch_size,
                "num_patches": num_patches,
                "parameters": params.get(self.model_size, 1300000),
                "is_loaded": self.is_loaded,
            }

    pat_stub.PATModel = MockPATModel
    pat_stub.PATFeatureExtractor = object

    with monkeypatch.context() as m:
        m.setitem(
            sys.modules,
            "big_mood_detector.infrastructure.ml_models.pat_model",
            pat_stub,
        )
        yield


# Clinical Configuration Fixtures
# ================================
# These fixtures help avoid duplicating clinical configuration in tests


@pytest.fixture(scope="session")
def clinical_config_path():
    """Path to the real clinical thresholds config file."""
    config_path = Path(__file__).parent.parent / "config" / "clinical_thresholds.yaml"
    if not config_path.exists():
        pytest.skip(f"Clinical config not found at {config_path}")
    return config_path


@pytest.fixture(scope="session")
def clinical_config_dict(clinical_config_path):
    """Load clinical config as a dictionary."""
    with open(clinical_config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def clinical_config(clinical_config_path):
    """
    Load clinical thresholds config from real file.

    Returns a fresh instance for each test to ensure test isolation.
    """
    from big_mood_detector.domain.services.clinical_thresholds import (
        load_clinical_thresholds,
    )
    return load_clinical_thresholds(clinical_config_path)


@pytest.fixture
def clinical_config_mutable(clinical_config_dict):
    """
    Mutable copy of clinical config dictionary for tests that need to modify values.

    Use this when you need to test with modified thresholds.
    """
    return deepcopy(clinical_config_dict)


@pytest.fixture
def clinical_config_factory(clinical_config_dict):
    """
    Factory for creating ClinicalThresholdsConfig with custom modifications.

    Example:
        config = clinical_config_factory(
            depression={'phq_cutoffs': {'moderate': {'min': 15}}}
        )
    """

    def _factory(**section_overrides):
        from big_mood_detector.domain.services.clinical_thresholds import (
            load_clinical_thresholds,
        )

        config_dict = deepcopy(clinical_config_dict)

        # Apply overrides at the section level
        for section, overrides in section_overrides.items():
            if section in config_dict:
                _deep_update(config_dict[section], overrides)

        # Write to temporary file and load
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = Path(f.name)

        try:
            return load_clinical_thresholds(temp_path)
        finally:
            temp_path.unlink()

    return _factory


def _deep_update(target: dict, source: dict) -> None:
    """Recursively update target dict with source dict."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
