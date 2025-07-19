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

import pytest  # noqa: E402


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "ml: Machine learning model tests")
    config.addinivalue_line("markers", "clinical: Clinical validation tests")
    config.addinivalue_line("markers", "slow: Slow tests (can be skipped in CI)")
    config.addinivalue_line("markers", "heavy: Tests that load real model weights or large data")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on directory structure
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


@pytest.fixture(autouse=True)
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
