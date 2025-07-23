"""
Test NHANES Binary Label Creation

Tests for creating binary labels from NHANES data matching PAT paper.
"""

import numpy as np
import pandas as pd
import pytest

from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import (
    NHANESProcessor,
)


class TestNHANESBinaryLabels:
    """Test binary label creation from NHANES data."""

    @pytest.fixture
    def processor(self, tmp_path):
        """Create NHANES processor with temp directory."""
        return NHANESProcessor(data_dir=tmp_path)

    @pytest.fixture
    def sample_cohort(self):
        """Create sample cohort data for testing."""
        return pd.DataFrame({
            'SEQN': [1, 2, 3, 4, 5, 6],
            'PHQ9_total': [2, 12, 5, 15, 3, 8],  # Depression scores
            'depressed': [0, 1, 0, 1, 0, 0],     # PHQ-9 >= 10
            'benzodiazepine': [0, 0, 1, 1, 0, 0],
            'ssri': [0, 1, 0, 1, 0, 0],
            'antidepressant': [0, 1, 0, 1, 0, 0]
        })

    def test_create_binary_labels_method_exists(self, processor):
        """NHANES processor should have create_binary_labels method."""
        assert hasattr(processor, 'create_binary_labels')

    def test_binary_depression_label(self, processor, sample_cohort):
        """Depression label based on PHQ-9 >= 10."""
        labels = processor.create_binary_labels(sample_cohort)

        # Subject 1: PHQ-9=2 -> Not depressed (0)
        assert labels.loc[0, 'depression_label'] == 0
        # Subject 2: PHQ-9=12 -> Depressed (1)
        assert labels.loc[1, 'depression_label'] == 1
        # Subject 4: PHQ-9=15 -> Depressed (1)
        assert labels.loc[3, 'depression_label'] == 1
        # Subject 6: PHQ-9=8 -> Not depressed (0)
        assert labels.loc[5, 'depression_label'] == 0

    def test_binary_benzodiazepine_label(self, processor, sample_cohort):
        """Benzodiazepine label is direct from medication flag."""
        labels = processor.create_binary_labels(sample_cohort)

        # Direct mapping
        assert labels.loc[0, 'benzodiazepine_label'] == 0
        assert labels.loc[2, 'benzodiazepine_label'] == 1
        assert labels.loc[3, 'benzodiazepine_label'] == 1

    def test_binary_ssri_label(self, processor, sample_cohort):
        """SSRI label is direct from medication flag."""
        labels = processor.create_binary_labels(sample_cohort)

        # Direct mapping
        assert labels.loc[0, 'ssri_label'] == 0
        assert labels.loc[1, 'ssri_label'] == 1
        assert labels.loc[3, 'ssri_label'] == 1

    def test_labels_are_independent(self, processor, sample_cohort):
        """Binary labels should be independent (not mutually exclusive)."""
        labels = processor.create_binary_labels(sample_cohort)

        # Subject 4 can be both depressed AND on benzodiazepines
        assert labels.loc[3, 'depression_label'] == 1
        assert labels.loc[3, 'benzodiazepine_label'] == 1
        assert labels.loc[3, 'ssri_label'] == 1

    def test_returns_dataframe_with_all_columns(self, processor, sample_cohort):
        """Should return DataFrame with original data plus label columns."""
        labels = processor.create_binary_labels(sample_cohort)

        assert isinstance(labels, pd.DataFrame)
        assert len(labels) == len(sample_cohort)
        assert 'depression_label' in labels.columns
        assert 'benzodiazepine_label' in labels.columns
        assert 'ssri_label' in labels.columns
        assert all(col in labels.columns for col in sample_cohort.columns)