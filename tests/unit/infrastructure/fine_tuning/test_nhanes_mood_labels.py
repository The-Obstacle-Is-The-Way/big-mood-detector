"""
Test NHANES Mood Label Creation

Tests for creating 3-class mood labels from NHANES data.
Maps PHQ-9 scores and medications to our mood states.
"""

import numpy as np
import pandas as pd
import pytest

from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import NHANESProcessor


class TestNHANESMoodLabels:
    """Test mood label creation from NHANES data."""
    
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
    
    def test_create_mood_labels_method_exists(self, processor):
        """NHANES processor should have create_mood_labels method."""
        assert hasattr(processor, 'create_mood_labels')
    
    def test_create_mood_labels_normal_state(self, processor, sample_cohort):
        """Normal: PHQ-9 < 5, no medications."""
        labels = processor.create_mood_labels(sample_cohort)
        
        # Subject 1: PHQ-9=2, no meds -> Normal
        assert labels.loc[0, 'mood_label'] == 'normal'
        assert labels.loc[0, 'mood_class'] == 0
    
    def test_create_mood_labels_depression_state(self, processor, sample_cohort):
        """Depression: PHQ-9 >= 10 or on antidepressants."""
        labels = processor.create_mood_labels(sample_cohort)
        
        # Subject 2: PHQ-9=12, on SSRI -> Depression
        assert labels.loc[1, 'mood_label'] == 'depression'
        assert labels.loc[1, 'mood_class'] == 1
        
        # Subject 4: PHQ-9=15, multiple meds -> Depression
        assert labels.loc[3, 'mood_label'] == 'depression'
        assert labels.loc[3, 'mood_class'] == 1
    
    def test_create_mood_labels_mania_state(self, processor, sample_cohort):
        """Mania/Hypomania: Benzodiazepine use (proxy for mood stabilization)."""
        labels = processor.create_mood_labels(sample_cohort)
        
        # Subject 3: PHQ-9=5, on benzodiazepine -> Mania (proxy)
        assert labels.loc[2, 'mood_label'] == 'mania'
        assert labels.loc[2, 'mood_class'] == 2
    
    def test_create_mood_labels_subclinical_depression(self, processor, sample_cohort):
        """Subclinical depression (PHQ-9 5-9) without meds -> Normal."""
        labels = processor.create_mood_labels(sample_cohort)
        
        # Subject 6: PHQ-9=8, no meds -> Normal (not severe enough)
        assert labels.loc[5, 'mood_label'] == 'normal'
        assert labels.loc[5, 'mood_class'] == 0
    
    def test_create_mood_labels_returns_dataframe(self, processor, sample_cohort):
        """Should return DataFrame with original data plus labels."""
        labels = processor.create_mood_labels(sample_cohort)
        
        assert isinstance(labels, pd.DataFrame)
        assert len(labels) == len(sample_cohort)
        assert 'mood_label' in labels.columns
        assert 'mood_class' in labels.columns
        assert all(col in labels.columns for col in sample_cohort.columns)
    
    def test_mood_label_distribution(self, processor, sample_cohort):
        """Check distribution of mood labels."""
        labels = processor.create_mood_labels(sample_cohort)
        
        label_counts = labels['mood_label'].value_counts()
        assert 'normal' in label_counts.index
        assert 'depression' in label_counts.index
        assert 'mania' in label_counts.index
        
        # Should have all 3 classes represented
        assert len(label_counts) == 3
    
    def test_extract_pat_sequences_method_exists(self, processor):
        """NHANES processor should have extract_pat_sequences method."""
        assert hasattr(processor, 'extract_pat_sequences')
    
    def test_extract_pat_sequences_shape(self, processor):
        """PAT sequences should be 7 days × 1440 minutes."""
        # Create mock actigraphy data for one subject
        actigraphy = pd.DataFrame({
            'SEQN': [1] * (7 * 1440),  # 7 days of data
            'PAXDAY': np.repeat(range(1, 8), 1440),
            'PAXMINUTE': np.tile(range(1440), 7),
            'PAXINTEN': np.random.randint(0, 1000, 7 * 1440)
        })
        
        sequences = processor.extract_pat_sequences(actigraphy, subject_id=1)
        
        assert sequences.shape == (7, 1440)
        assert sequences.dtype == np.float32
    
    def test_extract_pat_sequences_missing_days(self, processor):
        """Should handle missing days by padding with zeros."""
        # Only 5 days of data
        actigraphy = pd.DataFrame({
            'SEQN': [1] * (5 * 1440),
            'PAXDAY': np.repeat([1, 2, 4, 6, 7], 1440),  # Missing days 3 and 5
            'PAXMINUTE': np.tile(range(1440), 5),
            'PAXINTEN': np.random.randint(0, 1000, 5 * 1440)
        })
        
        sequences = processor.extract_pat_sequences(actigraphy, subject_id=1)
        
        assert sequences.shape == (7, 1440)
        # Check that missing days are zeros
        assert np.all(sequences[2, :] == 0)  # Day 3
        assert np.all(sequences[4, :] == 0)  # Day 5
    
    def test_extract_pat_sequences_normalization(self, processor):
        """Activity values should be normalized appropriately."""
        actigraphy = pd.DataFrame({
            'SEQN': [1] * 1440,
            'PAXDAY': [1] * 1440,
            'PAXMINUTE': range(1440),
            'PAXINTEN': [1000] * 1440  # Constant high activity
        })
        
        sequences = processor.extract_pat_sequences(actigraphy, subject_id=1)
        
        # Should be normalized (not raw counts)
        assert np.all(sequences >= 0)
        assert np.all(sequences <= 10)  # Reasonable upper bound after normalization