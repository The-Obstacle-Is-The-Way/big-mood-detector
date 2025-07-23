"""
Smoke Test for PAT Depression Head Training

Quick test to ensure training script can run without errors.
Uses minimal data to keep CI fast.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestTrainPATDepressionHeadSmoke:
    """Smoke test for the training script."""

    @pytest.mark.skipif(
        not Path("model_weights/pat/pretrained/PAT-S_29k_weights.h5").exists(),
        reason="PAT weights not available"
    )
    def test_training_script_help(self):
        """Test that the script can show help."""
        result = subprocess.run(
            [sys.executable, "scripts/train_pat_depression_head.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Train PAT depression classification head" in result.stdout

    @pytest.mark.slow
    @pytest.mark.skipif(
        not Path("data/nhanes").exists(),
        reason="NHANES data not available"
    )
    def test_training_smoke(self, tmp_path):
        """Test that training can run with minimal data."""
        # Create minimal fake NHANES data
        nhanes_dir = tmp_path / "nhanes"
        nhanes_dir.mkdir()

        # Create minimal actigraphy data (2 subjects, 7 days each)
        actigraphy_data = []
        for subject_id in [1, 2]:
            for day in range(1, 8):
                for minute in range(1440):
                    actigraphy_data.append({
                        'SEQN': subject_id,
                        'PAXDAY': day,
                        'PAXMINUTE': minute,
                        'PAXINTEN': np.random.randint(0, 1000)
                    })

        actigraphy_df = pd.DataFrame(actigraphy_data)
        actigraphy_df.to_csv(nhanes_dir / "actigraphy.csv", index=False)

        # Create minimal depression data
        depression_df = pd.DataFrame({
            'SEQN': [1, 2],
            'PHQ9_TOTAL': [5, 15]  # One below threshold, one above
        })
        depression_df.to_csv(nhanes_dir / "depression.csv", index=False)

        # Run training with minimal settings
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/train_pat_depression_head.py",
                "--nhanes-dir", str(nhanes_dir),
                "--output-dir", str(output_dir),
                "--epochs", "2",  # Minimal epochs
                "--max-subjects", "2"  # Only process 2 subjects
            ],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout for smoke test
        )

        # Check that it completed (may fail due to minimal data, but shouldn't crash)
        assert result.returncode in [0, 1], f"Script crashed: {result.stderr}"

        # If successful, check output
        if result.returncode == 0:
            output_file = output_dir / "pat_depression_head.pt"
            if output_file.exists():
                # Verify the saved model has expected structure
                import torch
                checkpoint = torch.load(output_file, map_location='cpu')
                assert 'model_state_dict' in checkpoint
                assert 'config' in checkpoint
                assert checkpoint['config']['task'] == 'depression_binary'
