#!/usr/bin/env python3
"""Monitor PAT training progress"""

import json
from pathlib import Path
import subprocess
import time

def check_training():
    output_dir = Path("model_weights/pat/pytorch/pat_l_training")
    
    # Check for checkpoints
    checkpoints = list(output_dir.glob("*.pt"))
    if checkpoints:
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Latest checkpoint: {latest.name}")
        
    # Check if process is running
    result = subprocess.run(["pgrep", "-f", "train_pat"], capture_output=True, text=True)
    if result.stdout:
        print("✅ Training is still running")
    else:
        print("❌ Training has stopped")
        
    # Check summary
    summary_path = output_dir / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n📊 Training Summary:")
        print(f"Best AUC: {summary['best_auc']:.4f}")
        print(f"Completed at: {summary['timestamp']}")

if __name__ == "__main__":
    check_training()