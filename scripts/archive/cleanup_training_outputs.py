#!/usr/bin/env python3
"""
Clean up and reorganize training outputs according to new structure.
This script will:
1. Create new directory structure
2. Move production model to proper location
3. Archive old experiments
4. Clean up scattered logs
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import tarfile
import json

def create_directory_structure():
    """Create the new organized directory structure."""
    dirs = [
        "training/experiments/archived",
        "training/experiments/active",
        "training/logs",
        "training/results",
        "model_weights/production",
        "model_weights/pretrained",
        "docs/training",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def move_production_model():
    """Move the best PAT-Conv-L model to production."""
    source = Path("model_weights/pat/pytorch/pat_conv_l_simple_best.pth")
    dest = Path("model_weights/production/pat_conv_l_v0.5929.pth")
    
    if source.exists():
        shutil.copy2(source, dest)
        print(f"✅ Copied production model: {dest}")
        
        # Create metadata file
        metadata = {
            "model_type": "PAT-Conv-L",
            "auc": 0.5929,
            "date_trained": "2025-07-25",
            "training_log": "pat_conv_l_simple.log",
            "parameters": 1984289,
            "architecture": "PAT-L with Conv1d patch embedding",
            "dataset": "NHANES 2013-2014",
            "task": "Depression classification (PHQ-9 >= 10)"
        }
        
        with open("model_weights/production/pat_conv_l_v0.5929.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("✅ Created model metadata")

def archive_old_experiments():
    """Archive old experiment files."""
    # Create archive directory
    archive_dir = Path("training/experiments/archived")
    
    # Archive old PyTorch models
    old_models = [
        "model_weights/pat/pytorch/pat_l_corrected_best.pth",
        "model_weights/pat/pytorch/pat_l_final_best.pth",
        "model_weights/pat/pytorch/pat_l_ft_best.pth",
        "model_weights/pat/pytorch/pat_l_gentle_best.pth",
        "model_weights/pat/pytorch/pat_l_higher_lr_best.pth",
        "model_weights/pat/pytorch/pat_l_optimized_best.pth",
        "model_weights/pat/pytorch/pat_conv_l_best_0e5.pth",
        "model_weights/pat/pytorch/pat_conv_l_best_3e5.pth",
        "model_weights/pat/pytorch/pat_conv_l_best_5e5.pth",
        "model_weights/pat/pytorch/pat_conv_l_best_10e5.pth",
    ]
    
    # Create tar archive for old models
    tar_path = archive_dir / f"old_models_{datetime.now():%Y%m%d}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for model_path in old_models:
            if Path(model_path).exists():
                tar.add(model_path, arcname=Path(model_path).name)
                print(f"  Archived: {Path(model_path).name}")
    
    print(f"✅ Created archive: {tar_path}")

def organize_logs():
    """Move and organize training logs."""
    # Copy important logs to canonical location
    important_logs = {
        "docs/archive/pat_experiments/pat_conv_l_simple.log": 
            "training/logs/pat_conv_l_v0.5929_20250725.log",
        "logs/pat_training/pat_l_corrected_20250724_160538.log":
            "training/logs/pat_l_v0.5888_20250724.log",
    }
    
    for source, dest in important_logs.items():
        if Path(source).exists():
            shutil.copy2(source, dest)
            print(f"✅ Copied log: {dest}")

def create_summary_report():
    """Create a summary report of the cleanup."""
    report = f"""# Training Cleanup Report
Generated: {datetime.now():%Y-%m-%d %H:%M:%S}

## Production Model
- PAT-Conv-L v0.5929: `model_weights/production/pat_conv_l_v0.5929.pth`
- Metadata: `model_weights/production/pat_conv_l_v0.5929.json`

## Key Training Logs
- PAT-Conv-L (best): `training/logs/pat_conv_l_v0.5929_20250725.log`
- PAT-L (previous best): `training/logs/pat_l_v0.5888_20250724.log`

## Archived Files
- Old models: `training/experiments/archived/old_models_*.tar.gz`
- Old logs: Original locations preserved for reference

## Next Steps
1. Update training scripts to use new paths
2. Update model loading code to use production path
3. Delete old files after verifying archives

## Directory Structure
```
training/
├── experiments/
│   ├── active/      # Current experiments
│   └── archived/    # Old experiments
├── logs/           # Canonical training logs  
└── results/        # Training summaries

model_weights/
├── production/     # Production-ready models
└── pretrained/     # Original pretrained weights
```
"""
    
    with open("CLEANUP_REPORT.md", "w") as f:
        f.write(report)
    print("✅ Created cleanup report: CLEANUP_REPORT.md")

def main():
    """Run the full cleanup process."""
    print("🧹 Starting training output cleanup...")
    
    # Create new structure
    create_directory_structure()
    
    # Move production model
    move_production_model()
    
    # Archive old experiments
    print("\n📦 Archiving old experiments...")
    archive_old_experiments()
    
    # Organize logs
    print("\n📋 Organizing logs...")
    organize_logs()
    
    # Create summary
    create_summary_report()
    
    print("\n✨ Cleanup complete! Check CLEANUP_REPORT.md for details.")
    print("\n⚠️  Old files are still in place. Delete manually after verifying archives.")

if __name__ == "__main__":
    main()