#!/usr/bin/env python3
"""
Canonical PAT Training Launcher
Trains PAT-S, PAT-M, or PAT-L with consistent settings
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch canonical PAT training")
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large", "all"],
        default="large",
        help="Which model size to train (default: large)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    scripts_dir = Path(__file__).parent

    # Map size to script
    size_to_script = {
        "small": scripts_dir / "train_pat_s_canonical.py",
        "medium": scripts_dir / "train_pat_m_canonical.py",
        "large": scripts_dir / "train_pat_l_run_now.py"
    }

    if args.model_size == "all":
        # Train all models sequentially
        for size in ["small", "medium", "large"]:
            print(f"\n{'='*60}")
            print(f"Training PAT-{size.upper()}")
            print(f"{'='*60}\n")

            script = size_to_script[size]
            cmd = [sys.executable, str(script)]

            # Add resume if provided
            if args.resume and size == "large":
                # For advanced script, we need to modify the command
                cmd = [
                    sys.executable,
                    str(scripts_dir / "train_pat_l_advanced.py"),
                    "--resume", args.resume,
                    "--unfreeze-last-n", "4",
                    "--head-lr", "3e-4",
                    "--encoder-lr", "3e-5",
                    "--epochs", "60",
                    "--scheduler", "cosine",
                    "--patience", "10",
                    "--output-dir", "model_weights/pat/pytorch/pat_l_retry"
                ]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"❌ PAT-{size.upper()} training failed!")
                sys.exit(1)
    else:
        # Train single model
        script = size_to_script[args.model_size]
        cmd = [sys.executable, str(script)]

        if args.resume and args.model_size == "large":
            # Use advanced script for resume
            cmd = [
                sys.executable,
                str(scripts_dir / "train_pat_l_advanced.py"),
                "--resume", args.resume,
                "--unfreeze-last-n", "4",
                "--head-lr", "3e-4",
                "--encoder-lr", "3e-5",
                "--epochs", "60",
                "--scheduler", "cosine",
                "--patience", "10",
                "--output-dir", "model_weights/pat/pytorch/pat_l_retry"
            ]

        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
