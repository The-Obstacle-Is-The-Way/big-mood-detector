#!/usr/bin/env python3
"""
Download Model Weights Script

Downloads pretrained model weights for PAT and prepares directory structure
for XGBoost models.

Usage:
    python scripts/download_model_weights.py --model pat
    python scripts/download_model_weights.py --model all
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PAT model download URLs from the official repository
PAT_MODELS = {
    "PAT-S_29k_weights.h5": {
        "url": "https://www.dropbox.com/scl/fi/12ip8owx1psc4o7b2uqff/PAT-S_29k_weights.h5?rlkey=ffaf1z45a74cbxrl7c9i2b32h&st=mfk6f0y5&dl=1",
        "size_mb": 0.285,
        "description": "PAT Small (285K parameters)"
    },
    "PAT-M_29k_weights.h5": {
        "url": "https://www.dropbox.com/scl/fi/hlfbni5bzsfq0pynarjcn/PAT-M_29k_weights.h5?rlkey=frbkjtbgliy9vq2kvzkquruvg&st=mxc4uet9&dl=1", 
        "size_mb": 1.0,
        "description": "PAT Medium (1M parameters)"
    },
    "PAT-L_29k_weights.h5": {
        "url": "https://www.dropbox.com/scl/fi/exk40hu1nxc1zr1prqrtp/PAT-L_29k_weights.h5?rlkey=t1e5h54oob0e1k4frqzjt1kmz&st=7a20pcox&dl=1",
        "size_mb": 1.99,
        "description": "PAT Large (1.99M parameters)"
    }
}


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """
    Download a file from URL to destination path.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {description}...")
        logger.info(f"From: {url[:50]}...")
        logger.info(f"To: {dest_path}")
        
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(f'\rProgress: {percent:.1f}%')
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
        print()  # New line after progress
        
        logger.info(f"Successfully downloaded {description}")
        return True
        
    except urllib.error.URLError as e:
        logger.error(f"Failed to download {description}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {description}: {e}")
        return False


def download_pat_models(model_dir: Path) -> Dict[str, bool]:
    """
    Download all PAT model weights.
    
    Args:
        model_dir: Directory to save models
        
    Returns:
        Dictionary of download results
    """
    results = {}
    
    for filename, info in PAT_MODELS.items():
        dest_path = model_dir / "pat" / "pretrained" / filename
        
        # Skip if already exists
        if dest_path.exists():
            logger.info(f"{filename} already exists, skipping")
            results[filename] = True
            continue
        
        # Download
        success = download_file(
            info["url"], 
            dest_path,
            f"{info['description']} ({info['size_mb']} MB)"
        )
        results[filename] = success
    
    return results


def setup_xgboost_directory(model_dir: Path) -> None:
    """
    Set up XGBoost model directory structure.
    
    Args:
        model_dir: Base model directory
    """
    xgboost_dirs = [
        model_dir / "xgboost" / "pretrained",
        model_dir / "xgboost" / "finetuned"
    ]
    
    for dir_path in xgboost_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    # Create info file
    info_path = model_dir / "xgboost" / "pretrained" / "README.txt"
    if not info_path.exists():
        info_path.write_text(
            "XGBoost models need to be trained using the mood prediction pipeline.\n"
            "Run: python scripts/train_xgboost_models.py\n"
            "\nExpected files:\n"
            "- depression_model.pkl\n"
            "- hypomanic_model.pkl\n"
            "- manic_model.pkl\n"
        )
        logger.info("Created XGBoost README")


def check_reference_repo_models() -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Check if model weights exist in reference repositories.
    
    Returns:
        Tuple of (pat_models, xgboost_models) found
    """
    pat_models = {}
    xgboost_models = {}
    
    # Check PAT reference repo
    pat_ref_dir = Path("reference_repos/Pretrained-Actigraphy-Transformer/model_weights")
    if pat_ref_dir.exists():
        for model_file in pat_ref_dir.glob("*.h5"):
            pat_models[model_file.name] = model_file
            logger.info(f"Found PAT model in reference repo: {model_file.name}")
    
    # Check for any existing XGBoost models
    # (These would need to be trained, but check common locations)
    potential_xgboost_paths = [
        Path("reference_repos/mood_ml/models"),
        Path("output/models"),
    ]
    
    for path in potential_xgboost_paths:
        if path.exists():
            for model_file in path.glob("*.pkl"):
                if any(name in model_file.name for name in ["depression", "hypomanic", "manic"]):
                    xgboost_models[model_file.name] = model_file
                    logger.info(f"Found XGBoost model: {model_file.name}")
    
    return pat_models, xgboost_models


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download model weights for Big Mood Detector"
    )
    parser.add_argument(
        "--model",
        choices=["pat", "xgboost", "all"],
        default="all",
        help="Which models to download/setup"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model_weights"),
        help="Base directory for model weights"
    )
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="Check reference repositories for existing models"
    )
    
    args = parser.parse_args()
    
    # Check reference repos first if requested
    if args.check_reference:
        logger.info("Checking reference repositories...")
        pat_ref, xgb_ref = check_reference_repo_models()
        
        if pat_ref:
            logger.info(f"Found {len(pat_ref)} PAT models in reference repo")
            response = input("Copy from reference repo instead of downloading? [y/N]: ")
            if response.lower() == 'y':
                for filename, source_path in pat_ref.items():
                    dest_path = args.model_dir / "pat" / "pretrained" / filename
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    import shutil
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"Copied {filename} from reference repo")
                
                # Skip PAT download
                if args.model == "pat":
                    return
    
    # Download/setup based on selection
    if args.model in ["pat", "all"]:
        logger.info("Setting up PAT models...")
        results = download_pat_models(args.model_dir)
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"PAT models: {successful}/{len(results)} successful")
        
        if successful < len(results):
            failed = [k for k, v in results.items() if not v]
            logger.error(f"Failed to download: {failed}")
    
    if args.model in ["xgboost", "all"]:
        logger.info("Setting up XGBoost directories...")
        setup_xgboost_directory(args.model_dir)
        
        logger.info(
            "\nXGBoost models need to be trained. "
            "Run: python scripts/train_xgboost_models.py"
        )
    
    logger.info("\nSetup complete!")
    logger.info(f"Model weights directory: {args.model_dir.absolute()}")


if __name__ == "__main__":
    main()