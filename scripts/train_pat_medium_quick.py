#!/usr/bin/env python3
"""
Quick training script for PAT-Medium depression head.
Simplified version to get started quickly.
"""

import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Check device availability
    if torch.backends.mps.is_available():
        device = 'mps'
        logger.info("✅ Apple M1 GPU acceleration available!")
    else:
        device = 'cpu'
        logger.info("ℹ️ Using CPU (MPS not available)")
    
    # Check data availability
    nhanes_dir = Path("data/nhanes/2013-2014")
    required_files = ["DPQ_H.xpt", "PAXMIN_H.xpt"]
    
    logger.info("\nChecking NHANES data files:")
    all_present = True
    for file in required_files:
        file_path = nhanes_dir / file
        if file_path.exists():
            logger.info(f"✅ {file} found")
        else:
            logger.info(f"❌ {file} missing")
            all_present = False
    
    if not all_present:
        logger.error("\n❌ Missing required data files!")
        return
    
    # Check model weights
    weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")
    if weights_path.exists():
        logger.info(f"\n✅ PAT-M weights found at {weights_path}")
    else:
        logger.error(f"\n❌ PAT-M weights not found at {weights_path}")
        return
    
    logger.info("\n🚀 Ready to train! Run the following command:")
    logger.info("\npython scripts/train_pat_depression_head_full.py \\")
    logger.info("    --model-size medium \\")
    logger.info(f"    --device {device} \\")
    logger.info("    --epochs 50 \\")
    logger.info("    --batch-size 32")
    
    logger.info("\nExpected training time on M1 Pro: 2-3 hours")
    logger.info("Expected AUC (from paper): ~0.56-0.59")
    logger.info("\nFor better performance, also train PAT-L after this completes.")

if __name__ == "__main__":
    main()