#!/usr/bin/env python3
"""
Validate PAT Depression Head Integration

This script validates that the PAT-Conv-L depression model is properly integrated
and can make predictions with the actual production weights.

Usage:
    python scripts/validate_pat_depression_integration.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
    ProductionPATLoader,
)
from big_mood_detector.infrastructure.ml_models.nhanes_normalizer import (
    NHANESNormalizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_activity(pattern: str = "normal") -> np.ndarray:
    """Create synthetic 7-day activity data for testing."""
    # 7 days * 24 hours * 60 minutes = 10,080 timesteps
    data = np.zeros(10080, dtype=np.float32)
    
    if pattern == "normal":
        # Normal circadian rhythm
        for day in range(7):
            day_start = day * 1440  # 1440 minutes per day
            
            # Low activity at night (0-6 AM)
            data[day_start:day_start+360] = np.random.normal(0.5, 0.2, 360)
            
            # Increasing activity in morning (6 AM - 12 PM)
            data[day_start+360:day_start+720] = np.random.normal(3.0, 1.0, 360)
            
            # High activity during day (12 PM - 6 PM)
            data[day_start+720:day_start+1080] = np.random.normal(5.0, 1.5, 360)
            
            # Decreasing activity in evening (6 PM - 12 AM)
            data[day_start+1080:day_start+1440] = np.random.normal(2.0, 0.8, 360)
            
    elif pattern == "depressed":
        # Disrupted circadian rhythm (common in depression)
        for day in range(7):
            day_start = day * 1440
            
            # Higher night activity (sleep disruption)
            data[day_start:day_start+360] = np.random.normal(2.0, 1.0, 360)
            
            # Lower morning activity (fatigue)
            data[day_start+360:day_start+720] = np.random.normal(1.5, 0.8, 360)
            
            # Irregular day activity
            data[day_start+720:day_start+1080] = np.random.normal(2.5, 2.0, 360)
            
            # Irregular evening activity
            data[day_start+1080:day_start+1440] = np.random.normal(2.0, 1.5, 360)
    
    # Ensure non-negative
    data = np.clip(data, 0, None)
    
    return data


def validate_with_mock_normalizer():
    """Validate using a mock normalizer (when NHANES stats aren't available)."""
    logger.info("=" * 60)
    logger.info("Validating with mock normalizer")
    logger.info("=" * 60)
    
    # Create mock normalizer with reasonable stats
    normalizer = NHANESNormalizer()
    normalizer.mean = np.full(10080, 2.5, dtype=np.float32)  # Typical mean activity
    normalizer.std = np.full(10080, 2.0, dtype=np.float32)   # Typical std
    normalizer.fitted = True
    
    try:
        # Create loader
        loader = ProductionPATLoader(normalizer=normalizer)
        logger.info("✅ Successfully created ProductionPATLoader")
        
        # Test with normal pattern
        normal_activity = create_synthetic_activity("normal")
        normal_prob = loader.predict_depression(normal_activity)
        logger.info(f"Normal pattern - Depression probability: {normal_prob:.4f}")
        
        # Test with depressed pattern
        depressed_activity = create_synthetic_activity("depressed")
        depressed_prob = loader.predict_depression(depressed_activity)
        logger.info(f"Depressed pattern - Depression probability: {depressed_prob:.4f}")
        
        # Validate outputs
        assert 0 <= normal_prob <= 1, f"Invalid probability: {normal_prob}"
        assert 0 <= depressed_prob <= 1, f"Invalid probability: {depressed_prob}"
        assert not np.isnan(normal_prob), "Probability is NaN"
        assert not np.isnan(depressed_prob), "Probability is NaN"
        
        logger.info("✅ All probability checks passed")
        
        # Test embeddings interface
        embeddings = loader.get_embeddings(normal_activity)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        assert embeddings.shape == (96,), f"Wrong embedding size: {embeddings.shape}"
        
        # Test predict_from_embeddings
        predictions = loader.predict_from_embeddings(embeddings)
        logger.info(f"Interface predictions - Depression: {predictions.depression_probability:.4f}")
        logger.info(f"Interface predictions - Confidence: {predictions.confidence:.4f}")
        
        logger.info("✅ All interface methods working correctly")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"❌ Model weights not found: {e}")
        logger.error("Please ensure pat_conv_l_v0.5929.pth is in model_weights/production/")
        return False
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False


def validate_with_real_normalizer():
    """Validate using real NHANES normalizer stats (if available)."""
    logger.info("=" * 60)
    logger.info("Validating with real NHANES normalizer")
    logger.info("=" * 60)
    
    try:
        # Try to load with default normalizer
        loader = ProductionPATLoader()
        logger.info("✅ Successfully loaded with NHANES normalizer")
        
        # Run same tests as above
        normal_activity = create_synthetic_activity("normal")
        normal_prob = loader.predict_depression(normal_activity)
        logger.info(f"Normal pattern - Depression probability: {normal_prob:.4f}")
        
        depressed_activity = create_synthetic_activity("depressed")
        depressed_prob = loader.predict_depression(depressed_activity)
        logger.info(f"Depressed pattern - Depression probability: {depressed_prob:.4f}")
        
        return True
        
    except ValueError as e:
        if "Normalizer not fitted" in str(e):
            logger.warning("⚠️  NHANES normalizer stats not found")
            logger.info("This is expected if nhanes_scaler_stats.json hasn't been created yet")
            return None
        else:
            raise
    except FileNotFoundError as e:
        logger.error(f"❌ Model weights not found: {e}")
        return False


def check_model_info():
    """Display information about the model."""
    logger.info("=" * 60)
    logger.info("Model Information")
    logger.info("=" * 60)
    
    model_path = Path("model_weights/production/pat_conv_l_v0.5929.pth")
    
    if model_path.exists():
        logger.info(f"✅ Model file exists: {model_path}")
        logger.info(f"   Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Load checkpoint to show info
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'val_auc' in checkpoint:
            logger.info(f"   Validation AUC: {checkpoint['val_auc']:.4f}")
        if 'epoch' in checkpoint:
            logger.info(f"   Training epochs: {checkpoint['epoch']}")
        
        # Check state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        logger.info(f"   Total parameters: {sum(p.numel() for p in state_dict.values()):,}")
        
        # Check for Conv architecture
        has_conv = any('conv' in key for key in state_dict.keys())
        logger.info(f"   Conv architecture: {'Yes' if has_conv else 'No'}")
        
    else:
        logger.error(f"❌ Model file not found: {model_path}")
        logger.info("Please download or copy pat_conv_l_v0.5929.pth to model_weights/production/")
        

def main():
    """Run all validation checks."""
    logger.info("PAT Depression Head Integration Validation")
    logger.info("=" * 60)
    
    # Check model info
    check_model_info()
    
    # Try real normalizer first
    real_result = validate_with_real_normalizer()
    
    if real_result is None:
        # Real normalizer not available, use mock
        mock_result = validate_with_mock_normalizer()
        if mock_result:
            logger.info("\n✅ VALIDATION PASSED (with mock normalizer)")
            logger.info("To use real NHANES normalization, create nhanes_scaler_stats.json")
        else:
            logger.error("\n❌ VALIDATION FAILED")
            sys.exit(1)
    elif real_result:
        logger.info("\n✅ VALIDATION PASSED (with NHANES normalizer)")
    else:
        logger.error("\n❌ VALIDATION FAILED")
        sys.exit(1)
    
    logger.info("\nNext steps:")
    logger.info("1. Wire up the DI container")
    logger.info("2. Add API endpoint for depression predictions")
    logger.info("3. Update CLI to show PAT depression scores")


if __name__ == "__main__":
    main()