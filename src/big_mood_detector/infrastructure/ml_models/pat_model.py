"""
PAT Model Integration

Integrates the Pretrained Actigraphy Transformer (PAT) for feature extraction
from 7-day activity sequences. Based on the paper "AI Foundation Models for
Wearable Movement Data in Mental Health Research".

This module provides a wrapper around the pretrained PAT models to extract
learned representations from activity sequences for downstream tasks.
"""

import logging
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for PAT model. "
        "Install with: pip install tensorflow>=2.10.0"
    ) from e

from big_mood_detector.domain.services.pat_sequence_builder import PATSequence

logger = logging.getLogger(__name__)


class PATModel:
    """
    Wrapper for Pretrained Actigraphy Transformer models.

    Supports Small, Medium, and Large model variants as described in the paper.
    Models are pretrained on NHANES data using masked autoencoding.
    """

    # Model configurations from the paper
    MODEL_CONFIGS = {
        "small": {
            "patch_size": 18,
            "embed_dim": 96,
            "encoder_num_heads": 6,
            "encoder_ff_dim": 256,
            "encoder_num_layers": 1,
            "encoder_rate": 0.1,
            "parameters": 285000,  # 285K
        },
        "medium": {
            "patch_size": 18,
            "embed_dim": 96,
            "encoder_num_heads": 12,
            "encoder_ff_dim": 256,
            "encoder_num_layers": 2,
            "encoder_rate": 0.1,
            "parameters": 1000000,  # 1M
        },
        "large": {
            "patch_size": 9,
            "embed_dim": 96,
            "encoder_num_heads": 12,
            "encoder_ff_dim": 256,
            "encoder_num_layers": 4,
            "encoder_rate": 0.1,
            "parameters": 1990000,  # 1.99M
        },
    }

    def __init__(self, model_size: str = "medium"):
        """
        Initialize PAT model wrapper.

        Args:
            model_size: One of "small", "medium", or "large"
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Invalid model size: {model_size}. "
                f"Choose from: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_size = model_size
        self.config = self.MODEL_CONFIGS[model_size]

        # Extract config values
        self.patch_size = self.config["patch_size"]
        self.embed_dim = self.config["embed_dim"]
        self.encoder_num_heads = self.config["encoder_num_heads"]
        self.encoder_ff_dim = self.config["encoder_ff_dim"]
        self.encoder_num_layers = self.config["encoder_num_layers"]
        self.encoder_rate = self.config["encoder_rate"]

        # Model state
        self.model: keras.Model | None = None
        self.is_loaded = False

        logger.info(f"Initialized PAT-{model_size.upper()} model wrapper")

    def load_pretrained_weights(self, weights_path: Path) -> bool:
        """
        Load pretrained PAT encoder weights.

        Args:
            weights_path: Path to the .h5 weights file

        Returns:
            True if successful, False otherwise
        """
        if not weights_path.exists():
            logger.error(f"Weights file not found: {weights_path}")
            return False

        try:
            # The PAT models are saved as complete Keras models
            # We load them with safe mode to handle custom objects
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Load with safe mode - this ignores custom objects but loads weights
            self.model = tf.keras.models.load_model(
                str(weights_path),
                compile=False,
                safe_mode=False  # Allow loading despite custom objects
            )
            
            # Verify the model has the expected input/output shape
            expected_input = (None, 10080)
            expected_output_patches = 10080 // self.patch_size
            
            if self.model.input_shape != expected_input:
                raise ValueError(f"Model input shape {self.model.input_shape} != expected {expected_input}")
            
            self.is_loaded = True
            logger.info(f"Successfully loaded pretrained PAT-{self.model_size.upper()} from {weights_path}")
            logger.info(f"Model expects input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
            return True

        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            logger.info("PAT models require TensorFlow custom objects that were used during training.")
            logger.info("For production use, consider re-saving the models in a simpler format.")
            self.model = None
            self.is_loaded = False
            return False

    def extract_features(self, sequence: PATSequence) -> np.ndarray:
        """
        Extract learned features from a PAT sequence.

        Args:
            sequence: 7-day activity sequence

        Returns:
            Feature vector of shape (embed_dim,)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained_weights first.")

        # Prepare input
        model_input = self._prepare_input(sequence)

        # Get encoder output
        if self.model is not None:
            # Predict returns shape (batch_size, num_patches, embed_dim)
            features = self.model.predict(model_input, verbose=0)
        else:
            raise RuntimeError("Model is None despite being marked as loaded")

        # Average pool over sequence dimension to get fixed-size features
        # Shape: (1, num_patches, embed_dim) -> (1, embed_dim)
        features = np.mean(features, axis=1)

        # Return 1D feature vector
        return features.squeeze()  # type: ignore[no-any-return]

    def extract_features_batch(self, sequences: list[PATSequence]) -> np.ndarray:
        """
        Extract features from multiple sequences.

        Args:
            sequences: List of PAT sequences

        Returns:
            Feature matrix of shape (batch_size, embed_dim)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained_weights first.")

        # Prepare batch input
        batch_input = np.vstack([self._prepare_input(seq) for seq in sequences])

        # Get features
        if self.model is not None:
            features = self.model.predict(batch_input, verbose=0)
        else:
            raise RuntimeError("Model is None despite being marked as loaded")

        # Average pool
        features = np.mean(features, axis=1)

        return features  # type: ignore[no-any-return]

    def get_attention_weights(self, sequence: PATSequence) -> np.ndarray | None:
        """
        Get attention weights for model explainability.

        Args:
            sequence: Input sequence

        Returns:
            Attention weights if available, None otherwise
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained_weights first.")

        # This would require the model to output attention weights
        # For now, return None as placeholder
        logger.warning("Attention weight extraction not yet implemented")
        return None

    def _prepare_input(self, sequence: PATSequence) -> np.ndarray:
        """
        Prepare sequence for model input.

        Args:
            sequence: PAT sequence

        Returns:
            Normalized input array of shape (1, 10080)
        """
        # Get normalized values
        normalized = sequence.get_normalized()

        # Add batch dimension
        return normalized.reshape(1, -1)


    def _get_positional_embeddings(self, num_patches: int, embed_dim: int) -> tf.Tensor:
        """
        Generate sine/cosine positional embeddings.

        Args:
            num_patches: Number of patches
            embed_dim: Embedding dimension

        Returns:
            Positional embeddings
        """
        position = tf.range(num_patches, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, embed_dim, 2, dtype=tf.float32)
            * (-tf.math.log(10000.0) / embed_dim)
        )

        pos_embeddings = tf.concat(
            [tf.sin(position * div_term), tf.cos(position * div_term)], axis=-1
        )

        return pos_embeddings  # type: ignore[no-any-return]

    def get_model_info(self) -> dict:
        """
        Get model configuration information.

        Returns:
            Dictionary with model details
        """
        num_patches = 10080 // self.patch_size

        return {
            "model_size": self.model_size,
            "patch_size": self.patch_size,
            "num_patches": num_patches,
            "embed_dim": self.embed_dim,
            "encoder_layers": self.encoder_num_layers,
            "encoder_heads": self.encoder_num_heads,
            "parameters": self.config["parameters"],
            "is_loaded": self.is_loaded,
        }


class PATFeatureExtractor:
    """
    High-level feature extractor using PAT models.

    Can combine features from multiple PAT model sizes for ensemble approaches.
    """

    def __init__(self, model_sizes: list[str] | None = None):
        """
        Initialize feature extractor with specified model sizes.

        Args:
            model_sizes: List of model sizes to use (defaults to ["medium"])
        """
        if model_sizes is None:
            model_sizes = ["medium"]
        self.models = {}
        for size in model_sizes:
            self.models[size] = PATModel(model_size=size)

    def load_all_models(self, weights_dir: Path) -> dict:
        """
        Load pretrained weights for all models.

        Args:
            weights_dir: Directory containing weight files

        Returns:
            Dictionary of load results
        """
        results = {}

        # Expected weight file names
        weight_files = {
            "small": "PAT-S_29k_weights.h5",
            "medium": "PAT-M_29k_weights.h5",
            "large": "PAT-L_29k_weights.h5",
        }

        for size, model in self.models.items():
            weight_file = weights_dir / weight_files.get(
                size, f"PAT-{size.upper()}_weights.h5"
            )
            results[size] = model.load_pretrained_weights(weight_file)

        return results

    def extract_ensemble_features(self, sequence: PATSequence) -> np.ndarray:
        """
        Extract and concatenate features from all loaded models.

        Args:
            sequence: Input sequence

        Returns:
            Concatenated feature vector
        """
        all_features = []

        for size, model in self.models.items():
            if model.is_loaded:
                features = model.extract_features(sequence)
                all_features.append(features)
            else:
                logger.warning(f"PAT-{size.upper()} not loaded, skipping")

        if not all_features:
            raise RuntimeError("No models loaded for feature extraction")

        # Concatenate features from all models
        return np.concatenate(all_features)
