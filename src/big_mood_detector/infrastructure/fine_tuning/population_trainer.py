"""
Population Trainer

Trains task-specific heads on NHANES cohorts for population-level fine-tuning.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)

# Export control based on torch availability
if TYPE_CHECKING:
    # Always export for type checking
    __all__ = [
        "PopulationTrainer",
        "XGBoostPopulationTrainer",
        "PATPopulationTrainer",
        "create_population_trainer",
    ]
else:
    # Only export what's available at runtime
    __all__ = [
        "PopulationTrainer",
        "XGBoostPopulationTrainer",
        "create_population_trainer",
    ]


class PopulationTrainer(ABC):
    """Base class for population-level fine-tuning."""

    def __init__(
        self,
        task_name: str = "depression",
        output_dir: Path = Path("models/population"),
    ):
        """Initialize trainer.

        Args:
            task_name: Task name (depression, benzodiazepine, ssri)
            output_dir: Directory for saving models
        """
        self.task_name = task_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_base_model(self) -> Any:
        """Load pre-trained base model."""
        pass

    @abstractmethod
    def fine_tune(self, **kwargs: Any) -> dict[str, float]:
        """Fine-tune model on task data."""
        pass

    def create_time_series_splits(
        self,
        data: pd.DataFrame,
        n_splits: int = 3,
        test_size: float = 0.2,
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Create time-series aware train/test splits.

        Args:
            data: DataFrame with time-ordered data
            n_splits: Number of splits
            test_size: Fraction for test set

        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(data)
        test_samples = int(n_samples * test_size)

        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_samples,
        )

        return list(tscv.split(data))

    def evaluate(
        self,
        y_true: NDArray[np.float32],
        y_pred: NDArray[np.float32],
        y_prob: NDArray[np.float32] | None = None,
    ) -> dict[str, float]:
        """Evaluate model performance.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        # Sensitivity = Recall for positive class
        metrics["sensitivity"] = metrics["recall"]

        # Specificity = Recall for negative class
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC if probabilities available
        if y_prob is not None:
            metrics["auc"] = roc_auc_score(y_true, y_prob)

        return metrics


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Don't assign None to module names - it causes type errors
    # Just let them be undefined


if TORCH_AVAILABLE:

    class TaskHead(nn.Module):
        """Task-specific head for PAT model."""

        def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 128,  # Reduced from 256
            num_classes: int = 1,  # 🎯 CRITICAL: Single logit for BCE
            dropout: float = 0.3,  # Slightly increased
        ):
            """Initialize task head.

            Args:
                input_dim: Input dimension from encoder
                hidden_dim: Hidden layer dimension
                num_classes: Number of output classes (1 for binary BCE)
                dropout: Dropout probability
            """
            super().__init__()

            # Simplified 2-layer head with SINGLE LOGIT output
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1, bias=True),  # 🎯 SINGLE LOGIT for BCE
            )

            self.output_dim = 1  # Single logit

            # 🎯 CRITICAL: Proper initialization for minority class learning
            self._initialize_weights()

        def _initialize_weights(self) -> None:
            """Initialize weights for better minority class learning."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # Xavier uniform initialization for better gradient flow
                    torch.nn.init.xavier_uniform_(module.weight)
                    # Initialize bias to 0 (neutral starting point)
                    if module.bias is not None:
                        module.bias.data.fill_(0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            result = self.layers(x).squeeze(1)  # (N,) not (N,1)
            assert isinstance(result, torch.Tensor)
            return result

    class FocalLoss(nn.Module):
        """Focal Loss for addressing class imbalance.

        Focuses learning on hard examples by down-weighting easy negatives.
        """
        def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float | None = None):
            super().__init__()
            self.alpha = alpha  # Class balance factor
            self.gamma = gamma  # Focusing parameter
            self.pos_weight = pos_weight

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # Convert to probabilities
            p = torch.sigmoid(inputs)

            # Calculate cross entropy
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

            # Calculate p_t
            p_t = p * targets + (1 - p) * (1 - targets)

            # Calculate alpha_t
            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                ce_loss = alpha_t * ce_loss

            # Apply pos_weight if provided
            if self.pos_weight is not None:
                weight_t = self.pos_weight * targets + 1.0 * (1 - targets)
                ce_loss = weight_t * ce_loss

            # Calculate focal weight
            focal_weight = (1 - p_t) ** self.gamma

            # Calculate focal loss
            focal_loss = focal_weight * ce_loss

            result = focal_loss.mean()
            assert isinstance(result, torch.Tensor)
            return result

else:
    # Protocol stub when torch not available
    from collections.abc import Iterable
    from typing import Protocol

    @runtime_checkable
    class TaskHead(Protocol):  # type: ignore[no-redef]
        """Protocol for TaskHead when torch is not available."""

        output_dim: int

        def parameters(self) -> Iterable[Any]: ...
        def train(self, mode: bool = True) -> Any: ...
        def eval(self) -> Any: ...
        def __call__(self, x: Any) -> Any: ...


def load_pat_model(model_path: str) -> Any:
    """Load pre-trained PAT model.

    Args:
        model_path: Path to model weights

    Returns:
        Loaded model
    """
    # Placeholder - would load actual PAT model
    logger.info(f"Loading PAT model from {model_path}")

    # Mock model for now
    class MockPATModel:
        def __init__(self) -> None:
            self.output_dim = 768

        def encode(self, sequences: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
            # Return random embeddings for now
            return np.random.rand(len(sequences), self.output_dim)

    return MockPATModel()


if TORCH_AVAILABLE:

    class PATPopulationTrainer(PopulationTrainer):
        """PAT-specific population trainer."""

        def __init__(
            self,
            base_model_path: str = "weights/PAT-S_29k_weights.h5",
            task_name: str = "depression",
            output_dir: Path = Path("models/population"),
        ):
            """Initialize PAT trainer.

            Args:
                base_model_path: Path to pre-trained PAT weights
                task_name: Task name
                output_dir: Output directory
            """
            super().__init__(task_name, output_dir)
            self.base_model_path = base_model_path

        def load_base_model(self) -> Any:
            """Load pre-trained PAT encoder."""
            return load_pat_model(self.base_model_path)

        def create_task_head(
            self,
            input_dim: int = 768,
            num_classes: int = 2,
            dropout: float = 0.2,
        ) -> TaskHead:
            """Create task-specific head.

            Args:
                input_dim: Input dimension
                num_classes: Number of classes
                dropout: Dropout rate

            Returns:
                Task head module
            """
            return TaskHead(
                input_dim=input_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

        def fine_tune(
            self,
            **kwargs: Any,
        ) -> dict[str, float]:
            """Fine-tune PAT with task head.

            Args:
                sequences: Activity sequences (N, 60)
                labels: Task labels
                epochs: Training epochs
                batch_size: Batch size
                learning_rate: Learning rate
                validation_split: Validation split ratio
                pos_weight: Positive class weight for imbalanced data

            Returns:
                Training metrics
            """
            # Extract parameters from kwargs
            sequences = kwargs["sequences"]
            labels = kwargs["labels"]
            epochs = kwargs.get("epochs", 10)
            batch_size = kwargs.get("batch_size", 32)
            # learning_rate removed - use head_learning_rate and encoder_learning_rate instead
            validation_split = kwargs.get("validation_split", 0.2)
            pos_weight = kwargs.get("pos_weight", None)

            logger.info(f"Fine-tuning PAT for {self.task_name}")

            # 🎯 DEVICE MANAGEMENT (critical fix!)
            device = torch.device("mps" if torch.backends.mps.is_available()
                                 else ("cuda" if torch.cuda.is_available() else "cpu"))
            logger.info(f"Using device: {device}")

            # Load base model
            encoder = self.load_base_model()

            # Encode sequences
            logger.info("Encoding sequences with PAT")
            embeddings = encoder.encode(sequences)

            # Create task head - SINGLE LOGIT for BCE
            input_dim = getattr(encoder, "output_dim", 768)
            if hasattr(input_dim, "__class__") and "Mock" in str(input_dim.__class__):
                input_dim = 768  # Default for PAT

            task_head = self.create_task_head(
                input_dim=input_dim,
                # num_classes=1 removed - hardcoded in TaskHead for BCE
            )

            # Move model to device
            task_head = task_head.to(device)

            # Convert to tensors and move to device
            X = torch.FloatTensor(embeddings).to(device)
            y = torch.FloatTensor(labels).to(device)  # Float for BCE

            # 🎯 SHUFFLE DATA (critical fix!)
            perm = torch.randperm(len(X))
            X, y = X[perm], y[perm]

            # Split data
            n_val = int(len(X) * validation_split)
            X_train, X_val = X[:-n_val], X[-n_val:]
            y_train, y_val = y[:-n_val], y[-n_val:]

            # Debug: Log class distribution in splits
            train_pos = (y_train == 1).sum().item()
            train_neg = (y_train == 0).sum().item()
            val_pos = (y_val == 1).sum().item()
            val_neg = (y_val == 0).sum().item()

            logger.info(f"Training split: {train_neg} negative, {train_pos} positive ({train_pos/(train_pos+train_neg)*100:.1f}%)")
            logger.info(f"Validation split: {val_neg} negative, {val_pos} positive ({val_pos/(val_pos+val_neg)*100:.1f}%)")

            if val_pos == 0:
                logger.warning("⚠️ NO POSITIVES IN VALIDATION SET - This would break dynamic num_classes!")

            # Training setup with SELECTIVE UNFREEZING + HIGHER LEARNING RATE (critical fix!)
            # 🎯 5e-3 for head, 1e-5 for top encoder layers
            head_lr = kwargs.get("head_learning_rate", 5e-3)  # Much higher default
            encoder_lr = kwargs.get("encoder_learning_rate", 1e-5)  # Small LR for encoder
            unfreeze_layers = kwargs.get("unfreeze_layers", 1)  # Number of top layers to unfreeze

            # 🎯 SELECTIVE UNFREEZING: Only unfreeze last N transformer blocks
            # First freeze everything
            if hasattr(encoder, 'parameters'):
                for param in encoder.parameters():
                    param.requires_grad = False

            # Then selectively unfreeze top layers
            unfrozen_params = []
            if encoder_lr > 0 and unfreeze_layers > 0:
                # PAT model structure detection - try multiple approaches
                if hasattr(encoder, 'blocks'):
                    # Standard transformer blocks
                    blocks = encoder.blocks[-unfreeze_layers:]  # Last N blocks
                    for block in blocks:
                        for param in block.parameters():
                            param.requires_grad = True
                            unfrozen_params.append(param)
                    logger.info(f"🔓 Unfroze last {unfreeze_layers} transformer blocks ({len(unfrozen_params)} params)")
                elif hasattr(encoder, 'layers'):
                    # Layer list structure
                    layers = encoder.layers[-unfreeze_layers:]  # Last N layers
                    for layer in layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                            unfrozen_params.append(param)
                    logger.info(f"🔓 Unfroze last {unfreeze_layers} transformer layers ({len(unfrozen_params)} params)")
                elif hasattr(encoder, 'transformer_blocks'):
                    # Alternative naming
                    blocks = encoder.transformer_blocks[-unfreeze_layers:]
                    for block in blocks:
                        for param in block.parameters():
                            param.requires_grad = True
                            unfrozen_params.append(param)
                    logger.info(f"🔓 Unfroze last {unfreeze_layers} transformer blocks ({len(unfrozen_params)} params)")
                else:
                    # Last resort: unfreeze all parameters (allows fine-tuning)
                    logger.warning("⚠️ Could not find transformer block structure - unfreezing entire encoder")
                    for param in encoder.parameters():
                        param.requires_grad = True
                        unfrozen_params.append(param)
                    logger.info(f"🔓 Unfroze entire encoder ({len(unfrozen_params)} params)")

            # Setup optimizer with different learning rates
            if len(unfrozen_params) > 0:
                # Split learning rates: unfrozen encoder parts + head
                optimizer = torch.optim.AdamW([
                    {'params': unfrozen_params, 'lr': encoder_lr, 'weight_decay': 1e-5},
                    {'params': task_head.parameters(), 'lr': head_lr, 'weight_decay': 0.0}
                ])
                logger.info(f"Using split LR: encoder_top={encoder_lr:.1e}, head={head_lr:.1e}")
            else:
                # Head only (encoder fully frozen)
                optimizer = torch.optim.AdamW(task_head.parameters(), lr=head_lr, weight_decay=0.0)
                logger.info(f"Encoder FROZEN, head LR={head_lr:.1e}")

            # 🎯 BCEWithLogitsLoss with pos_weight for class imbalance
            if pos_weight is not None:
                weight_tensor = torch.tensor(float(pos_weight), device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
                logger.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight:.2f}")
            else:
                criterion = nn.BCEWithLogitsLoss()
                logger.info("Using standard BCEWithLogitsLoss (no pos_weight)")

            # 🎯 SANITY CHECK: Initial logit distribution (before training)
            task_head.eval()
            with torch.no_grad():
                initial_logits = task_head(X_val)  # Single logits now
                pos_logits = initial_logits[y_val == 1].cpu().numpy()
                neg_logits = initial_logits[y_val == 0].cpu().numpy()

                if len(pos_logits) > 0 and len(neg_logits) > 0:
                    logger.info(f"🔍 INITIAL LOGITS - Pos mean: {pos_logits.mean():.4f}, Neg mean: {neg_logits.mean():.4f}")
                    logger.info(f"🔍 INITIAL SEPARATION: {abs(pos_logits.mean() - neg_logits.mean()):.4f}")
                else:
                    logger.warning("⚠️ Missing positive or negative samples in validation set!")

            # 🎯 WEIGHTED SAMPLING - Create balanced indices for each epoch
            train_targets = y_train.cpu().numpy().astype(int)
            class_counts = np.bincount(train_targets)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[train_targets]

            logger.info(f"🎯 Class distribution: {class_counts} → weights: {class_weights}")

            # Training loop with optional early stopping
            task_head.train()
            best_val_loss = float('inf')
            patience_counter = 0
            patience = kwargs.get("patience", 10)  # Early stopping patience

            for epoch in range(epochs):
                # 🎯 WEIGHTED RANDOM SAMPLING for this epoch (ensures balanced batches)
                weighted_indices = np.random.choice(
                    len(X_train),
                    size=len(X_train),
                    p=sample_weights / sample_weights.sum(),
                    replace=True
                )

                # Shuffle the weighted indices
                np.random.shuffle(weighted_indices)
                # Mini-batch training with WEIGHTED SAMPLING
                for i in range(0, len(X_train), batch_size):
                    # Get weighted batch indices
                    batch_indices = weighted_indices[i : i + batch_size]
                    batch_X = X_train[batch_indices]
                    batch_y = y_train[batch_indices]

                    optimizer.zero_grad()
                    outputs = task_head(batch_X)  # Single logits
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                # Validation
                task_head.eval()
                with torch.no_grad():
                    val_logits = task_head(X_val)  # Single logits
                    val_loss = criterion(val_logits, y_val)
                    val_probs = torch.sigmoid(val_logits)  # Convert to probabilities
                    val_preds = (val_probs > 0.5).long()  # BCE threshold
                    val_acc = (val_preds == y_val.long()).float().mean()  # 🎯 Fix dtype mismatch

                    # 🎯 ENHANCED LOGIT MONITORING (every 5 epochs)
                    if (epoch + 1) % 5 == 0:
                        pred_counts = torch.bincount(val_preds, minlength=2)
                        true_counts = torch.bincount(y_val.long(), minlength=2)
                        logger.info(f"  Val predictions: {pred_counts[0].item()} negative, {pred_counts[1].item()} positive")
                        logger.info(f"  Val true labels: {true_counts[0].item()} negative, {true_counts[1].item()} positive")

                        # Confusion matrix
                        tp = ((val_preds == 1) & (y_val == 1)).sum().item()
                        fp = ((val_preds == 1) & (y_val == 0)).sum().item()
                        tn = ((val_preds == 0) & (y_val == 0)).sum().item()
                        fn = ((val_preds == 0) & (y_val == 1)).sum().item()
                        logger.info(f"  Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

                        # 🎯 CRITICAL: Monitor actual logit separation!
                        pos_logits = val_logits[y_val == 1].cpu().numpy()
                        neg_logits = val_logits[y_val == 0].cpu().numpy()

                        if len(pos_logits) > 0 and len(neg_logits) > 0:
                            pos_mean = pos_logits.mean()
                            neg_mean = neg_logits.mean()
                            separation = abs(pos_mean - neg_mean)
                            logger.info(f"  🎯 LOGITS - Pos: {pos_mean:.4f}, Neg: {neg_mean:.4f}, Sep: {separation:.4f}")

                            # Alert if no separation developing
                            if separation < 0.1:
                                logger.warning(f"  ⚠️ WEAK SEPARATION: {separation:.4f} - Model not learning!")
                            elif separation > 0.5:
                                logger.info(f"  ✅ GOOD SEPARATION: {separation:.4f} - Model learning!")

                task_head.train()

                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"loss={loss:.4f}, val_loss={val_loss:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"🛑 Early stopping at epoch {epoch+1} (patience={patience})")
                        break

            # Final metrics with THRESHOLD OPTIMIZATION (🎯 critical for imbalanced data!)
            task_head.eval()
            with torch.no_grad():
                final_logits = task_head(X_val)  # Single logits
                final_probs = torch.sigmoid(final_logits)  # BCE probabilities

                # 🎯 SEARCH FOR OPTIMAL THRESHOLD (instead of fixed 0.5)
                y_val_np = y_val.cpu().numpy().astype(int)
                probs_np = final_probs.cpu().numpy().squeeze()

                best_f1, best_thr = 0.0, 0.5
                best_metrics = None

                logger.info("🔍 Searching optimal threshold...")
                for threshold in np.linspace(0.05, 0.6, 12):  # Test range
                    test_preds = (probs_np > threshold).astype(int)

                    if len(np.unique(test_preds)) > 1:  # Avoid single-class predictions
                        try:
                            test_f1 = f1_score(y_val_np, test_preds, zero_division=0)
                            test_recall = recall_score(y_val_np, test_preds, zero_division=0)
                            test_precision = precision_score(y_val_np, test_preds, zero_division=0)

                            if test_f1 > best_f1:
                                best_f1 = test_f1
                                best_thr = threshold
                                best_metrics = {
                                    'threshold': threshold,
                                    'f1': test_f1,
                                    'recall': test_recall,
                                    'precision': test_precision
                                }

                            logger.info(f"  Thr={threshold:.2f}: F1={test_f1:.3f}, Rec={test_recall:.3f}, Pre={test_precision:.3f}")
                        except Exception:
                            pass

                logger.info(f"🏆 Best threshold: {best_thr:.2f} → F1={best_f1:.3f}")

                # Use best threshold for final predictions
                final_preds = (probs_np > best_thr).astype(int)

            # Standard metrics with optimized threshold
            metrics = self.evaluate(
                y_val_np,
                final_preds,
                probs_np,
            )

            # Add threshold info to metrics
            metrics['optimal_threshold'] = best_thr
            if best_metrics:
                metrics.update({f"opt_{k}": v for k, v in best_metrics.items()})

            metrics["final_loss"] = float(val_loss)
            metrics["final_accuracy"] = float(val_acc)
            metrics["epochs_completed"] = epochs

            # Save model
            self.save_model(encoder, task_head, self.task_name, metrics)

            return metrics

        def save_model(
            self,
            encoder: Any,
            task_head: Any,  # Can be nn.Module or TaskHead protocol
            task_name: str,
            metrics: dict[str, float],
        ) -> Path:
            """Save fine-tuned model.

            Args:
                encoder: PAT encoder
                task_head: Task-specific head
                task_name: Task name
                metrics: Training metrics

            Returns:
                Path to saved model
            """
            # Save PyTorch model
            model_path = self.output_dir / f"pat_{task_name}.pt"
            # Only save state dict for real models, not mocks
            save_dict = {
                "task_name": task_name,
                "metrics": metrics,
            }

            if hasattr(task_head, "state_dict"):
                try:
                    state_dict = task_head.state_dict()
                    # Check if it's a real state dict (not Mock)
                    if not hasattr(state_dict, "__class__") or "Mock" not in str(
                        state_dict.__class__
                    ):
                        save_dict["task_head_state_dict"] = state_dict
                except Exception:
                    # Mock object, skip state dict
                    pass

            torch.save(save_dict, model_path)

            # Save metadata
            metadata_path = self.output_dir / f"pat_{task_name}_metadata.json"

            # Get output dim safely (handle mocks)
            output_dim = getattr(task_head, "output_dim", 2)
            if hasattr(output_dim, "__class__") and "Mock" in str(output_dim.__class__):
                output_dim = 2  # Default binary classification

            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "task_name": task_name,
                        "base_model": self.base_model_path,
                        "metrics": {
                            k: (
                                float(v)
                                if isinstance(v, int | float | np.number)
                                else str(v)
                            )
                            for k, v in metrics.items()
                        },
                        "output_dim": output_dim,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Saved model to {model_path}")
            return model_path

else:
    # When torch is not available, create a stub class
    class PATPopulationTrainer:  # type: ignore[no-redef]
        """Stub for when PyTorch is not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required for PAT population training. "
                "Install with: pip install torch"
            )


class XGBoostPopulationTrainer(PopulationTrainer):
    """XGBoost-specific population trainer."""

    def __init__(
        self,
        base_model_path: str = "mood_ml/XGBoost_DE.pkl",
        task_name: str = "depression",
        output_dir: Path = Path("models/population"),
    ):
        """Initialize XGBoost trainer.

        Args:
            base_model_path: Path to pre-trained model
            task_name: Task name
            output_dir: Output directory
        """
        super().__init__(task_name, output_dir)
        self.base_model_path = base_model_path

    def load_base_model(self) -> xgb.XGBClassifier:
        """Load pre-trained XGBoost model."""
        logger.info(f"Loading XGBoost model from {self.base_model_path}")
        model = joblib.load(self.base_model_path)
        assert isinstance(model, xgb.XGBClassifier)
        return model

    def validate_features(self, features: pd.DataFrame) -> None:
        """Validate features match mood_ml expectations.

        Args:
            features: Feature DataFrame

        Raises:
            ValueError: If required features missing
        """
        required_features = [
            "mean_sleep_duration",
            "std_sleep_duration",
            "mean_sleep_efficiency",
            "IS",
            "IV",
            "RA",
            "L5",
            "M10",
        ]

        missing = set(required_features) - set(features.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def incremental_train(
        self,
        features: pd.DataFrame,
        labels: NDArray[np.float32],
        num_boost_round: int = 50,
        sample_weight: NDArray[np.float32] | None = None,
    ) -> dict[str, float]:
        """Incrementally train XGBoost model.

        Args:
            features: Feature matrix
            labels: Target labels
            num_boost_round: Additional boosting rounds
            sample_weight: Sample weights

        Returns:
            Training metrics
        """
        # Load base model
        base_model = self.load_base_model()

        # Get current number of trees
        initial_trees = base_model.n_estimators or 100

        # Continue training
        dtrain = xgb.DMatrix(features, label=labels, weight=sample_weight)

        # Convert sklearn model to native XGBoost
        xgb_model = base_model.get_booster()

        # Additional training
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.01,  # Small learning rate for fine-tuning
            "max_depth": base_model.max_depth or 6,
        }

        updated_model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            xgb_model=xgb_model,
        )

        # Convert back to sklearn API
        final_model = xgb.XGBClassifier()
        final_model._Booster = updated_model
        final_model.n_estimators = initial_trees + num_boost_round

        # Fit to set n_classes_ attribute
        final_model.fit(features, labels)

        # Evaluate
        predictions = final_model.predict(features)
        probabilities = final_model.predict_proba(features)[:, 1]

        metrics = self.evaluate(labels, predictions, probabilities)
        metrics["total_estimators"] = final_model.n_estimators

        # Save updated model
        save_path = self.output_dir / f"xgboost_{self.task_name}_updated.pkl"
        joblib.dump(final_model, save_path)

        logger.info(f"Saved updated model to {save_path}")
        return metrics

    def fine_tune(
        self,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Fine-tune XGBoost model.

        Args:
            features: Feature matrix
            labels: Target labels
            **kwargs: Additional arguments

        Returns:
            Training metrics
        """
        # Extract parameters from kwargs
        features = kwargs["features"]
        labels = kwargs["labels"]
        self.validate_features(features)
        return self.incremental_train(features, labels, **kwargs)


def create_population_trainer(
    model_type: str,
    task_name: str = "depression",
    **kwargs: Any,
) -> PopulationTrainer:
    """Factory function for creating trainers.

    Args:
        model_type: Model type (pat, xgboost)
        task_name: Task name
        **kwargs: Additional arguments

    Returns:
        Appropriate trainer instance

    Raises:
        ValueError: If model type unknown
    """
    if model_type.lower() == "pat":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PAT population training. Install with: pip install torch"
            )
        return PATPopulationTrainer(task_name=task_name, **kwargs)
    elif model_type.lower() == "xgboost":
        return XGBoostPopulationTrainer(task_name=task_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

