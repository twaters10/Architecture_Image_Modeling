#!/usr/bin/env python3
"""
Unified Inference Script for Architectural Style Classification.

This script consolidates all inference functionality into a single,
configuration-driven tool:
  - Model evaluation with comprehensive metrics
  - Grad-CAM explainability (single, batch, misclassifications)
  - Production inference with quality checks
  - Flexible output generation based on config

Usage:
    # Use default config (../conf/inference.yaml)
    python inference.py

    # Use custom config
    python inference.py --config ../conf/my_config.yaml

    # Override mode via CLI
    python inference.py --mode evaluate
    python inference.py --mode explain_single --image path/to/image.jpg
    python inference.py --mode production --image_dir data/test_images

Configuration Modes:
    - evaluate: Full test set evaluation with metrics
    - explain_single: Grad-CAM for single image
    - explain_batch: Grad-CAM for directory of images
    - explain_misclassifications: Find and explain errors
    - inference: Flexible inference with configurable outputs
    - production: Complete pipeline with all features
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Local imports
from model import VanillaCNN, get_pretrained_model
from utils.config import get_data_paths, get_normalization_config, load_config
from utils.data_loaders import get_data_loaders
from explainability import (
    GradCAM,
    GradCAMVisualizer,
    GradCAMMetrics,
    get_target_layer_robust,
    MetadataStorage,
    ExplainabilityMetadata,
    extract_heatmap_statistics,
    extract_spatial_distribution,
    extract_top_activations
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_inference_config(config_path: Path) -> Dict[str, Any]:
    """Load inference configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Create one using the template in ../conf/inference.yaml or ../conf/inference_example.yaml"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: Dict[str, Any], output_dir: Path):
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))

    handlers = [logging.StreamHandler()]

    if log_config.get('save_log', True):
        log_file = output_dir / log_config.get('log_filename', 'inference.log')
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def extract_model_name_from_checkpoint(checkpoint_path: str) -> str:
    """Extract model architecture name from checkpoint directory."""
    dir_name = Path(checkpoint_path).parent.name

    known_models = [
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "vgg11", "vgg13", "vgg16", "vgg19",
        "vanilla",
    ]

    for model in known_models:
        if dir_name.startswith(model + "_") or dir_name == model:
            return model

    raise ValueError(
        f"Could not detect model type from directory '{dir_name}'. "
        f"Known models: {known_models}"
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    num_classes: int,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, str, Dict]:
    """Load trained model from checkpoint."""
    if device is None:
        device = get_device()

    checkpoint_path = str(checkpoint_path)

    if model_name is None:
        model_name = extract_model_name_from_checkpoint(checkpoint_path)

    logging.info(f"Model architecture: {model_name}")

    # Create model
    if model_name == "vanilla":
        model = VanillaCNN(num_classes=num_classes)
    else:
        model = get_pretrained_model(
            model_name,
            num_classes=num_classes,
            pretrained=False
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "val_loss": checkpoint.get("val_loss", "unknown"),
        "val_acc": checkpoint.get("val_acc", "unknown"),
    }

    logging.info(f"Loaded checkpoint from epoch {info['epoch']}")
    if isinstance(info["val_loss"], float):
        logging.info(f"  Validation loss: {info['val_loss']:.4f}")
    if isinstance(info["val_acc"], float):
        logging.info(f"  Validation accuracy: {info['val_acc']:.2f}%")

    return model, model_name, info


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Get inference preprocessing transform."""
    norm_config = get_normalization_config()
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_config.get("mean", [0.485, 0.456, 0.406]),
            std=norm_config.get("std", [0.229, 0.224, 0.225])
        ),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for display."""
    norm_config = get_normalization_config()
    mean = np.array(norm_config.get("mean", [0.485, 0.456, 0.406])).reshape(3, 1, 1)
    std = np.array(norm_config.get("std", [0.229, 0.224, 0.225])).reshape(3, 1, 1)

    img = tensor.detach().cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img.transpose(1, 2, 0)


def get_class_names_from_data_dir() -> List[str]:
    """Get class names from data directory."""
    paths = get_data_paths()
    train_dir = paths["train"]

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Please ensure data is prepared using prepare_data.py --split"
        )

    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return class_names


# ============================================================================
# INFERENCE PIPELINE CLASS (for Streamlit app compatibility)
# ============================================================================

class InferencePipeline:
    """
    Simple inference pipeline for single-image classification with Grad-CAM.

    This class provides a clean interface for the Streamlit app and other
    applications that need to classify individual images with explanations.

    Usage:
        pipeline = InferencePipeline("checkpoints/resnet18.../best_model.pth")
        result = pipeline.explain(image, top_k=5)
        # result contains: predicted_class, confidence, top_k, overlay
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize pipeline with a trained model checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
        """
        self.checkpoint_path = checkpoint_path
        self.device = get_device()

        # Get class names
        self.class_names = get_class_names_from_data_dir()
        self.num_classes = len(self.class_names)

        # Load model
        self.model, self.model_name, _ = load_model_from_checkpoint(
            checkpoint_path,
            num_classes=self.num_classes,
            model_name=None,
            device=self.device
        )
        self.model.eval()

        # Setup Grad-CAM
        self.target_layer = get_target_layer_robust(self.model, self.model_name)
        self.gradcam = GradCAM(self.model, self.target_layer, self.device)
        self.visualizer = GradCAMVisualizer(self.class_names)

        # Setup preprocessing
        self.transform = get_inference_transform()

    def predict(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify an image and return top-K predictions.

        Args:
            image: PIL Image in RGB format
            top_k: Number of top predictions to return

        Returns:
            Dictionary with:
                - predicted_class: str
                - confidence: float (0-1)
                - top_k: List[Tuple[str, float]] - (class_name, probability) pairs
        """
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        # Get top-K
        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        predicted_idx = top_indices[0]
        predicted_class = self.class_names[predicted_idx]
        confidence = float(top_probs[0])

        top_k_results = [
            (self.class_names[idx], float(prob))
            for idx, prob in zip(top_indices, top_probs)
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_k": top_k_results,
        }

    def explain(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify an image and generate Grad-CAM explanation.

        Args:
            image: PIL Image in RGB format
            top_k: Number of top predictions to return

        Returns:
            Dictionary with:
                - predicted_class: str
                - confidence: float (0-1)
                - top_k: List[Tuple[str, float]] - (class_name, probability) pairs
                - overlay: np.ndarray - Grad-CAM overlay image (H, W, 3) in [0, 1]
                - heatmap: np.ndarray - Raw heatmap (H, W) in [0, 1]
        """
        # Get predictions
        result = self.predict(image, top_k=top_k)

        # Preprocess for Grad-CAM
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate Grad-CAM
        predicted_class = result["predicted_class"]
        predicted_idx = self.class_names.index(predicted_class)

        heatmap = self.gradcam.generate(
            input_tensor,
            target_class=predicted_idx,
            handle_zero_gradients=True
        )

        # Create visualization
        original_array = np.array(image.resize((224, 224))) / 255.0
        overlay = self.visualizer.overlay_heatmap(
            original_array,
            heatmap,
            alpha=0.4
        )

        result["overlay"] = overlay
        result["heatmap"] = heatmap

        return result

    def __repr__(self):
        return (
            f"InferencePipeline(\n"
            f"  checkpoint={self.checkpoint_path}\n"
            f"  model={self.model_name}\n"
            f"  classes={self.num_classes}\n"
            f"  device={self.device}\n"
            f")"
        )


def flag_suspicious_with_thresholds(
    metrics: Dict[str, float],
    thresholds: Dict[str, float]
) -> Tuple[bool, str]:
    """Flag predictions using custom thresholds from config."""
    reasons = []

    min_drop = thresholds.get('min_confidence_drop', 10.0)
    if metrics['confidence_drop_pct'] < min_drop:
        reasons.append(f"Low confidence drop (<{min_drop}%)")

    max_increase = thresholds.get('max_confidence_increase', -20.0)
    if metrics['confidence_increase_pct'] < max_increase:
        reasons.append(f"Confidence decreased >{abs(max_increase)}%")

    min_concentration = thresholds.get('min_concentration', 0.25)
    if metrics['concentration'] < min_concentration:
        reasons.append(f"Very diffuse heatmap (<{min_concentration})")

    max_coverage = thresholds.get('max_coverage', 0.75)
    if metrics['coverage'] > max_coverage:
        reasons.append(f"Very high coverage (>{max_coverage*100:.0f}%)")

    min_coverage = thresholds.get('min_coverage', 0.05)
    if metrics['coverage'] < min_coverage:
        reasons.append(f"Very low coverage (<{min_coverage*100:.0f}%)")

    is_suspicious = len(reasons) > 0
    reason = "; ".join(reasons) if is_suspicious else "No issues detected"

    return is_suspicious, reason


# ============================================================================
# MODE HANDLERS
# ============================================================================

def calculate_per_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Calculate precision, recall, and F1 score for each class."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }

    return per_class


def calculate_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """Calculate confusion matrix."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(labels, predictions, labels=range(num_classes))


# ============================================================================
# VISUALIZATION FUNCTIONS (from evaluate.py)
# ============================================================================

def save_text_summary(
    results: Dict,
    save_path: str,
    checkpoint_path: str,
    model_type: str,
    timestamp: str
) -> None:
    """
    Save a text summary of evaluation results.

    Args:
        results: Dictionary containing evaluation results.
        save_path: Path to save the text file.
        checkpoint_path: Path to the model checkpoint.
        model_type: Model type ('best' or 'final').
        timestamp: Evaluation timestamp.
    """
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ARCHITECTURAL STYLE CLASSIFICATION - EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Model Type: {model_type}\n\n")

        f.write("=" * 70 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test Loss: {results['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.2f}%\n")

        if 'top_5_accuracy' in results:
            f.write(f"Top-5 Accuracy: {results['top_5_accuracy']:.2f}%\n")

        f.write(f"\nTotal Samples: {results.get('total_samples', len(results.get('labels', [])))}\n")
        f.write(f"Number of Classes: {len(results.get('class_names', results.get('per_class_metrics', {}).keys()))}\n\n")

        if 'per_class_metrics' in results:
            f.write("=" * 70 + "\n")
            f.write("PER-CLASS METRICS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 70 + "\n")

            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"{class_name:<20} {metrics['precision']:<12.3f} "
                       f"{metrics['recall']:<12.3f} {metrics['f1']:<12.3f} "
                       f"{metrics['support']:<10}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Evaluation Summary\n")
        f.write("=" * 70 + "\n")


def plot_confusion_matrix(
    results: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        results: Dictionary containing evaluation results.
        save_path: Path to save the plot. If None, displays interactively.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logging.warning("matplotlib/seaborn not installed. Skipping plot.")
        return

    # Handle confusion matrix from either format
    if 'confusion_matrix' in results:
        cm = results['confusion_matrix']
        if isinstance(cm, list):
            cm = np.array(cm)
    else:
        logging.warning("No confusion matrix found in results")
        return

    class_names = results.get('class_names', [f"Class {i}" for i in range(len(cm))])

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Confusion matrix plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history (loss and accuracy curves).

    Args:
        history_path: Path to training_history.json file.
        save_path: Path to save the plot. If None, displays interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed. Skipping plot.")
        return

    if not Path(history_path).exists():
        logging.warning(f"Training history not found: {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logging.info(f"Training history plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_per_class_metrics(
    results: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot per-class precision, recall, and F1-score as bar charts.

    Args:
        results: Dictionary containing evaluation results.
        save_path: Path to save the plot. If None, displays interactively.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logging.warning("matplotlib not installed. Skipping plot.")
        return

    metrics = results['per_class_metrics']
    class_names = list(metrics.keys())

    precision = [metrics[c]['precision'] for c in class_names]
    recall = [metrics[c]['recall'] for c in class_names]
    f1_score = [metrics[c]['f1'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Per-class metrics plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_class_accuracy_heatmap(
    results: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a heatmap showing class-wise accuracy.

    Args:
        results: Dictionary containing evaluation results.
        save_path: Path to save the plot. If None, displays interactively.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError:
        logging.warning("matplotlib/seaborn not installed. Skipping plot.")
        return

    cm = results.get('confusion_matrix')
    if cm is None:
        logging.warning("No confusion matrix in results")
        return

    if isinstance(cm, list):
        cm = np.array(cm)

    class_names = results.get('class_names', [f"Class {i}" for i in range(len(cm))])

    # Calculate per-class accuracy
    class_accuracy = np.diag(cm) / cm.sum(axis=1)

    # Create a grid for heatmap
    accuracy_grid = class_accuracy.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(8, len(class_names) * 0.4 + 1))

    sns.heatmap(accuracy_grid, annot=True, fmt='.2%', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'},
                yticklabels=class_names, xticklabels=['Accuracy'],
                ax=ax)

    ax.set_title('Per-Class Accuracy')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Class accuracy heatmap saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comprehensive_analysis(
    results: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive multi-panel visualization of evaluation results.

    Args:
        results: Dictionary containing evaluation results.
        save_path: Path to save the plot. If None, displays interactively.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError:
        logging.warning("matplotlib/seaborn not installed. Skipping plot.")
        return

    cm = results.get('confusion_matrix')
    if cm is None:
        logging.warning("No confusion matrix in results")
        return

    if isinstance(cm, list):
        cm = np.array(cm)

    class_names = results.get('class_names', [f"Class {i}" for i in range(len(cm))])
    metrics = results['per_class_metrics']

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Normalized Confusion Matrix
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 2. Per-Class Metrics Bar Chart
    ax2 = fig.add_subplot(gs[0, 2])
    precision = [metrics[c]['precision'] for c in class_names]
    recall = [metrics[c]['recall'] for c in class_names]
    f1_score = [metrics[c]['f1'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    ax2.barh(x - width, precision, width, label='Precision', alpha=0.8)
    ax2.barh(x, recall, width, label='Recall', alpha=0.8)
    ax2.barh(x + width, f1_score, width, label='F1-Score', alpha=0.8)

    ax2.set_yticks(x)
    ax2.set_yticklabels(class_names, fontsize=8)
    ax2.set_xlabel('Score')
    ax2.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim([0, 1.1])

    # 3. Class Distribution
    ax3 = fig.add_subplot(gs[1, 2])
    support = [metrics[c]['support'] for c in class_names]
    ax3.barh(class_names, support, color='steelblue', alpha=0.7)
    ax3.set_xlabel('Number of Samples')
    ax3.set_title('Test Set Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(support):
        ax3.text(v + max(support)*0.01, i, str(v), va='center', fontsize=8)

    # 4. Class-wise Accuracy
    ax4 = fig.add_subplot(gs[2, 0])
    class_accuracy = np.diag(cm) / cm.sum(axis=1)
    colors = ['green' if acc >= 0.7 else 'orange' if acc >= 0.5 else 'red'
              for acc in class_accuracy]
    ax4.bar(class_names, class_accuracy * 100, color=colors, alpha=0.7)
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
    ax4.axhline(y=results['test_accuracy'], color='r', linestyle='--',
                label=f"Overall: {results['test_accuracy']:.1f}%")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # 5. Top Misclassifications
    ax5 = fig.add_subplot(gs[2, 1])
    misclass = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclass.append((f"{class_names[i][:8]}\n→{class_names[j][:8]}", cm[i, j]))

    misclass.sort(key=lambda x: x[1], reverse=True)
    top_misclass = misclass[:8]

    if top_misclass:
        labels, counts = zip(*top_misclass)
        ax5.barh(range(len(labels)), counts, color='coral', alpha=0.7)
        ax5.set_yticks(range(len(labels)))
        ax5.set_yticklabels(labels, fontsize=7)
        ax5.set_xlabel('Count')
        ax5.set_title('Top Misclassifications', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(counts):
            ax5.text(v + max(counts)*0.01, i, str(v), va='center', fontsize=8)

    # 6. Overall Metrics Summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    total_samples = results.get('total_samples', len(results.get('labels', [])))

    summary_text = f"""
    EVALUATION SUMMARY
    {'='*30}

    Overall Accuracy: {results['test_accuracy']:.2f}%
    Overall Loss: {results['test_loss']:.4f}

    Total Samples: {total_samples}
    Number of Classes: {len(class_names)}

    Best Class Accuracy:
      {class_names[np.argmax(class_accuracy)]}
      ({class_accuracy.max()*100:.1f}%)

    Worst Class Accuracy:
      {class_names[np.argmin(class_accuracy)]}
      ({class_accuracy.min()*100:.1f}%)
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

    plt.suptitle('Comprehensive Evaluation Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logging.info(f"Comprehensive analysis plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def run_evaluate_mode(
    model: nn.Module,
    class_names: List[str],
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path
):
    """Run evaluation mode on test set."""
    logging.info("=" * 70)
    logging.info("EVALUATION MODE")
    logging.info("=" * 70)

    eval_config = config['evaluation']
    eval_mode_config = config['evaluate_mode']

    # Get test loader
    paths = get_data_paths()
    test_dir = paths["test"]

    if not test_dir.exists():
        logging.error(f"Test directory not found: {test_dir}")
        return

    # Create data loader
    _, _, test_loader, _, _ = get_data_loaders(
        batch_size=eval_mode_config['test_batch_size'],
        num_workers=eval_mode_config['num_workers']
    )

    # Run evaluation
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    correct = 0
    total = 0

    logging.info("Evaluating on test set...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    test_loss = running_loss / total
    test_accuracy = 100.0 * correct / total

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "total_samples": total
    }

    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.2f}%")

    # Per-class metrics
    if eval_config['metrics']['per_class_metrics']:
        per_class = calculate_per_class_metrics(all_labels, all_predictions, class_names)
        results['per_class_metrics'] = per_class

        logging.info("\nPer-class metrics:")
        for class_name, metrics in per_class.items():
            logging.info(f"  {class_name}:")
            logging.info(f"    Precision: {metrics['precision']:.3f}")
            logging.info(f"    Recall: {metrics['recall']:.3f}")
            logging.info(f"    F1: {metrics['f1']:.3f}")
            logging.info(f"    Support: {metrics['support']}")

    # Confusion matrix
    if eval_config['metrics']['confusion_matrix']:
        cm = calculate_confusion_matrix(all_labels, all_predictions, len(class_names))
        results['confusion_matrix'] = cm.tolist()

    # Top-K accuracy
    if eval_config['metrics']['top_k_accuracy']:
        top_k = eval_config['metrics']['top_k']
        top_k_acc = 0
        for i, probs in enumerate(all_probs):
            top_k_preds = np.argsort(probs)[-top_k:]
            if all_labels[i] in top_k_preds:
                top_k_acc += 1
        top_k_acc = 100.0 * top_k_acc / len(all_labels)
        results[f'top_{top_k}_accuracy'] = top_k_acc
        logging.info(f"Top-{top_k} Accuracy: {top_k_acc:.2f}%")

    # Add class_names and labels to results for visualization functions
    results['class_names'] = class_names
    results['labels'] = all_labels
    results['predictions'] = all_predictions

    # Save results as JSON
    if eval_config['save']['metrics_json']:
        results_to_save = {
            "test_loss": results['test_loss'],
            "test_accuracy": results['test_accuracy'],
            "per_class_metrics": results.get('per_class_metrics', {}),
            "confusion_matrix": results.get('confusion_matrix', []),
            "total_samples": total,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        if f'top_{eval_config["metrics"]["top_k"]}_accuracy' in results:
            results_to_save[f'top_{eval_config["metrics"]["top_k"]}_accuracy'] = results[f'top_{eval_config["metrics"]["top_k"]}_accuracy']

        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logging.info(f"Saved evaluation results to: {results_path}")

    # Save text summary
    summary_path = output_dir / 'evaluation_summary.txt'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = config['model']['checkpoint']
    model_type = "best" if "best_model" in checkpoint_path else "final"
    save_text_summary(results, str(summary_path), checkpoint_path, model_type, timestamp)

    if eval_config['save']['classification_report_txt']:
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_predictions,
                                      target_names=class_names)
        report_path = output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logging.info(f"Saved classification report to: {report_path}")

    # Generate visualizations
    if eval_config['visualize'].get('per_class_bars', True):
        logging.info("Generating visualizations...")

        # 1. Comprehensive analysis (all-in-one)
        comprehensive_path = output_dir / 'comprehensive_analysis.png'
        plot_comprehensive_analysis(results, save_path=str(comprehensive_path))

        # 2. Individual plots
        # Confusion matrix (enhanced version)
        if eval_config['save']['confusion_matrix_plot']:
            confusion_matrix_path = output_dir / 'confusion_matrix.png'
            plot_confusion_matrix(results, save_path=str(confusion_matrix_path))

        # Per-class metrics bar chart
        if 'per_class_metrics' in results:
            per_class_path = output_dir / 'per_class_metrics.png'
            plot_per_class_metrics(results, save_path=str(per_class_path))

        # Class accuracy heatmap
        class_accuracy_path = output_dir / 'class_accuracy.png'
        plot_class_accuracy_heatmap(results, save_path=str(class_accuracy_path))

        # 3. Training history (if available)
        checkpoint_dir = Path(checkpoint_path).parent
        history_path = checkpoint_dir / "training_history.json"
        if history_path.exists():
            training_history_path = output_dir / 'train_history.png'
            plot_training_history(str(history_path), save_path=str(training_history_path))
        else:
            logging.info("Note: training_history.json not found - skipping training history plot")

    logging.info("=" * 70)
    logging.info("EVALUATION COMPLETE")
    logging.info(f"All results saved to: {output_dir}")
    logging.info("=" * 70)


def run_explain_single_mode(
    model: nn.Module,
    model_name: str,
    class_names: List[str],
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path
):
    """Run Grad-CAM explanation for single image."""
    logging.info("=" * 70)
    logging.info("EXPLAIN SINGLE IMAGE MODE")
    logging.info("=" * 70)

    # Get image path
    single_config = config['explain_single_mode']
    image_path = single_config.get('image_path') or config['paths'].get('image_path')

    if not image_path:
        logging.error("No image_path specified in config")
        return

    image_path = Path(image_path)
    if not image_path.exists():
        logging.error(f"Image not found: {image_path}")
        return

    logging.info(f"Image: {image_path}")

    # Load and preprocess
    original_image = Image.open(image_path)
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    transform = get_inference_transform()
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze(0)

    pred_idx = probabilities.argmax().item()
    predicted_class = class_names[pred_idx]
    confidence = probabilities[pred_idx].item()

    # Get top-K
    top_k = config['gradcam']['top_k']
    top_k_probs, top_k_indices = probabilities.topk(top_k)
    top_k_predictions = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_k_indices, top_k_probs)
    ]

    logging.info(f"Predicted: {predicted_class} ({confidence:.1%})")
    logging.info(f"Top-{top_k}: {top_k_predictions}")

    # Generate Grad-CAM if enabled
    if config['gradcam']['enabled']:
        target_layer = get_target_layer_robust(model, model_name)
        gradcam = GradCAM(model, target_layer, device)

        input_tensor.requires_grad_(True)
        heatmap = gradcam.generate(
            input_tensor,
            target_class=pred_idx,
            handle_zero_gradients=config['gradcam']['handle_zero_gradients']
        )

        # Create visualization
        visualizer = GradCAMVisualizer(class_names)
        preprocessed_image = denormalize(input_tensor.squeeze(0))

        output_path = output_dir / f"{image_path.stem}_gradcam.png"
        fig = visualizer.create_single_explanation(
            original_image=original_image,
            preprocessed_tensor=preprocessed_image,
            heatmap=heatmap,
            predicted_class=predicted_class,
            confidence=confidence,
            true_class=single_config.get('true_class'),
            top_k_predictions=top_k_predictions,
            output_path=output_path
        )
        visualizer.close_figure(fig)

        gradcam.remove_hooks()

        logging.info(f"Saved visualization to: {output_path}")

    logging.info("=" * 70)


def run_explain_batch_mode(
    model: nn.Module,
    model_name: str,
    class_names: List[str],
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path
):
    """Run Grad-CAM explanations for batch of images."""
    logging.info("=" * 70)
    logging.info("EXPLAIN BATCH MODE")
    logging.info("=" * 70)

    image_dir = Path(config['paths']['image_dir'])
    if not image_dir.exists():
        logging.error(f"Image directory not found: {image_dir}")
        return

    # Find images (recursively search subdirectories)
    image_files = []
    for ext in config['processing']['image_extensions']:
        image_files.extend(image_dir.rglob(ext))
    image_files = sorted(image_files)

    limit = config['processing'].get('limit')
    if limit:
        image_files = image_files[:limit]

    logging.info(f"Found {len(image_files)} images")

    if not config['gradcam']['enabled']:
        logging.warning("Grad-CAM disabled in config")
        return

    # Setup
    target_layer = get_target_layer_robust(model, model_name)
    gradcam = GradCAM(model, target_layer, device)
    visualizer = GradCAMVisualizer(class_names)
    transform = get_inference_transform()

    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    # Process images
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing")):
        try:
            original_image = Image.open(img_path)
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            input_tensor = transform(original_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1).squeeze(0)

            pred_idx = probabilities.argmax().item()
            predicted_class = class_names[pred_idx]
            confidence = probabilities[pred_idx].item()

            # Generate Grad-CAM
            input_tensor.requires_grad_(True)
            heatmap = gradcam.generate(input_tensor, target_class=pred_idx)

            # Save visualization
            preprocessed_image = denormalize(input_tensor.squeeze(0))
            output_path = heatmap_dir / f"{img_path.stem}_gradcam.png"

            fig = visualizer.create_single_explanation(
                original_image=original_image,
                preprocessed_tensor=preprocessed_image,
                heatmap=heatmap,
                predicted_class=predicted_class,
                confidence=confidence,
                output_path=output_path
            )
            visualizer.close_figure(fig)

        except Exception as e:
            logging.error(f"Error processing {img_path.name}: {e}")
            if not config['processing']['skip_errors']:
                raise

    gradcam.remove_hooks()
    logging.info(f"Saved {len(image_files)} visualizations to: {heatmap_dir}")


def run_explain_misclassifications_mode(
    model: nn.Module,
    model_name: str,
    class_names: List[str],
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path
):
    """Find and explain misclassified test images."""
    logging.info("=" * 70)
    logging.info("EXPLAIN MISCLASSIFICATIONS MODE")
    logging.info("=" * 70)

    paths = get_data_paths()
    test_dir = paths["test"]

    if not test_dir.exists():
        logging.error(f"Test directory not found: {test_dir}")
        return

    misclass_config = config['explain_misclassifications_mode']
    limit = misclass_config['limit']

    # Find misclassifications
    transform = get_inference_transform()
    misclassifications = []

    logging.info("Scanning test set for misclassifications...")

    for true_class_idx, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue

        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

        for img_file in image_files:
            try:
                original_image = Image.open(img_file)
                if original_image.mode != "RGB":
                    original_image = original_image.convert("RGB")

                input_tensor = transform(original_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1).squeeze(0)

                pred_idx = probabilities.argmax().item()

                if pred_idx != true_class_idx:
                    misclassifications.append({
                        'image_path': img_file,
                        'true_class': class_name,
                        'true_idx': true_class_idx,
                        'pred_class': class_names[pred_idx],
                        'pred_idx': pred_idx,
                        'confidence': probabilities[pred_idx].item(),
                        'original_image': original_image
                    })
            except Exception as e:
                logging.warning(f"Error processing {img_file.name}: {e}")

    if len(misclassifications) == 0:
        logging.info("No misclassifications found! Model has 100% accuracy on test set.")
        return

    logging.info(f"Found {len(misclassifications)} misclassifications")

    # Sort if requested
    if misclass_config['sort_by'] == 'confidence':
        misclassifications.sort(key=lambda x: x['confidence'], reverse=True)

    # Limit
    misclassifications = misclassifications[:limit]

    # Generate Grad-CAM
    if not config['gradcam']['enabled']:
        logging.warning("Grad-CAM disabled, skipping visualizations")
        return

    target_layer = get_target_layer_robust(model, model_name)
    gradcam = GradCAM(model, target_layer, device)
    visualizer = GradCAMVisualizer(class_names)

    misclass_dir = output_dir / "misclassifications"
    misclass_dir.mkdir(exist_ok=True)

    for idx, misclass in enumerate(tqdm(misclassifications, desc="Generating explanations")):
        try:
            input_tensor = transform(misclass['original_image']).unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            heatmap = gradcam.generate(input_tensor, target_class=misclass['pred_idx'])

            preprocessed_image = denormalize(input_tensor.squeeze(0))
            output_path = misclass_dir / f"misclass_{idx:03d}_{misclass['true_class']}_as_{misclass['pred_class']}.png"

            fig = visualizer.create_single_explanation(
                original_image=misclass['original_image'],
                preprocessed_tensor=preprocessed_image,
                heatmap=heatmap,
                predicted_class=misclass['pred_class'],
                confidence=misclass['confidence'],
                true_class=misclass['true_class'],
                output_path=output_path
            )
            visualizer.close_figure(fig)

        except Exception as e:
            logging.error(f"Error generating explanation: {e}")

    gradcam.remove_hooks()
    logging.info(f"Saved {len(misclassifications)} explanations to: {misclass_dir}")


def run_inference_mode(
    model: nn.Module,
    model_name: str,
    class_names: List[str],
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path
):
    """Run flexible inference mode with configurable outputs."""
    logging.info("=" * 70)
    logging.info("INFERENCE MODE")
    logging.info("=" * 70)

    # Determine if single image or batch
    image_path = config['paths'].get('image_path')
    image_dir = config['paths'].get('image_dir')

    if image_path:
        # Single image
        run_explain_single_mode(model, model_name, class_names, config, device, output_dir)
    elif image_dir:
        # Batch inference with configurable outputs
        run_production_mode(model, model_name, class_names, config, device, output_dir)
    else:
        logging.error("No image_path or image_dir specified")


def run_production_mode(
    model: nn.Module,
    model_name: str,
    class_names: List[str],
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path
):
    """Run full production pipeline with all features."""
    logging.info("=" * 70)
    logging.info("PRODUCTION MODE")
    logging.info("=" * 70)

    prod_config = config['production_mode']

    # Run evaluation if requested
    if prod_config.get('run_evaluation', False):
        paths = get_data_paths()
        if paths["test"].exists():
            run_evaluate_mode(model, class_names, config, device, output_dir)

    # Run batch inference with quality checks
    image_dir = Path(config['paths']['image_dir'])
    if not image_dir.exists():
        logging.error(f"Image directory not found: {image_dir}")
        return

    # Find images (recursively search subdirectories)
    image_files = []
    for ext in config['processing']['image_extensions']:
        image_files.extend(image_dir.rglob(ext))
    image_files = sorted(image_files)

    limit = config['processing'].get('limit')
    if limit:
        image_files = image_files[:limit]

    logging.info(f"Found {len(image_files)} images to process")

    # Setup
    transform = get_inference_transform()

    # Setup Grad-CAM if enabled
    gradcam = None
    visualizer = None
    if config['gradcam']['enabled']:
        target_layer = get_target_layer_robust(model, model_name)
        gradcam = GradCAM(model, target_layer, device)
        visualizer = GradCAMVisualizer(class_names)

    # Setup output directories
    outputs_config = config['outputs']
    heatmap_config = outputs_config['gradcam_heatmaps']

    if heatmap_config['enabled'] and heatmap_config['separate_dirs']:
        heatmap_dir = output_dir / "heatmaps"
        suspicious_dir = output_dir / "suspicious"
        heatmap_dir.mkdir(exist_ok=True)
        suspicious_dir.mkdir(exist_ok=True)
    else:
        heatmap_dir = output_dir / "heatmaps"
        suspicious_dir = heatmap_dir
        heatmap_dir.mkdir(exist_ok=True)

    # Setup metadata storage
    metadata_config = outputs_config['metadata']
    if metadata_config['enabled']:
        metadata_path = output_dir / metadata_config['filename']
        metadata_storage = MetadataStorage(metadata_path, format=metadata_config['format'])

    # Process images
    num_visualizations_saved = 0
    num_flagged = 0
    num_errors = 0

    quality_enabled = config['quality_metrics']['enabled']
    flag_suspicious = config['quality_metrics']['flag_suspicious']

    iterator = tqdm(image_files, desc="Processing") if config['processing']['show_progress'] else image_files

    for idx, img_path in enumerate(iterator):
        try:
            # Load image
            original_image = Image.open(img_path)
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            input_tensor = transform(original_image).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1).squeeze(0)

            pred_idx = probabilities.argmax().item()
            predicted_class = class_names[pred_idx]
            confidence = probabilities[pred_idx].item()

            # Get top-K
            top_k = config['gradcam']['top_k']
            top_k_probs, top_k_indices = probabilities.topk(top_k)
            top_k_classes = [class_names[i.item()] for i in top_k_indices]
            top_k_confidences = [p.item() for p in top_k_probs]

            # Generate Grad-CAM with quality metrics if enabled
            heatmap = None
            metrics = {}
            is_suspicious = False
            suspicious_reason = ""

            if gradcam:
                input_tensor.requires_grad_(True)

                if quality_enabled:
                    heatmap, metrics = gradcam.generate_with_metrics(
                        input_tensor,
                        target_class=pred_idx,
                        compute_metrics=True
                    )

                    if flag_suspicious:
                        is_suspicious, suspicious_reason = flag_suspicious_with_thresholds(
                            metrics, config['quality_metrics']['thresholds']
                        )
                        if is_suspicious:
                            num_flagged += 1
                else:
                    heatmap = gradcam.generate(input_tensor, target_class=pred_idx)

            # Save metadata if enabled
            if metadata_config['enabled']:
                heatmap_stats = extract_heatmap_statistics(heatmap) if heatmap is not None else {}
                spatial_dist = extract_spatial_distribution(heatmap) if heatmap is not None else {}
                top_activations_data = None

                if metadata_config['include_top_activations'] and heatmap is not None:
                    top_activations_data = extract_top_activations(
                        heatmap, top_n=metadata_config['top_activations_count']
                    )

                metadata = ExplainabilityMetadata(
                    image_path=str(img_path),
                    image_id=img_path.stem,
                    model_name=model_name,
                    checkpoint_path=str(config['model']['checkpoint']),
                    predicted_class=predicted_class,
                    predicted_class_idx=pred_idx,
                    confidence=confidence,
                    top_k_classes=top_k_classes,
                    top_k_confidences=top_k_confidences,
                    **heatmap_stats,
                    **spatial_dist,
                    top_activations=top_activations_data,
                    is_suspicious=is_suspicious,
                    suspicious_reason=suspicious_reason,
                    **metrics if quality_enabled else {}
                )

                metadata_storage.append_to_file(metadata)

            # Save visualization if configured
            save_viz = False
            save_count = heatmap_config.get('save_count')

            if heatmap_config['enabled'] and heatmap is not None:
                if heatmap_config['save_suspicious_only']:
                    save_viz = is_suspicious
                elif save_count is not None:
                    save_viz = num_visualizations_saved < save_count
                else:
                    save_viz = True

                if save_viz:
                    preprocessed_image = denormalize(input_tensor.squeeze(0))

                    if is_suspicious and heatmap_config['separate_dirs']:
                        output_path = suspicious_dir / f"{img_path.stem}_FLAGGED.png"
                    else:
                        output_path = heatmap_dir / f"{img_path.stem}_gradcam.png"

                    fig = visualizer.create_single_explanation(
                        original_image=original_image,
                        preprocessed_tensor=preprocessed_image,
                        heatmap=heatmap,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        top_k_predictions=list(zip(top_k_classes, top_k_confidences)),
                        output_path=output_path
                    )
                    visualizer.close_figure(fig)
                    num_visualizations_saved += 1

        except Exception as e:
            num_errors += 1
            logging.error(f"Error processing {img_path.name}: {e}")
            if not config['processing']['skip_errors']:
                raise

    # Cleanup
    if gradcam:
        gradcam.remove_hooks()

    # Summary
    logging.info("=" * 70)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Total images: {len(image_files)}")
    logging.info(f"Successfully processed: {len(image_files) - num_errors}")
    if num_errors > 0:
        logging.warning(f"Errors: {num_errors}")
    logging.info(f"Visualizations saved: {num_visualizations_saved}")
    if flag_suspicious:
        flagged_pct = (num_flagged / len(image_files) * 100) if len(image_files) > 0 else 0
        logging.info(f"Suspicious predictions flagged: {num_flagged} ({flagged_pct:.1f}%)")

    if metadata_config['enabled']:
        logging.info(f"Metadata saved to: {metadata_path}")
    logging.info(f"Output directory: {output_dir}")

    # Generate summary report if requested
    if prod_config.get('generate_report', False):
        report_path = output_dir / prod_config.get('report_filename', 'inference_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PRODUCTION INFERENCE REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Checkpoint: {config['model']['checkpoint']}\n\n")
            f.write(f"Total images processed: {len(image_files)}\n")
            f.write(f"Successful: {len(image_files) - num_errors}\n")
            f.write(f"Errors: {num_errors}\n\n")
            if flag_suspicious:
                f.write(f"Flagged predictions: {num_flagged} ({flagged_pct:.1f}%)\n\n")
            f.write(f"Visualizations saved: {num_visualizations_saved}\n")
            if metadata_config['enabled']:
                f.write(f"Metadata file: {metadata_path.name}\n")

        logging.info(f"Report saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified inference script for architectural style classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config", type=str, default="../conf/inference.yaml",
        help="Path to YAML configuration file (default: ../conf/inference.yaml)"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["evaluate", "explain_single", "explain_batch",
                "explain_misclassifications", "inference", "production"],
        help="Override mode from config"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override checkpoint path"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Override image path (for single image modes)"
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="Override image directory (for batch modes)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    try:
        config = load_inference_config(config_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"\nERROR: Invalid YAML: {e}")
        sys.exit(1)

    # Apply CLI overrides
    if args.mode:
        config['mode'] = args.mode
    if args.checkpoint:
        config['model']['checkpoint'] = args.checkpoint
    if args.image:
        config['paths']['image_path'] = args.image
    if args.image_dir:
        config['paths']['image_dir'] = args.image_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir

    # Determine output directory
    # Priority: CLI arg > config > checkpoint_dir/results/
    checkpoint_path = Path(config['model']['checkpoint'])

    if args.output_dir:
        # CLI override
        output_dir = Path(args.output_dir)
    elif config['paths'].get('output_dir'):
        # From config
        output_dir = Path(config['paths']['output_dir'])
    else:
        # Default: use checkpoint directory + results/
        checkpoint_dir = checkpoint_path.parent
        output_dir = checkpoint_dir / "results"

    config['paths']['output_dir'] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, output_dir)

    logging.info("=" * 70)
    logging.info("UNIFIED INFERENCE PIPELINE")
    logging.info("=" * 70)
    logging.info(f"Config: {config_path}")
    logging.info(f"Mode: {config['mode']}")
    logging.info(f"Output: {output_dir}")

    # Load model
    device = get_device()
    logging.info(f"Device: {device}")

    checkpoint_path = Path(config['model']['checkpoint'])
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    class_names = get_class_names_from_data_dir()
    num_classes = len(class_names)
    logging.info(f"Classes: {num_classes}")

    model, model_name, checkpoint_info = load_model_from_checkpoint(
        str(checkpoint_path),
        num_classes,
        config['model'].get('name'),
        device
    )

    # Route to appropriate mode handler
    mode = config['mode']

    if mode == "evaluate":
        run_evaluate_mode(model, class_names, config, device, output_dir)
    elif mode == "explain_single":
        run_explain_single_mode(model, model_name, class_names, config, device, output_dir)
    elif mode == "explain_batch":
        run_explain_batch_mode(model, model_name, class_names, config, device, output_dir)
    elif mode == "explain_misclassifications":
        run_explain_misclassifications_mode(model, model_name, class_names, config, device, output_dir)
    elif mode == "inference":
        run_inference_mode(model, model_name, class_names, config, device, output_dir)
    elif mode == "production":
        run_production_mode(model, model_name, class_names, config, device, output_dir)
    else:
        logging.error(f"Unknown mode: {mode}")
        sys.exit(1)

    logging.info("=" * 70)
    logging.info("DONE")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
