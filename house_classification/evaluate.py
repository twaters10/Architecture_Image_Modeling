#!/usr/bin/env python3
"""
Evaluation Script for Architectural Style Classification CNN.

This script provides comprehensive model evaluation including:
- Test set accuracy and loss
- Per-class precision, recall, and F1 scores
- Confusion matrix generation
- Classification report
- Visualization of results

Usage:
    # Evaluate a trained model
    python 05_evaluate.py --checkpoint checkpoints/best_model.pth --model vanilla

    # Evaluate with confusion matrix visualization
    python 05_evaluate.py --checkpoint checkpoints/best_model.pth --model resnet18 --plot

Requirements:
    - torch
    - torchvision
    - scikit-learn
    - matplotlib (optional, for plotting)
    - seaborn (optional, for plotting)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Local imports
from utils.config import load_config, get_data_paths
from utils.data_loaders import get_data_loaders
from model import VanillaCNN, get_pretrained_model


def get_device() -> torch.device:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """
    Evaluate the model on the test set.

    Args:
        model: Trained neural network model.
        test_loader: DataLoader for test data.
        device: Device to run evaluation on.
        class_names: List of class names for reporting.

    Returns:
        dict: Evaluation metrics including accuracy, per-class metrics,
              predictions, and ground truth labels.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    correct = 0
    total = 0

    print("\nEvaluating on test set...")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Accumulate statistics
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate overall metrics
    test_loss = running_loss / total
    test_accuracy = 100.0 * correct / total

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate per-class metrics
    per_class_metrics = calculate_per_class_metrics(
        all_labels, all_predictions, class_names
    )

    # Calculate confusion matrix
    confusion_matrix = calculate_confusion_matrix(
        all_labels, all_predictions, len(class_names)
    )

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probs,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "class_names": class_names
    }


def calculate_per_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Calculate precision, recall, and F1 score for each class.

    Args:
        labels: Ground truth labels.
        predictions: Model predictions.
        class_names: List of class names.

    Returns:
        dict: Per-class metrics.
    """
    num_classes = len(class_names)
    metrics = {}

    for i, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
        tp = np.sum((predictions == i) & (labels == i))
        fp = np.sum((predictions == i) & (labels != i))
        fn = np.sum((predictions != i) & (labels == i))

        # Calculate metrics (handle division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Support (number of true samples)
        support = np.sum(labels == i)

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(support)
        }

    return metrics


def calculate_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Calculate the confusion matrix.

    Args:
        labels: Ground truth labels.
        predictions: Model predictions.
        num_classes: Number of classes.

    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(labels, predictions):
        cm[true_label, pred_label] += 1
    return cm


def print_classification_report(results: Dict) -> None:
    """
    Print a formatted classification report.

    Args:
        results: Dictionary containing evaluation results.
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(f"\nOverall Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Overall Test Loss: {results['test_loss']:.4f}")

    print("\n" + "-" * 70)
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)

    metrics = results['per_class_metrics']
    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for class_name, class_metrics in metrics.items():
        print(f"{class_name:<25} "
              f"{class_metrics['precision']:>10.3f} "
              f"{class_metrics['recall']:>10.3f} "
              f"{class_metrics['f1_score']:>10.3f} "
              f"{class_metrics['support']:>10}")

        support = class_metrics['support']
        total_support += support
        weighted_precision += class_metrics['precision'] * support
        weighted_recall += class_metrics['recall'] * support
        weighted_f1 += class_metrics['f1_score'] * support

    # Calculate weighted averages
    weighted_precision /= total_support
    weighted_recall /= total_support
    weighted_f1 /= total_support

    print("-" * 70)
    print(f"{'Weighted Average':<25} "
          f"{weighted_precision:>10.3f} "
          f"{weighted_recall:>10.3f} "
          f"{weighted_f1:>10.3f} "
          f"{total_support:>10}")
    print("=" * 70)


def print_confusion_matrix(results: Dict) -> None:
    """
    Print the confusion matrix in text format.

    Args:
        results: Dictionary containing evaluation results.
    """
    cm = results['confusion_matrix']
    class_names = results['class_names']

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    # Print header
    max_len = max(len(name) for name in class_names)
    header = " " * (max_len + 2) + "  ".join(f"{name[:8]:>8}" for name in class_names)
    print(f"\n{'Predicted ->':>{max_len + 2}}")
    print(header)
    print("-" * len(header))

    # Print rows
    for i, class_name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:>8}" for j in range(len(class_names)))
        print(f"{class_name:>{max_len}} | {row}")

    print("=" * 70)


def save_text_summary(results: Dict, save_path: str, checkpoint_path: str, model_type: str, timestamp: str) -> None:
    """
    Save a comprehensive text summary of evaluation results.

    Args:
        results: Dictionary containing evaluation results.
        save_path: Path to save the text summary.
        checkpoint_path: Path to the checkpoint used.
        model_type: Type of model (best or final).
        timestamp: Timestamp of evaluation.
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ARCHITECTURAL STYLE CLASSIFICATION - EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Evaluation metadata
        f.write("EVALUATION METADATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Evaluation Date: {timestamp}\n")
        f.write(f"Number of Classes: {len(results['class_names'])}\n")
        f.write(f"Total Test Samples: {len(results['labels'])}\n")
        f.write("\n")

        # Overall metrics
        f.write("=" * 80 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.2f}%\n")
        f.write(f"Test Loss: {results['test_loss']:.4f}\n")
        f.write("\n")

        # Per-class metrics
        f.write("=" * 80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Class':<30} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}\n")
        f.write("-" * 80 + "\n")

        metrics = results['per_class_metrics']
        total_support = 0
        weighted_precision = 0
        weighted_recall = 0
        weighted_f1 = 0

        for class_name, class_metrics in metrics.items():
            f.write(f"{class_name:<30} "
                   f"{class_metrics['precision']:>12.4f} "
                   f"{class_metrics['recall']:>12.4f} "
                   f"{class_metrics['f1_score']:>12.4f} "
                   f"{class_metrics['support']:>12}\n")

            support = class_metrics['support']
            total_support += support
            weighted_precision += class_metrics['precision'] * support
            weighted_recall += class_metrics['recall'] * support
            weighted_f1 += class_metrics['f1_score'] * support

        # Weighted averages
        weighted_precision /= total_support
        weighted_recall /= total_support
        weighted_f1 /= total_support

        f.write("-" * 80 + "\n")
        f.write(f"{'Weighted Average':<30} "
               f"{weighted_precision:>12.4f} "
               f"{weighted_recall:>12.4f} "
               f"{weighted_f1:>12.4f} "
               f"{total_support:>12}\n")
        f.write("\n")

        # Confusion Matrix
        cm = results['confusion_matrix']
        class_names = results['class_names']

        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 80 + "\n\n")

        max_len = max(len(name) for name in class_names)
        header = " " * (max_len + 2) + "  ".join(f"{name[:10]:>10}" for name in class_names)
        f.write(f"{'True \\ Predicted':>{max_len + 2}}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for i, class_name in enumerate(class_names):
            row = "  ".join(f"{cm[i, j]:>10}" for j in range(len(class_names)))
            f.write(f"{class_name:>{max_len}} | {row}\n")

        f.write("\n")

        # Top confused pairs
        f.write("=" * 80 + "\n")
        f.write("TOP 10 MISCLASSIFICATION PAIRS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'True Class':<25} {'Predicted As':<25} {'Count':>10} {'% of True':>10}\n")
        f.write("-" * 80 + "\n")

        # Find top misclassifications
        misclassifications = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    total_for_class = cm[i].sum()
                    percentage = (cm[i, j] / total_for_class) * 100 if total_for_class > 0 else 0
                    misclassifications.append((class_names[i], class_names[j], cm[i, j], percentage))

        # Sort by count
        misclassifications.sort(key=lambda x: x[2], reverse=True)

        for true_class, pred_class, count, pct in misclassifications[:10]:
            f.write(f"{true_class:<25} {pred_class:<25} {count:>10} {pct:>9.1f}%\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def plot_confusion_matrix(
    results: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the confusion matrix using matplotlib and seaborn.

    Args:
        results: Dictionary containing evaluation results.
        save_path: Path to save the plot. If None, displays interactively.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib and/or seaborn not installed. Skipping plot.")
        return

    cm = results['confusion_matrix']
    class_names = results['class_names']

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nConfusion matrix saved to: {save_path}")
    else:
        plt.show()


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
        print("Warning: matplotlib not installed. Skipping plot.")
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
        print(f"\nTraining history plot saved to: {save_path}")
    else:
        plt.show()


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
        print("Warning: matplotlib not installed. Skipping plot.")
        return

    metrics = results['per_class_metrics']
    class_names = list(metrics.keys())

    precision = [metrics[c]['precision'] for c in class_names]
    recall = [metrics[c]['recall'] for c in class_names]
    f1_score = [metrics[c]['f1_score'] for c in class_names]

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
        print(f"\nPer-class metrics plot saved to: {save_path}")
    else:
        plt.show()


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
        print("Warning: matplotlib/seaborn not installed. Skipping plot.")
        return

    cm = results['confusion_matrix']
    class_names = results['class_names']

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
        print(f"\nClass accuracy heatmap saved to: {save_path}")
    else:
        plt.show()


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
        print("Warning: matplotlib/seaborn not installed. Skipping plot.")
        return

    cm = results['confusion_matrix']
    class_names = results['class_names']
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
    f1_score = [metrics[c]['f1_score'] for c in class_names]

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

    summary_text = f"""
    EVALUATION SUMMARY
    {'='*30}

    Overall Accuracy: {results['test_accuracy']:.2f}%
    Overall Loss: {results['test_loss']:.4f}

    Total Samples: {len(results['labels'])}
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
        print(f"\nComprehensive analysis plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def prompt_checkpoint_selection(checkpoints_dir: Path) -> Optional[Path]:
    """
    Prompt user to select a checkpoint from available checkpoints.
    Two-step process: first select directory, then select best/final model.

    Args:
        checkpoints_dir: Directory containing model checkpoints.

    Returns:
        Path to selected checkpoint or None if no checkpoints found.
    """
    # Find all checkpoint directories
    checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()],
                           key=lambda x: x.stat().st_mtime, reverse=True)

    if not checkpoint_dirs:
        print(f"\nNo checkpoint directories found in {checkpoints_dir}")
        return None

    # Step 1: Select checkpoint directory
    print("\n" + "=" * 70)
    print("AVAILABLE MODEL CHECKPOINT DIRECTORIES")
    print("=" * 70)

    for idx, checkpoint_dir in enumerate(checkpoint_dirs, 1):
        dir_name = checkpoint_dir.name
        # Check which models are available in this directory
        available_models = []
        if (checkpoint_dir / "best_model.pth").exists():
            available_models.append("best")
        if (checkpoint_dir / "final_model.pth").exists():
            available_models.append("final")
        models_str = ", ".join(available_models) if available_models else "no models"
        print(f"{idx}. {dir_name} [{models_str}]")

    print("=" * 70)

    # Get directory selection
    selected_dir = None
    while True:
        try:
            selection = input(f"\nSelect checkpoint directory (1-{len(checkpoint_dirs)}): ").strip()
            selection_idx = int(selection) - 1
            if 0 <= selection_idx < len(checkpoint_dirs):
                selected_dir = checkpoint_dirs[selection_idx]
                print(f"Selected directory: {selected_dir.name}")
                break
            else:
                print(f"Please enter a number between 1 and {len(checkpoint_dirs)}")
        except (ValueError, KeyboardInterrupt):
            print("\nSelection cancelled.")
            return None

    # Step 2: Select best or final model
    print("\n" + "=" * 70)
    print("SELECT MODEL CHECKPOINT")
    print("=" * 70)

    available_checkpoints = []
    if (selected_dir / "best_model.pth").exists():
        available_checkpoints.append(("best_model.pth", "Best model (lowest validation loss)"))
    if (selected_dir / "final_model.pth").exists():
        available_checkpoints.append(("final_model.pth", "Final model (last epoch)"))

    if not available_checkpoints:
        print(f"\nNo checkpoint files found in {selected_dir}")
        return None

    for idx, (filename, description) in enumerate(available_checkpoints, 1):
        print(f"{idx}. {filename} - {description}")

    print("=" * 70)

    # Get checkpoint selection
    while True:
        try:
            selection = input(f"\nSelect checkpoint (1-{len(available_checkpoints)}): ").strip()
            selection_idx = int(selection) - 1
            if 0 <= selection_idx < len(available_checkpoints):
                selected_checkpoint = selected_dir / available_checkpoints[selection_idx][0]
                print(f"Selected: {selected_checkpoint}")
                return selected_checkpoint
            else:
                print(f"Please enter a number between 1 and {len(available_checkpoints)}")
        except (ValueError, KeyboardInterrupt):
            print("\nSelection cancelled.")
            return None


def extract_model_type_from_path(checkpoint_path: Path) -> str:
    """
    Extract model type from checkpoint directory name.

    Checkpoint directories follow the format:
    {model}_ep{epochs}_bs{batch_size}_lr{lr}_{timestamp}

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Model architecture name extracted from the directory.
    """
    # Get the directory name
    checkpoint_dir = checkpoint_path.parent
    dir_name = checkpoint_dir.name

    # Known model types
    known_models = ["vanilla", "resnet18", "resnet34", "resnet50",
                   "efficientnet_b0", "mobilenet_v2"]

    # Try to match the directory name with known models
    for model in known_models:
        if dir_name.startswith(model + "_"):
            print(f"Detected model type from checkpoint path: {model}")
            return model

    # If we can't extract it, default to vanilla and warn
    print(f"Warning: Could not extract model type from directory name '{dir_name}'")
    print("Defaulting to 'vanilla'. If this is incorrect, please specify --model explicitly.")
    return "vanilla"


def prompt_plot_generation() -> bool:
    """
    Prompt user whether to generate plots.

    Returns:
        True if plots should be generated, False otherwise.
    """
    print("\n" + "=" * 70)
    print("GENERATE VISUALIZATION PLOTS?")
    print("=" * 70)
    print("1. Yes - Generate confusion matrix and training history plots")
    print("2. No - Skip plot generation")
    print("=" * 70)

    while True:
        try:
            selection = input("\nGenerate plots? (1-2): ").strip()
            if selection == "1":
                print("Plots will be generated")
                return True
            elif selection == "2":
                print("Skipping plot generation")
                return False
            else:
                print("Please enter 1 or 2")
        except (ValueError, KeyboardInterrupt):
            print("\nDefaulting to no plots")
            return False


def main():
    """Main entry point for evaluation script."""
    # Load config for defaults
    config = load_config()
    paths = get_data_paths()

    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN for architectural style classification. "
                    "Can be run interactively or with command-line arguments."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (if not provided, will prompt)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["vanilla", "resnet18", "resnet34", "resnet50",
                 "efficientnet_b0", "mobilenet_v2"],
        help="Model architecture (auto-detected from checkpoint path if not specified)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.get("data_loader", {}).get("batch_size", 32),
        help="Batch size for evaluation (default from config)"
    )
    parser.add_argument(
        "--plot", action="store_true", default=None,
        help="Generate and save visualization plots (if not provided, will prompt)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save evaluation results (default from config)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Architectural Style Classification - Evaluation")
    print("=" * 60)

    # Interactive prompts if arguments not provided
    if args.checkpoint is None:
        checkpoint_path = prompt_checkpoint_selection(paths["checkpoints"])
        if checkpoint_path is None:
            print("\nNo checkpoint selected. Exiting.")
            return
        args.checkpoint = str(checkpoint_path)

    # Extract model type from checkpoint path if not provided
    if args.model is None:
        args.model = extract_model_type_from_path(Path(args.checkpoint))

    if args.plot is None:
        args.plot = prompt_plot_generation()

    print("\n" + "=" * 60)
    print("Configuration loaded from conf/data.yaml")

    print("\n" + "=" * 60)
    print("FINAL CONFIGURATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Generate plots: {'Yes' if args.plot else 'No'}")

    # Determine output directory and filenames
    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent

    # Determine if this is best or final model
    model_type = "best" if checkpoint_path.name == "best_model.pth" else "final"

    # Create timestamp for metadata (saved in JSON)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results subdirectory with model type only (no timestamp)
    # This will overwrite previous evaluations for the same model type
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        results_dir_name = f"results_{model_type}"
        output_path = checkpoint_dir / results_dir_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Create simple filenames
    results_filename = "evaluation_results.json"
    summary_filename = "evaluation_summary.txt"
    confusion_matrix_filename = "confusion_matrix.png"
    training_history_filename = "training_history.png"
    per_class_metrics_filename = "per_class_metrics.png"
    class_accuracy_filename = "class_accuracy.png"
    comprehensive_analysis_filename = "comprehensive_analysis.png"

    print(f"Results directory: {output_path}")
    print(f"Results will overwrite existing {model_type} evaluation results")
    print("=" * 60)

    # Load data
    print("\nLoading test data...")
    _, _, test_loader, class_names = get_data_loaders(
        batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == "vanilla":
        model = VanillaCNN(num_classes=num_classes)
    else:
        model = get_pretrained_model(args.model, num_classes=num_classes, pretrained=False)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Evaluate
    results = evaluate_model(model, test_loader, device, class_names)

    # Print reports
    print_classification_report(results)
    print_confusion_matrix(results)

    # Save JSON results with model type and timestamp in filename
    results_to_save = {
        "test_loss": results['test_loss'],
        "test_accuracy": results['test_accuracy'],
        "per_class_metrics": results['per_class_metrics'],
        "confusion_matrix": results['confusion_matrix'].tolist(),
        "model_type": model_type,
        "checkpoint_path": str(checkpoint_path),
        "timestamp": timestamp
    }
    results_file_path = output_path / results_filename
    with open(results_file_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nJSON results saved to: {results_file_path}")

    # Save text summary
    summary_file_path = output_path / summary_filename
    save_text_summary(results, str(summary_file_path), str(checkpoint_path), model_type, timestamp)
    print(f"Text summary saved to: {summary_file_path}")

    # Generate plots if requested
    if args.plot:
        print("\nGenerating visualizations...")

        # Comprehensive analysis (all-in-one)
        comprehensive_path = output_path / comprehensive_analysis_filename
        plot_comprehensive_analysis(results, save_path=str(comprehensive_path))

        # Individual plots
        confusion_matrix_path = output_path / confusion_matrix_filename
        plot_confusion_matrix(results, save_path=str(confusion_matrix_path))

        per_class_path = output_path / per_class_metrics_filename
        plot_per_class_metrics(results, save_path=str(per_class_path))

        class_accuracy_path = output_path / class_accuracy_filename
        plot_class_accuracy_heatmap(results, save_path=str(class_accuracy_path))

        # Check for training history
        history_path = checkpoint_dir / "training_history.json"
        if history_path.exists():
            training_history_path = output_path / training_history_filename
            plot_training_history(str(history_path), save_path=str(training_history_path))
        else:
            print("\nNote: training_history.json not found - skipping training history plot")

    print("\nEvaluation complete!")
    print(f"\nAll results saved to: {output_path}")


if __name__ == "__main__":
    main()