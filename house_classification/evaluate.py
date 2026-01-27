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
from config import load_config, get_data_paths
from data_loaders import get_data_loaders
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


def main():
    """Main entry point for evaluation script."""
    # Load config for defaults
    config = load_config()
    paths = get_data_paths()

    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN for architectural style classification. "
                    "Default values are loaded from conf/data.yaml."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model", type=str, default=config.get("model", {}).get("default", "vanilla"),
        choices=["vanilla", "resnet18", "resnet34", "resnet50",
                 "efficientnet_b0", "mobilenet_v2"],
        help="Model architecture used for training (default from config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.get("data_loader", {}).get("batch_size", 32),
        help="Batch size for evaluation (default from config)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate and save visualization plots"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save evaluation results (default from config)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Architectural Style Classification - Evaluation")
    print("=" * 60)
    print("\nConfiguration loaded from conf/data.yaml")

    # Create output directory (use config path if not specified)
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = paths["results"]
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_path}")

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

    # Save results
    results_to_save = {
        "test_loss": results['test_loss'],
        "test_accuracy": results['test_accuracy'],
        "per_class_metrics": results['per_class_metrics'],
        "confusion_matrix": results['confusion_matrix'].tolist()
    }
    with open(output_path / "evaluation_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {output_path / 'evaluation_results.json'}")

    # Generate plots if requested
    if args.plot:
        plot_confusion_matrix(
            results,
            save_path=str(output_path / "confusion_matrix.png")
        )

        # Check for training history
        checkpoint_dir = Path(args.checkpoint).parent
        history_path = checkpoint_dir / "training_history.json"
        if history_path.exists():
            plot_training_history(
                str(history_path),
                save_path=str(output_path / "training_history.png")
            )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()