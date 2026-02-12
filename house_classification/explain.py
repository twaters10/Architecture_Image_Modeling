#!/usr/bin/env python3
"""
Model Explainability Script for Architectural Style Classification.

This script provides visual explanations of model predictions using Grad-CAM
(Gradient-weighted Class Activation Mapping).

Usage:
    # Interactive mode (recommended)
    python explain.py

    # Single image explanation
    python explain.py --checkpoint checkpoints/resnet18.../best_model.pth --image path/to/image.jpg

    # Analyze misclassifications from evaluation results
    python explain.py --checkpoint checkpoints/resnet18.../best_model.pth --mode misclassifications

    # Batch processing
    python explain.py --checkpoint checkpoints/resnet18.../best_model.pth --mode batch --image_dir path/to/images/

Requirements:
    - torch
    - torchvision
    - PIL
    - matplotlib
    - numpy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Local imports
from utils.config import get_data_paths, get_normalization_config, load_config, get_mlflow_config
from model import VanillaCNN, get_pretrained_model
from explainability import GradCAM, GradCAMVisualizer, get_target_layer
from utils.mlflow_utils import (
    MLflowLogger,
    log_single_explanation,
    log_misclassification_analysis,
    log_batch_analysis
)


def get_device() -> torch.device:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def extract_model_name_from_checkpoint(checkpoint_path: str) -> str:
    """
    Extract model architecture name from checkpoint directory naming convention.

    Directories follow: {model}_ep{epochs}_bs{batch_size}_lr{lr}
    """
    dir_name = Path(checkpoint_path).parent.name

    # Check multi-word names first to avoid partial matches
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
) -> Tuple[torch.nn.Module, str, Dict]:
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        num_classes: Number of output classes.
        model_name: Model architecture name. If None, auto-detected from path.
        device: Device to load model on. If None, uses best available.

    Returns:
        Tuple of (model, model_name, checkpoint_info_dict).
    """
    if device is None:
        device = get_device()

    checkpoint_path = str(checkpoint_path)

    if model_name is None:
        model_name = extract_model_name_from_checkpoint(checkpoint_path)

    print(f"Model architecture: {model_name}")

    # Create model (pretrained=False since we load our own weights)
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

    print(f"Loaded checkpoint from epoch {info['epoch']}")
    if isinstance(info["val_loss"], float):
        print(f"  Validation loss: {info['val_loss']:.4f}")
    if isinstance(info["val_acc"], float):
        print(f"  Validation accuracy: {info['val_acc']:.2f}%")

    return model, model_name, info


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Get the inference preprocessing transform matching training pipeline."""
    norm_config = get_normalization_config()
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # 256 when image_size=224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_config.get("mean", [0.485, 0.456, 0.406]),
            std=norm_config.get("std", [0.229, 0.224, 0.225])
        ),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for display.

    Returns HWC numpy array in [0,1].
    """
    norm_config = get_normalization_config()
    mean = np.array(norm_config.get("mean", [0.485, 0.456, 0.406])).reshape(3, 1, 1)
    std = np.array(norm_config.get("std", [0.229, 0.224, 0.225])).reshape(3, 1, 1)

    img = tensor.detach().cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img.transpose(1, 2, 0)  # CHW -> HWC


def load_image(image_path: str) -> Image.Image:
    """Load an image and convert to RGB."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_class_names_from_data_dir() -> List[str]:
    """Get class names from the data directory structure."""
    paths = get_data_paths()
    train_dir = paths["train"]

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Please ensure data is prepared using 01b_image_train_val_test_split.py"
        )

    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return class_names


def explain_single_image(
    model: torch.nn.Module,
    model_name: str,
    image_path: Path,
    class_names: List[str],
    device: torch.device,
    output_dir: Path,
    true_class: Optional[str] = None,
    top_k: int = 5
):
    """
    Generate Grad-CAM explanation for a single image.

    Args:
        model: Trained model.
        model_name: Model architecture name.
        image_path: Path to input image.
        class_names: List of class names.
        device: Device to run on.
        output_dir: Directory to save outputs.
        true_class: Ground truth class (optional).
        top_k: Number of top predictions to show.
    """
    print(f"\n{'=' * 60}")
    print("SINGLE IMAGE EXPLANATION")
    print(f"{'=' * 60}")
    print(f"Image: {image_path.name}")

    # Load and preprocess image
    original_image = load_image(str(image_path))
    print(f"Original size: {original_image.size}")

    transform = get_inference_transform()
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")

    # Get target layer for Grad-CAM
    target_layer = get_target_layer(model, model_name)
    print(f"Target layer: {type(target_layer).__name__}")

    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer, device)

    # Forward pass to get predictions
    input_tensor.requires_grad_(True)
    with torch.enable_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze(0)

    # Get prediction results
    predicted_idx = probabilities.argmax().item()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    # Get top-K predictions
    top_k_probs, top_k_indices = probabilities.topk(top_k)
    top_k_predictions = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_k_indices, top_k_probs)
    ]

    print(f"\n{'=' * 60}")
    print("PREDICTION RESULTS")
    print(f"{'=' * 60}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.1%}")
    if true_class:
        match = "CORRECT ✓" if true_class == predicted_class else "INCORRECT ✗"
        print(f"True class: {true_class} ({match})")
    print(f"\nTop-{top_k} predictions:")
    for rank, (name, prob) in enumerate(top_k_predictions, 1):
        marker = " <-- PREDICTED" if name == predicted_class else ""
        print(f"  {rank}. {name}: {prob:.1%}{marker}")

    # Generate Grad-CAM heatmap
    print(f"\nGenerating Grad-CAM for: {predicted_class}")
    heatmap = gradcam.generate(input_tensor, target_class=predicted_idx)
    print(f"Heatmap shape: {heatmap.shape}")

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
        true_class=true_class,
        top_k_predictions=top_k_predictions,
        output_path=output_path
    )
    visualizer.close_figure(fig)

    # Generate class comparison
    print(f"\nGenerating class comparison across top-{top_k} predictions...")
    heatmaps_by_class = {}
    for class_name, prob in top_k_predictions:
        class_idx = class_names.index(class_name)
        input_fresh = transform(original_image).unsqueeze(0).to(device)
        input_fresh.requires_grad_(True)
        heatmap_k = gradcam.generate(input_fresh, target_class=class_idx)
        heatmaps_by_class[class_name] = (heatmap_k, prob)

    comparison_path = output_dir / f"{image_path.stem}_class_comparison.png"
    fig = visualizer.create_class_comparison(
        preprocessed_tensor=preprocessed_image,
        heatmaps_by_class=heatmaps_by_class,
        predicted_class=predicted_class,
        output_path=comparison_path
    )
    visualizer.close_figure(fig)

    # Clean up
    gradcam.remove_hooks()

    print(f"\n{'=' * 60}")
    print("OUTPUTS SAVED")
    print(f"{'=' * 60}")
    print(f"Main visualization: {output_path}")
    print(f"Class comparison: {comparison_path}")


def explain_misclassifications(
    model: torch.nn.Module,
    model_name: str,
    checkpoint_dir: Path,
    class_names: List[str],
    device: torch.device,
    output_dir: Path,
    limit: int = 20
):
    """
    Analyze misclassifications by running inference on test set.

    Args:
        model: Trained model.
        model_name: Model architecture name.
        checkpoint_dir: Checkpoint directory (used for reference).
        class_names: List of class names.
        device: Device to run on.
        output_dir: Directory to save outputs.
        limit: Maximum number of misclassifications to analyze.
    """
    print(f"\n{'=' * 60}")
    print("MISCLASSIFICATION ANALYSIS")
    print(f"{'=' * 60}")
    print("\nRunning inference on test set to find misclassifications...")

    # Get test data directory
    paths = get_data_paths()
    test_dir = paths["test"]

    if not test_dir.exists():
        print(f"\nERROR: Test directory not found: {test_dir}")
        print("Please ensure data is prepared using 01b_image_train_val_test_split.py")
        return

    # Initialize
    gradcam = GradCAM(model, get_target_layer(model, model_name), device)
    visualizer = GradCAMVisualizer(class_names)
    transform = get_inference_transform()

    # Process test set and find misclassifications
    misclassifications = []
    processed_count = 0

    print("\nScanning test set...")
    for true_class_idx, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue

        image_files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))

        for img_file in image_files:
            # Load and predict
            try:
                original_image = load_image(str(img_file))
                input_tensor = transform(original_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1).squeeze(0)

                pred_idx = probabilities.argmax().item()
                predicted_class = class_names[pred_idx]

                # Check if misclassified
                if pred_idx != true_class_idx:
                    misclassifications.append({
                        'image_path': img_file,
                        'true_class': class_name,
                        'true_idx': true_class_idx,
                        'pred_class': predicted_class,
                        'pred_idx': pred_idx,
                        'confidence': probabilities[pred_idx].item(),
                        'original_image': original_image
                    })

            except Exception as e:
                print(f"\nWarning: Failed to process {img_file.name}: {e}")
                continue

    if len(misclassifications) == 0:
        print("\n✓ No misclassifications found! Model has 100% accuracy on test set.")
        gradcam.remove_hooks()
        return

    print(f"\nFound {len(misclassifications)} misclassifications")
    print(f"Analyzing up to {min(limit, len(misclassifications))} samples...\n")

    # Generate Grad-CAM for misclassified samples
    for idx, misclass in enumerate(misclassifications[:limit]):
        print(f"Processing {idx + 1}/{min(limit, len(misclassifications))}: "
              f"{misclass['true_class']} → {misclass['pred_class']} "
              f"(confidence: {misclass['confidence']:.1%})")

        # Load and process image
        input_tensor = transform(misclass['original_image']).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # Generate Grad-CAM
        with torch.enable_grad():
            heatmap = gradcam.generate(input_tensor, target_class=misclass['pred_idx'])

        # Create visualization
        preprocessed_image = denormalize(input_tensor.squeeze(0))
        output_path = output_dir / f"misclass_{idx:03d}_{misclass['true_class']}_as_{misclass['pred_class']}.png"

        fig = visualizer.create_single_explanation(
            original_image=misclass['original_image'],
            preprocessed_tensor=preprocessed_image,
            heatmap=heatmap,
            predicted_class=misclass['pred_class'],
            confidence=misclass['confidence'],
            true_class=misclass['true_class'],
            top_k_predictions=None,
            output_path=output_path
        )
        visualizer.close_figure(fig)

        processed_count += 1

    gradcam.remove_hooks()

    print(f"\n{'=' * 60}")
    print("MISCLASSIFICATION ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total misclassifications found: {len(misclassifications)}")
    print(f"Visualizations created: {processed_count}")
    print(f"Results saved to: {output_dir}")


def explain_batch(
    model: torch.nn.Module,
    model_name: str,
    image_dir: Path,
    class_names: List[str],
    device: torch.device,
    output_dir: Path,
    true_class: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    Generate Grad-CAM explanations for a batch of images.

    Args:
        model: Trained model.
        model_name: Model architecture name.
        image_dir: Directory containing images.
        class_names: List of class names.
        device: Device to run on.
        output_dir: Directory to save outputs.
        true_class: True class label if all images are from same class.
        limit: Maximum number of images to process.
    """
    print(f"\n{'=' * 60}")
    print("BATCH PROCESSING")
    print(f"{'=' * 60}")
    print(f"Image directory: {image_dir}")

    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(image_dir.glob(ext))

    image_files = sorted(image_files)

    if limit:
        image_files = image_files[:limit]

    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Initialize
    gradcam = GradCAM(model, get_target_layer(model, model_name), device)
    visualizer = GradCAMVisualizer(class_names)
    transform = get_inference_transform()

    # Process each image
    for idx, img_file in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Load and process
            original_image = load_image(str(img_file))
            input_tensor = transform(original_image).unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)

            # Get prediction
            with torch.enable_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1).squeeze(0)

            pred_idx = probabilities.argmax().item()
            predicted_class = class_names[pred_idx]
            confidence = probabilities[pred_idx].item()

            # Generate Grad-CAM
            heatmap = gradcam.generate(input_tensor, target_class=pred_idx)

            # Create visualization
            preprocessed_image = denormalize(input_tensor.squeeze(0))
            output_path = output_dir / f"batch_{idx:04d}_{img_file.stem}_gradcam.png"

            fig = visualizer.create_single_explanation(
                original_image=original_image,
                preprocessed_tensor=preprocessed_image,
                heatmap=heatmap,
                predicted_class=predicted_class,
                confidence=confidence,
                true_class=true_class,
                top_k_predictions=None,
                output_path=output_path
            )
            visualizer.close_figure(fig)

        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            continue

    gradcam.remove_hooks()

    print(f"\n{'=' * 60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Processed {len(image_files)} images")
    print(f"Results saved to: {output_dir}")


def prompt_checkpoint_selection() -> Path:
    """Prompt user to select a checkpoint directory."""
    paths = get_data_paths()
    checkpoints_dir = paths["checkpoints"]

    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_dir}\n"
            "Please train a model first using train.py"
        )

    # Get available checkpoint directories
    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No checkpoint directories found in {checkpoints_dir}\n"
            "Please train a model first using train.py"
        )

    print("\n" + "=" * 70)
    print("AVAILABLE MODEL CHECKPOINT DIRECTORIES")
    print("=" * 70)

    for i, ckpt_dir in enumerate(checkpoint_dirs, 1):
        has_best = (ckpt_dir / "best_model.pth").exists()
        has_final = (ckpt_dir / "final_model.pth").exists()
        markers = []
        if has_best:
            markers.append("best")
        if has_final:
            markers.append("final")
        marker_str = f"[{', '.join(markers)}]" if markers else ""
        print(f"{i}. {ckpt_dir.name} {marker_str}")

    print("=" * 70)

    while True:
        try:
            choice = input(f"\nSelect checkpoint directory (1-{len(checkpoint_dirs)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(checkpoint_dirs):
                return checkpoint_dirs[choice_num - 1]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(checkpoint_dirs)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(checkpoint_dirs)}.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)


def prompt_model_checkpoint_file(checkpoint_dir: Path) -> Path:
    """Prompt user to select best or final model checkpoint."""
    best_model = checkpoint_dir / "best_model.pth"
    final_model = checkpoint_dir / "final_model.pth"

    available_models = []
    if best_model.exists():
        available_models.append(("best_model.pth", "Best model (lowest validation loss)", best_model))
    if final_model.exists():
        available_models.append(("final_model.pth", "Final model (last epoch)", final_model))

    if not available_models:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")

    if len(available_models) == 1:
        print(f"\nUsing {available_models[0][0]}")
        return available_models[0][2]

    print("\n" + "=" * 70)
    print("SELECT MODEL CHECKPOINT")
    print("=" * 70)
    for i, (name, description, _) in enumerate(available_models, 1):
        print(f"{i}. {name} - {description}")
    print("=" * 70)

    while True:
        try:
            choice = input(f"\nSelect checkpoint (1-{len(available_models)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                return available_models[choice_num - 1][2]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(available_models)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(available_models)}.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)


def prompt_analysis_mode() -> str:
    """Prompt user to select analysis mode."""
    modes = [
        ("single", "Single image explanation - Analyze one specific image"),
        ("misclassifications", "Misclassification analysis - Explain model errors from evaluation results"),
        ("batch", "Batch processing - Analyze multiple images from a directory"),
    ]

    print("\n" + "=" * 70)
    print("SELECT ANALYSIS MODE")
    print("=" * 70)
    for i, (mode_key, description) in enumerate(modes, 1):
        print(f"{i}. {description}")
    print("=" * 70)

    while True:
        try:
            choice = input(f"\nSelect mode (1-{len(modes)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(modes):
                return modes[choice_num - 1][0]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(modes)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(modes)}.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)


def prompt_mlflow_tracking() -> bool:
    """
    Prompt user whether to enable MLflow tracking.

    Returns:
        bool: True if MLflow should be enabled, False otherwise.
    """
    print("\n" + "=" * 70)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 70)
    print("MLflow tracks all explainability metrics, parameters, and artifacts.")
    print("You can view results later with: mlflow ui")
    print("=" * 70)
    print("1. Yes - Enable MLflow tracking (recommended)")
    print("2. No - Skip MLflow tracking")
    print("=" * 70)

    while True:
        try:
            choice = input("\nEnable MLflow tracking? (1-2): ").strip()
            if choice == "1":
                print("✓ MLflow tracking enabled")
                return True
            elif choice == "2":
                print("✗ MLflow tracking disabled")
                return False
            else:
                print("Please enter 1 or 2.")
        except (ValueError, KeyboardInterrupt):
            print("\nDefaulting to no MLflow tracking")
            return False


def main():
    """Main entry point for explainability script."""
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explanations for architectural style classification model."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model architecture (auto-detected if not specified)"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["single", "misclassifications", "batch"],
        help="Analysis mode (will prompt if not specified)"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to single image (for single mode)"
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="Directory of images (for batch mode)"
    )
    parser.add_argument(
        "--true_class", type=str, default=None,
        help="True class label (optional, for single/batch mode)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Custom output directory (default: checkpoint_dir/explanations/)"
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of images to process (for misclassifications/batch mode)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predictions to show (for single mode)"
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Enable MLflow experiment tracking"
    )
    parser.add_argument(
        "--no_mlflow", action="store_true",
        help="Disable MLflow experiment tracking"
    )
    parser.add_argument(
        "--mlflow_tracking_uri", type=str, default=None,
        help="MLflow tracking server URI (default: from config or local ./mlruns)"
    )
    parser.add_argument(
        "--mlflow_experiment", type=str, default=None,
        help="MLflow experiment name (default: from config)"
    )

    args = parser.parse_args()

    # Load MLflow configuration
    config = load_config()
    mlflow_config = get_mlflow_config(config)

    # Determine if MLflow should be enabled (priority: CLI > interactive > config)
    mlflow_enabled = False
    if args.no_mlflow:
        # Explicitly disabled via CLI
        mlflow_enabled = False
    elif args.mlflow:
        # Explicitly enabled via CLI
        mlflow_enabled = True
    elif mlflow_config.get("enabled", True):
        # Config says enabled, prompt user
        mlflow_enabled = prompt_mlflow_tracking()
    else:
        # Config says disabled
        mlflow_enabled = False

    # Use config defaults if not provided via CLI
    mlflow_tracking_uri = args.mlflow_tracking_uri or mlflow_config.get("tracking_uri")
    mlflow_experiment = args.mlflow_experiment or mlflow_config.get("experiment_name", "architectural-style-explainability")

    print("\n" + "=" * 70)
    print("Architectural Style Classification - Model Explainability (Grad-CAM)")
    print("=" * 70)
    print(f"MLflow tracking: {'enabled' if mlflow_enabled else 'disabled'}")

    # Interactive checkpoint selection if not provided
    if args.checkpoint is None:
        checkpoint_dir = prompt_checkpoint_selection()
        checkpoint_path = prompt_model_checkpoint_file(checkpoint_dir)
    else:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_dir = checkpoint_path.parent

    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\nCheckpoint: {checkpoint_path}")

    # Get class names
    class_names = get_class_names_from_data_dir()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    # Load model
    device = get_device()
    print(f"\nDevice: {device}")

    print("\nLoading model...")
    model, model_name, checkpoint_info = load_model_from_checkpoint(
        str(checkpoint_path),
        num_classes=num_classes,
        model_name=args.model,
        device=device
    )

    # Interactive mode selection if not provided
    if args.mode is None:
        mode = prompt_analysis_mode()
    else:
        mode = args.mode

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_dir / "explanations" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalysis mode: {mode}")
    print(f"Output directory: {output_dir}")

    # Initialize MLflow logger if enabled
    mlflow_logger = MLflowLogger(
        experiment_name=mlflow_experiment,
        tracking_uri=mlflow_tracking_uri,
        enable_logging=mlflow_enabled
    )

    # Start MLflow run if enabled
    if mlflow_enabled:
        run_name = f"{model_name}_{mode}"
        run_tags = {
            "model_architecture": model_name,
            "analysis_mode": mode,
            "checkpoint": str(checkpoint_path)
        }
        mlflow_logger.start_run(run_name=run_name, tags=run_tags)

        # Log common parameters
        mlflow_logger.log_params({
            "model_name": model_name,
            "num_classes": num_classes,
            "device": str(device),
            "analysis_mode": mode
        })

    try:
        # Execute based on mode
        if mode == "single":
            # Get image path
            if args.image is None:
                print("\nPlease provide the path to the image you want to analyze:")
                image_path = input("Image path: ").strip()
            else:
                image_path = args.image

            image_path = Path(image_path)
            if not image_path.exists():
                print(f"\nERROR: Image not found: {image_path}")
                sys.exit(1)

            # Log mode-specific parameters
            if mlflow_enabled:
                mlflow_logger.log_params({
                    "image_name": image_path.name,
                    "true_class": args.true_class or "unknown",
                    "top_k": args.top_k
                })

            explain_single_image(
                model=model,
                model_name=model_name,
                image_path=image_path,
                class_names=class_names,
                device=device,
                output_dir=output_dir,
                true_class=args.true_class,
                top_k=args.top_k
            )

    elif mode == "misclassifications":
            # Log mode-specific parameters
            if mlflow_enabled:
                mlflow_logger.log_params({
                    "limit": args.limit,
                    "checkpoint_dir": str(checkpoint_dir)
                })

            explain_misclassifications(
                model=model,
                model_name=model_name,
                checkpoint_dir=checkpoint_dir,
                class_names=class_names,
                device=device,
                output_dir=output_dir,
                limit=args.limit
            )

        elif mode == "batch":
            # Get image directory
            if args.image_dir is None:
                print("\nPlease provide the path to the directory containing images:")
                image_dir = input("Image directory: ").strip()
            else:
                image_dir = args.image_dir

            image_dir = Path(image_dir)
            if not image_dir.exists():
                print(f"\nERROR: Directory not found: {image_dir}")
                sys.exit(1)

            # Log mode-specific parameters
            if mlflow_enabled:
                mlflow_logger.log_params({
                    "image_dir": str(image_dir),
                    "true_class": args.true_class or "unknown",
                    "limit": args.limit
                })

            explain_batch(
                model=model,
                model_name=model_name,
                image_dir=image_dir,
                class_names=class_names,
                device=device,
                output_dir=output_dir,
                true_class=args.true_class,
                limit=args.limit
            )

    finally:
        # End MLflow run if enabled
        if mlflow_enabled:
            print("\n" + "=" * 70)
            print("MLflow Logging")
            print("=" * 70)
            print("Note: Basic run parameters logged to MLflow.")
            print("Detailed per-image logging pending full integration.")
            print(f"View results with: mlflow ui")
            mlflow_logger.end_run()

    print("\n" + "=" * 70)
    print("EXPLANATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
