#!/usr/bin/env python3
"""
Data Loaders for Architectural Style Image Classification.

This module provides PyTorch data loaders with image augmentation pipelines
for training, validation, and testing CNN models on architectural style images.

Augmentations Applied (Training):
    - Random horizontal flip
    - Random rotation (up to 15 degrees)
    - Color jitter (brightness, contrast, saturation)
    - Random resized crop
    - Normalization (ImageNet statistics)
    - CutMix (optional): Image-mixing augmentation that replaces random patches

Augmentations Applied (Validation/Test):
    - Resize and center crop
    - Normalization (ImageNet statistics)

CutMix Augmentation:
    CutMix is an advanced image-mixing technique that:
    - Replaces a random rectangular region of one image with a patch from another
    - Mixes labels proportionally to the area of the cut region
    - Prevents overconfidence and improves generalization

    Enable in conf/training_config.yaml:
        augmentation:
          cutmix:
            enabled: true
            alpha: 1.0
            prob: 0.5

Basic Usage:
    from data_loaders import get_data_loaders

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir='../architectural_style_images',
        batch_size=32,
        image_size=224
    )

CutMix Usage in Training Loop:
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        use_cutmix=True
    )

    for images, targets_info in train_loader:
        images = images.to(device)

        if isinstance(targets_info, dict):
            # CutMix was applied
            target_a = targets_info['target_a'].to(device)
            target_b = targets_info['target_b'].to(device)
            lam = targets_info['lam']

            outputs = model(images)
            loss = criterion(outputs, target_a) * lam + \
                   criterion(outputs, target_b) * (1 - lam)
        else:
            # Normal batch (no CutMix)
            targets = targets_info.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

Requirements:
    - torch
    - torchvision
    - pillow
    - numpy
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
try:
    from torchvision.transforms import v2 as transforms_v2
    CUTMIX_AVAILABLE = True
except ImportError:
    CUTMIX_AVAILABLE = False

# Local imports
from .config import (
    load_config,
    get_data_paths,
    get_data_loader_config,
    get_augmentation_config,
    get_normalization_config,
)


class ConvertImageMode:
    """
    Custom transform to handle palette images with transparency.

    Converts palette mode images (mode 'P') with transparency to RGBA,
    then converts all images to RGB. This prevents the warning:
    "Palette images with Transparency expressed in bytes should be converted to RGBA images"
    """

    def __call__(self, img):
        """
        Convert image to RGB, handling palette images with transparency.

        Args:
            img: PIL Image

        Returns:
            PIL Image in RGB mode
        """
        # Convert all non-RGB images directly to RGB
        # This handles palette images with transparency without triggering warnings
        # PIL's convert('RGB') internally handles transparency correctly
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img


# ============================================================================
# CUTMIX AUGMENTATION (using PyTorch built-in)
# ============================================================================

class CutMixTransform:
    """
    Wrapper for PyTorch's built-in CutMix transform.

    This provides a simple interface compatible with our training loop.
    Uses torchvision.transforms.v2.CutMix for the actual implementation.

    Usage:
        cutmix = CutMixTransform(num_classes=10, alpha=1.0)

        # In training loop:
        for images, labels in train_loader:
            images, labels = cutmix(images, labels)
            # labels is now a dict with 'target_a', 'target_b', 'lam' if applied
    """

    def __init__(self, num_classes, alpha=1.0):
        """
        Initialize CutMix transform.

        Args:
            num_classes: Number of classes in the dataset
            alpha: Beta distribution parameter (default: 1.0)
        """
        if not CUTMIX_AVAILABLE:
            raise ImportError(
                "CutMix requires torchvision >= 0.15.0 with transforms v2. "
                "Please upgrade: pip install --upgrade torchvision"
            )

        self.num_classes = num_classes
        self.cutmix = transforms_v2.CutMix(num_classes=num_classes, alpha=alpha)

    def __call__(self, images, labels):
        """
        Apply CutMix to a batch.

        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,) as integers

        Returns:
            Tuple of (images, labels_dict) where labels_dict contains:
                - Original labels mixed according to CutMix
        """
        # PyTorch's CutMix expects one-hot encoded labels
        # It returns mixed images and mixed one-hot labels
        mixed_images, mixed_labels = self.cutmix(images, labels)

        # For compatibility with our training loop, we need to extract
        # the mixing information. PyTorch's CutMix returns soft labels
        # which are already mixed, so we can use them directly
        return mixed_images, mixed_labels


def cutmix_criterion(outputs, targets, criterion):
    """
    Calculate loss for batches with CutMix augmentation.

    Compatible with PyTorch's built-in CutMix which returns soft labels.

    Args:
        outputs: Model outputs (B, num_classes) - logits
        targets: Either soft labels (B, num_classes) from CutMix or
                 hard labels (B,) for normal batches
        criterion: Loss function (e.g., nn.CrossEntropyLoss())

    Returns:
        loss: Computed loss value

    Example:
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = cutmix_criterion(outputs, labels, criterion)
            loss.backward()
    """
    # Check if targets are soft labels (CutMix applied) or hard labels
    if targets.dim() == 2 and targets.size(1) > 1:
        # Soft labels from CutMix - use cross entropy with soft targets
        loss = -(targets * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
    else:
        # Hard labels - normal cross entropy
        loss = criterion(outputs, targets)

    return loss


def get_train_transforms(
    image_size: int = 224,
    aug_config: Optional[Dict[str, Any]] = None,
    norm_config: Optional[Dict[str, Any]] = None
) -> transforms.Compose:
    """
    Get image transformations for training data.

    Includes data augmentation to prevent overfitting:
    - RandomResizedCrop: Random scale and aspect ratio crops
    - RandomHorizontalFlip: 50% chance of horizontal flip
    - RandomRotation: Up to 15 degrees rotation
    - ColorJitter: Random brightness, contrast, saturation changes

    Args:
        image_size: Target image size (width and height). Defaults to 224.
        aug_config: Augmentation config from YAML. If None, loads from config file.
        norm_config: Normalization config from YAML. If None, loads from config file.

    Returns:
        transforms.Compose: Composed transformation pipeline for training.
    """
    # Load config if not provided
    if aug_config is None:
        aug_config = get_augmentation_config()
    if norm_config is None:
        norm_config = get_normalization_config()

    color_jitter = aug_config.get("color_jitter", {})

    return transforms.Compose([
        ConvertImageMode(),  # Handle palette images with transparency
        transforms.RandomResizedCrop(
            image_size,
            scale=tuple(aug_config.get("random_crop_scale", [0.8, 1.0]))
        ),
        transforms.RandomHorizontalFlip(p=aug_config.get("horizontal_flip_prob", 0.5)),
        transforms.RandomRotation(degrees=aug_config.get("rotation_degrees", 15)),
        transforms.ColorJitter(
            brightness=color_jitter.get("brightness", 0.2),
            contrast=color_jitter.get("contrast", 0.2),
            saturation=color_jitter.get("saturation", 0.2),
            hue=color_jitter.get("hue", 0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_config.get("mean", [0.485, 0.456, 0.406]),
            std=norm_config.get("std", [0.229, 0.224, 0.225])
        )
    ])


def get_val_test_transforms(
    image_size: int = 224,
    norm_config: Optional[Dict[str, Any]] = None
) -> transforms.Compose:
    """
    Get image transformations for validation and test data.

    No augmentation applied - only resize, crop, and normalization
    to ensure consistent evaluation.

    Args:
        image_size: Target image size (width and height). Defaults to 224.
        norm_config: Normalization config from YAML. If None, loads from config file.

    Returns:
        transforms.Compose: Composed transformation pipeline for evaluation.
    """
    if norm_config is None:
        norm_config = get_normalization_config()

    return transforms.Compose([
        ConvertImageMode(),  # Handle palette images with transparency
        transforms.Resize(int(image_size * 1.14)),  # Resize to 256 if image_size=224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_config.get("mean", [0.485, 0.456, 0.406]),
            std=norm_config.get("std", [0.229, 0.224, 0.225])
        )
    ])


def get_data_loaders(
    data_dir: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    image_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    use_cutmix: Optional[bool] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Optional[CutMixTransform]]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Loads settings from conf/training_config.yaml by default. Parameters passed to this
    function override the config file values.

    Expects directory structure created by data_prep/splitter.py:
        data_dir/
        ├── train/
        │   ├── american_craftsman/
        │   ├── colonial/
        │   └── ...
        ├── validation/
        │   └── ...
        └── test/
            └── ...

    Args:
        data_dir: Root directory containing train/validation/test subdirectories.
            Defaults to path from conf/training_config.yaml.
        batch_size: Number of images per batch. Defaults to config value.
        image_size: Target image size for CNN input. Defaults to config value.
        num_workers: Number of worker processes for data loading. Defaults to config value.
        pin_memory: Whether to pin memory for faster GPU transfer. Defaults to config value.
            Note: Automatically disabled on MPS/CPU devices (only works with CUDA).
        use_cutmix: Whether to use CutMix augmentation. Defaults to config value.

    Returns:
        Tuple containing:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - test_loader: DataLoader for test data
            - class_names: List of class names (architectural styles)
            - cutmix_transform: CutMixTransform object if enabled, None otherwise

    Raises:
        FileNotFoundError: If the data directory or split directories don't exist.

    Note:
        When CutMix is enabled, apply the returned cutmix_transform in your training loop:
            train_loader, val_loader, test_loader, class_names, cutmix = get_data_loaders(use_cutmix=True)

            for images, labels in train_loader:
                if cutmix is not None:
                    images, labels = cutmix(images, labels)
                # labels are now soft (mixed) if CutMix was applied
    """
    # Load configuration
    config = load_config()
    loader_config = get_data_loader_config()
    aug_config = get_augmentation_config()
    paths = get_data_paths()

    # Use config values as defaults, override with function parameters
    batch_size = batch_size if batch_size is not None else loader_config["batch_size"]
    image_size = image_size if image_size is not None else loader_config["image_size"]
    num_workers = num_workers if num_workers is not None else loader_config["num_workers"]
    pin_memory = pin_memory if pin_memory is not None else loader_config["pin_memory"]

    # CutMix configuration
    cutmix_config = aug_config.get("cutmix", {})
    use_cutmix = use_cutmix if use_cutmix is not None else cutmix_config.get("enabled", False)

    # Only use pin_memory with CUDA (not supported on MPS/CPU)
    # This avoids the warning: "pin_memory argument is set as true but not supported on MPS"
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False

    # Get data paths from config or parameter
    if data_dir is not None:
        data_path = Path(data_dir)
        train_dir = data_path / "train"
        val_dir = data_path / "validation"
        test_dir = data_path / "test"
    else:
        train_dir = paths["train"]
        val_dir = paths["validation"]
        test_dir = paths["test"]

    # Verify directories exist
    for split_dir, split_name in [(train_dir, "train"), (val_dir, "validation"), (test_dir, "test")]:
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_name} directory not found at {split_dir}. "
                "Run prepare_data.py --split first."
            )

    # Create datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(image_size)
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_test_transforms(image_size)
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_val_test_transforms(image_size)
    )

    # Get class names from training dataset
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Setup CutMix transform if enabled
    cutmix_transform = None
    if use_cutmix:
        if not CUTMIX_AVAILABLE:
            print("Warning: CutMix requires torchvision >= 0.15.0. Skipping CutMix.")
        else:
            cutmix_alpha = cutmix_config.get("alpha", 1.0)
            cutmix_transform = CutMixTransform(
                num_classes=num_classes,
                alpha=cutmix_alpha
            )
            print(f"CutMix enabled: alpha={cutmix_alpha} (apply in training loop)")

    # Create data loaders (no collate_fn needed - CutMix applied in training loop)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, class_names, cutmix_transform


def get_dataset_info(data_dir: Optional[Union[str, Path]] = None) -> dict:
    """
    Get information about the dataset without loading images.

    Args:
        data_dir: Root directory containing train/validation/test subdirectories.
            Defaults to path from conf/training_config.yaml.

    Returns:
        dict: Dataset statistics including class counts and sample distribution.
    """
    if data_dir is None:
        paths = get_data_paths()
        data_path = paths["raw_data"]
    else:
        data_path = Path(data_dir)

    info = {"splits": {}}

    for split_name in ["train", "validation", "test"]:
        split_dir = data_path / split_name
        if not split_dir.exists():
            continue

        split_info = {"classes": {}, "total": 0}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*")))
                split_info["classes"][class_dir.name] = count
                split_info["total"] += count

        info["splits"][split_name] = split_info

    return info


def save_sample_batches(
    train_loader: DataLoader,
    class_names: List[str],
    output_dir: Union[str, Path],
    samples_per_class: int = 12,
    norm_mean: List[float] = [0.485, 0.456, 0.406],
    norm_std: List[float] = [0.229, 0.224, 0.225]
) -> None:
    """
    Save visual samples of augmented training images for each class.

    Creates one image file per class showing a grid of sample images
    with augmentations applied. Useful for visualizing what the model
    sees during training.

    Args:
        train_loader: DataLoader for training data.
        class_names: List of class names.
        output_dir: Directory to save sample images (will create image_batches/ subdirectory).
        samples_per_class: Number of sample images to show per class. Defaults to 12.
        norm_mean: Mean values used for normalization (for denormalization).
        norm_std: Std values used for normalization (for denormalization).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Skipping sample batch visualization.")
        return

    output_path = Path(output_dir) / "image_batches"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating sample batch visualizations...")
    print(f"Collecting {samples_per_class} samples per class...")

    # Convert norm values to numpy arrays for denormalization
    mean = np.array(norm_mean).reshape(3, 1, 1)
    std = np.array(norm_std).reshape(3, 1, 1)

    # Collect samples for each class
    class_samples = {class_name: [] for class_name in class_names}
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

    # Iterate through batches to collect samples
    # Note: CutMix is applied in training loop, not in data loader,
    # so we always get normal (images, labels) format here
    for images, labels in train_loader:
        for img, label in zip(images, labels):
            class_name = class_names[label.item()]
            if len(class_samples[class_name]) < samples_per_class:
                # Denormalize image: img_denorm = img * std + mean
                img_np = img.numpy()
                img_denorm = img_np * std + mean
                # Clip to [0, 1] range and transpose to HWC format
                img_denorm = np.clip(img_denorm, 0, 1).transpose(1, 2, 0)
                class_samples[class_name].append(img_denorm)

        # Check if we have enough samples for all classes
        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break

    # Create and save subplot for each class
    grid_size = int(np.ceil(np.sqrt(samples_per_class)))
    rows = grid_size
    cols = grid_size

    for class_name, samples in class_samples.items():
        if not samples:
            print(f"Warning: No samples found for class '{class_name}'")
            continue

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        fig.suptitle(f"Training Samples: {class_name}", fontsize=16, fontweight='bold')

        # Flatten axes array for easier iteration
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < len(samples):
                ax.imshow(samples[idx])
                ax.set_title(f"Sample {idx + 1}", fontsize=10)
            else:
                # Hide empty subplots
                ax.axis('off')

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        # Save the figure
        output_file = output_path / f"{class_name.replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Sample batch visualizations saved to: {output_path}/")
    print(f"Created {len([s for s in class_samples.values() if s])} class visualization files")


def save_training_diagnostics(
    output_dir: Union[str, Path],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: List[str],
    model_name: str,
    model_params: Dict[str, int],
    config: Dict[str, Any]
) -> None:
    """
    Save comprehensive training diagnostics to a text file.

    Creates a detailed report of the training setup including dataset statistics,
    model architecture, hyperparameters, and data augmentation settings.

    Args:
        output_dir: Directory to save diagnostics file.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        class_names: List of class names.
        model_name: Name of the model architecture.
        model_params: Dictionary with parameter counts (total, trainable, frozen).
        config: Full configuration dictionary.
    """
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    diagnostics_file = output_path / "training_diagnostics.txt"

    # Get dataset info
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    # Count samples per class in training set
    class_counts = {class_name: 0 for class_name in class_names}
    for _, label in train_dataset:
        class_counts[class_names[label]] += 1

    with open(diagnostics_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING DIAGNOSTICS AND CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")

        # Timestamp
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint Directory: {output_path.name}\n")
        f.write("\n")

        # Model Architecture
        f.write("=" * 80 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Parameters: {model_params['total']:,}\n")
        f.write(f"Trainable Parameters: {model_params['trainable']:,}\n")
        if model_params.get('frozen', 0) > 0:
            f.write(f"Frozen Parameters: {model_params['frozen']:,}\n")
        f.write("\n")

        # Dataset Statistics
        f.write("=" * 80 + "\n")
        f.write("DATASET STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")

        f.write(f"Total Training Samples: {len(train_dataset)}\n")
        f.write(f"Total Validation Samples: {len(val_dataset)}\n")
        f.write(f"Total Test Samples: {len(test_dataset)}\n")
        f.write(f"Total Dataset Size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}\n\n")

        f.write(f"Training Batches per Epoch: {len(train_loader)}\n")
        f.write(f"Validation Batches: {len(val_loader)}\n")
        f.write(f"Test Batches: {len(test_loader)}\n")
        f.write("\n")

        # Per-Class Training Counts
        f.write("Per-Class Training Sample Distribution:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<30} {'Count':>10} {'Percentage':>15}\n")
        f.write("-" * 80 + "\n")
        total_train = len(train_dataset)
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = (count / total_train) * 100
            f.write(f"{class_name:<30} {count:>10} {percentage:>14.1f}%\n")
        f.write("-" * 80 + "\n\n")

        # Data Loading Configuration
        f.write("=" * 80 + "\n")
        f.write("DATA LOADING CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        loader_config = config.get('data_loader', {})
        f.write(f"Batch Size: {loader_config.get('batch_size', 'N/A')}\n")
        f.write(f"Number of Workers: {loader_config.get('num_workers', 'N/A')}\n")
        f.write(f"Pin Memory: {loader_config.get('pin_memory', 'N/A')}\n")
        f.write(f"Image Size: {loader_config.get('image_size', 'N/A')}x{loader_config.get('image_size', 'N/A')}\n")
        f.write("\n")

        # Data Augmentation (Training)
        f.write("=" * 80 + "\n")
        f.write("DATA AUGMENTATION (Training Only)\n")
        f.write("=" * 80 + "\n")
        f.write("Note: Augmentation is applied on-the-fly during training.\n")
        f.write("Each epoch sees different augmented versions of the same images.\n\n")

        aug_config = config.get('augmentation', {})
        f.write(f"Random Crop Scale: {aug_config.get('random_crop_scale', 'N/A')}\n")
        f.write(f"Horizontal Flip Probability: {aug_config.get('horizontal_flip_prob', 'N/A')}\n")
        f.write(f"Rotation Degrees: ±{aug_config.get('rotation_degrees', 'N/A')}°\n")

        color_jitter = aug_config.get('color_jitter', {})
        f.write(f"Color Jitter:\n")
        f.write(f"  - Brightness: ±{color_jitter.get('brightness', 'N/A')}\n")
        f.write(f"  - Contrast: ±{color_jitter.get('contrast', 'N/A')}\n")
        f.write(f"  - Saturation: ±{color_jitter.get('saturation', 'N/A')}\n")
        f.write(f"  - Hue: ±{color_jitter.get('hue', 'N/A')}\n")
        f.write("\n")

        # Normalization
        f.write("=" * 80 + "\n")
        f.write("NORMALIZATION (All Splits)\n")
        f.write("=" * 80 + "\n")
        norm_config = config.get('normalization', {})
        f.write(f"Mean (RGB): {norm_config.get('mean', 'N/A')}\n")
        f.write(f"Std (RGB): {norm_config.get('std', 'N/A')}\n")
        f.write("Note: Using ImageNet statistics\n\n")

        # Training Hyperparameters
        f.write("=" * 80 + "\n")
        f.write("TRAINING HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n")
        train_config = config.get('training', {})
        f.write(f"Epochs: {train_config.get('epochs', 'N/A')}\n")
        f.write(f"Learning Rate: {train_config.get('learning_rate', 'N/A')}\n")
        f.write(f"Weight Decay (L2 Regularization): {train_config.get('weight_decay', 'N/A')}\n")
        f.write(f"Early Stopping Patience: {train_config.get('early_stopping_patience', 'N/A')} epochs\n")

        lr_scheduler = train_config.get('lr_scheduler', {})
        f.write(f"Learning Rate Scheduler:\n")
        f.write(f"  - Type: ReduceLROnPlateau\n")
        f.write(f"  - Factor: {lr_scheduler.get('factor', 'N/A')}\n")
        f.write(f"  - Patience: {lr_scheduler.get('patience', 'N/A')} epochs\n")
        f.write("\n")

        # Loss Function & Optimizer
        f.write("=" * 80 + "\n")
        f.write("OPTIMIZATION\n")
        f.write("=" * 80 + "\n")
        f.write("Loss Function: CrossEntropyLoss\n")
        f.write("Optimizer: Adam\n")
        f.write("\n")

        # Visualization Settings
        f.write("=" * 80 + "\n")
        f.write("VISUALIZATION\n")
        f.write("=" * 80 + "\n")
        viz_config = config.get('visualization', {})
        f.write(f"Samples per Class (image_batches/): {viz_config.get('samples_per_class', 'N/A')}\n")
        f.write("\n")

        # Footer
        f.write("=" * 80 + "\n")
        f.write("END OF DIAGNOSTICS\n")
        f.write("=" * 80 + "\n")

    print(f"Training diagnostics saved to: {diagnostics_file}")


def append_training_runtime(
    output_dir: Union[str, Path],
    total_time_seconds: float
) -> None:
    """
    Append training runtime information to the diagnostics file.

    This function adds the total training time to an existing training_diagnostics.txt
    file after training completes.

    Args:
        output_dir: Directory containing the training_diagnostics.txt file.
        total_time_seconds: Total training time in seconds.
    """
    output_path = Path(output_dir)
    diagnostics_file = output_path / "training_diagnostics.txt"

    if not diagnostics_file.exists():
        print(f"Warning: Diagnostics file not found: {diagnostics_file}")
        return

    # Calculate time in different units
    total_seconds = total_time_seconds
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    # Read existing content and remove the footer
    with open(diagnostics_file, 'r') as f:
        content = f.read()

    # Remove the END OF DIAGNOSTICS footer
    footer_marker = "=" * 80 + "\nEND OF DIAGNOSTICS\n" + "=" * 80 + "\n"
    if footer_marker in content:
        content = content.replace(footer_marker, "")

    # Append runtime section and new footer
    with open(diagnostics_file, 'w') as f:
        f.write(content)
        f.write("=" * 80 + "\n")
        f.write("TRAINING RUNTIME\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Training Time: {total_seconds:.2f} seconds\n")
        f.write(f"                     {total_minutes:.2f} minutes\n")
        f.write(f"                     {total_hours:.2f} hours\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF DIAGNOSTICS\n")
        f.write("=" * 80 + "\n")

    print(f"Training runtime appended to: {diagnostics_file}")


if __name__ == "__main__":
    # Test the data loaders
    print("=" * 50)
    print("Data Loader Test")
    print("=" * 50)

    try:
        # Show loaded configuration
        config = load_config()
        loader_config = get_data_loader_config()
        paths = get_data_paths()

        print("\nConfiguration loaded from conf/training_config.yaml:")
        print(f"  Batch size: {loader_config['batch_size']}")
        print(f"  Image size: {loader_config['image_size']}")
        print(f"  Num workers: {loader_config['num_workers']}")

        print("\nData paths:")
        print(f"  Train: {paths['train']}")
        print(f"  Validation: {paths['validation']}")
        print(f"  Test: {paths['test']}")

        # Get dataset info first
        info = get_dataset_info()
        print("\nDataset Information:")
        for split_name, split_info in info["splits"].items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total images: {split_info['total']}")
            for class_name, count in split_info["classes"].items():
                print(f"    {class_name}: {count}")

        # Create data loaders (using config values)
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader, class_names, cutmix = get_data_loaders(
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )

        print(f"\nClass names: {class_names}")
        print(f"Number of classes: {len(class_names)}")
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Test loading a batch
        print("\nLoading a sample batch...")
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

        print("\nData loaders ready!")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run prepare_data.py --split first to create the splits.")