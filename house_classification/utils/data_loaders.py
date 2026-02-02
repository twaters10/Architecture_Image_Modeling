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

Augmentations Applied (Validation/Test):
    - Resize and center crop
    - Normalization (ImageNet statistics)

Usage:
    from data_loaders import get_data_loaders

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir='../architectural_style_images',
        batch_size=32,
        image_size=224
    )

Requirements:
    - torch
    - torchvision
    - pillow
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
    pin_memory: Optional[bool] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Loads settings from conf/data.yaml by default. Parameters passed to this
    function override the config file values.

    Expects directory structure created by 01b_image_train_val_test_split.py:
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
            Defaults to path from conf/data.yaml.
        batch_size: Number of images per batch. Defaults to config value.
        image_size: Target image size for CNN input. Defaults to config value.
        num_workers: Number of worker processes for data loading. Defaults to config value.
        pin_memory: Whether to pin memory for faster GPU transfer. Defaults to config value.
            Note: Automatically disabled on MPS/CPU devices (only works with CUDA).

    Returns:
        Tuple containing:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - test_loader: DataLoader for test data
            - class_names: List of class names (architectural styles)

    Raises:
        FileNotFoundError: If the data directory or split directories don't exist.
    """
    # Load configuration
    loader_config = get_data_loader_config()
    paths = get_data_paths()

    # Use config values as defaults, override with function parameters
    batch_size = batch_size if batch_size is not None else loader_config["batch_size"]
    image_size = image_size if image_size is not None else loader_config["image_size"]
    num_workers = num_workers if num_workers is not None else loader_config["num_workers"]
    pin_memory = pin_memory if pin_memory is not None else loader_config["pin_memory"]

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
                "Run 01b_image_train_val_test_split.py first."
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

    # Create data loaders
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

    return train_loader, val_loader, test_loader, class_names


def get_dataset_info(data_dir: Optional[Union[str, Path]] = None) -> dict:
    """
    Get information about the dataset without loading images.

    Args:
        data_dir: Root directory containing train/validation/test subdirectories.
            Defaults to path from conf/data.yaml.

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

        print("\nConfiguration loaded from conf/data.yaml:")
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
        train_loader, val_loader, test_loader, class_names = get_data_loaders(
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
        print("Please run 01b_image_train_val_test_split.py first to create the splits.")