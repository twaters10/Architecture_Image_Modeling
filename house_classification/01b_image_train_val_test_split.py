#!/usr/bin/env python3
"""
Image Dataset Train/Validation/Test Splitter for Architectural Style Classification.

This script splits images organized by architectural style into train, validation,
and test sets for machine learning model training. It preserves the class structure
by creating subdirectories for each house style within the split directories.

Directory Structure:
    Input (expected):
        architectural_style_images/
        ├── american_craftsman/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── colonial/
        │   └── ...
        └── ...

    Output (generated):
        architectural_style_images/
        ├── train/
        │   ├── american_craftsman/
        │   └── colonial/
        ├── validation/
        │   ├── american_craftsman/
        │   └── colonial/
        └── test/
            ├── american_craftsman/
            └── colonial/

Usage:
    Run interactively:
        $ python 01b_image_train_val_test_split.py

    The script will prompt for:
        1. Random seed (optional, for reproducibility)
        2. Train/validation/test split percentages (must sum to 100)

Example:
    $ python 01b_image_train_val_test_split.py
    Enter random seed (press Enter for random): 42
    Train split percentage (e.g., 70): 70
    Validation split percentage (e.g., 20): 20
    Test split percentage (e.g., 10): 10

Note:
    - Images are copied (not moved) to preserve the original dataset
    - Supported image formats: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
"""

import shutil
import random
from pathlib import Path
from typing import Optional, Tuple, Union


def get_split_ratios() -> Tuple[float, float, float]:
    """
    Prompt user interactively for train/validation/test split ratios.

    Continuously prompts the user until valid percentages that sum to 100 are provided.
    Handles invalid input gracefully with error messages.

    Returns:
        Tuple[float, float, float]: A tuple of (train_ratio, val_ratio, test_ratio)
            as decimal values between 0 and 1 (e.g., 0.7, 0.2, 0.1).

    Raises:
        KeyboardInterrupt: If user cancels input with Ctrl+C.

    Example:
        >>> train, val, test = get_split_ratios()
        Enter split ratios (must sum to 100):
        Train split percentage (e.g., 70): 70
        Validation split percentage (e.g., 20): 20
        Test split percentage (e.g., 10): 10
        >>> print(train, val, test)
        0.7 0.2 0.1
    """
    print("\nEnter split ratios (must sum to 100):")

    while True:
        try:
            train = float(input("Train split percentage (e.g., 70): "))
            val = float(input("Validation split percentage (e.g., 20): "))
            test = float(input("Test split percentage (e.g., 10): "))

            if abs(train + val + test - 100) < 0.01:  # Allow for floating point errors
                return train / 100, val / 100, test / 100
            else:
                print(f"Error: Splits must sum to 100. You entered {train + val + test}")
        except ValueError:
            print("Error: Please enter valid numbers")

def split_dataset(
    source_dir: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> None:
    """
    Split images from source directory into train/validation/test sets.

    Iterates through each house style subdirectory, reads images from the 'images'
    subfolder, randomly shuffles them, and copies them into train/validation/test
    directories while preserving the class (house style) structure.

    Args:
        source_dir: Path to the root directory containing house style subdirectories.
            Each subdirectory should contain image files directly (e.g.,
            american_craftsman/image1.jpg). Defaults to the architectural_style_images
            directory relative to this script's location.
        train_ratio: Proportion of images to allocate to training set.
            Must be between 0 and 1. Defaults to 0.7 (70%).
        val_ratio: Proportion of images to allocate to validation set.
            Must be between 0 and 1. Defaults to 0.2 (20%).
        test_ratio: Proportion of images to allocate to test set.
            Must be between 0 and 1. Defaults to 0.1 (10%).

    Returns:
        None. Creates directories and copies files as a side effect.

    Note:
        - The sum of train_ratio, val_ratio, and test_ratio should equal 1.0
        - Images are copied (not moved) to preserve the original dataset
        - If a house style directory contains no images, it is skipped
        - Random shuffling should be seeded externally for reproducibility
    """
    # Default to architectural_style_images directory relative to this script
    if source_dir is None:
        source_dir = Path(__file__).parent.parent / "architectural_style_images"
    source_path = Path(source_dir)
    
    # Find all housestyle subdirectories
    housestyle_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not housestyle_dirs:
        print(f"Error: No subdirectories found in {source_dir}")
        return
    
    print(f"\nFound {len(housestyle_dirs)} house style(s):")
    for d in housestyle_dirs:
        print(f"  - {d.name}")
    
    # Create output directory structure
    splits = ['train', 'validation', 'test']
    
    for housestyle_dir in housestyle_dirs:
        # Get all image files directly from the style directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        images = [f for f in housestyle_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not images:
            print(f"\nWarning: No images found in {housestyle_dir}, skipping...")
            continue
        
        # Shuffle images randomly
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        print(f"\n{housestyle_dir.name}:")
        print(f"  Total images: {n_total}")
        print(f"  Train: {len(train_images)} | Validation: {len(val_images)} | Test: {len(test_images)}")
        
        # Copy images to new structure
        split_data = {
            'train': train_images,
            'validation': val_images,
            'test': test_images
        }
        
        for split_name, split_images in split_data.items():
            # Create destination directory: data/train/housestyle/
            dest_dir = source_path / split_name / housestyle_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for img in split_images:
                dest_file = dest_dir / img.name
                shutil.copy2(img, dest_file)
    
    print("\n✓ Dataset split complete!")
    print(f"\nNew structure:")
    print(f"  {source_dir}/train/<housestyle>/")
    print(f"  {source_dir}/validation/<housestyle>/")
    print(f"  {source_dir}/test/<housestyle>/")

if __name__ == "__main__":
    # ==========================================================================
    # Main Entry Point
    # ==========================================================================
    # This script is designed to be run interactively from the command line.
    # It guides the user through configuring and executing the dataset split.
    # ==========================================================================

    print("=" * 50)
    print("Image Dataset Splitter")
    print("=" * 50)

    # Set random seed for reproducibility
    # Using a consistent seed ensures the same split can be recreated
    seed = input("\nEnter random seed (press Enter for random): ").strip()
    if seed:
        random.seed(int(seed))
        print(f"Random seed set to: {seed}")

    # Get split ratios from user
    train_ratio, val_ratio, test_ratio = get_split_ratios()

    # Confirm before proceeding (destructive operation - copies many files)
    print(f"\nSplit ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    confirm = input("Proceed with split? (y/n): ").strip().lower()

    if confirm == 'y':
        split_dataset(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    else:
        print("Operation cancelled.")