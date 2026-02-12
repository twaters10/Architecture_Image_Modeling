#!/usr/bin/env python3
"""
Image Dataset Train/Validation/Test Splitter for Architectural Style Classification.

This script splits images organized by architectural style into train, validation,
and test sets for machine learning model training. It preserves the class structure
by creating subdirectories for each house style within the split directories.

Optimizations over previous version:
    - Symlinks by default (saves disk space), with copy fallback option
    - Rerun-safe: filters out train/validation/test dirs from input, cleans old splits
    - Minimum images guard: warns when a class has too few images for reliable splits
    - Per-class statistics table printed after splitting
    - Pathlib throughout for consistent path handling

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
        │   ├── american_craftsman/ -> symlinks to originals
        │   └── colonial/
        ├── validation/
        │   └── ...
        └── test/
            └── ...

Usage:
    Run as a standalone script:
        $ python -m data_prep.splitter

    Or via the unified pipeline:
        $ python prepare_data.py --split
"""

import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

# Ensure parent directory (house_classification/) is on sys.path for standalone execution
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
SPLIT_NAMES = {'train', 'validation', 'test'}
MIN_IMAGES_WARNING = 10  # Warn if a class has fewer than this many images


# ---------------------------------------------------------------------------
# User Input
# ---------------------------------------------------------------------------

def get_split_ratios() -> Tuple[float, float, float]:
    """
    Prompt user interactively for train/validation/test split ratios.

    Returns:
        Tuple of (train_ratio, val_ratio, test_ratio) as decimals.
    """
    print("\nEnter split ratios (must sum to 100):")

    while True:
        try:
            train = float(input("Train split percentage (e.g., 70): "))
            val = float(input("Validation split percentage (e.g., 20): "))
            test = float(input("Test split percentage (e.g., 10): "))

            if abs(train + val + test - 100) < 0.01:
                return train / 100, val / 100, test / 100
            else:
                print(f"Error: Splits must sum to 100. You entered {train + val + test}")
        except ValueError:
            print("Error: Please enter valid numbers")


def get_link_mode() -> str:
    """
    Prompt user to choose between symlinks (saves space) and copies.

    Returns:
        'symlink' or 'copy'.
    """
    print("\nFile mode for split:")
    print("  1. Symlinks (recommended) - saves disk space, links to originals")
    print("  2. Copies - duplicates files (uses more disk space)")

    while True:
        try:
            choice = input("Select mode (1-2): ").strip()
            if choice == "1":
                return "symlink"
            elif choice == "2":
                return "copy"
            else:
                print("Please enter 1 or 2.")
        except (ValueError, KeyboardInterrupt):
            print("\nDefaulting to symlinks")
            return "symlink"


# ---------------------------------------------------------------------------
# Split Logic
# ---------------------------------------------------------------------------

def split_dataset(
    source_dir: Optional[Union[str, Path]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    link_mode: str = "symlink",
) -> None:
    """
    Split images from source directory into train/validation/test sets.

    Args:
        source_dir: Path to the root directory containing house style subdirectories.
            Defaults to the architectural_style_images directory.
        train_ratio: Proportion for training set (0-1).
        val_ratio: Proportion for validation set (0-1).
        test_ratio: Proportion for test set (0-1).
        link_mode: 'symlink' to create symbolic links, 'copy' to duplicate files.
    """
    if source_dir is None:
        source_dir = Path(__file__).parent.parent / "architectural_style_images"
    source_path = Path(source_dir)

    # Find style subdirectories, filtering out split dirs from previous runs
    housestyle_dirs = sorted([
        d for d in source_path.iterdir()
        if d.is_dir() and d.name not in SPLIT_NAMES
    ])

    if not housestyle_dirs:
        print(f"Error: No style subdirectories found in {source_dir}")
        print(f"  (Excluded reserved names: {SPLIT_NAMES})")
        return

    print(f"\nFound {len(housestyle_dirs)} house style(s):")
    for d in housestyle_dirs:
        print(f"  - {d.name}")

    # Clean up previous split directories
    for split_name in SPLIT_NAMES:
        split_dir = source_path / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
            print(f"\nCleaned previous split: {split_dir}")

    # Track statistics for summary table
    stats = []

    for housestyle_dir in housestyle_dirs:
        # Get all image files
        images = sorted([
            f for f in housestyle_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not images:
            print(f"\nWarning: No images found in {housestyle_dir.name}, skipping...")
            continue

        n_total = len(images)

        # Minimum images guard
        if n_total < MIN_IMAGES_WARNING:
            print(f"\nWarning: {housestyle_dir.name} has only {n_total} images "
                  f"(< {MIN_IMAGES_WARNING}). Splits may be unreliable.")

        # Check that each split gets at least 1 image
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, n_total - n_train - n_val)

        # If we over-allocated due to max(1, ...), adjust train down
        if n_train + n_val + n_test > n_total:
            n_train = n_total - n_val - n_test
            if n_train < 1:
                # Not enough images for 3-way split
                print(f"\nWarning: {housestyle_dir.name} has only {n_total} images, "
                      f"cannot split into 3 sets. Putting all in train.")
                n_train = n_total
                n_val = 0
                n_test = 0

        # Shuffle and split
        random.shuffle(images)
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:n_train + n_val + n_test]

        stats.append({
            'style': housestyle_dir.name,
            'total': n_total,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images),
        })

        # Create split directories and link/copy images
        split_data = {
            'train': train_images,
            'validation': val_images,
            'test': test_images
        }

        for split_name, split_images in split_data.items():
            if not split_images:
                continue

            dest_dir = source_path / split_name / housestyle_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                dest_file = dest_dir / img.name

                if link_mode == "symlink":
                    try:
                        os.symlink(img.resolve(), dest_file)
                    except OSError:
                        # Fallback to copy if symlinks not supported
                        shutil.copy2(img, dest_file)
                else:
                    shutil.copy2(img, dest_file)

    # Print summary statistics table
    if stats:
        print("\n" + "=" * 65)
        print("SPLIT SUMMARY")
        print("=" * 65)
        print(f"{'Style':<30} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
        print("-" * 65)

        total_all = total_train = total_val = total_test = 0
        for s in stats:
            print(f"{s['style']:<30} {s['total']:>6} {s['train']:>6} {s['val']:>6} {s['test']:>6}")
            total_all += s['total']
            total_train += s['train']
            total_val += s['val']
            total_test += s['test']

        print("-" * 65)
        print(f"{'TOTAL':<30} {total_all:>6} {total_train:>6} {total_val:>6} {total_test:>6}")
        print("=" * 65)

    print(f"\nFile mode: {link_mode}")
    print(f"\nNew structure:")
    print(f"  {source_dir}/train/<housestyle>/")
    print(f"  {source_dir}/validation/<housestyle>/")
    print(f"  {source_dir}/test/<housestyle>/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("Image Dataset Splitter")
    print("=" * 50)

    # Set random seed for reproducibility
    seed = input("\nEnter random seed (press Enter for random): ").strip()
    if seed:
        random.seed(int(seed))
        print(f"Random seed set to: {seed}")

    # Get split ratios from user
    train_ratio, val_ratio, test_ratio = get_split_ratios()

    # Get file mode (symlink vs copy)
    link_mode = get_link_mode()

    # Confirm before proceeding
    print(f"\nSplit ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"File mode: {link_mode}")
    confirm = input("Proceed with split? (y/n): ").strip().lower()

    if confirm == 'y':
        split_dataset(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            link_mode=link_mode,
        )
    else:
        print("Operation cancelled.")
