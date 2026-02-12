"""
Image Cleanup Module - Duplicate Detection and Removal

This module identifies and removes duplicate or near-duplicate images from a
directory structure organized by architectural style folders. It uses perceptual
hashing (pHash) to detect visually similar images, even if they have different
file sizes, formats, or minor modifications.

Optimizations over previous version:
    - Parallel hash computation via multiprocessing.Pool (scales with CPU cores)
    - Uses phash (perceptual hash) instead of average_hash for better accuracy
    - Corruption detection separated from hashing (no surprise file deletions)
    - Pathlib throughout for consistent path handling
    - Reuses shared config loader from utils

Workflow:
    1. User provides a similarity threshold (0-20)
    2. Script scans each style folder in the configured download directory
    3. Removes corrupted/unreadable images (with reporting)
    4. Calculates perceptual hash for each valid image (in parallel)
    5. Groups images with similar hashes (within threshold)
    6. Keeps the first image in each group, removes duplicates

Dependencies:
    - PIL (Pillow): Image processing
    - imagehash: Perceptual hashing algorithms
    - PyYAML: Configuration file parsing

Usage:
    Run as a standalone script:
        $ python -m data_prep.cleanup

    Or via the unified pipeline:
        $ python prepare_data.py --cleanup
"""

import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure parent directory (house_classification/) is on sys.path for standalone execution
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import imagehash
import yaml
from PIL import Image

from utils.config import get_project_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
DEFAULT_HASH_WORKERS = min(cpu_count(), 8)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_user_input() -> Optional[Dict]:
    """
    Prompt the user interactively for cleanup configuration settings.

    Returns:
        Configuration dictionary with 'config_path' and 'threshold',
        or None if config files not found.
    """
    print("=" * 60)
    print("Duplicate Image Finder and Remover")
    print("=" * 60)

    conf_dir = get_project_root() / "conf"

    if not conf_dir.exists():
        print(f"Error: 'conf' folder not found at {conf_dir}")
        return None

    config_path = conf_dir / "apikeys.yaml"

    if not config_path.exists():
        print(f"Error: apikeys.yaml not found in {conf_dir}")
        return None

    print(f"Using config file: apikeys.yaml")

    print("\nSimilarity threshold determines how similar images must be to be considered duplicates.")
    print("Lower values = more strict (only very similar images), Higher values = more lenient")
    print("Recommended: 5-10 for near-duplicates, 0-3 for exact duplicates")

    while True:
        try:
            threshold = int(input("Enter similarity threshold (0-20, recommended 5-7): ").strip())
            if 0 <= threshold <= 20:
                break
            else:
                print("Please enter a number between 0 and 20.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return {
        'config_path': str(config_path),
        'threshold': threshold
    }


# ---------------------------------------------------------------------------
# Corruption Detection (separate pass)
# ---------------------------------------------------------------------------

def find_corrupted_images(image_paths: List[Path]) -> List[Path]:
    """
    Identify corrupted or unreadable image files.

    Opens each image with PIL to verify it can be read. This is a separate
    pass from hashing so that corruption detection doesn't have side effects.

    Args:
        image_paths: List of image file paths to check.

    Returns:
        List of paths to corrupted files.
    """
    corrupted = []
    for filepath in image_paths:
        try:
            with Image.open(filepath) as img:
                img.verify()
        except Exception:
            corrupted.append(filepath)
    return corrupted


def remove_corrupted_images(image_paths: List[Path]) -> Tuple[List[Path], int]:
    """
    Find and remove corrupted images, returning the clean list.

    Args:
        image_paths: List of image file paths to check.

    Returns:
        Tuple of (clean_paths, corrupted_count).
    """
    corrupted = find_corrupted_images(image_paths)

    for filepath in corrupted:
        try:
            filepath.unlink()
            print(f"   Removed corrupted: {filepath.name}")
        except Exception as e:
            print(f"   Failed to remove {filepath.name}: {e}")

    clean_paths = [p for p in image_paths if p not in set(corrupted)]
    return clean_paths, len(corrupted)


# ---------------------------------------------------------------------------
# Parallel Hashing
# ---------------------------------------------------------------------------

def _hash_single_image(filepath_str: str) -> Tuple[str, Optional[str]]:
    """
    Compute perceptual hash for a single image. Worker function for multiprocessing.

    Uses phash (perceptual hash) which is more robust than average_hash against
    resizing, minor edits, and compression differences.

    Args:
        filepath_str: String path to the image file.

    Returns:
        Tuple of (filepath_str, hash_hex_string) or (filepath_str, None) on error.
    """
    try:
        with Image.open(filepath_str) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            h = imagehash.phash(img)
            return (filepath_str, str(h))
    except Exception:
        return (filepath_str, None)


def compute_hashes_parallel(
    image_paths: List[Path],
    workers: Optional[int] = None
) -> Tuple[Dict, List[Path]]:
    """
    Compute perceptual hashes for all images using multiprocessing.

    Args:
        image_paths: List of image file paths to hash.
        workers: Number of parallel workers. Defaults to min(cpu_count, 8).

    Returns:
        Tuple of:
            - hash_dict: defaultdict mapping ImageHash -> list of Paths
            - failed: list of paths that failed to hash
    """
    if workers is None:
        workers = DEFAULT_HASH_WORKERS

    str_paths = [str(p) for p in image_paths]

    hash_dict = defaultdict(list)
    failed = []

    with Pool(workers) as pool:
        results = pool.map(_hash_single_image, str_paths)

    for filepath_str, hash_hex in results:
        filepath = Path(filepath_str)
        if hash_hex is not None:
            hash_obj = imagehash.hex_to_hash(hash_hex)
            hash_dict[hash_obj].append(filepath)
        else:
            failed.append(filepath)

    return hash_dict, failed


# ---------------------------------------------------------------------------
# Duplicate Detection
# ---------------------------------------------------------------------------

def find_duplicate_groups(
    hash_dict: Dict,
    threshold: int
) -> List[List[Path]]:
    """
    Find groups of duplicate/near-duplicate images from pre-computed hashes.

    Uses greedy clustering: for each unprocessed hash, finds all other hashes
    within the Hamming distance threshold and groups their files together.

    Args:
        hash_dict: Mapping of ImageHash -> list of file paths.
        threshold: Maximum Hamming distance to consider images as duplicates.

    Returns:
        List of duplicate groups. Each group is a list of file paths where
        the first element is the "original" to keep.
    """
    duplicates = []
    processed_hashes = set()

    hash_items = list(hash_dict.items())

    for i, (hash1, files1) in enumerate(hash_items):
        if hash1 in processed_hashes:
            continue

        # Start group with files sharing this exact hash
        group = list(files1)

        # Compare with remaining hashes
        for j in range(i + 1, len(hash_items)):
            hash2, files2 = hash_items[j]
            if hash2 in processed_hashes:
                continue

            if hash1 - hash2 <= threshold:
                group.extend(files2)
                processed_hashes.add(hash2)

        if len(group) > 1:
            duplicates.append(group)
            processed_hashes.add(hash1)

    return duplicates


# ---------------------------------------------------------------------------
# Folder Processing
# ---------------------------------------------------------------------------

def process_folder(folder_path: Path, threshold: int) -> Tuple[int, int]:
    """
    Process a single style folder: remove corrupted, detect and remove duplicates.

    Args:
        folder_path: Path to the style folder.
        threshold: Hamming distance threshold for duplicate detection.

    Returns:
        Tuple of (duplicates_removed, corrupted_removed).
    """
    print(f"\nScanning folder: {folder_path.name}")

    # Get all image files
    image_files = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print("   No images found in folder")
        return 0, 0

    print(f"   Found {len(image_files)} images")

    # Step 1: Remove corrupted images (separate pass)
    clean_files, corrupted_count = remove_corrupted_images(image_files)

    if corrupted_count:
        print(f"   Removed {corrupted_count} corrupted files")

    if not clean_files:
        return 0, corrupted_count

    # Step 2: Compute hashes in parallel
    print(f"   Computing perceptual hashes ({DEFAULT_HASH_WORKERS} workers)...")
    hash_dict, hash_failures = compute_hashes_parallel(clean_files)

    if hash_failures:
        print(f"   {len(hash_failures)} files failed to hash")

    # Step 3: Find duplicate groups
    duplicate_groups = find_duplicate_groups(hash_dict, threshold)

    # Step 4: Remove duplicates (keep first in each group)
    duplicates_removed = 0

    if duplicate_groups:
        print(f"   Found {len(duplicate_groups)} duplicate groups")

        for group in duplicate_groups:
            keep_file = group[0]
            remove_files = group[1:]

            print(f"   Keeping: {keep_file.name}")

            for dup_file in remove_files:
                try:
                    dup_file.unlink()
                    print(f"   Removed: {dup_file.name}")
                    duplicates_removed += 1
                except Exception as e:
                    print(f"   Error removing {dup_file.name}: {e}")
    else:
        print("   No duplicates found")

    return duplicates_removed, corrupted_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def remove_duplicates():
    """
    Main entry point for the duplicate image cleanup process.

    Orchestrates: config loading -> folder scanning -> corruption removal ->
    parallel hashing -> duplicate detection -> duplicate removal.
    """
    user_config = get_user_input()
    if user_config is None:
        return

    config_path = user_config['config_path']
    threshold = user_config['threshold']

    # Load configuration
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        base_download_path = Path(config['paths']['download_base'])
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Make sure your YAML has 'paths.download_base'")
        return

    if not base_download_path.exists():
        print(f"Error: Download path not found at {base_download_path}")
        return

    print(f"\nScanning directory: {base_download_path}")
    print(f"Similarity threshold: {threshold}")
    print(f"Hash algorithm: phash (perceptual hash)")
    print(f"Hash workers: {DEFAULT_HASH_WORKERS}")
    print("=" * 60)

    # Get all style folders (skip train/validation/test split dirs)
    skip_dirs = {'train', 'validation', 'test'}
    style_folders = sorted([
        d for d in base_download_path.iterdir()
        if d.is_dir() and d.name not in skip_dirs
    ])

    if not style_folders:
        print("No style folders found.")
        return

    print(f"Found {len(style_folders)} style folders to process\n")

    total_duplicates_removed = 0
    total_corrupted_removed = 0

    for style_folder in style_folders:
        dups, corrupted = process_folder(style_folder, threshold)
        total_duplicates_removed += dups
        total_corrupted_removed += corrupted

    print("\n" + "=" * 60)
    print(f"Cleanup complete!")
    print(f"Total duplicates removed: {total_duplicates_removed}")
    print(f"Total corrupted files removed: {total_corrupted_removed}")
    print("=" * 60)


if __name__ == "__main__":
    remove_duplicates()
