"""
Image Cleanup Module - Duplicate Detection and Removal

This module provides functionality to identify and remove duplicate or near-duplicate
images from a directory structure organized by architectural style folders. It uses
perceptual hashing (pHash) to detect visually similar images, even if they have
different file sizes, formats, or minor modifications.

Workflow:
    1. User provides a similarity threshold (0-20)
    2. Script scans each style folder in the configured download directory
    3. Calculates perceptual hash for each image
    4. Groups images with similar hashes (within threshold)
    5. Keeps the first image in each group, removes duplicates
    6. Automatically removes corrupted/unreadable images

Configuration:
    Requires a YAML config file at '../conf/apikeys.yaml' with the following structure:
        paths:
            download_base: /path/to/image/folders

Dependencies:
    - PIL (Pillow): Image processing
    - imagehash: Perceptual hashing algorithms
    - PyYAML: Configuration file parsing

Usage:
    Run as a standalone script:
        $ python 01a_image_cleanup.py

    Or import and call programmatically:
        from 01a_image_cleanup import remove_duplicates
        remove_duplicates()

Author: Architecture Image Modeling Project
"""

import os
import yaml
from PIL import Image
import imagehash
from collections import defaultdict


def load_config(config_file):
    """
    Load and parse settings from a YAML configuration file.

    Args:
        config_file (str): Absolute or relative path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary containing project settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def get_user_input():
    """
    Prompt the user interactively for cleanup configuration settings.

    Guides the user through selecting a similarity threshold for duplicate
    detection. The threshold determines how visually similar two images must
    be to be considered duplicates.

    Returns:
        dict: Configuration dictionary with keys:
            - 'config_path' (str): Path to the apikeys.yaml config file
            - 'threshold' (int): Similarity threshold (0-20)
        None: If the config directory or apikeys.yaml file is not found.

    Note:
        Threshold guidelines:
            - 0-3: Only exact or nearly exact duplicates
            - 5-10: Near-duplicates (recommended for most use cases)
            - 10-20: More aggressive matching, may catch similar but distinct images
    """
    print("=" * 60)
    print("Duplicate Image Finder and Remover")
    print("=" * 60)
    
    # Get config file from conf folder (one level up from script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conf_dir = os.path.join(os.path.dirname(script_dir), 'conf')
    
    if not os.path.exists(conf_dir):
        print(f"Error: 'conf' folder not found at {conf_dir}")
        return None
    
    # Look for apikeys.yaml specifically
    config_path = os.path.join(conf_dir, 'apikeys.yaml')
    
    if not os.path.exists(config_path):
        print(f"Error: apikeys.yaml not found in {conf_dir}")
        return None
    
    print(f"Using config file: apikeys.yaml")
    
    # Get similarity threshold
    print("\nSimilarity threshold determines how similar images must be to be considered duplicates.")
    print("Lower values = more strict (only very similar images), Higher values = more lenient")
    print("Recommended: 5-10 for near-duplicates, 0-3 for exact duplicates")
    
    while True:
        try:
            threshold = int(input("Enter similarity threshold (0-20, recommended 5): ").strip())
            if 0 <= threshold <= 20:
                break
            else:
                print("Please enter a number between 0 and 20.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    return {
        'config_path': config_path,
        'threshold': threshold
    }

def calculate_image_hash(image_path):
    """
    Calculate the perceptual hash (aHash) for an image file.

    Uses the average hash algorithm which is fast and effective for detecting
    duplicates. The hash represents a fingerprint of the image's visual content,
    allowing comparison independent of file format, size, or minor edits.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        imagehash.ImageHash: The computed perceptual hash object.
        None: If the image is corrupted or cannot be processed (file is deleted).

    Side Effects:
        - Corrupted or unreadable images are automatically deleted from disk.
        - Progress and error messages are printed to stdout.

    Note:
        Images are converted to RGB mode before hashing to ensure consistent
        results across different image formats (RGBA, grayscale, etc.).
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Use average hash for speed, or dhash/phash for more accuracy
            return imagehash.average_hash(img)
    except Exception as e:
        print(f"   Error processing {os.path.basename(image_path)}: {e}")
        # Delete the corrupted file
        try:
            os.remove(image_path)
            print(f"   Deleted corrupted file: {os.path.basename(image_path)}")
        except Exception as delete_error:
            print(f"   Failed to delete {os.path.basename(image_path)}: {delete_error}")
        return None

def find_duplicates_in_folder(folder_path, threshold):
    """
    Scan a folder and identify groups of duplicate or near-duplicate images.

    Uses perceptual hashing to compare images. Two images are considered
    duplicates if the Hamming distance between their hashes is less than
    or equal to the specified threshold.

    Args:
        folder_path (str): Path to the folder containing images to scan.
        threshold (int): Maximum hash difference to consider images as duplicates.
            A threshold of 0 means exact matches only; higher values allow
            more visual difference between matched images.

    Returns:
        tuple: A tuple containing:
            - list[list[str]]: List of duplicate groups, where each group is a
              list of file paths. The first file in each group is considered
              the "original" to keep.
            - int: Count of corrupted files that were detected and removed.

    Supported Formats:
        .jpg, .jpeg, .png, .gif, .bmp, .tiff
    """
    print(f"\nScanning folder: {os.path.basename(folder_path)}")
    
    # Dictionary to store hash -> list of image paths
    hash_dict = defaultdict(list)
    corrupted_count = 0
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print("   No images found in folder")
        return [], 0
    
    print(f"   Found {len(image_files)} images to analyze...")
    
    # Calculate hash for each image
    for filename in image_files:
        filepath = os.path.join(folder_path, filename)
        img_hash = calculate_image_hash(filepath)
        
        if img_hash:
            hash_dict[img_hash].append(filepath)
        else:
            # File was corrupted and deleted
            corrupted_count += 1
    
    # Find duplicates and near-duplicates using pairwise hash comparison
    # Uses a greedy clustering approach: once a hash is assigned to a group, it's not reconsidered
    duplicates = []
    processed_hashes = set()  # Track hashes already assigned to a duplicate group

    for hash1, files1 in hash_dict.items():
        if hash1 in processed_hashes:
            continue

        # Start a new potential duplicate group with files sharing this exact hash
        duplicate_group = list(files1)

        # Compare with other hashes to find near-duplicates within threshold
        for hash2, files2 in hash_dict.items():
            if hash1 == hash2 or hash2 in processed_hashes:
                continue

            # Hamming distance between hashes (number of differing bits)
            hash_diff = hash1 - hash2

            if hash_diff <= threshold:
                # Hashes are similar enough - merge into the same duplicate group
                duplicate_group.extend(files2)
                processed_hashes.add(hash2)

        # Only record groups with more than one file (actual duplicates)
        if len(duplicate_group) > 1:
            duplicates.append(duplicate_group)
            processed_hashes.add(hash1)

    return duplicates, corrupted_count

def remove_duplicates():
    """
    Main entry point for the duplicate image cleanup process.

    Orchestrates the complete cleanup workflow:
        1. Collects user configuration (threshold setting)
        2. Loads the project configuration from apikeys.yaml
        3. Iterates through all style subfolders in the download directory
        4. Identifies and removes duplicate images in each folder
        5. Reports summary statistics

    The function processes folders organized by architectural style, where each
    subfolder in the base download path represents a different house style
    category (e.g., 'craftsman', 'victorian', 'colonial').

    Returns:
        None

    Side Effects:
        - Duplicate image files are permanently deleted from disk
        - Corrupted image files are permanently deleted from disk
        - Progress and summary information printed to stdout

    Example:
        >>> remove_duplicates()
        ============================================================
        Duplicate Image Finder and Remover
        ============================================================
        ...
    """
    # Get user input
    user_config = get_user_input()
    if user_config is None:
        return
    
    config_path = user_config['config_path']
    threshold = user_config['threshold']
    
    # Load configuration
    try:
        config = load_config(config_path)
        base_download_path = config['paths']['download_base']
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Make sure your YAML has 'paths.download_base'")
        return
    
    if not os.path.exists(base_download_path):
        print(f"Error: Download path not found at {base_download_path}")
        return
    
    print(f"\nScanning directory: {base_download_path}")
    print(f"Similarity threshold: {threshold}")
    print("=" * 60)
    
    # Get all style folders
    style_folders = [f for f in os.listdir(base_download_path) 
                     if os.path.isdir(os.path.join(base_download_path, f))]
    
    if not style_folders:
        print("No style folders found.")
        return
    
    print(f"Found {len(style_folders)} style folders to process\n")
    
    total_duplicates_removed = 0
    total_corrupted_removed = 0
    
    # Process each style folder
    for style_folder in style_folders:
        folder_path = os.path.join(base_download_path, style_folder)
        duplicates, corrupted_count = find_duplicates_in_folder(folder_path, threshold)
        
        total_corrupted_removed += corrupted_count
        
        if duplicates:
            print(f"   Found {len(duplicates)} duplicate groups")
            
            for group in duplicates:
                # Keep the first image, remove the rest
                keep_file = group[0]
                remove_files = group[1:]
                
                print(f"   Keeping: {os.path.basename(keep_file)}")
                
                for dup_file in remove_files:
                    try:
                        os.remove(dup_file)
                        print(f"   Removed: {os.path.basename(dup_file)}")
                        total_duplicates_removed += 1
                    except Exception as e:
                        print(f"   Error removing {os.path.basename(dup_file)}: {e}")
        else:
            print("   No duplicates found")
    
    print("\n" + "=" * 60)
    print(f"Cleanup complete!")
    print(f"Total duplicates removed: {total_duplicates_removed}")
    print(f"Total corrupted files removed: {total_corrupted_removed}")
    print("=" * 60)

if __name__ == "__main__":
    # Entry point when run as a standalone script
    remove_duplicates()