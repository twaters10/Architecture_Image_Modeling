"""
Image Scraper Module - Architectural House Style Dataset Builder

This module automates the collection of house exterior images for training
machine learning models to classify architectural styles. It uses the SerpAPI
Google Images search to find and download relevant images organized by style.

Optimizations over previous version:
    - Concurrent downloads via ThreadPoolExecutor (8 workers)
    - Image validation on download (PIL verify) to catch corrupt/non-image files
    - Resume capability: skips styles that already have enough images
    - SerpAPI pagination to fetch more than 100 images per style
    - Pathlib throughout for consistent path handling

Workflow:
    1. User selects a configuration file (containing API key and paths)
    2. User selects a CSV file containing house style names
    3. User specifies how many images to download per style
    4. Script searches Google Images for each style and downloads results
    5. Images are saved in style-specific subdirectories

Directory Structure Created:
    download_base/
    ├── craftsman/
    │   ├── craftsman_00001.jpg
    │   └── craftsman_00002.jpg
    ├── victorian/
    │   ├── victorian_00001.jpg
    │   └── victorian_00002.jpg
    └── ...

Dependencies:
    - requests: HTTP library for downloading images
    - pandas: CSV file parsing
    - serpapi: Google Search API client
    - PyYAML: Configuration file parsing
    - Pillow: Image validation

Usage:
    Run as a standalone script:
        $ python -m data_prep.scraper

    Or via the unified pipeline:
        $ python prepare_data.py --scrape
"""

import io
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure parent directory (house_classification/) is on sys.path for standalone execution
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import pandas as pd
import requests
import yaml
from PIL import Image
from serpapi import GoogleSearch

from utils.config import get_project_root

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
MAX_DOWNLOAD_WORKERS = 8


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_file: str) -> Dict:
    """
    Load and parse settings from a YAML configuration file.

    Args:
        config_file: Absolute or relative path to the YAML config file.

    Returns:
        Parsed configuration dictionary containing API keys and paths.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def get_user_input() -> Optional[Dict]:
    """
    Prompt the user interactively for download configuration settings.

    Guides the user through:
        1. Choose a YAML configuration file from the conf/ directory
        2. Choose a CSV file containing house styles from the data/ directory
        3. Specify the number of images to download per architectural style

    Returns:
        Configuration dictionary, or None if required files not found.
    """
    print("=" * 60)
    print("House Style Image Downloader")
    print("=" * 60)

    # Get config file from conf folder
    conf_dir = get_project_root() / "conf"

    if not conf_dir.exists():
        print(f"Error: 'conf' folder not found at {conf_dir}")
        return None

    yaml_files = sorted([f.name for f in conf_dir.iterdir()
                         if f.suffix in ('.yaml', '.yml')])

    if not yaml_files:
        print(f"Error: No YAML files found in {conf_dir}")
        return None

    print(f"\nAvailable config files in 'conf' folder:")
    for i, filename in enumerate(yaml_files, 1):
        print(f"  {i}. {filename}")

    while True:
        try:
            selection = int(input(f"\nSelect a config file (1-{len(yaml_files)}): ").strip())
            if 1 <= selection <= len(yaml_files):
                config_path = conf_dir / yaml_files[selection - 1]
                print(f"Selected: {yaml_files[selection - 1]}")
                break
            else:
                print(f"Please enter a number between 1 and {len(yaml_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get CSV file from data folder
    data_dir = get_project_root() / "data"

    if not data_dir.exists():
        print(f"Error: 'data' folder not found at {data_dir}")
        return None

    csv_files = sorted([f.name for f in data_dir.iterdir() if f.suffix == '.csv'])

    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        return None

    print(f"\nAvailable CSV files in 'data' folder:")
    for i, filename in enumerate(csv_files, 1):
        print(f"  {i}. {filename}")

    while True:
        try:
            selection = int(input(f"\nSelect a CSV file (1-{len(csv_files)}): ").strip())
            if 1 <= selection <= len(csv_files):
                csv_path = data_dir / csv_files[selection - 1]
                print(f"Selected: {csv_files[selection - 1]}")
                break
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            num_images = int(input("Enter number of images to download per house style: ").strip())
            if num_images > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return {
        'config_path': str(config_path),
        'csv_path': str(csv_path),
        'num_images': num_images
    }


# ---------------------------------------------------------------------------
# Image Validation
# ---------------------------------------------------------------------------

def validate_image_bytes(content: bytes) -> bool:
    """
    Check if downloaded content is a valid image.

    Uses PIL to open and verify the image data. Catches corrupt downloads,
    HTML error pages saved as .jpg, and other non-image content.

    Args:
        content: Raw bytes of the downloaded file.

    Returns:
        True if content is a valid image, False otherwise.
    """
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Resume Support
# ---------------------------------------------------------------------------

def count_existing_images(style_dir: Path) -> int:
    """
    Count valid image files already present in a style directory.

    Args:
        style_dir: Path to the style-specific image directory.

    Returns:
        Number of existing image files.
    """
    if not style_dir.exists():
        return 0
    return sum(1 for f in style_dir.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


# ---------------------------------------------------------------------------
# SerpAPI Pagination
# ---------------------------------------------------------------------------

def fetch_image_urls(api_key: str, query: str, num_images: int) -> List[str]:
    """
    Fetch image URLs from SerpAPI with automatic pagination.

    A single SerpAPI call returns ~100 results. This function paginates
    using the 'ijn' parameter to collect larger sets.

    Args:
        api_key: SerpAPI authentication key.
        query: Search query string.
        num_images: Target number of image URLs to collect.

    Returns:
        List of image URLs (up to num_images).
    """
    urls = []
    page = 0

    while len(urls) < num_images:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
            "num": 100,
            "ijn": str(page),
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            images = results.get("images_results", [])

            if not images:
                break

            for img in images:
                url = img.get("original")
                if url:
                    urls.append(url)

            page += 1
        except Exception as e:
            print(f"   API error on page {page}: {e}")
            break

    return urls[:num_images]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_single_image(url: str, dest_path: Path) -> bool:
    """
    Download a single image, validate it with PIL, and save to disk.

    Args:
        url: URL of the image to download.
        dest_path: Destination file path.

    Returns:
        True if download and validation succeeded, False otherwise.
    """
    try:
        response = requests.get(
            url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'}
        )
        response.raise_for_status()

        if not validate_image_bytes(response.content):
            return False

        dest_path.write_bytes(response.content)
        return True
    except Exception:
        return False


def download_images_for_style(
    api_key: str,
    style_name: str,
    style_dir: Path,
    num_images: int,
    max_workers: int = MAX_DOWNLOAD_WORKERS
) -> int:
    """
    Download images for a single architectural style with concurrent downloads.

    Supports resume: checks how many valid images already exist and only
    downloads enough to reach the target count.

    Args:
        api_key: SerpAPI authentication key.
        style_name: Human-readable style name (e.g., "Gothic Revival").
        style_dir: Path to the style-specific image directory.
        num_images: Target number of images for this style.
        max_workers: Maximum concurrent download threads.

    Returns:
        Number of newly downloaded images.
    """
    style_folder = style_name.replace(" ", "_").lower()
    query = f"{style_name} house exterior architecture"

    # Resume: check existing images
    existing = count_existing_images(style_dir)
    if existing >= num_images:
        print(f"   Already have {existing}/{num_images} images, skipping")
        return 0

    needed = num_images - existing
    print(f"   Have {existing}, need {needed} more")

    # Fetch URLs with pagination (request extras to account for download failures)
    buffer = min(needed, 20)
    urls = fetch_image_urls(api_key, query, needed + buffer)
    print(f"   Fetched {len(urls)} URLs from SerpAPI")

    if not urls:
        print("   No image URLs found")
        return 0

    # Pre-generate sequential filenames (avoids collision issues)
    style_dir.mkdir(parents=True, exist_ok=True)
    counter = existing
    tasks = []
    for url in urls:
        counter += 1
        filename = f"{style_folder}_{counter:05d}.jpg"
        dest_path = style_dir / filename
        tasks.append((url, dest_path))

    # Download concurrently
    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dest = {
            executor.submit(download_single_image, url, dest): dest
            for url, dest in tasks
        }

        for future in as_completed(future_to_dest):
            dest_path = future_to_dest[future]

            try:
                if future.result():
                    downloaded += 1
                    print(f"   [+] Saved {dest_path.name} ({downloaded}/{needed})")
                else:
                    failed += 1
                    # Clean up any partial file
                    if dest_path.exists():
                        dest_path.unlink()
            except Exception:
                failed += 1
                if dest_path.exists():
                    dest_path.unlink()

    if failed:
        print(f"   {failed} downloads failed")

    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_images_from_csv():
    """
    Main entry point for the image downloading process.

    Orchestrates the complete download workflow:
        1. Collects user configuration (files and image count)
        2. Loads API credentials and path settings from YAML config
        3. Reads house styles from the selected CSV file
        4. For each style, searches Google Images via SerpAPI
        5. Downloads and validates images concurrently
    """
    user_config = get_user_input()
    if user_config is None:
        return

    config_path = user_config['config_path']
    csv_path = user_config['csv_path']
    num_images = user_config['num_images']

    # Load configuration
    try:
        config = load_config(config_path)
        api_key = config['api_keys']['serpapi']
        base_download_path = Path(config['paths']['download_base'])
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Make sure your YAML has 'api_keys.serpapi' and 'paths.download_base'")
        return

    # Load house styles from CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    style_col = 'style'
    if style_col not in df.columns:
        print(f"Error: CSV must have a '{style_col}' column.")
        print(f"Found columns: {df.columns.tolist()}")
        return

    # Display summary
    print(f"\nFound {len(df)} house styles in CSV")
    print(f"Will download {num_images} images per style")
    print(f"Images will be saved to: {base_download_path}")
    print(f"Concurrent downloads: {MAX_DOWNLOAD_WORKERS} workers")
    print(f"Resume: existing images will be preserved\n")

    total_downloaded = 0

    for index, row in df.iterrows():
        style_name = str(row[style_col]).strip()
        style_folder = style_name.replace(" ", "_").lower()
        style_dir = base_download_path / style_folder

        print(f"\n--- Processing Style: {style_name} ---")
        print(f"Search query: {style_name} house exterior architecture")

        downloaded = download_images_for_style(
            api_key=api_key,
            style_name=style_name,
            style_dir=style_dir,
            num_images=num_images
        )

        total_downloaded += downloaded
        print(f"Successfully downloaded {downloaded} new images for {style_name}")

    print("\n" + "=" * 60)
    print(f"Dataset creation complete! Total new images: {total_downloaded}")
    print("=" * 60)


if __name__ == "__main__":
    download_images_from_csv()
