"""
Image Scraper Module - Architectural House Style Dataset Builder

This module automates the collection of house exterior images for training
machine learning models to classify architectural styles. It uses the SerpAPI
Google Images search to find and download relevant images organized by style.

Workflow:
    1. User selects a configuration file (containing API key and paths)
    2. User selects a CSV file containing house style names
    3. User specifies how many images to download per style
    4. Script searches Google Images for each style and downloads results
    5. Images are saved in style-specific subdirectories

Directory Structure Created:
    download_base/
    ├── craftsman/
    │   ├── craftsman_12345.jpg
    │   └── craftsman_67890.jpg
    ├── victorian/
    │   ├── victorian_11111.jpg
    │   └── victorian_22222.jpg
    └── ...

Configuration:
    Requires a YAML config file in '../conf/' with the following structure:
        api_keys:
            serpapi: YOUR_SERPAPI_KEY
        paths:
            download_base: /path/to/save/images

    Requires a CSV file in '../data/' with a 'style' column containing
    architectural style names (e.g., "Craftsman", "Victorian", "Colonial").

Dependencies:
    - requests: HTTP library for downloading images
    - pandas: CSV file parsing
    - serpapi: Google Search API client
    - PyYAML: Configuration file parsing

API Requirements:
    - SerpAPI account with valid API key (https://serpapi.com)
    - API usage is subject to SerpAPI rate limits and pricing

Usage:
    Run as a standalone script:
        $ python 01_image_scrapper.py

    Follow the interactive prompts to configure and start downloading.

Author: Architecture Image Modeling Project
"""

import os
import requests
import yaml
import pandas as pd
from serpapi import GoogleSearch
import time
import random


def load_config(config_file):
    """
    Load and parse settings from a YAML configuration file.

    Args:
        config_file (str): Absolute or relative path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary containing API keys and paths.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def get_user_input():
    """
    Prompt the user interactively for download configuration settings.

    Guides the user through a series of selections:
        1. Choose a YAML configuration file from the conf/ directory
        2. Choose a CSV file containing house styles from the data/ directory
        3. Specify the number of images to download per architectural style

    Returns:
        dict: Configuration dictionary with keys:
            - 'config_path' (str): Full path to the selected YAML config file
            - 'csv_path' (str): Full path to the selected CSV file
            - 'num_images' (int): Number of images to download per style
        None: If required directories or files are not found.

    Note:
        The function expects the following directory structure relative to
        the script location:
            ../conf/   - Contains YAML configuration files
            ../data/   - Contains CSV files with house style definitions
    """
    print("=" * 60)
    print("House Style Image Downloader")
    print("=" * 60)
    
    # Get config file from conf folder (one level up from script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conf_dir = os.path.join(os.path.dirname(script_dir), 'conf')
    
    if not os.path.exists(conf_dir):
        print(f"Error: 'conf' folder not found at {conf_dir}")
        return None
    
    # List all YAML files in conf directory
    yaml_files = [f for f in os.listdir(conf_dir) if f.endswith(('.yaml', '.yml'))]
    
    if not yaml_files:
        print(f"Error: No YAML files found in {conf_dir}")
        return None
    
    # Display available config files
    print(f"\nAvailable config files in 'conf' folder:")
    for i, filename in enumerate(yaml_files, 1):
        print(f"  {i}. {filename}")
    
    # Get user selection
    while True:
        try:
            selection = int(input(f"\nSelect a config file (1-{len(yaml_files)}): ").strip())
            if 1 <= selection <= len(yaml_files):
                config_path = os.path.join(conf_dir, yaml_files[selection - 1])
                print(f"Selected: {yaml_files[selection - 1]}")
                break
            else:
                print(f"Please enter a number between 1 and {len(yaml_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get CSV file from data folder (one level up from script location)
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    if not os.path.exists(data_dir):
        print(f"Error: 'data' folder not found at {data_dir}")
        return None
    
    # List all CSV files in data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        return None
    
    # Display available CSV files
    print(f"\nAvailable CSV files in 'data' folder:")
    for i, filename in enumerate(csv_files, 1):
        print(f"  {i}. {filename}")
    
    # Get user selection
    while True:
        try:
            selection = int(input(f"\nSelect a CSV file (1-{len(csv_files)}): ").strip())
            if 1 <= selection <= len(csv_files):
                csv_path = os.path.join(data_dir, csv_files[selection - 1])
                print(f"Selected: {csv_files[selection - 1]}")
                break
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get number of images per style
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
        'config_path': config_path,
        'csv_path': csv_path,
        'num_images': num_images
    }

def download_images_from_csv():
    """
    Main entry point for the image downloading process.

    Orchestrates the complete download workflow:
        1. Collects user configuration (files and image count)
        2. Loads API credentials and path settings from YAML config
        3. Reads house styles from the selected CSV file
        4. For each style, searches Google Images via SerpAPI
        5. Downloads and saves images to style-specific folders

    The function creates a directory structure where each architectural style
    has its own subfolder containing downloaded images. Filenames include the
    style name and a random 5-digit number for uniqueness.

    Returns:
        None

    Side Effects:
        - Creates directories for each house style in the download path
        - Downloads and saves image files to disk
        - Progress and status messages printed to stdout

    Error Handling:
        - Invalid configurations are reported and function exits gracefully
        - Individual image download failures are logged but don't stop the process
        - API errors for a style are logged, then processing continues to next style

    Rate Limiting:
        Includes a 0.5 second delay between downloads to avoid IP blocking
        and respect server resources.
    """
    # Get user input
    user_config = get_user_input()
    if user_config is None:
        return
    
    # Extract configuration values from user input
    config_path = user_config['config_path']
    csv_path = user_config['csv_path']
    num_images = user_config['num_images']

    # 1. Load configuration and extract required settings
    # Expected YAML structure: api_keys.serpapi and paths.download_base
    try:
        config = load_config(config_path)
        api_key = config['api_keys']['serpapi']           # SerpAPI authentication key
        base_download_path = config['paths']['download_base']  # Root folder for downloads
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Make sure your YAML has 'api_keys.serpapi' and 'paths.download_base'")
        return

    # 2. Load and validate the CSV file containing house styles
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Verify the CSV contains the required 'style' column
    style_col = 'style'
    if style_col not in df.columns:
        print(f"Error: CSV must have a '{style_col}' column.")
        print(f"Found columns: {df.columns.tolist()}")
        return

    # Display summary of planned download operation
    print(f"\nFound {len(df)} house styles in CSV")
    print(f"Will download {num_images} images per style")
    print(f"Images will be saved to: {base_download_path}\n")
    
    # 4. Loop through each row in the CSV, processing one style at a time
    for index, row in df.iterrows():
        style_name = str(row[style_col]).strip()
        # Normalize folder name: lowercase with underscores (e.g., "Gothic Revival" -> "gothic_revival")
        style_folder = style_name.replace(" ", "_").lower()

        # Construct search query optimized for exterior architectural photos
        search_query = f"{style_name} house exterior architecture"

        # Create a dedicated directory for this architectural style
        style_dir = os.path.join(base_download_path, style_folder)
        if not os.path.exists(style_dir):
            os.makedirs(style_dir)

        print(f"\n--- Processing Style: {style_name} ---")
        print(f"Search query: {search_query}")

        # Configure SerpAPI Google Images search parameters
        params = {
            "engine": "google_images",  # Use Google Images search engine
            "q": search_query,          # Search query string
            "api_key": api_key,         # SerpAPI authentication key
            "num": num_images           # Requested number of results
        }

        try:
            # Execute the search and parse results
            search = GoogleSearch(params)
            results = search.get_dict()
            images = results.get("images_results", [])

            print(f"Found {len(images)} images for: {style_name}")

            downloaded = 0
            used_numbers = set()  # Track used random numbers to ensure unique filenames

            for i, img in enumerate(images):
                # Extract the original (full-resolution) image URL
                img_url = img.get("original")
                if not img_url:
                    continue

                try:
                    # Download image with timeout and browser-like User-Agent header
                    response = requests.get(img_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                    response.raise_for_status()  # Raise exception for HTTP errors (4xx, 5xx)

                    # Generate unique 5-digit random number for filename uniqueness
                    while True:
                        random_num = random.randint(10000, 99999)
                        if random_num not in used_numbers:
                            used_numbers.add(random_num)
                            break

                    # Construct filename: {style}_{random}.jpg (e.g., gothic_revival_47382.jpg)
                    filename = f"{style_folder}_{random_num}.jpg"
                    filepath = os.path.join(style_dir, filename)

                    # Write image bytes to disk
                    with open(filepath, 'wb') as f:
                        f.write(response.content)

                    downloaded += 1
                    print(f"   [+] Saved {filename} ({downloaded}/{num_images})")

                    # Rate limiting: delay between requests to avoid IP blocking
                    time.sleep(0.5)

                except Exception as e:
                    # Log individual download failures but continue processing
                    print(f"   [-] Failed image {i+1}: {e}")

            print(f"Successfully downloaded {downloaded} images for {style_name}")

        except Exception as e:
            # Log API errors but continue to next style
            print(f"API Search Error for '{style_name}': {e}")

    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Entry point when run as a standalone script
    download_images_from_csv()