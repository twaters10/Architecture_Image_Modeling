import os
import requests
import yaml
import pandas as pd
from serpapi import GoogleSearch
import time
import random

def load_config(config_file):
    """Loads settings from the YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def get_user_input():
    """Prompts user for required file paths and configuration."""
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
    # Get user input
    user_config = get_user_input()
    if user_config is None:
        return
    
    config_path = user_config['config_path']
    csv_path = user_config['csv_path']
    num_images = user_config['num_images']
    
    # 1. Load configuration and API Key
    try:
        config = load_config(config_path)
        api_key = config['api_keys']['serpapi']
        base_download_path = config['paths']['download_base']
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Make sure your YAML has 'api_keys.serpapi' and 'paths.download_base'")
        return
    
    # 2. Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Check for 'style' column
    style_col = 'style'
    if style_col not in df.columns:
        print(f"Error: CSV must have a '{style_col}' column.")
        print(f"Found columns: {df.columns.tolist()}")
        return

    print(f"\nFound {len(df)} house styles in CSV")
    print(f"Will download {num_images} images per style")
    print(f"Images will be saved to: {base_download_path}\n")
    
    # 4. Loop through each row in the CSV
    for index, row in df.iterrows():
        style_name = str(row[style_col]).strip()
        style_folder = style_name.replace(" ", "_").lower()
        
        # Create search query with exterior emphasis
        search_query = f"{style_name} house exterior architecture"
        
        # Create a specific directory for this style
        style_dir = os.path.join(base_download_path, style_folder)
        if not os.path.exists(style_dir):
            os.makedirs(style_dir)
        
        print(f"\n--- Processing Style: {style_name} ---")
        print(f"Search query: {search_query}")

        # Setup SerpApi Search
        params = {
            "engine": "google_images",
            "q": search_query,
            "api_key": api_key,
            "num": num_images
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            images = results.get("images_results", [])
            
            print(f"Found {len(images)} images for: {style_name}")

            downloaded = 0
            used_numbers = set()
            
            for i, img in enumerate(images):
                img_url = img.get("original")
                if not img_url:
                    continue
                    
                try:
                    # Download the image
                    response = requests.get(img_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                    response.raise_for_status()

                    # Generate unique random number for filename
                    while True:
                        random_num = random.randint(10000, 99999)
                        if random_num not in used_numbers:
                            used_numbers.add(random_num)
                            break
                    
                    # Create filename with random number (e.g., gothic_revival_47382.jpg)
                    filename = f"{style_folder}_{random_num}.jpg"
                    filepath = os.path.join(style_dir, filename)

                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    downloaded += 1
                    print(f"   [+] Saved {filename} ({downloaded}/{num_images})")
                    
                    # Polite delay to avoid IP blocks
                    time.sleep(0.5)

                except Exception as e:
                    print(f"   [-] Failed image {i+1}: {e}")

            print(f"Successfully downloaded {downloaded} images for {style_name}")

        except Exception as e:
            print(f"API Search Error for '{style_name}': {e}")

    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)

if __name__ == "__main__":
    download_images_from_csv()