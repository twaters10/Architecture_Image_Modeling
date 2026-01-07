import os
import yaml
from PIL import Image
import imagehash
from collections import defaultdict

def load_config(config_file):
    """Loads settings from the YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def get_user_input():
    """Prompts user for configuration."""
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
    """Calculate perceptual hash for an image. Returns None for corrupted images."""
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
    """Find duplicate images in a folder based on perceptual hashing."""
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
    
    # Find duplicates and near-duplicates
    duplicates = []
    processed_hashes = set()
    
    for hash1, files1 in hash_dict.items():
        if hash1 in processed_hashes:
            continue
        
        duplicate_group = list(files1)
        
        # Compare with other hashes to find near-duplicates
        for hash2, files2 in hash_dict.items():
            if hash1 == hash2 or hash2 in processed_hashes:
                continue
            
            # Calculate hash difference
            hash_diff = hash1 - hash2
            
            if hash_diff <= threshold:
                duplicate_group.extend(files2)
                processed_hashes.add(hash2)
        
        # If we found duplicates, keep the first and mark others for removal
        if len(duplicate_group) > 1:
            duplicates.append(duplicate_group)
            processed_hashes.add(hash1)
    
    return duplicates, corrupted_count

def remove_duplicates():
    """Main function to find and remove duplicate images."""
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
    remove_duplicates()