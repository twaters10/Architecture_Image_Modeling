import os
import shutil
import random
from pathlib import Path

def get_split_ratios():
    """Prompt user for train/validation/test split ratios."""
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

def split_dataset(source_dir="../architectural_style_images", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split images from ../architectural_style_images/housestyle/images into train/val/test splits.
    
    Args:
        source_dir: Root data directory
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
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
        images_dir = housestyle_dir / "images"
        
        if not images_dir.exists():
            print(f"\nWarning: {images_dir} not found, skipping...")
            continue
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        images = [f for f in images_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not images:
            print(f"\nWarning: No images found in {images_dir}, skipping...")
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
    
    # Confirm before proceeding
    print(f"\nSplit ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    confirm = input("Proceed with split? (y/n): ").strip().lower()
    
    if confirm == 'y':
        split_dataset(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    else:
        print("Operation cancelled.")