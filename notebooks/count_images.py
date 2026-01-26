#!/usr/bin/env python3
"""Count the number of images in each subfolder under architectural_style_images."""

import os
from pathlib import Path

def count_images_in_subfolders(base_dir: str) -> dict:
    """Count image files in each subfolder of the given directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    base_path = Path(base_dir)

    counts = {}
    for subfolder in sorted(base_path.iterdir()):
        if subfolder.is_dir() and not subfolder.name.startswith('.'):
            image_count = sum(
                1 for f in subfolder.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            counts[subfolder.name] = image_count

    return counts

def main():
    base_dir = Path(__file__).parent / "architectural_style_images"

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return

    counts = count_images_in_subfolders(base_dir)

    print("\nImage counts by architectural style:")
    print("-" * 40)

    total = 0
    for style, count in counts.items():
        print(f"{style:<25} {count:>5}")
        total += count

    print("-" * 40)
    print(f"{'TOTAL':<25} {total:>5}")

if __name__ == "__main__":
    main()