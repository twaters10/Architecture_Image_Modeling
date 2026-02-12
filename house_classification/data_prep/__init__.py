"""
Data Preparation Package for Architectural Style Classification.

Submodules:
    scraper  - Download images from Google Images via SerpAPI
    cleanup  - Remove corrupted and duplicate images (parallel phash)
    splitter - Create train/validation/test sets (symlinks by default)

The unified pipeline runner is in the parent module: prepare_data.py
"""
