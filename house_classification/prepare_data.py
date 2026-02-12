#!/usr/bin/env python3
"""
Unified Data Preparation Pipeline for Architectural Style Classification.

Chains the three data preparation steps into a single reproducible workflow:
    1. Scrape   - Download images from Google Images via SerpAPI
    2. Cleanup  - Remove corrupted and duplicate images (parallel phash)
    3. Split    - Create train/validation/test sets (symlinks by default)

Each step can be skipped via CLI flags or interactive prompts.

Usage:
    # Interactive mode - prompts for each step
    python prepare_data.py

    # Run all steps with defaults
    python prepare_data.py --all

    # Run specific steps
    python prepare_data.py --scrape --cleanup --split

    # Skip a step
    python prepare_data.py --all --skip-scrape

    # Full CLI mode (no prompts)
    python prepare_data.py --all --num-images 200 --threshold 5 \\
        --train-ratio 70 --val-ratio 20 --test-ratio 10 --seed 42

    # Cleanup + split only (common after adding images manually)
    python prepare_data.py --cleanup --split --threshold 7
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

from utils.config import get_project_root


# ---------------------------------------------------------------------------
# Lazy imports (so prepare_data.py loads fast even if deps are missing)
# ---------------------------------------------------------------------------


def _get_scraper():
    from data_prep import scraper
    return scraper


def _get_cleanup():
    from data_prep import cleanup
    return cleanup


def _get_splitter():
    from data_prep import splitter
    return splitter


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no with a default."""
    suffix = "(Y/n)" if default else "(y/N)"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print("Please enter y or n.")


def prompt_int(question: str, default: int) -> int:
    """Prompt user for an integer with a default."""
    while True:
        answer = input(f"{question} [{default}]: ").strip()
        if not answer:
            return default
        try:
            return int(answer)
        except ValueError:
            print("Please enter a valid number.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified data preparation pipeline: scrape -> cleanup -> split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py                     # Interactive mode
  python prepare_data.py --all               # All steps, default settings
  python prepare_data.py --cleanup --split   # Skip scraping
  python prepare_data.py --all --skip-scrape # Same as above
  python prepare_data.py --all --num-images 200 --threshold 5 --seed 42
        """,
    )

    # Step selection
    step_group = parser.add_argument_group("steps", "Select which steps to run")
    step_group.add_argument("--all", action="store_true",
                            help="Run all steps (scrape, cleanup, split)")
    step_group.add_argument("--scrape", action="store_true",
                            help="Run scraping step")
    step_group.add_argument("--cleanup", action="store_true",
                            help="Run cleanup step")
    step_group.add_argument("--split", action="store_true",
                            help="Run split step")
    step_group.add_argument("--skip-scrape", action="store_true",
                            help="Skip scraping when using --all")
    step_group.add_argument("--skip-cleanup", action="store_true",
                            help="Skip cleanup when using --all")
    step_group.add_argument("--skip-split", action="store_true",
                            help="Skip split when using --all")

    # Scrape options
    scrape_group = parser.add_argument_group("scrape options")
    scrape_group.add_argument("--num-images", type=int, default=100,
                              help="Images to download per style (default: 100)")
    scrape_group.add_argument("--csv", type=str, default=None,
                              help="Path to CSV with styles (default: data/house_style_list.csv)")
    scrape_group.add_argument("--config", type=str, default=None,
                              help="Path to apikeys.yaml (default: conf/apikeys.yaml)")

    # Cleanup options
    cleanup_group = parser.add_argument_group("cleanup options")
    cleanup_group.add_argument("--threshold", type=int, default=5,
                               help="Duplicate similarity threshold 0-20 (default: 5)")

    # Split options
    split_group = parser.add_argument_group("split options")
    split_group.add_argument("--train-ratio", type=float, default=70,
                             help="Train split percentage (default: 70)")
    split_group.add_argument("--val-ratio", type=float, default=20,
                             help="Validation split percentage (default: 20)")
    split_group.add_argument("--test-ratio", type=float, default=10,
                             help="Test split percentage (default: 10)")
    split_group.add_argument("--seed", type=int, default=None,
                             help="Random seed for reproducible splits")
    split_group.add_argument("--copy", action="store_true",
                             help="Copy files instead of symlinking")

    args = parser.parse_args()

    # Determine which steps to run
    any_step_explicit = args.scrape or args.cleanup or args.split
    interactive = not args.all and not any_step_explicit

    if args.all:
        do_scrape = not args.skip_scrape
        do_cleanup = not args.skip_cleanup
        do_split = not args.skip_split
    elif any_step_explicit:
        do_scrape = args.scrape
        do_cleanup = args.cleanup
        do_split = args.split
    else:
        # Interactive mode - ask for each step
        do_scrape = None
        do_cleanup = None
        do_split = None

    # Load shared configuration
    root = get_project_root()
    config_path = Path(args.config) if args.config else root / "conf" / "apikeys.yaml"
    csv_path = Path(args.csv) if args.csv else root / "data" / "house_style_list.csv"

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        download_dir = Path(config['paths']['download_base'])
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except (KeyError, TypeError) as e:
        print(f"Error: Invalid config file - missing 'paths.download_base': {e}")
        sys.exit(1)

    link_mode = "copy" if args.copy else "symlink"

    # Header
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    print(f"  Config:       {config_path.name}")
    print(f"  CSV:          {csv_path.name}")
    print(f"  Download dir: {download_dir}")
    print("=" * 60)

    # --- Step 1: Scrape ---
    if interactive:
        do_scrape = prompt_yes_no("\nStep 1: Scrape images from Google?")

    if do_scrape:
        print("\n" + "=" * 60)
        print("STEP 1: SCRAPE IMAGES")
        print("=" * 60)

        api_key = config.get('api_keys', {}).get('serpapi')
        if not api_key:
            print("Error: No SerpAPI key found in config (api_keys.serpapi)")
            print("Skipping scrape step.")
        else:
            if not csv_path.exists():
                print(f"Error: CSV not found: {csv_path}")
                print("Skipping scrape step.")
            else:
                num_images = args.num_images
                if interactive:
                    num_images = prompt_int("  Images per style?", num_images)

                # Lazy-load scraper module
                scraper = _get_scraper()
                run_scrape_impl(scraper, api_key, csv_path, download_dir, num_images)
    else:
        print("\nStep 1: Scrape - SKIPPED")

    # --- Step 2: Cleanup ---
    if interactive:
        do_cleanup = prompt_yes_no("\nStep 2: Clean up duplicates and corrupted images?")

    if do_cleanup:
        print("\n" + "=" * 60)
        print("STEP 2: CLEANUP (duplicates + corrupted)")
        print("=" * 60)

        if not download_dir.exists():
            print(f"Error: Download dir not found: {download_dir}")
            print("Skipping cleanup step.")
        else:
            threshold = args.threshold
            if interactive:
                threshold = prompt_int("  Similarity threshold (0-20)?", threshold)

            cleanup = _get_cleanup()
            run_cleanup_impl(cleanup, download_dir, threshold)
    else:
        print("\nStep 2: Cleanup - SKIPPED")

    # --- Step 3: Split ---
    if interactive:
        do_split = prompt_yes_no("\nStep 3: Split into train/validation/test?")

    if do_split:
        print("\n" + "=" * 60)
        print("STEP 3: SPLIT DATASET")
        print("=" * 60)

        if not download_dir.exists():
            print(f"Error: Download dir not found: {download_dir}")
            print("Skipping split step.")
        else:
            train_r = args.train_ratio / 100
            val_r = args.val_ratio / 100
            test_r = args.test_ratio / 100
            seed = args.seed

            if interactive:
                seed_input = input(f"  Random seed (Enter for none) [{seed or ''}]: ").strip()
                if seed_input:
                    seed = int(seed_input)

                print(f"\n  Current ratios: train={train_r:.0%} val={val_r:.0%} test={test_r:.0%}")
                if prompt_yes_no("  Change ratios?", default=False):
                    train_r = float(input("    Train %: ")) / 100
                    val_r = float(input("    Val %: ")) / 100
                    test_r = float(input("    Test %: ")) / 100

                if prompt_yes_no("  Use symlinks (saves disk space)?", default=True):
                    link_mode = "symlink"
                else:
                    link_mode = "copy"

            splitter = _get_splitter()
            run_split_impl(splitter, download_dir, train_r, val_r, test_r, link_mode, seed)
    else:
        print("\nStep 3: Split - SKIPPED")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Step implementations (use lazy-loaded modules)
# ---------------------------------------------------------------------------

def run_scrape_impl(scraper, api_key, csv_path, download_dir, num_images):
    """Run scraping using functions from the scraper module."""
    df = pd.read_csv(csv_path)
    if 'style' not in df.columns:
        print(f"Error: CSV must have a 'style' column. Found: {df.columns.tolist()}")
        return

    print(f"  Styles: {len(df)}")
    print(f"  Images per style: {num_images}")
    print(f"  Download dir: {download_dir}\n")

    total = 0
    for _, row in df.iterrows():
        style_name = str(row['style']).strip()
        style_folder = style_name.replace(" ", "_").lower()
        style_dir = download_dir / style_folder

        print(f"--- {style_name} ---")
        downloaded = scraper.download_images_for_style(
            api_key=api_key,
            style_name=style_name,
            style_dir=style_dir,
            num_images=num_images,
        )
        total += downloaded

    print(f"\nScrape complete: {total} new images downloaded")


def run_cleanup_impl(cleanup, download_dir, threshold):
    """Run cleanup using functions from the cleanup module."""
    skip_dirs = {'train', 'validation', 'test'}
    style_folders = sorted([
        d for d in download_dir.iterdir()
        if d.is_dir() and d.name not in skip_dirs
    ])

    if not style_folders:
        print("  No style folders found to clean.")
        return

    print(f"  Folders: {len(style_folders)}")
    print(f"  Threshold: {threshold}\n")

    total_dups = 0
    total_corrupt = 0
    for folder in style_folders:
        dups, corrupt = cleanup.process_folder(folder, threshold)
        total_dups += dups
        total_corrupt += corrupt

    print(f"\nCleanup complete: {total_dups} duplicates, {total_corrupt} corrupted removed")


def run_split_impl(splitter, download_dir, train_ratio, val_ratio, test_ratio, link_mode, seed):
    """Run splitting using functions from the splitter module."""
    if seed is not None:
        random.seed(seed)
        print(f"  Random seed: {seed}")

    print(f"  Ratios: train={train_ratio:.0%} val={val_ratio:.0%} test={test_ratio:.0%}")
    print(f"  Mode: {link_mode}\n")

    splitter.split_dataset(
        source_dir=download_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        link_mode=link_mode,
    )


if __name__ == "__main__":
    main()
