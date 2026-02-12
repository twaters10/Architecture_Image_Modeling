#!/usr/bin/env python3
"""
Configuration Loader for Architectural Style Classification.

This module provides utilities for loading configuration from YAML files,
ensuring consistent settings across all training pipeline scripts.

Usage:
    from config import load_config, get_project_root

    config = load_config()
    data_dir = get_project_root() / config['paths']['train']

Requirements:
    - PyYAML
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_project_root() -> Path:
    """
    Get the project root directory.

    The project root is determined by looking for the 'conf' directory,
    starting from this file's location and moving up.

    Returns:
        Path: Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the project root cannot be determined.
    """
    current = Path(__file__).parent

    # Move up to find the project root (contains 'conf' directory)
    for _ in range(5):  # Limit search depth
        if (current / "conf").exists():
            return current
        current = current.parent

    # Fallback: assume parent of house_classification is root
    return Path(__file__).parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file. If None, loads from
            the default location (conf/training_config.yaml in project root).

    Returns:
        dict: Configuration dictionary with all settings.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    if config_path is None:
        config_path = get_project_root() / "conf" / "training_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_data_paths(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """
    Get resolved data paths from configuration.

    Converts relative paths in config to absolute paths based on project root.

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        dict: Dictionary with absolute Path objects for each data location.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    paths = config.get("paths", {})

    return {
        "raw_data": root / paths.get("raw_data", "architectural_style_images"),
        "train": root / paths.get("train", "architectural_style_images/train"),
        "validation": root / paths.get("validation", "architectural_style_images/validation"),
        "test": root / paths.get("test", "architectural_style_images/test"),
        "checkpoints": root / paths.get("checkpoints", "checkpoints"),
        "results": root / paths.get("results", "evaluation_results"),
    }


def get_data_loader_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get data loader configuration.

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        dict: Data loader settings (batch_size, num_workers, etc.)
    """
    if config is None:
        config = load_config()

    defaults = {
        "batch_size": 32,
        "num_workers": 4,
        "pin_memory": True,
        "image_size": 224,
    }

    loader_config = config.get("data_loader", {})
    return {**defaults, **loader_config}


def get_augmentation_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get image augmentation configuration.

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        dict: Augmentation settings.
    """
    if config is None:
        config = load_config()

    defaults = {
        "random_crop_scale": [0.8, 1.0],
        "horizontal_flip_prob": 0.5,
        "rotation_degrees": 15,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
    }

    aug_config = config.get("augmentation", {})
    return {**defaults, **aug_config}


def get_training_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get training configuration.

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        dict: Training settings (epochs, learning_rate, etc.)
    """
    if config is None:
        config = load_config()

    defaults = {
        "epochs": 50,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "early_stopping_patience": 10,
        "lr_scheduler": {"factor": 0.5, "patience": 5},
    }

    train_config = config.get("training", {})
    return {**defaults, **train_config}


def get_normalization_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get image normalization configuration (ImageNet statistics by default).

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        dict: Normalization mean and std values.
    """
    if config is None:
        config = load_config()

    defaults = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    norm_config = config.get("normalization", {})
    return {**defaults, **norm_config}


def get_visualization_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get visualization configuration.

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        dict: Visualization settings (samples_per_class, etc.)
    """
    if config is None:
        config = load_config()

    defaults = {
        "samples_per_class": 12,
    }

    viz_config = config.get("visualization", {})
    return {**defaults, **viz_config}


def get_model_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get model configuration.

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        dict: Model settings (default model, dropout_rate, freeze_features, etc.)
    """
    if config is None:
        config = load_config()

    defaults = {
        "default": "vanilla",
        "dropout_rate": 0.5,
        "freeze_features": False,
    }

    model_config = config.get("model", {})
    return {**defaults, **model_config}


def get_mlflow_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get MLflow configuration.

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        dict: MLflow settings (enabled, tracking_uri, experiment_name).
    """
    if config is None:
        config = load_config()

    defaults = {
        "enabled": True,
        "tracking_uri": None,
        "experiment_name": "architectural-style-training"
    }

    mlflow_config = config.get("mlflow", {})
    return {**defaults, **mlflow_config}


def load_apikeys_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load API keys and paths configuration from apikeys.yaml.

    Args:
        config_path: Path to the YAML config file. If None, loads from
            the default location (conf/apikeys.yaml in project root).

    Returns:
        dict: Configuration dictionary with api_keys and paths.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    if config_path is None:
        config_path = get_project_root() / "conf" / "apikeys.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"API keys config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_tuning_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load tuning configuration from YAML file.

    Args:
        config_path: Path to the tuning YAML config file. If None, loads from
            the default location (conf/tuning_config.yaml in project root).

    Returns:
        dict: Tuning configuration with all settings and defaults applied.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    if config_path is None:
        config_path = get_project_root() / "conf" / "tuning_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Tuning config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply defaults for tuning section
    tuning_defaults = {
        "search_method": "grid_search",
        "model": "resnet18",
        "epochs_per_trial": 15,
        "patience": 7,
        "weight_decay": 0.0001,
        "freeze_features": False,
        "output_dir": "checkpoints_tuning",
        "no_plots": False,
    }
    config["tuning"] = {**tuning_defaults, **config.get("tuning", {})}

    # Apply defaults for search space
    search_defaults = {
        "learning_rate": {
            "values": [0.0001, 0.001, 0.01],
            "min": 0.0001, "max": 0.01, "log_scale": True,
        },
        "dropout_rate": {
            "values": [0.3, 0.5, 0.7],
            "min": 0.1, "max": 0.8, "log_scale": False,
        },
        "batch_size": {
            "values": [16, 32, 64],
            "choices": [8, 16, 32, 64, 128],
        },
    }
    search_space = config.get("search_space", {})
    for param, defaults in search_defaults.items():
        search_space[param] = {**defaults, **search_space.get(param, {})}
    config["search_space"] = search_space

    # Apply defaults for bayesian settings
    bayesian_defaults = {"n_trials": 20, "n_initial": 5}
    config["bayesian"] = {**bayesian_defaults, **config.get("bayesian", {})}

    # Apply defaults for genetic settings
    genetic_defaults = {
        "population_size": 10,
        "n_generations": 5,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
    }
    config["genetic"] = {**genetic_defaults, **config.get("genetic", {})}

    return config


if __name__ == "__main__":
    # Test configuration loading
    print("=" * 50)
    print("Configuration Loader Test")
    print("=" * 50)

    print(f"\nProject root: {get_project_root()}")

    config = load_config()
    print(f"\nLoaded configuration sections: {list(config.keys())}")

    paths = get_data_paths()
    print("\nData paths:")
    for name, path in paths.items():
        exists = "exists" if path.exists() else "NOT FOUND"
        print(f"  {name}: {path} [{exists}]")

    print("\nData loader config:")
    for key, value in get_data_loader_config().items():
        print(f"  {key}: {value}")

    print("\nTraining config:")
    for key, value in get_training_config().items():
        print(f"  {key}: {value}")

    print("\nConfiguration loaded successfully!")