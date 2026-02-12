#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Architectural Style Classification.

Supports three search methods configured via conf/tuning_config.yaml:
    - Grid Search: exhaustive search over all value combinations
    - Bayesian Optimization: GP-based sequential optimization
    - Genetic Algorithm: evolutionary optimization

Usage:
    # Run with default config (conf/tuning_config.yaml)
    python tune_hyperparameters.py

    # Override model from config
    python tune_hyperparameters.py --model vanilla

    # Use custom config file
    python tune_hyperparameters.py --config path/to/tuning_config.yaml

    # Quick grid search (reduced grid)
    python tune_hyperparameters.py --quick

Requirements:
    - torch, torchvision, tqdm
    - matplotlib, seaborn (for visualizations)
    - scikit-learn (for Bayesian optimization)
"""

import argparse
import itertools
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Local imports
from utils.config import load_tuning_config, get_data_paths, load_config, get_mlflow_config
from utils.data_loaders import get_data_loaders
from model import VanillaCNN, get_pretrained_model, count_parameters
from utils.mlflow_training import log_hyperparameter_tuning
from train import train, get_device, prompt_model_selection


# ---------------------------------------------------------------------------
# Shared Functions
# ---------------------------------------------------------------------------

def create_model_with_dropout(
    model_name: str,
    num_classes: int,
    dropout_rate: float,
    freeze_features: bool = False
) -> torch.nn.Module:
    """
    Create a model with the specified dropout rate.

    Args:
        model_name: Model architecture name.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability for the classifier.
        freeze_features: Whether to freeze pretrained feature layers.

    Returns:
        The configured model.
    """
    if model_name == "vanilla":
        return VanillaCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        return get_pretrained_model(
            model_name,
            num_classes=num_classes,
            freeze_features=freeze_features,
            dropout_rate=dropout_rate
        )


def run_single_trial(
    trial_id: str,
    model_name: str,
    hyperparams: Dict[str, Any],
    num_classes: int,
    num_epochs: int,
    weight_decay: float,
    patience: int,
    checkpoint_dir: str,
    freeze_features: bool,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run a single training trial with the given hyperparameters.

    Args:
        trial_id: Identifier for this trial.
        model_name: Model architecture name.
        hyperparams: Dict with learning_rate, dropout_rate, batch_size.
        num_classes: Number of output classes.
        num_epochs: Maximum epochs per trial.
        weight_decay: L2 regularization strength.
        patience: Early stopping patience.
        checkpoint_dir: Directory to save trial checkpoints.
        freeze_features: Whether to freeze pretrained feature layers.
        device: Device to train on.

    Returns:
        Dict with trial results including metrics and hyperparameters.
    """
    lr = hyperparams["learning_rate"]
    dr = hyperparams["dropout_rate"]
    bs = hyperparams["batch_size"]

    trial_dir = str(Path(checkpoint_dir) / "trials" / trial_id)

    try:
        # Create data loaders with trial-specific batch size
        train_loader, val_loader, _, _ = get_data_loaders(batch_size=bs)

        # Create model with trial-specific dropout
        model = create_model_with_dropout(
            model_name, num_classes, dr, freeze_features
        )
        params = count_parameters(model)

        # Run training using existing train() function
        history, total_time = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            patience=patience,
            checkpoint_dir=trial_dir,
            device=device
        )

        best_val_acc = max(history["val_acc"])
        best_val_loss = min(history["val_loss"])
        best_epoch = history["val_loss"].index(best_val_loss) + 1

        return {
            "trial_id": trial_id,
            "status": "completed",
            "hyperparameters": {
                "learning_rate": lr,
                "dropout_rate": dr,
                "batch_size": bs
            },
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_train_accuracy": history["train_acc"][-1],
            "final_train_loss": history["train_loss"][-1],
            "total_epochs_run": len(history["train_loss"]),
            "training_time_seconds": total_time,
            "model_parameters": params
        }

    except RuntimeError as e:
        error_msg = str(e)
        print(f"\n  Trial {trial_id} FAILED: {error_msg[:200]}")
        return {
            "trial_id": trial_id,
            "status": "failed",
            "hyperparameters": {
                "learning_rate": lr,
                "dropout_rate": dr,
                "batch_size": bs
            },
            "error": error_msg[:500]
        }


def build_results_dict(
    model_name: str,
    search_method: str,
    all_results: List[Dict[str, Any]],
    search_space: Dict[str, Any],
    total_time: float,
    tuning_config: Dict[str, Any],
    method_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the standardized results dictionary from trial results.

    Args:
        model_name: Model architecture name.
        search_method: The search method used.
        all_results: List of all trial result dicts.
        search_space: Search space configuration from YAML.
        total_time: Total tuning wall-clock time in seconds.
        tuning_config: The tuning section of the YAML config.
        method_config: Method-specific config (bayesian or genetic section).

    Returns:
        Standardized results dict with metadata, best trial, and all trials.
    """
    completed = [r for r in all_results if r["status"] == "completed"]
    failed = [r for r in all_results if r["status"] == "failed"]
    ranked = sorted(completed, key=lambda x: x["best_val_accuracy"], reverse=True)
    ranked_ids = [r["trial_id"] for r in ranked]

    best_trial = None
    if ranked:
        best = ranked[0]
        best_trial = {
            "trial_id": best["trial_id"],
            "hyperparameters": best["hyperparameters"],
            "best_val_accuracy": best["best_val_accuracy"],
            "best_val_loss": best["best_val_loss"],
            "best_epoch": best["best_epoch"]
        }

    metadata = {
        "model_name": model_name,
        "search_method": search_method,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "total_tuning_time_seconds": round(total_time, 2),
        "total_tuning_time_minutes": round(total_time / 60, 2),
        "total_trials": len(all_results),
        "successful_trials": len(completed),
        "failed_trials": len(failed),
        "epochs_per_trial": tuning_config["epochs_per_trial"],
        "weight_decay": tuning_config["weight_decay"],
        "early_stopping_patience": tuning_config["patience"],
        "freeze_features": tuning_config["freeze_features"],
        "device": str(torch.device("cpu")),  # updated by caller
    }
    # Add method-specific metadata
    metadata.update(method_config)

    # Build search space summary for JSON
    search_space_summary = {
        "learning_rates": search_space["learning_rate"].get("values", []),
        "dropout_rates": search_space["dropout_rate"].get("values", []),
        "batch_sizes": search_space["batch_size"].get("values", []),
    }
    if search_method in ("bayesian", "genetic"):
        search_space_summary["learning_rate_range"] = [
            search_space["learning_rate"]["min"],
            search_space["learning_rate"]["max"]
        ]
        search_space_summary["dropout_rate_range"] = [
            search_space["dropout_rate"]["min"],
            search_space["dropout_rate"]["max"]
        ]
        search_space_summary["batch_size_choices"] = search_space["batch_size"]["choices"]

    return {
        "metadata": metadata,
        "search_space": search_space_summary,
        "best_trial": best_trial,
        "all_trials": all_results,
        "trials_ranked_by_val_accuracy": ranked_ids
    }


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def build_hyperparameter_grid(
    search_space: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Build all hyperparameter combinations for grid search.

    Args:
        search_space: Search space config with 'values' lists per parameter.

    Returns:
        List of dicts with trial_id and hyperparameter values.
    """
    learning_rates = search_space["learning_rate"]["values"]
    dropout_rates = search_space["dropout_rate"]["values"]
    batch_sizes = search_space["batch_size"]["values"]

    combinations = list(itertools.product(learning_rates, dropout_rates, batch_sizes))
    grid = []
    for i, (lr, dr, bs) in enumerate(combinations, start=1):
        grid.append({
            "trial_id": f"trial_{i:03d}",
            "learning_rate": lr,
            "dropout_rate": dr,
            "batch_size": int(bs)
        })
    return grid


def run_grid_search(
    model_name: str,
    search_space: Dict[str, Any],
    num_classes: int,
    tuning_config: Dict[str, Any],
    output_dir: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run exhaustive grid search over all hyperparameter combinations.
    """
    grid = build_hyperparameter_grid(search_space)
    total_trials = len(grid)
    all_results = []
    tuning_start = time.time()

    for i, trial in enumerate(grid, start=1):
        trial_id = trial["trial_id"]
        lr = trial["learning_rate"]
        dr = trial["dropout_rate"]
        bs = trial["batch_size"]

        print("\n" + "=" * 60)
        print(f"[Grid Search] Trial {i}/{total_trials}: lr={lr}, dropout={dr}, batch_size={bs}")
        print("=" * 60)

        result = run_single_trial(
            trial_id=trial_id,
            model_name=model_name,
            hyperparams=trial,
            num_classes=num_classes,
            num_epochs=tuning_config["epochs_per_trial"],
            weight_decay=tuning_config["weight_decay"],
            patience=tuning_config["patience"],
            checkpoint_dir=output_dir,
            freeze_features=tuning_config["freeze_features"],
            device=device
        )
        all_results.append(result)

        if result["status"] == "completed":
            print(f"\n  Best val accuracy: {result['best_val_accuracy']:.2f}%")
            print(f"  Training time: {result['training_time_seconds']:.1f}s")

    total_time = time.time() - tuning_start

    results = build_results_dict(
        model_name=model_name,
        search_method="grid_search",
        all_results=all_results,
        search_space=search_space,
        total_time=total_time,
        tuning_config=tuning_config,
        method_config={}
    )
    results["metadata"]["device"] = str(device)
    return results


# ---------------------------------------------------------------------------
# Bayesian Optimization
# ---------------------------------------------------------------------------

def _sample_random_point(search_space: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, Any]:
    """Sample a random hyperparameter configuration from the search space."""
    lr_cfg = search_space["learning_rate"]
    dr_cfg = search_space["dropout_rate"]
    bs_cfg = search_space["batch_size"]

    if lr_cfg.get("log_scale", False):
        lr = float(np.exp(rng.uniform(np.log(lr_cfg["min"]), np.log(lr_cfg["max"]))))
    else:
        lr = float(rng.uniform(lr_cfg["min"], lr_cfg["max"]))

    dr = float(rng.uniform(dr_cfg["min"], dr_cfg["max"]))
    bs = int(rng.choice(bs_cfg["choices"]))

    return {"learning_rate": lr, "dropout_rate": round(dr, 4), "batch_size": bs}


def _encode_point(hp: Dict[str, Any], search_space: Dict[str, Any]) -> np.ndarray:
    """Encode hyperparameters as a normalized [0, 1] vector for the GP."""
    lr_cfg = search_space["learning_rate"]
    dr_cfg = search_space["dropout_rate"]
    bs_choices = search_space["batch_size"]["choices"]

    # Learning rate (log-scale if configured)
    if lr_cfg.get("log_scale", False):
        lr_norm = (np.log(hp["learning_rate"]) - np.log(lr_cfg["min"])) / \
                  (np.log(lr_cfg["max"]) - np.log(lr_cfg["min"]))
    else:
        lr_norm = (hp["learning_rate"] - lr_cfg["min"]) / (lr_cfg["max"] - lr_cfg["min"])

    # Dropout rate (linear)
    dr_norm = (hp["dropout_rate"] - dr_cfg["min"]) / (dr_cfg["max"] - dr_cfg["min"])

    # Batch size (index in choices, normalized)
    bs_idx = bs_choices.index(hp["batch_size"]) if hp["batch_size"] in bs_choices else 0
    bs_norm = bs_idx / max(len(bs_choices) - 1, 1)

    return np.array([
        np.clip(lr_norm, 0, 1),
        np.clip(dr_norm, 0, 1),
        np.clip(bs_norm, 0, 1)
    ])


def _decode_point(x: np.ndarray, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a normalized [0, 1] vector back to hyperparameters."""
    lr_cfg = search_space["learning_rate"]
    dr_cfg = search_space["dropout_rate"]
    bs_choices = search_space["batch_size"]["choices"]

    x = np.clip(x, 0, 1)

    # Learning rate
    if lr_cfg.get("log_scale", False):
        lr = float(np.exp(x[0] * (np.log(lr_cfg["max"]) - np.log(lr_cfg["min"])) + np.log(lr_cfg["min"])))
    else:
        lr = float(x[0] * (lr_cfg["max"] - lr_cfg["min"]) + lr_cfg["min"])

    # Dropout rate
    dr = float(x[1] * (dr_cfg["max"] - dr_cfg["min"]) + dr_cfg["min"])

    # Batch size (snap to nearest choice)
    bs_idx = int(round(x[2] * (len(bs_choices) - 1)))
    bs_idx = max(0, min(bs_idx, len(bs_choices) - 1))
    bs = bs_choices[bs_idx]

    return {"learning_rate": round(lr, 6), "dropout_rate": round(dr, 4), "batch_size": int(bs)}


def _expected_improvement(X_candidates: np.ndarray, gp, y_best: float) -> np.ndarray:
    """Compute Expected Improvement acquisition function."""
    from scipy.stats import norm

    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    z = (mu - y_best) / sigma
    ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei


def run_bayesian_search(
    model_name: str,
    search_space: Dict[str, Any],
    num_classes: int,
    tuning_config: Dict[str, Any],
    bayesian_config: Dict[str, Any],
    output_dir: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run Bayesian optimization using a Gaussian Process surrogate model.

    Starts with random exploration, then uses Expected Improvement to
    select promising hyperparameter configurations.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    n_trials = bayesian_config["n_trials"]
    n_initial = bayesian_config["n_initial"]
    rng = np.random.RandomState(42)

    all_results = []
    X_observed = []
    y_observed = []
    tuning_start = time.time()

    for i in range(1, n_trials + 1):
        trial_id = f"trial_{i:03d}"

        if i <= n_initial:
            # Random exploration phase
            hp = _sample_random_point(search_space, rng)
            print("\n" + "=" * 60)
            print(f"[Bayesian] Trial {i}/{n_trials} (random exploration): "
                  f"lr={hp['learning_rate']:.6f}, dropout={hp['dropout_rate']:.4f}, "
                  f"batch_size={hp['batch_size']}")
            print("=" * 60)
        else:
            # GP-guided phase
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                n_restarts_optimizer=5,
                random_state=rng
            )
            X_arr = np.array(X_observed)
            y_arr = np.array(y_observed)
            gp.fit(X_arr, y_arr)

            # Generate candidate points and pick best EI
            n_candidates = 1000
            X_candidates = rng.uniform(0, 1, size=(n_candidates, 3))
            ei_values = _expected_improvement(X_candidates, gp, y_best=max(y_observed))
            best_candidate = X_candidates[np.argmax(ei_values)]

            hp = _decode_point(best_candidate, search_space)
            print("\n" + "=" * 60)
            print(f"[Bayesian] Trial {i}/{n_trials} (GP-guided): "
                  f"lr={hp['learning_rate']:.6f}, dropout={hp['dropout_rate']:.4f}, "
                  f"batch_size={hp['batch_size']}")
            print("=" * 60)

        result = run_single_trial(
            trial_id=trial_id,
            model_name=model_name,
            hyperparams=hp,
            num_classes=num_classes,
            num_epochs=tuning_config["epochs_per_trial"],
            weight_decay=tuning_config["weight_decay"],
            patience=tuning_config["patience"],
            checkpoint_dir=output_dir,
            freeze_features=tuning_config["freeze_features"],
            device=device
        )
        all_results.append(result)

        # Record observation for GP
        x_encoded = _encode_point(hp, search_space)
        X_observed.append(x_encoded)
        if result["status"] == "completed":
            y_observed.append(result["best_val_accuracy"])
            print(f"\n  Best val accuracy: {result['best_val_accuracy']:.2f}%")
            print(f"  Training time: {result['training_time_seconds']:.1f}s")
        else:
            y_observed.append(0.0)  # Penalize failed trials

    total_time = time.time() - tuning_start

    results = build_results_dict(
        model_name=model_name,
        search_method="bayesian",
        all_results=all_results,
        search_space=search_space,
        total_time=total_time,
        tuning_config=tuning_config,
        method_config={"n_trials": n_trials, "n_initial": n_initial}
    )
    results["metadata"]["device"] = str(device)
    return results


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

def _init_population(
    pop_size: int,
    search_space: Dict[str, Any],
    rng: np.random.RandomState
) -> List[Dict[str, Any]]:
    """Initialize a random population of hyperparameter configurations."""
    return [_sample_random_point(search_space, rng) for _ in range(pop_size)]


def _tournament_select(
    population: List[Dict[str, Any]],
    fitness: List[float],
    tournament_size: int,
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """Select an individual using tournament selection."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmax([fitness[i] for i in indices])]
    return population[best_idx].copy()


def _crossover(
    parent1: Dict[str, Any],
    parent2: Dict[str, Any],
    crossover_rate: float,
    search_space: Dict[str, Any],
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """Uniform crossover between two parents."""
    if rng.random() > crossover_rate:
        return parent1.copy()

    child = {}
    for key in ["learning_rate", "dropout_rate", "batch_size"]:
        if rng.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child


def _mutate(
    individual: Dict[str, Any],
    mutation_rate: float,
    search_space: Dict[str, Any],
    rng: np.random.RandomState
) -> Dict[str, Any]:
    """Mutate an individual's hyperparameters."""
    ind = individual.copy()

    # Mutate learning rate
    if rng.random() < mutation_rate:
        lr_cfg = search_space["learning_rate"]
        if lr_cfg.get("log_scale", False):
            log_lr = np.log(ind["learning_rate"])
            log_lr += rng.normal(0, 0.5)
            log_lr = np.clip(log_lr, np.log(lr_cfg["min"]), np.log(lr_cfg["max"]))
            ind["learning_rate"] = round(float(np.exp(log_lr)), 6)
        else:
            ind["learning_rate"] += rng.normal(0, (lr_cfg["max"] - lr_cfg["min"]) * 0.1)
            ind["learning_rate"] = round(
                float(np.clip(ind["learning_rate"], lr_cfg["min"], lr_cfg["max"])), 6
            )

    # Mutate dropout rate
    if rng.random() < mutation_rate:
        dr_cfg = search_space["dropout_rate"]
        ind["dropout_rate"] += rng.normal(0, (dr_cfg["max"] - dr_cfg["min"]) * 0.1)
        ind["dropout_rate"] = round(
            float(np.clip(ind["dropout_rate"], dr_cfg["min"], dr_cfg["max"])), 4
        )

    # Mutate batch size
    if rng.random() < mutation_rate:
        bs_choices = search_space["batch_size"]["choices"]
        ind["batch_size"] = int(rng.choice(bs_choices))

    return ind


def run_genetic_search(
    model_name: str,
    search_space: Dict[str, Any],
    num_classes: int,
    tuning_config: Dict[str, Any],
    genetic_config: Dict[str, Any],
    output_dir: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run genetic algorithm optimization.

    Evolves a population of hyperparameter configurations over multiple
    generations using tournament selection, crossover, and mutation.
    Elitism ensures the best individual always survives.
    """
    pop_size = genetic_config["population_size"]
    n_generations = genetic_config["n_generations"]
    mutation_rate = genetic_config["mutation_rate"]
    crossover_rate = genetic_config["crossover_rate"]
    tournament_size = min(3, pop_size)
    rng = np.random.RandomState(42)

    all_results = []
    trial_counter = 0
    tuning_start = time.time()

    # Initialize population
    population = _init_population(pop_size, search_space, rng)
    best_ever_fitness = -1
    best_ever_hp = None

    for gen in range(1, n_generations + 1):
        print("\n" + "#" * 60)
        print(f"GENERATION {gen}/{n_generations}")
        print("#" * 60)

        # Evaluate population
        fitness = []
        for j, individual in enumerate(population):
            trial_counter += 1
            trial_id = f"trial_{trial_counter:03d}"

            print("\n" + "=" * 60)
            print(f"[Genetic] Gen {gen}, Individual {j + 1}/{pop_size}: "
                  f"lr={individual['learning_rate']:.6f}, "
                  f"dropout={individual['dropout_rate']:.4f}, "
                  f"batch_size={individual['batch_size']}")
            print("=" * 60)

            result = run_single_trial(
                trial_id=trial_id,
                model_name=model_name,
                hyperparams=individual,
                num_classes=num_classes,
                num_epochs=tuning_config["epochs_per_trial"],
                weight_decay=tuning_config["weight_decay"],
                patience=tuning_config["patience"],
                checkpoint_dir=output_dir,
                freeze_features=tuning_config["freeze_features"],
                device=device
            )
            all_results.append(result)

            if result["status"] == "completed":
                fit = result["best_val_accuracy"]
                print(f"\n  Best val accuracy: {fit:.2f}%")
                print(f"  Training time: {result['training_time_seconds']:.1f}s")
            else:
                fit = 0.0
            fitness.append(fit)

            if fit > best_ever_fitness:
                best_ever_fitness = fit
                best_ever_hp = individual.copy()

        # Print generation summary
        gen_best = max(fitness)
        gen_avg = np.mean(fitness)
        print(f"\n  Generation {gen} summary: best={gen_best:.2f}%, avg={gen_avg:.2f}%")
        print(f"  Best ever: {best_ever_fitness:.2f}%")

        # Skip evolution after last generation
        if gen == n_generations:
            break

        # Create next generation
        new_population = []

        # Elitism: keep best individual
        elite_idx = np.argmax(fitness)
        new_population.append(population[elite_idx].copy())

        # Fill rest with offspring
        while len(new_population) < pop_size:
            parent1 = _tournament_select(population, fitness, tournament_size, rng)
            parent2 = _tournament_select(population, fitness, tournament_size, rng)
            child = _crossover(parent1, parent2, crossover_rate, search_space, rng)
            child = _mutate(child, mutation_rate, search_space, rng)
            new_population.append(child)

        population = new_population

    total_time = time.time() - tuning_start

    results = build_results_dict(
        model_name=model_name,
        search_method="genetic",
        all_results=all_results,
        search_space=search_space,
        total_time=total_time,
        tuning_config=tuning_config,
        method_config={
            "population_size": pop_size,
            "n_generations": n_generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate
        }
    )
    results["metadata"]["device"] = str(device)
    return results


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------

def plot_tuning_heatmaps(results: Dict[str, Any], output_dir: str) -> None:
    """
    Create 2D heatmaps of validation accuracy for each pair of hyperparameters.

    Generates three heatmaps, each marginalizing over the third hyperparameter.
    Most useful for grid search; for bayesian/genetic it bins results.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Skipping heatmaps.")
        return

    completed = [t for t in results["all_trials"] if t["status"] == "completed"]
    if not completed:
        print("No completed trials to plot.")
        return

    search_space = results["search_space"]
    method = results["metadata"]["search_method"]

    # For grid search, use the exact values; for others, bin into the grid values
    lrs = sorted(search_space.get("learning_rates", []))
    drs = sorted(search_space.get("dropout_rates", []))
    bss = sorted(search_space.get("batch_sizes", []))

    if not lrs or not drs or not bss:
        print("Skipping heatmaps: insufficient search space values.")
        return

    def _snap_to_nearest(val, options):
        return min(options, key=lambda x: abs(x - val))

    output_path = Path(output_dir)
    pairs = [
        ("learning_rate", "dropout_rate", lrs, drs, "Batch Size",
         "heatmap_lr_vs_dropout.png"),
        ("learning_rate", "batch_size", lrs, bss, "Dropout Rate",
         "heatmap_lr_vs_batchsize.png"),
        ("dropout_rate", "batch_size", drs, bss, "Learning Rate",
         "heatmap_dropout_vs_batchsize.png"),
    ]

    for y_name, x_name, y_vals, x_vals, z_name, filename in pairs:
        data = np.zeros((len(y_vals), len(x_vals)))
        counts = np.zeros((len(y_vals), len(x_vals)))

        for t in completed:
            hp = t["hyperparameters"]
            y_val = _snap_to_nearest(hp[y_name], y_vals)
            x_val = _snap_to_nearest(hp[x_name], x_vals)
            yi = y_vals.index(y_val)
            xi = x_vals.index(x_val)
            data[yi, xi] += t["best_val_accuracy"]
            counts[yi, xi] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.where(counts > 0, data / counts, 0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            data, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=[str(v) for v in x_vals],
            yticklabels=[str(v) for v in y_vals],
            ax=ax
        )
        ax.set_xlabel(x_name.replace("_", " ").title())
        ax.set_ylabel(y_name.replace("_", " ").title())
        ax.set_title(
            f"Mean Val Accuracy: {y_name.replace('_', ' ').title()} vs "
            f"{x_name.replace('_', ' ').title()}\n"
            f"(averaged over {z_name}) [{method}]"
        )
        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"  Saved {filename}")


def plot_top_configurations(
    results: Dict[str, Any], output_dir: str, top_n: int = 10
) -> None:
    """Create a horizontal bar chart of top N configurations by validation accuracy."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("Warning: matplotlib not installed. Skipping top configurations plot.")
        return

    completed = [t for t in results["all_trials"] if t["status"] == "completed"]
    if not completed:
        return

    ranked = sorted(completed, key=lambda x: x["best_val_accuracy"], reverse=True)
    top = ranked[:top_n]
    top.reverse()

    labels = []
    accuracies = []
    for t in top:
        hp = t["hyperparameters"]
        labels.append(f"lr={hp['learning_rate']}, do={hp['dropout_rate']}, bs={hp['batch_size']}")
        accuracies.append(t["best_val_accuracy"])

    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.6)))
    colors = cm.YlOrRd(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)), accuracies, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Best Validation Accuracy (%)")
    ax.set_title(f"Top {len(top)} Hyperparameter Configurations [{results['metadata']['search_method']}]")

    for i, acc in enumerate(accuracies):
        ax.text(acc + 0.3, i, f"{acc:.2f}%", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "top_configurations.png", dpi=150)
    plt.close()
    print("  Saved top_configurations.png")


def plot_best_training_curves(results: Dict[str, Any], output_dir: str) -> None:
    """Plot training loss and accuracy curves for the best trial."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping training curves.")
        return

    best_trial = results.get("best_trial")
    if not best_trial:
        return

    trial_dir = Path(output_dir) / "trials" / best_trial["trial_id"]
    history_file = trial_dir / "training_history.json"
    if not history_file.exists():
        print(f"  Warning: History file not found for best trial: {history_file}")
        return

    with open(history_file) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss (Best Trial)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], 'b-', label='Train Accuracy')
    ax2.plot(epochs, history["val_acc"], 'r-', label='Val Accuracy')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy (Best Trial)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    hp = best_trial["hyperparameters"]
    fig.suptitle(
        f"Best Config: lr={hp['learning_rate']}, dropout={hp['dropout_rate']}, "
        f"batch_size={hp['batch_size']}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "best_training_curves.png", dpi=150)
    plt.close()
    print("  Saved best_training_curves.png")


def plot_tuning_summary(results: Dict[str, Any], output_dir: str) -> None:
    """
    Create a comprehensive multi-panel tuning summary figure.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Skipping summary.")
        return

    completed = [t for t in results["all_trials"] if t["status"] == "completed"]
    if not completed:
        return

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    meta = results["metadata"]
    search_space = results["search_space"]

    # --- Panel 1: LR vs Dropout Heatmap (top-left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    lrs = sorted(search_space.get("learning_rates", []))
    drs = sorted(search_space.get("dropout_rates", []))

    if lrs and drs:
        def _snap(val, opts):
            return min(opts, key=lambda x: abs(x - val))

        data = np.zeros((len(lrs), len(drs)))
        counts = np.zeros((len(lrs), len(drs)))
        for t in completed:
            hp = t["hyperparameters"]
            yi = lrs.index(_snap(hp["learning_rate"], lrs))
            xi = drs.index(_snap(hp["dropout_rate"], drs))
            data[yi, xi] += t["best_val_accuracy"]
            counts[yi, xi] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.where(counts > 0, data / counts, 0)

        sns.heatmap(
            data, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=[str(v) for v in drs],
            yticklabels=[str(v) for v in lrs],
            ax=ax1
        )
        ax1.set_xlabel("Dropout Rate")
        ax1.set_ylabel("Learning Rate")
        ax1.set_title("Mean Val Accuracy: LR vs Dropout")
    else:
        ax1.text(0.5, 0.5, "Insufficient data for heatmap",
                 ha='center', va='center', transform=ax1.transAxes)

    # --- Panel 2: Top 10 Bar Chart (top-right) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ranked = sorted(completed, key=lambda x: x["best_val_accuracy"], reverse=True)
    top = ranked[:10]
    top.reverse()

    labels = []
    accuracies = []
    for t in top:
        hp = t["hyperparameters"]
        labels.append(f"lr={hp['learning_rate']}, do={hp['dropout_rate']}, bs={hp['batch_size']}")
        accuracies.append(t["best_val_accuracy"])

    colors = cm.YlOrRd(np.linspace(0.3, 0.9, len(top)))
    ax2.barh(range(len(top)), accuracies, color=colors)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Best Val Accuracy (%)")
    ax2.set_title(f"Top {len(top)} Configurations")

    # --- Panel 3: Best Trial Training Curves (bottom-left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    best_trial = results.get("best_trial")
    if best_trial:
        trial_dir = Path(output_dir) / "trials" / best_trial["trial_id"]
        history_file = trial_dir / "training_history.json"
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
            epochs = range(1, len(history["train_loss"]) + 1)
            ax3.plot(epochs, history["train_acc"], 'b-', label='Train Acc')
            ax3.plot(epochs, history["val_acc"], 'r-', label='Val Acc')
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Accuracy (%)")
            ax3.set_title("Best Trial Training Curves")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "History not found", ha='center', va='center',
                     transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, "No completed trials", ha='center', va='center',
                 transform=ax3.transAxes)

    # --- Panel 4: Text Summary (bottom-right) ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary_lines = [
        f"Model: {meta['model_name']}",
        f"Search Method: {meta['search_method']}",
        f"Total Trials: {meta['total_trials']}",
        f"Successful: {meta['successful_trials']}  |  Failed: {meta['failed_trials']}",
        f"Epochs per Trial: {meta['epochs_per_trial']}",
        f"Total Tuning Time: {meta['total_tuning_time_minutes']:.1f} min",
    ]

    if meta["search_method"] == "bayesian":
        summary_lines.append(f"Initial Random Trials: {meta.get('n_initial', 'N/A')}")
    elif meta["search_method"] == "genetic":
        summary_lines.append(f"Population: {meta.get('population_size', 'N/A')}")
        summary_lines.append(f"Generations: {meta.get('n_generations', 'N/A')}")

    if best_trial:
        hp = best_trial["hyperparameters"]
        summary_lines += [
            "",
            "Best Configuration:",
            f"  Learning Rate: {hp['learning_rate']}",
            f"  Dropout Rate: {hp['dropout_rate']}",
            f"  Batch Size: {hp['batch_size']}",
            f"  Val Accuracy: {best_trial['best_val_accuracy']:.2f}%",
            f"  Val Loss: {best_trial['best_val_loss']:.4f}",
            f"  Best Epoch: {best_trial['best_epoch']}",
        ]

    ax4.text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=ax4.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    ax4.set_title("Tuning Summary")

    fig.suptitle(
        f"Hyperparameter Tuning Results: {meta['model_name']} [{meta['search_method']}]",
        fontsize=16, fontweight='bold'
    )
    plt.savefig(Path(output_dir) / "tuning_summary.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved tuning_summary.png")


# ---------------------------------------------------------------------------
# Diagnostics Summary
# ---------------------------------------------------------------------------

def write_diagnostics_summary(results: Dict[str, Any], output_dir: str) -> None:
    """
    Write a diagnostics_summary.txt with tuning results and runtime.

    Args:
        results: The standardized results dict from a tuning run.
        output_dir: Directory to write the summary file.
    """
    meta = results["metadata"]
    best = results.get("best_trial")
    search_space = results["search_space"]
    output_path = Path(output_dir) / "diagnostics_summary.txt"

    lines = []
    lines.append("=" * 60)
    lines.append("HYPERPARAMETER TUNING DIAGNOSTICS SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Timestamp:              {meta['timestamp']}")
    lines.append(f"Model:                  {meta['model_name']}")
    lines.append(f"Search Method:          {meta['search_method']}")
    lines.append(f"Device:                 {meta['device']}")
    lines.append(f"Freeze Features:        {meta['freeze_features']}")
    lines.append(f"Epochs per Trial:       {meta['epochs_per_trial']}")
    lines.append(f"Early Stop Patience:    {meta['early_stopping_patience']}")
    lines.append(f"Weight Decay:           {meta['weight_decay']}")
    lines.append("")

    # Search method specific info
    lines.append("-" * 60)
    lines.append("SEARCH CONFIGURATION")
    lines.append("-" * 60)
    if meta["search_method"] == "grid_search":
        lines.append(f"Learning Rates:         {search_space.get('learning_rates', [])}")
        lines.append(f"Dropout Rates:          {search_space.get('dropout_rates', [])}")
        lines.append(f"Batch Sizes:            {search_space.get('batch_sizes', [])}")
    elif meta["search_method"] == "bayesian":
        lines.append(f"Total Trials:           {meta.get('n_trials', 'N/A')}")
        lines.append(f"Initial Random Trials:  {meta.get('n_initial', 'N/A')}")
        lr_range = search_space.get("learning_rate_range", [])
        dr_range = search_space.get("dropout_rate_range", [])
        lines.append(f"LR Range:               {lr_range}")
        lines.append(f"Dropout Range:          {dr_range}")
        lines.append(f"Batch Size Choices:     {search_space.get('batch_size_choices', [])}")
    elif meta["search_method"] == "genetic":
        lines.append(f"Population Size:        {meta.get('population_size', 'N/A')}")
        lines.append(f"Generations:            {meta.get('n_generations', 'N/A')}")
        lines.append(f"Mutation Rate:          {meta.get('mutation_rate', 'N/A')}")
        lines.append(f"Crossover Rate:         {meta.get('crossover_rate', 'N/A')}")
        lr_range = search_space.get("learning_rate_range", [])
        dr_range = search_space.get("dropout_rate_range", [])
        lines.append(f"LR Range:               {lr_range}")
        lines.append(f"Dropout Range:          {dr_range}")
        lines.append(f"Batch Size Choices:     {search_space.get('batch_size_choices', [])}")
    lines.append("")

    # Trial results
    lines.append("-" * 60)
    lines.append("TRIAL RESULTS")
    lines.append("-" * 60)
    lines.append(f"Total Trials:           {meta['total_trials']}")
    lines.append(f"Successful Trials:      {meta['successful_trials']}")
    lines.append(f"Failed Trials:          {meta['failed_trials']}")
    lines.append("")

    # Best configuration
    if best:
        hp = best["hyperparameters"]
        lines.append("-" * 60)
        lines.append("BEST CONFIGURATION")
        lines.append("-" * 60)
        lines.append(f"Trial ID:               {best['trial_id']}")
        lines.append(f"Learning Rate:          {hp['learning_rate']}")
        lines.append(f"Dropout Rate:           {hp['dropout_rate']}")
        lines.append(f"Batch Size:             {hp['batch_size']}")
        lines.append(f"Best Val Accuracy:      {best['best_val_accuracy']:.2f}%")
        lines.append(f"Best Val Loss:          {best['best_val_loss']:.4f}")
        lines.append(f"Best Epoch:             {best['best_epoch']}")
        lines.append("")

    # Top 5 configurations
    completed = [r for r in results["all_trials"] if r["status"] == "completed"]
    ranked = sorted(completed, key=lambda x: x["best_val_accuracy"], reverse=True)
    if len(ranked) > 1:
        lines.append("-" * 60)
        lines.append("TOP 5 CONFIGURATIONS")
        lines.append("-" * 60)
        for i, trial in enumerate(ranked[:5], start=1):
            hp = trial["hyperparameters"]
            lines.append(
                f"  {i}. lr={hp['learning_rate']}, dropout={hp['dropout_rate']}, "
                f"bs={hp['batch_size']} -> {trial['best_val_accuracy']:.2f}% "
                f"(epoch {trial['best_epoch']})"
            )
        lines.append("")

    # Runtime
    total_sec = meta["total_tuning_time_seconds"]
    total_min = meta["total_tuning_time_minutes"]
    total_hrs = round(total_sec / 3600, 2)
    lines.append("-" * 60)
    lines.append("RUNTIME")
    lines.append("-" * 60)
    lines.append(f"Total Tuning Time:      {total_sec:.2f} seconds")
    lines.append(f"                        {total_min:.2f} minutes")
    lines.append(f"                        {total_hrs:.2f} hours")
    if completed:
        avg_time = sum(t["training_time_seconds"] for t in completed) / len(completed)
        lines.append(f"Avg Time per Trial:     {avg_time:.1f} seconds")
    lines.append("")
    lines.append("=" * 60)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Saved diagnostics_summary.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prompt_mlflow_tracking() -> bool:
    """
    Prompt user whether to enable MLflow tracking.

    Returns:
        bool: True if MLflow should be enabled, False otherwise.
    """
    print("\n" + "=" * 70)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 70)
    print("MLflow tracks all tuning trials, metrics, and configurations.")
    print("You can view results later with: mlflow ui")
    print("=" * 70)
    print("1. Yes - Enable MLflow tracking (recommended)")
    print("2. No - Skip MLflow tracking")
    print("=" * 70)

    while True:
        try:
            choice = input("\nEnable MLflow tracking? (1-2): ").strip()
            if choice == "1":
                print("✓ MLflow tracking enabled")
                return True
            elif choice == "2":
                print("✗ MLflow tracking disabled")
                return False
            else:
                print("Please enter 1 or 2.")
        except (ValueError, KeyboardInterrupt):
            print("\nDefaulting to no MLflow tracking")
            return False


def main():
    """Main entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for architectural style classification. "
                    "Configure via conf/tuning_config.yaml."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to tuning config YAML (default: conf/tuning_config.yaml)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced grid for grid_search: lr=[0.001,0.01], do=[0.3,0.5], bs=[16,32]"
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Enable MLflow experiment tracking"
    )
    parser.add_argument(
        "--no_mlflow", action="store_true",
        help="Disable MLflow experiment tracking"
    )
    parser.add_argument(
        "--mlflow_tracking_uri", type=str, default=None,
        help="MLflow tracking server URI (default: from config or local ./mlruns)"
    )
    parser.add_argument(
        "--mlflow_experiment", type=str, default=None,
        help="MLflow experiment name (default: from config)"
    )

    args = parser.parse_args()

    # Load tuning config
    config = load_tuning_config(args.config)
    tuning = config["tuning"]
    search_space = config["search_space"]
    bayesian_cfg = config["bayesian"]
    genetic_cfg = config["genetic"]

    # Load MLflow configuration from training config
    training_config = load_config()
    mlflow_config = get_mlflow_config(training_config)

    # Determine if MLflow should be enabled (priority: CLI > interactive > config)
    mlflow_enabled = False
    if args.no_mlflow:
        # Explicitly disabled via CLI
        mlflow_enabled = False
    elif args.mlflow:
        # Explicitly enabled via CLI
        mlflow_enabled = True
    elif mlflow_config.get("enabled", True):
        # Config says enabled, prompt user
        mlflow_enabled = prompt_mlflow_tracking()
    else:
        # Config says disabled
        mlflow_enabled = False

    # Use config defaults if not provided via CLI
    mlflow_tracking_uri = args.mlflow_tracking_uri or mlflow_config.get("tracking_uri")
    mlflow_experiment = args.mlflow_experiment or mlflow_config.get("experiment_name", "architectural-style-tuning")

    # Interactive model selection
    model_name = prompt_model_selection()
    no_plots = args.no_plots or tuning["no_plots"]
    search_method = tuning["search_method"]

    # Quick mode overrides search space for grid search
    if args.quick:
        search_method = "grid_search"
        search_space["learning_rate"]["values"] = [0.001, 0.01]
        search_space["dropout_rate"]["values"] = [0.3, 0.5]
        search_space["batch_size"]["values"] = [16, 32]

    # Output directory: checkpoints_tuning/<model>/<search_method>/
    paths = get_data_paths()
    output_dir = paths["checkpoints"].parent / tuning["output_dir"] / model_name / search_method
    trials_dir = output_dir / "trials"

    device = get_device()

    # Get num_classes from dataset
    _, _, _, class_names = get_data_loaders(batch_size=32)
    num_classes = len(class_names)

    # Print configuration
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"\nSearch method: {search_method}")
    print(f"Model: {model_name}")
    print(f"Freeze features: {tuning['freeze_features']}")
    print(f"Epochs per trial: {tuning['epochs_per_trial']}")
    print(f"Early stopping patience: {tuning['patience']}")
    print(f"Weight decay: {tuning['weight_decay']}")
    print(f"Device: {device}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"MLflow tracking: {'enabled' if mlflow_enabled else 'disabled'}")

    if search_method == "grid_search":
        print(f"\nGrid Search Space:")
        print(f"  Learning rates: {search_space['learning_rate']['values']}")
        print(f"  Dropout rates: {search_space['dropout_rate']['values']}")
        print(f"  Batch sizes: {search_space['batch_size']['values']}")
        total = (len(search_space['learning_rate']['values']) *
                 len(search_space['dropout_rate']['values']) *
                 len(search_space['batch_size']['values']))
        print(f"  Total trials: {total}")
    elif search_method == "bayesian":
        print(f"\nBayesian Optimization:")
        print(f"  Total trials: {bayesian_cfg['n_trials']}")
        print(f"  Initial random: {bayesian_cfg['n_initial']}")
        print(f"  LR range: [{search_space['learning_rate']['min']}, {search_space['learning_rate']['max']}]")
        print(f"  Dropout range: [{search_space['dropout_rate']['min']}, {search_space['dropout_rate']['max']}]")
        print(f"  Batch size choices: {search_space['batch_size']['choices']}")
    elif search_method == "genetic":
        print(f"\nGenetic Algorithm:")
        print(f"  Population size: {genetic_cfg['population_size']}")
        print(f"  Generations: {genetic_cfg['n_generations']}")
        print(f"  Mutation rate: {genetic_cfg['mutation_rate']}")
        print(f"  Crossover rate: {genetic_cfg['crossover_rate']}")
        total = genetic_cfg['population_size'] * genetic_cfg['n_generations']
        print(f"  Total trials: {total}")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)

    # Clear previous tuning results for this model/search_method (overwrite)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleared previous results in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch to search method
    if search_method == "grid_search":
        results = run_grid_search(
            model_name=model_name,
            search_space=search_space,
            num_classes=num_classes,
            tuning_config=tuning,
            output_dir=str(output_dir),
            device=device
        )
    elif search_method == "bayesian":
        results = run_bayesian_search(
            model_name=model_name,
            search_space=search_space,
            num_classes=num_classes,
            tuning_config=tuning,
            bayesian_config=bayesian_cfg,
            output_dir=str(output_dir),
            device=device
        )
    elif search_method == "genetic":
        results = run_genetic_search(
            model_name=model_name,
            search_space=search_space,
            num_classes=num_classes,
            tuning_config=tuning,
            genetic_config=genetic_cfg,
            output_dir=str(output_dir),
            device=device
        )
    else:
        raise ValueError(
            f"Unknown search method: {search_method}. "
            "Supported: grid_search, bayesian, genetic"
        )

    # Save results JSON
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "tuning_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Write diagnostics summary
    write_diagnostics_summary(results, str(output_dir))

    # Generate visualizations
    if not no_plots:
        print("\nGenerating visualizations...")
        plot_tuning_heatmaps(results, str(output_dir))
        plot_top_configurations(results, str(output_dir))
        plot_best_training_curves(results, str(output_dir))
        plot_tuning_summary(results, str(output_dir))

    # Print final summary
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    meta = results["metadata"]
    print(f"Search method: {meta['search_method']}")
    print(f"Total trials: {meta['total_trials']}")
    print(f"Successful: {meta['successful_trials']}")
    print(f"Failed: {meta['failed_trials']}")
    print(f"Total time: {meta['total_tuning_time_minutes']:.1f} minutes")

    best = results.get("best_trial")
    if best:
        hp = best["hyperparameters"]
        print(f"\nBest Configuration:")
        print(f"  Learning rate: {hp['learning_rate']}")
        print(f"  Dropout rate: {hp['dropout_rate']}")
        print(f"  Batch size: {hp['batch_size']}")
        print(f"  Val accuracy: {best['best_val_accuracy']:.2f}%")
        print(f"  Val loss: {best['best_val_loss']:.4f}")

    print(f"\nResults saved to: {output_dir}/")

    # Log to MLflow if enabled
    if mlflow_enabled:
        print("\n" + "=" * 60)
        print("Logging to MLflow...")
        print("=" * 60)

        log_hyperparameter_tuning(
            model_name=model_name,
            search_method=search_method,
            search_space=search_space,
            results=results.get("trials", []),
            best_config=best["hyperparameters"] if best else {},
            output_dir=Path(output_dir),
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment
        )


if __name__ == "__main__":
    main()
