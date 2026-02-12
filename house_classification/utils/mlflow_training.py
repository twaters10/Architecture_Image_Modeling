#!/usr/bin/env python3
"""
MLflow Integration for Training Pipeline.

Provides functions to log training runs, metrics, and artifacts to MLflow.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from .system_metrics import get_system_info
    SYSTEM_METRICS_AVAILABLE = True
except ImportError:
    SYSTEM_METRICS_AVAILABLE = False


def prompt_experiment_selection(default_name: str) -> str:
    """
    Prompt user to select an existing MLflow experiment or create a new one.

    Args:
        default_name: Default experiment name if user picks the default option.

    Returns:
        Selected or created experiment name.
    """
    if not MLFLOW_AVAILABLE:
        return default_name

    print("\n" + "=" * 70)
    print("MLFLOW EXPERIMENT SELECTION")
    print("=" * 70)

    # List existing experiments
    try:
        experiments = mlflow.search_experiments(order_by=["last_update_time DESC"])
        # Filter out the Default experiment if empty
        named_experiments = [
            e for e in experiments
            if e.name != "Default"
        ]
    except Exception:
        named_experiments = []

    print(f"\n  Default for this task: {default_name}")
    print(f"\n  1. Use default ({default_name})")
    print(f"  2. Create a new experiment")

    if named_experiments:
        print(f"  3. Use an existing experiment")
        max_choice = 3
    else:
        max_choice = 2

    print("=" * 70)

    while True:
        try:
            choice = input(f"\nSelect option (1-{max_choice}): ").strip()

            if choice == "1":
                print(f"  Using experiment: {default_name}")
                return default_name

            elif choice == "2":
                name = input("  Enter new experiment name: ").strip()
                if name:
                    print(f"  Using experiment: {name}")
                    return name
                else:
                    print("  Name cannot be empty. Try again.")

            elif choice == "3" and named_experiments:
                print("\n  Existing experiments:")
                for i, exp in enumerate(named_experiments, start=1):
                    print(f"    {i}. {exp.name}")

                while True:
                    try:
                        exp_choice = input(f"\n  Select experiment (1-{len(named_experiments)}): ").strip()
                        exp_idx = int(exp_choice) - 1
                        if 0 <= exp_idx < len(named_experiments):
                            selected = named_experiments[exp_idx].name
                            print(f"  Using experiment: {selected}")
                            return selected
                        else:
                            print(f"  Please enter a number between 1 and {len(named_experiments)}.")
                    except ValueError:
                        print(f"  Please enter a valid number.")
                    except KeyboardInterrupt:
                        print(f"\n  Using default: {default_name}")
                        return default_name
            else:
                print(f"  Please enter a number between 1 and {max_choice}.")

        except (ValueError, KeyboardInterrupt):
            print(f"\n  Using default: {default_name}")
            return default_name


def log_training_run(
    model_name: str,
    model: Any,
    history: Dict,
    config: Dict,
    checkpoint_dir: Path,
    class_names: List[str],
    total_time: float,
):
    """
    Log a complete training run to the currently active MLflow run.

    This function expects an active MLflow run (created by the caller) so that
    MLflow's built-in system metrics logging can capture metrics during training.
    The caller should:
    1. Call mlflow.enable_system_metrics_logging() before starting the run
    2. Start the run with mlflow.start_run() before training begins
    3. Run training inside the run context
    4. Call this function to log parameters, metrics, and artifacts

    Args:
        model_name: Name of the model architecture.
        model: Trained PyTorch model.
        history: Training history dict with losses and accuracies.
        config: Training configuration dict.
        checkpoint_dir: Path to checkpoint directory.
        class_names: List of class names.
        total_time: Total training time in seconds.
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Skipping logging.")
        return

    # Log parameters
    params = {
        "model_architecture": model_name,
        "epochs": len(history.get("train_loss", [])),
        "num_classes": len(class_names),
        "batch_size": config.get("batch_size", "unknown"),
        "learning_rate": config.get("learning_rate", "unknown"),
        "weight_decay": config.get("weight_decay", "unknown"),
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau"
    }
    mlflow.log_params(params)

    # Log system information as parameters
    if SYSTEM_METRICS_AVAILABLE:
        try:
            system_info = get_system_info()
            for key, value in system_info.items():
                mlflow.log_param(f"system_{key}", value)
        except Exception as e:
            print(f"Warning: Could not log system info: {e}")

    # Log tags
    mlflow.set_tags({
        "model_type": model_name,
        "task": "image_classification",
        "dataset": "architectural_styles",
        "classes": ",".join(class_names)
    })

    # Log model metrics per epoch (grouped under model_metrics/ in MLflow UI)
    train_losses = history.get("train_loss", [])
    train_accs = history.get("train_acc", [])
    val_losses = history.get("val_loss", [])
    val_accs = history.get("val_acc", [])
    learning_rates = history.get("learning_rates", [])

    for epoch in range(len(train_losses)):
        metrics = {
            "model_metrics/train_loss": train_losses[epoch],
            "model_metrics/train_accuracy": train_accs[epoch],
        }
        if epoch < len(val_losses):
            metrics["model_metrics/val_loss"] = val_losses[epoch]
            metrics["model_metrics/val_accuracy"] = val_accs[epoch]
        if epoch < len(learning_rates):
            metrics["model_metrics/learning_rate"] = learning_rates[epoch]

        mlflow.log_metrics(metrics, step=epoch)

    # Log final model metrics
    final_metrics = {
        "model_metrics/final_train_loss": train_losses[-1] if train_losses else 0,
        "model_metrics/final_train_accuracy": train_accs[-1] if train_accs else 0,
        "model_metrics/best_val_loss": min(val_losses) if val_losses else 0,
        "model_metrics/best_val_accuracy": max(val_accs) if val_accs else 0,
        "model_metrics/training_time_seconds": total_time,
        "model_metrics/training_time_minutes": total_time / 60,
        "model_metrics/total_epochs": len(train_losses)
    }
    mlflow.log_metrics(final_metrics)

    # Log model
    try:
        mlflow.pytorch.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")

    # Log artifacts
    if checkpoint_dir.exists():
        # Log best model checkpoint
        best_model = checkpoint_dir / "best_model.pth"
        if best_model.exists():
            mlflow.log_artifact(str(best_model), artifact_path="checkpoints")

        # Log training history
        history_file = checkpoint_dir / "training_history.json"
        if history_file.exists():
            mlflow.log_artifact(str(history_file), artifact_path="metrics")

        # Log training diagnostics
        diagnostics_file = checkpoint_dir / "training_diagnostics.txt"
        if diagnostics_file.exists():
            mlflow.log_artifact(str(diagnostics_file), artifact_path="diagnostics")

        # Log sample batch images
        image_batches_dir = checkpoint_dir / "image_batches"
        if image_batches_dir.exists():
            for img_file in image_batches_dir.glob("*.png"):
                mlflow.log_artifact(str(img_file), artifact_path="sample_batches")

    # Create and log training curves
    fig = create_training_curves(history)
    mlflow.log_figure(fig, "training_curves.png")
    plt.close(fig)

    print(f"\n  Run ID: {mlflow.active_run().info.run_id}")


def create_training_curves(history: Dict) -> plt.Figure:
    """
    Create training curves visualization.

    Args:
        history: Training history dict.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history["val_loss"], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history["val_acc"], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def log_hyperparameter_tuning(
    model_name: str,
    search_method: str,
    search_space: Dict,
    results: List[Dict],
    best_config: Dict,
    output_dir: Path,
):
    """
    Log hyperparameter tuning results to the currently active MLflow run.

    This function expects an active MLflow run (created by the caller) so that
    MLflow's built-in system metrics logging can capture metrics during tuning.

    Args:
        model_name: Name of the model architecture.
        search_method: Search method (grid_search, bayesian, genetic).
        search_space: Search space configuration.
        results: List of trial results.
        best_config: Best configuration found.
        output_dir: Path to output directory.
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Skipping logging.")
        return

    # Log parent run parameters
    parent_params = {
        "model_architecture": model_name,
        "search_method": search_method,
        "n_trials": len(results),
        "search_space": str(search_space)
    }
    mlflow.log_params(parent_params)

    # Log system information as parameters
    if SYSTEM_METRICS_AVAILABLE:
        try:
            system_info = get_system_info()
            for key, value in system_info.items():
                mlflow.log_param(f"system_{key}", value)
        except Exception as e:
            print(f"Warning: Could not log system info: {e}")

    # Log best configuration
    for key, value in best_config.items():
        mlflow.log_param(f"best_{key}", value)

    # Log summary model metrics (grouped under model_metrics/ in MLflow UI)
    val_accs = [r.get("val_acc", 0) for r in results]
    if val_accs:
        mlflow.log_metrics({
            "model_metrics/best_val_accuracy": max(val_accs),
            "model_metrics/mean_val_accuracy": float(np.mean(val_accs)),
            "model_metrics/std_val_accuracy": float(np.std(val_accs)),
            "model_metrics/total_trials": len(results)
        })

    # Log each trial as a nested run
    for idx, trial in enumerate(results):
        with mlflow.start_run(run_name=f"trial_{idx+1:03d}", nested=True):
            # Log trial parameters
            trial_params = {
                "learning_rate": trial.get("learning_rate", 0),
                "dropout_rate": trial.get("dropout_rate", 0),
                "batch_size": trial.get("batch_size", 0),
                "trial_number": idx + 1
            }
            mlflow.log_params(trial_params)

            # Log trial metrics
            trial_metrics = {
                "model_metrics/val_accuracy": trial.get("val_acc", 0),
                "model_metrics/val_loss": trial.get("val_loss", 0),
                "model_metrics/best_epoch": trial.get("best_epoch", 0)
            }
            mlflow.log_metrics(trial_metrics)

    # Log artifacts from output directory
    if output_dir.exists():
        results_file = output_dir / "tuning_results.json"
        if results_file.exists():
            mlflow.log_artifact(str(results_file))

        diagnostics_file = output_dir / "diagnostics_summary.txt"
        if diagnostics_file.exists():
            mlflow.log_artifact(str(diagnostics_file))

        for viz_file in output_dir.glob("*.png"):
            mlflow.log_artifact(str(viz_file), artifact_path="visualizations")

    print(f"\n  Run ID: {mlflow.active_run().info.run_id}")


def log_evaluation_run(
    model_name: str,
    checkpoint_path: Path,
    results: Dict,
    output_dir: Path,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "architectural-style-evaluation"
):
    """
    Log evaluation results to MLflow.

    Args:
        model_name: Name of the model architecture.
        checkpoint_path: Path to model checkpoint.
        results: Evaluation results dict.
        output_dir: Path to output directory.
        tracking_uri: MLflow tracking server URI.
        experiment_name: Name of MLflow experiment.
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Skipping logging.")
        return

    # Set tracking URI and experiment
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_evaluation"):
        # Log parameters
        params = {
            "model_architecture": model_name,
            "checkpoint": str(checkpoint_path),
            "num_classes": len(results.get("class_names", []))
        }
        mlflow.log_params(params)

        # Log overall metrics
        metrics = {
            "test_accuracy": results.get("test_accuracy", 0),
            "test_loss": results.get("test_loss", 0)
        }
        mlflow.log_metrics(metrics)

        # Log per-class metrics
        per_class = results.get("per_class_metrics", {})
        for class_name, class_metrics in per_class.items():
            mlflow.log_metrics({
                f"{class_name}_precision": class_metrics.get("precision", 0),
                f"{class_name}_recall": class_metrics.get("recall", 0),
                f"{class_name}_f1_score": class_metrics.get("f1_score", 0)
            })

        # Log artifacts
        if output_dir.exists():
            # Log evaluation results
            eval_results_file = output_dir / "evaluation_results.json"
            if eval_results_file.exists():
                mlflow.log_artifact(str(eval_results_file))

            # Log summary
            summary_file = output_dir / "evaluation_summary.txt"
            if summary_file.exists():
                mlflow.log_artifact(str(summary_file))

            # Log visualizations
            for viz_file in output_dir.glob("*.png"):
                mlflow.log_artifact(str(viz_file), artifact_path="visualizations")

        print(f"\n✓ Evaluation logged to MLflow")
        print(f"  Experiment: {experiment_name}")
        print(f"  Test Accuracy: {results.get('test_accuracy', 0):.2f}%")
