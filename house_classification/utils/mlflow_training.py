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
    from .system_metrics import SystemMetricsMonitor, get_system_info
    SYSTEM_METRICS_AVAILABLE = True
except ImportError:
    SYSTEM_METRICS_AVAILABLE = False


def log_training_run(
    model_name: str,
    model: Any,
    history: Dict,
    config: Dict,
    checkpoint_dir: Path,
    class_names: List[str],
    total_time: float,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "architectural-style-training",
    system_metrics_summary: Optional[Dict] = None
):
    """
    Log a complete training run to MLflow.

    Args:
        model_name: Name of the model architecture.
        model: Trained PyTorch model.
        history: Training history dict with losses and accuracies.
        config: Training configuration dict.
        checkpoint_dir: Path to checkpoint directory.
        class_names: List of class names.
        total_time: Total training time in seconds.
        tracking_uri: MLflow tracking server URI.
        experiment_name: Name of MLflow experiment.
        system_metrics_summary: Optional dict of system metrics (avg/max CPU, GPU, memory, etc.)
            collected during training via TrainingMetricsMonitor.
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Skipping logging.")
        return

    # Set tracking URI and experiment
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_training"):
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

        # Log metrics per epoch
        train_losses = history.get("train_loss", [])
        train_accs = history.get("train_acc", [])
        val_losses = history.get("val_loss", [])
        val_accs = history.get("val_acc", [])
        learning_rates = history.get("learning_rates", [])

        for epoch in range(len(train_losses)):
            metrics = {
                "train_loss": train_losses[epoch],
                "train_accuracy": train_accs[epoch],
            }
            if epoch < len(val_losses):
                metrics["val_loss"] = val_losses[epoch]
                metrics["val_accuracy"] = val_accs[epoch]
            if epoch < len(learning_rates):
                metrics["learning_rate"] = learning_rates[epoch]

            mlflow.log_metrics(metrics, step=epoch)

        # Log final metrics
        final_metrics = {
            "final_train_loss": train_losses[-1] if train_losses else 0,
            "final_train_accuracy": train_accs[-1] if train_accs else 0,
            "best_val_loss": min(val_losses) if val_losses else 0,
            "best_val_accuracy": max(val_accs) if val_accs else 0,
            "training_time_seconds": total_time,
            "training_time_minutes": total_time / 60,
            "total_epochs": len(train_losses)
        }
        mlflow.log_metrics(final_metrics)

        # Log system metrics collected during training
        if system_metrics_summary:
            system_metrics_to_log = {}
            for key, value in system_metrics_summary.items():
                if isinstance(value, (int, float)):
                    system_metrics_to_log[f"system/{key}"] = value
            if system_metrics_to_log:
                mlflow.log_metrics(system_metrics_to_log)
                print(f"  Logged {len(system_metrics_to_log)} system metrics (CPU, GPU, memory, etc.)")

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

        print(f"\n✓ Training run logged to MLflow experiment: {experiment_name}")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")


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


class TrainingMetricsMonitor:
    """
    Context manager for monitoring system metrics during training.

    Example:
        with TrainingMetricsMonitor() as monitor:
            # Training code here
            pass

        # Get metrics summary after training
        metrics = monitor.get_summary()
    """

    def __init__(self, interval: float = 5.0, enable_gpu: bool = True):
        """
        Initialize the training metrics monitor.

        Args:
            interval: Time in seconds between metric collection (default: 5.0)
            enable_gpu: Whether to monitor GPU metrics (default: True)
        """
        self.interval = interval
        self.enable_gpu = enable_gpu
        self.monitor = None
        self.metrics_summary = None

    def __enter__(self):
        """Start monitoring."""
        if SYSTEM_METRICS_AVAILABLE:
            try:
                self.monitor = SystemMetricsMonitor(
                    interval=self.interval,
                    enable_gpu=self.enable_gpu
                )
                self.monitor.start()
            except Exception as e:
                print(f"Warning: Could not start system metrics monitoring: {e}")
                self.monitor = None
        else:
            print("⚠️  System metrics monitoring not available (install psutil and pynvml)")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and collect summary."""
        if self.monitor:
            try:
                self.monitor.stop()
                self.metrics_summary = self.monitor.get_summary_metrics()
            except Exception as e:
                print(f"Warning: Error collecting metrics summary: {e}")
                self.metrics_summary = None

    def get_summary(self) -> Optional[Dict]:
        """
        Get the summary metrics.

        Returns:
            Dictionary of summary metrics, or None if monitoring failed
        """
        return self.metrics_summary

    def get_current(self) -> Optional[Dict]:
        """
        Get current metrics snapshot.

        Returns:
            Dictionary of current metrics, or None if monitoring not active
        """
        if self.monitor:
            return self.monitor.get_current_metrics()
        return None


def log_hyperparameter_tuning(
    model_name: str,
    search_method: str,
    search_space: Dict,
    results: List[Dict],
    best_config: Dict,
    output_dir: Path,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "architectural-style-tuning"
):
    """
    Log hyperparameter tuning results to MLflow.

    Args:
        model_name: Name of the model architecture.
        search_method: Search method (grid_search, bayesian, genetic).
        search_space: Search space configuration.
        results: List of trial results.
        best_config: Best configuration found.
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

    # Create parent run for the tuning session
    with mlflow.start_run(run_name=f"{model_name}_{search_method}_tuning"):
        # Log parent run parameters
        parent_params = {
            "model_architecture": model_name,
            "search_method": search_method,
            "n_trials": len(results),
            "search_space": str(search_space)
        }
        mlflow.log_params(parent_params)

        # Log best configuration
        for key, value in best_config.items():
            mlflow.log_param(f"best_{key}", value)

        # Log summary metrics
        val_accs = [r.get("val_acc", 0) for r in results]
        mlflow.log_metrics({
            "best_val_accuracy": max(val_accs),
            "mean_val_accuracy": np.mean(val_accs),
            "std_val_accuracy": np.std(val_accs),
            "total_trials": len(results)
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
                    "val_accuracy": trial.get("val_acc", 0),
                    "val_loss": trial.get("val_loss", 0),
                    "best_epoch": trial.get("best_epoch", 0)
                }
                mlflow.log_metrics(trial_metrics)

        # Log artifacts from output directory
        if output_dir.exists():
            # Log result files
            results_file = output_dir / "tuning_results.json"
            if results_file.exists():
                mlflow.log_artifact(str(results_file))

            diagnostics_file = output_dir / "diagnostics_summary.txt"
            if diagnostics_file.exists():
                mlflow.log_artifact(str(diagnostics_file))

            # Log visualizations
            for viz_file in output_dir.glob("*.png"):
                mlflow.log_artifact(str(viz_file), artifact_path="visualizations")

        print(f"\n✓ Hyperparameter tuning logged to MLflow")
        print(f"  Experiment: {experiment_name}")
        print(f"  Total trials: {len(results)}")
        print(f"  Parent run ID: {mlflow.active_run().info.run_id}")


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
