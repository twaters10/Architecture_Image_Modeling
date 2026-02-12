#!/usr/bin/env python3
"""
MLflow Integration Utilities for Model Explainability.

Provides functions to log Grad-CAM visualizations, metrics, and artifacts
to MLflow for experiment tracking and model interpretability documentation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Install with: pip install mlflow")


class MLflowLogger:
    """
    MLflow logger for Grad-CAM explainability experiments.

    Handles logging of visualizations, metrics, parameters, and artifacts
    to MLflow tracking server.
    """

    def __init__(
        self,
        experiment_name: str = "architectural-style-explainability",
        tracking_uri: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of MLflow experiment.
            tracking_uri: MLflow tracking server URI (None for local).
            enable_logging: Whether to enable MLflow logging.
        """
        self.enabled = enable_logging and MLFLOW_AVAILABLE

        if not self.enabled:
            if enable_logging and not MLFLOW_AVAILABLE:
                print("MLflow logging disabled: MLflow not installed")
            return

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> Optional[Any]:
        """
        Start an MLflow run.

        Args:
            run_name: Name for the run.
            tags: Dictionary of tags to add to the run.
            nested: Whether this is a nested run.

        Returns:
            MLflow run object if enabled, None otherwise.
        """
        if not self.enabled:
            return None

        run = mlflow.start_run(run_name=run_name, nested=nested)

        if tags:
            mlflow.set_tags(tags)

        return run

    def end_run(self):
        """End the current MLflow run."""
        if self.enabled:
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log.
        """
        if not self.enabled:
            return

        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"Warning: Failed to log param {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log.
            step: Step number for the metrics.
        """
        if not self.enabled:
            return

        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                print(f"Warning: Failed to log metric {key}: {e}")

    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None):
        """
        Log an artifact file to MLflow.

        Args:
            local_path: Path to the local file to log.
            artifact_path: Subdirectory in artifacts to store the file.
        """
        if not self.enabled:
            return

        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        except Exception as e:
            print(f"Warning: Failed to log artifact {local_path}: {e}")

    def log_artifacts(self, local_dir: Path, artifact_path: Optional[str] = None):
        """
        Log an entire directory of artifacts to MLflow.

        Args:
            local_dir: Path to the local directory to log.
            artifact_path: Subdirectory in artifacts to store the files.
        """
        if not self.enabled:
            return

        try:
            mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
        except Exception as e:
            print(f"Warning: Failed to log artifacts from {local_dir}: {e}")

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure object.
            artifact_file: Name for the artifact file.
        """
        if not self.enabled:
            return

        try:
            mlflow.log_figure(figure, artifact_file)
        except Exception as e:
            print(f"Warning: Failed to log figure {artifact_file}: {e}")

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log.
            artifact_file: Name for the JSON file.
        """
        if not self.enabled:
            return

        try:
            mlflow.log_dict(dictionary, artifact_file)
        except Exception as e:
            print(f"Warning: Failed to log dict to {artifact_file}: {e}")

    def log_model(self, model, artifact_path: str = "model"):
        """
        Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model to log.
            artifact_path: Subdirectory in artifacts for the model.
        """
        if not self.enabled:
            return

        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            print(f"Warning: Failed to log model: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags on the current run.

        Args:
            tags: Dictionary of tags to set.
        """
        if not self.enabled:
            return

        try:
            mlflow.set_tags(tags)
        except Exception as e:
            print(f"Warning: Failed to set tags: {e}")


def log_single_explanation(
    logger: MLflowLogger,
    image_name: str,
    predicted_class: str,
    true_class: Optional[str],
    confidence: float,
    visualization_path: Path,
    comparison_path: Optional[Path] = None,
    top_k_predictions: Optional[List[tuple]] = None
):
    """
    Log a single image explanation to MLflow.

    Args:
        logger: MLflowLogger instance.
        image_name: Name of the analyzed image.
        predicted_class: Predicted class.
        true_class: True class (if known).
        confidence: Prediction confidence.
        visualization_path: Path to the main visualization.
        comparison_path: Path to the class comparison visualization.
        top_k_predictions: List of (class, probability) tuples.
    """
    if not logger.enabled:
        return

    # Log metrics
    metrics = {
        "confidence": confidence,
        "correct": 1.0 if (true_class and predicted_class == true_class) else 0.0
    }
    logger.log_metrics(metrics)

    # Log parameters
    params = {
        "image_name": image_name,
        "predicted_class": predicted_class,
        "analysis_mode": "single"
    }
    if true_class:
        params["true_class"] = true_class
    logger.log_params(params)

    # Log visualizations
    logger.log_artifact(visualization_path, artifact_path="gradcam_visualizations")
    if comparison_path and comparison_path.exists():
        logger.log_artifact(comparison_path, artifact_path="class_comparisons")

    # Log top-k predictions as JSON
    if top_k_predictions:
        top_k_dict = {
            class_name: float(prob)
            for class_name, prob in top_k_predictions
        }
        logger.log_dict(top_k_dict, "top_k_predictions.json")


def log_misclassification_analysis(
    logger: MLflowLogger,
    total_misclassifications: int,
    analyzed_count: int,
    misclassification_pairs: Dict[str, int],
    output_dir: Path
):
    """
    Log misclassification analysis results to MLflow.

    Args:
        logger: MLflowLogger instance.
        total_misclassifications: Total number of misclassifications found.
        analyzed_count: Number of misclassifications analyzed.
        misclassification_pairs: Dict mapping "true->pred" to count.
        output_dir: Directory containing the visualization artifacts.
    """
    if not logger.enabled:
        return

    # Log metrics
    metrics = {
        "total_misclassifications": float(total_misclassifications),
        "analyzed_misclassifications": float(analyzed_count),
        "misclassification_rate": float(total_misclassifications) / max(analyzed_count, 1)
    }
    logger.log_metrics(metrics)

    # Log parameters
    params = {
        "analysis_mode": "misclassifications",
        "limit": analyzed_count
    }
    logger.log_params(params)

    # Log confusion pairs
    logger.log_dict(misclassification_pairs, "misclassification_pairs.json")

    # Log all visualizations
    if output_dir.exists():
        logger.log_artifacts(output_dir, artifact_path="misclassifications")


def log_batch_analysis(
    logger: MLflowLogger,
    image_count: int,
    class_distribution: Dict[str, int],
    avg_confidence: float,
    output_dir: Path,
    true_class: Optional[str] = None
):
    """
    Log batch analysis results to MLflow.

    Args:
        logger: MLflowLogger instance.
        image_count: Number of images analyzed.
        class_distribution: Dict mapping predicted class to count.
        avg_confidence: Average prediction confidence.
        output_dir: Directory containing the visualization artifacts.
        true_class: True class if all images are from same class.
    """
    if not logger.enabled:
        return

    # Log metrics
    metrics = {
        "image_count": float(image_count),
        "avg_confidence": avg_confidence,
        "num_unique_predictions": float(len(class_distribution))
    }
    logger.log_metrics(metrics)

    # Log parameters
    params = {
        "analysis_mode": "batch",
        "batch_size": image_count
    }
    if true_class:
        params["true_class"] = true_class
        # Calculate accuracy if we know true class
        correct = class_distribution.get(true_class, 0)
        accuracy = correct / max(image_count, 1)
        metrics["batch_accuracy"] = accuracy

    logger.log_params(params)
    logger.log_metrics(metrics)

    # Log class distribution
    logger.log_dict(class_distribution, "class_distribution.json")

    # Log sample visualizations (not all to avoid clutter)
    if output_dir.exists():
        # Log first 10 visualizations
        viz_files = sorted(list(output_dir.glob("*.png")))[:10]
        for viz_file in viz_files:
            logger.log_artifact(viz_file, artifact_path="batch_samples")


def create_explainability_summary(
    model_name: str,
    checkpoint_path: Path,
    analysis_mode: str,
    metrics: Dict[str, float],
    output_dir: Path
) -> Dict:
    """
    Create a summary dictionary for the explainability run.

    Args:
        model_name: Name of the model architecture.
        checkpoint_path: Path to the model checkpoint.
        analysis_mode: Type of analysis performed.
        metrics: Dictionary of computed metrics.
        output_dir: Directory where outputs were saved.

    Returns:
        Dictionary with summary information.
    """
    summary = {
        "model_architecture": model_name,
        "checkpoint": str(checkpoint_path),
        "analysis_mode": analysis_mode,
        "output_directory": str(output_dir),
        "metrics": metrics
    }
    return summary
