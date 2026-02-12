#!/usr/bin/env python3
"""
Visualization Utilities for Grad-CAM.

Provides tools for creating publication-quality visualizations of Grad-CAM
heatmaps, including overlays, comparison panels, and comprehensive reports.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAMVisualizer:
    """
    Visualization utilities for Grad-CAM explanations.

    Provides methods for creating overlays, comparison panels, and
    comprehensive visualization reports.

    Args:
        class_names: List of class names for labeling predictions.
        cmap: Matplotlib colormap name for heatmaps (default: 'jet').
    """

    def __init__(self, class_names: List[str], cmap: str = "jet"):
        self.class_names = class_names
        self.cmap = cmap

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap on an image.

        Args:
            image: Original image as HWC numpy array in [0, 1].
            heatmap: Grad-CAM heatmap of shape (h, w) in [0, 1].
            alpha: Blending weight for the heatmap (0 = only image, 1 = only heatmap).

        Returns:
            Blended image as HWC numpy array in [0, 1].
        """
        # Resize heatmap to match image dimensions
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]),
                resample=Image.BILINEAR
            )
        ) / 255.0

        # Apply colormap
        colored_heatmap = cm.get_cmap(self.cmap)(heatmap_resized)[:, :, :3]

        # Blend with original image
        blended = (1 - alpha) * image + alpha * colored_heatmap
        return np.clip(blended, 0, 1)

    def create_single_explanation(
        self,
        original_image: Image.Image,
        preprocessed_tensor: np.ndarray,
        heatmap: np.ndarray,
        predicted_class: str,
        confidence: float,
        true_class: Optional[str] = None,
        top_k_predictions: Optional[List[Tuple[str, float]]] = None,
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create a comprehensive single-image Grad-CAM visualization.

        Args:
            original_image: Original PIL Image.
            preprocessed_tensor: Preprocessed image tensor (denormalized) as HWC array.
            heatmap: Grad-CAM heatmap.
            predicted_class: Predicted class name.
            confidence: Prediction confidence (0-1).
            true_class: Ground truth class name (optional).
            top_k_predictions: List of (class_name, probability) tuples.
            output_path: Path to save the figure (optional).

        Returns:
            matplotlib Figure object.
        """
        # Create overlay
        overlay = self.overlay_heatmap(preprocessed_tensor, heatmap, alpha=0.5)

        # Determine layout based on whether we have top-k predictions
        if top_k_predictions:
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: Original image
        axes[0].imshow(original_image)
        title = f"Original Image\nPredicted: {predicted_class} ({confidence:.1%})"
        if true_class:
            match = "✓ Correct" if true_class == predicted_class else "✗ Incorrect"
            title += f"\nTrue: {true_class} ({match})"
        axes[0].set_title(title, fontsize=11, fontweight='bold')
        axes[0].axis("off")

        # Panel 2: Grad-CAM heatmap
        heatmap_display = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (224, 224), resample=Image.BILINEAR
            )
        ) / 255.0
        im = axes[1].imshow(heatmap_display, cmap=self.cmap, vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap\n(Model Attention)", fontsize=11, fontweight='bold')
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM Overlay\n(Highlighted Regions)", fontsize=11, fontweight='bold')
        axes[2].axis("off")

        # Panel 4: Top-K confidence bar chart (if provided)
        if top_k_predictions:
            names, probs = zip(*top_k_predictions)
            names = list(reversed(names))
            probs = list(reversed(probs))
            colors = ["#2ecc71" if n == predicted_class else "#3498db" for n in names]
            axes[3].barh(names, probs, color=colors)
            axes[3].set_xlim(0, 1)
            axes[3].set_xlabel("Confidence", fontsize=10)
            axes[3].set_title("Top-K Predictions", fontsize=11, fontweight='bold')
            for i, p in enumerate(probs):
                axes[3].text(p + 0.01, i, f"{p:.1%}", va="center", fontsize=9)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")

        return fig

    def create_class_comparison(
        self,
        preprocessed_tensor: np.ndarray,
        heatmaps_by_class: Dict[str, Tuple[np.ndarray, float]],
        predicted_class: str,
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create a comparison visualization across multiple classes.

        Shows how the model's attention differs when considering different
        class predictions.

        Args:
            preprocessed_tensor: Preprocessed image (HWC) in [0, 1].
            heatmaps_by_class: Dict mapping class_name to (heatmap, probability).
            predicted_class: The top predicted class.
            output_path: Path to save the figure (optional).

        Returns:
            matplotlib Figure object.
        """
        n_classes = len(heatmaps_by_class)
        fig, axes = plt.subplots(2, n_classes, figsize=(5 * n_classes, 10))

        for col, (class_name, (hmap, prob)) in enumerate(heatmaps_by_class.items()):
            # Top row: heatmap overlay
            overlay = self.overlay_heatmap(preprocessed_tensor, hmap, alpha=0.5)
            axes[0, col].imshow(overlay)
            marker = " (predicted)" if class_name == predicted_class else ""
            axes[0, col].set_title(
                f"{class_name}{marker}\n{prob:.1%}",
                fontsize=10,
                fontweight='bold' if class_name == predicted_class else 'normal'
            )
            axes[0, col].axis("off")

            # Bottom row: raw heatmap
            hmap_resized = np.array(
                Image.fromarray((hmap * 255).astype(np.uint8)).resize(
                    (224, 224), resample=Image.BILINEAR
                )
            ) / 255.0
            axes[1, col].imshow(hmap_resized, cmap=self.cmap, vmin=0, vmax=1)
            axes[1, col].set_title(f"Heatmap: {class_name}", fontsize=10)
            axes[1, col].axis("off")

        plt.suptitle(
            "Grad-CAM Comparison Across Top Predicted Classes",
            fontsize=14,
            fontweight="bold"
        )
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved class comparison to: {output_path}")

        return fig

    def create_batch_grid(
        self,
        images: List[np.ndarray],
        heatmaps: List[np.ndarray],
        predictions: List[str],
        confidences: List[float],
        true_labels: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        max_cols: int = 4
    ) -> plt.Figure:
        """
        Create a grid visualization for multiple images.

        Args:
            images: List of preprocessed images (HWC) in [0, 1].
            heatmaps: List of Grad-CAM heatmaps.
            predictions: List of predicted class names.
            confidences: List of prediction confidences.
            true_labels: List of ground truth labels (optional).
            output_path: Path to save the figure (optional).
            max_cols: Maximum number of columns in the grid.

        Returns:
            matplotlib Figure object.
        """
        n_images = len(images)
        n_cols = min(n_images, max_cols)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        for idx in range(n_images):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Create overlay
            overlay = self.overlay_heatmap(images[idx], heatmaps[idx], alpha=0.5)
            ax.imshow(overlay)

            # Create title
            title = f"{predictions[idx]}\n{confidences[idx]:.1%}"
            if true_labels:
                match = "✓" if true_labels[idx] == predictions[idx] else "✗"
                title = f"{match} {title}\n(True: {true_labels[idx]})"

            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide empty subplots
        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.suptitle("Grad-CAM Batch Visualization", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved batch grid to: {output_path}")

        return fig

    @staticmethod
    def close_figure(fig: plt.Figure):
        """Close a matplotlib figure to free memory."""
        plt.close(fig)
