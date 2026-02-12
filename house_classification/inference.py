#!/usr/bin/env python3
"""
Inference Pipeline for Architectural Style Classification.

Provides a clean, reusable InferencePipeline class that wraps model loading,
prediction, and Grad-CAM explanation into a single interface.

Usage:
    # As a Python module
    from inference import InferencePipeline

    with InferencePipeline("checkpoints/resnet18_ep50_bs32_lr0.001/best_model.pth") as pipe:
        result = pipe.predict(image)        # predictions only
        result = pipe.explain(image)        # predictions + Grad-CAM

    # CLI
    python inference.py --checkpoint checkpoints/resnet18.../best_model.pth --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from explain import (
    denormalize,
    extract_model_name_from_checkpoint,
    get_class_names_from_data_dir,
    get_device,
    get_inference_transform,
    load_image,
    load_model_from_checkpoint,
)
from explainability import GradCAM, GradCAMVisualizer, get_target_layer


class InferencePipeline:
    """
    End-to-end inference pipeline for architectural style classification.

    Bundles model loading, preprocessing, prediction, and Grad-CAM
    explainability into a single reusable object.

    Args:
        checkpoint_path: Path to a .pth checkpoint file.
        device: Device to run on. If None, auto-selects best available.

    Example:
        >>> with InferencePipeline("checkpoints/resnet18.../best_model.pth") as pipe:
        ...     result = pipe.explain(Image.open("house.jpg"))
        ...     print(result["predicted_class"], result["confidence"])
    """

    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None):
        self.checkpoint_path = str(checkpoint_path)
        self.device = device or get_device()

        # Load class names
        self.class_names = get_class_names_from_data_dir()
        self.num_classes = len(self.class_names)

        # Load model
        self.model, self.model_name, self.checkpoint_info = load_model_from_checkpoint(
            self.checkpoint_path,
            num_classes=self.num_classes,
            device=self.device,
        )

        # Set up Grad-CAM
        self.target_layer = get_target_layer(self.model, self.model_name)
        self.gradcam = GradCAM(self.model, self.target_layer, self.device)
        self.visualizer = GradCAMVisualizer(self.class_names)

        # Preprocessing transform
        self.transform = get_inference_transform()

    def predict(self, image: Image.Image, top_k: int = 5) -> Dict:
        """
        Classify an image without generating Grad-CAM.

        Args:
            image: PIL RGB Image.
            top_k: Number of top predictions to return.

        Returns:
            Dict with keys:
                predicted_class (str): Top-1 class name.
                confidence (float): Top-1 probability.
                top_k (list[tuple[str, float]]): Top-K (class, prob) pairs.
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1).squeeze(0)

        top_k_probs, top_k_indices = probabilities.topk(min(top_k, self.num_classes))
        top_k_predictions = [
            (self.class_names[idx.item()], prob.item())
            for idx, prob in zip(top_k_indices, top_k_probs)
        ]

        return {
            "predicted_class": top_k_predictions[0][0],
            "confidence": top_k_predictions[0][1],
            "top_k": top_k_predictions,
        }

    def explain(self, image: Image.Image, top_k: int = 5) -> Dict:
        """
        Classify an image and generate a Grad-CAM explanation.

        Args:
            image: PIL RGB Image.
            top_k: Number of top predictions to return.

        Returns:
            Dict with keys:
                predicted_class (str): Top-1 class name.
                confidence (float): Top-1 probability.
                top_k (list[tuple[str, float]]): Top-K (class, prob) pairs.
                heatmap (np.ndarray): Raw Grad-CAM heatmap (h, w) in [0, 1].
                overlay (np.ndarray): Blended HWC numpy image in [0, 1].
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        with torch.enable_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1).squeeze(0)

        top_k_probs, top_k_indices = probabilities.topk(min(top_k, self.num_classes))
        top_k_predictions = [
            (self.class_names[idx.item()], prob.item())
            for idx, prob in zip(top_k_indices, top_k_probs)
        ]

        predicted_idx = top_k_indices[0].item()

        # Generate Grad-CAM heatmap for the predicted class
        heatmap = self.gradcam.generate(input_tensor, target_class=predicted_idx)

        # Create overlay on the preprocessed image
        preprocessed_image = denormalize(input_tensor.squeeze(0))
        overlay = self.visualizer.overlay_heatmap(preprocessed_image, heatmap, alpha=0.5)

        return {
            "predicted_class": top_k_predictions[0][0],
            "confidence": top_k_predictions[0][1],
            "top_k": top_k_predictions,
            "heatmap": heatmap,
            "overlay": overlay,
        }

    def close(self):
        """Remove Grad-CAM hooks to free memory."""
        self.gradcam.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single image with optional Grad-CAM output."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predictions to display (default: 5)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save Grad-CAM overlay PNG (default: same dir as image)"
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    with InferencePipeline(str(checkpoint_path)) as pipe:
        image = load_image(str(image_path))
        result = pipe.explain(image, top_k=args.top_k)

        print(f"\nPredicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nTop-{args.top_k} predictions:")
        for rank, (name, prob) in enumerate(result["top_k"], 1):
            print(f"  {rank}. {name}: {prob:.1%}")

        # Save overlay
        output_dir = Path(args.output_dir) if args.output_dir else image_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_gradcam.png"

        overlay_uint8 = (result["overlay"] * 255).astype(np.uint8)
        Image.fromarray(overlay_uint8).save(output_path)
        print(f"\nGrad-CAM overlay saved to: {output_path}")


if __name__ == "__main__":
    main()
