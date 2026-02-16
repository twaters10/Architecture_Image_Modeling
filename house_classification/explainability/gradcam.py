#!/usr/bin/env python3
"""
Grad-CAM: Gradient-weighted Class Activation Mapping.

Production-ready implementation for generating visual explanations of
CNN predictions by highlighting important regions in input images.

Enhanced for production inference pipelines with:
- Memory-efficient batch processing
- Automated sanity metrics
- Zero gradient handling
- Metadata extraction

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (2017)
"""

from typing import Optional, List, Tuple, Dict
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on a target convolutional layer to
    capture activations and gradients, then computes the gradient-weighted
    activation map.

    Args:
        model: Trained PyTorch model in eval mode.
        target_layer: The convolutional layer (nn.Module) to visualize.
        device: Device to run computations on.

    Example:
        >>> model = load_model(checkpoint_path)
        >>> target_layer = model.layer4[-1]  # For ResNet
        >>> gradcam = GradCAM(model, target_layer, device)
        >>> heatmap = gradcam.generate(input_tensor, target_class=0)
        >>> gradcam.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device
    ):
        self.model = model.to(device)
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

        # Set model to eval mode
        self.model.eval()

    def _save_activation(self, module, input, output):
        """Forward hook: capture the output activations of the target layer."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: capture the gradients flowing back through the target layer."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        handle_zero_gradients: bool = True
    ) -> np.ndarray:
        """
        Generate the Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class: Class index to generate the heatmap for.
                         If None, uses the predicted (argmax) class.
            handle_zero_gradients: If True, return empty heatmap on zero gradients
                                  instead of raising error (default: True).

        Returns:
            np.ndarray: Heatmap of shape (h, w) with values in [0, 1].

        Raises:
            RuntimeError: If no gradients are captured and handle_zero_gradients=False.
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        # Forward pass
        with torch.enable_grad():
            output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backpropagate the target class score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        if self.gradients is None:
            if handle_zero_gradients:
                warnings.warn(
                    "No gradients captured. Returning empty heatmap. "
                    "This may indicate dead neurons or incorrect target layer.",
                    UserWarning
                )
                # Return empty heatmap with same spatial dims as activations
                if self.activations is not None:
                    h, w = self.activations.shape[-2:]
                    return np.zeros((h, w), dtype=np.float32)
                else:
                    return np.zeros((7, 7), dtype=np.float32)  # Default size
            else:
                raise RuntimeError(
                    "No gradients captured. Ensure the target layer is part of "
                    "the forward pass and the model supports gradient computation."
                )

        # Global average pool the gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3)).squeeze(0)  # (C,)
        activations = self.activations.squeeze(0)  # (C, H, W)

        # Check for all-zero weights (dead neurons)
        if torch.all(weights == 0):
            if handle_zero_gradients:
                warnings.warn(
                    "All gradient weights are zero (dead neurons). "
                    "Returning empty heatmap.",
                    UserWarning
                )
                return np.zeros(activations.shape[1:], dtype=np.float32)
            else:
                raise RuntimeError("All gradient weights are zero (dead neurons).")

        # Weighted combination of activation maps
        cam = torch.zeros(
            activations.shape[1:],
            dtype=activations.dtype,
            device=activations.device
        )
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU - only keep positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            # All-zero CAM after ReLU (all negative contributions)
            if handle_zero_gradients:
                warnings.warn(
                    "CAM is all zeros after ReLU (all negative contributions). "
                    "This may indicate the model is not using the target layer for this prediction.",
                    UserWarning
                )

        return cam

    def generate_batch(
        self,
        input_tensors: torch.Tensor,
        target_classes: Optional[List[int]] = None,
        micro_batch_size: int = 8,
        clear_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM heatmaps for a batch of images with memory management.

        Processes large batches in smaller micro-batches to avoid OOM errors.
        Automatically clears CUDA cache between micro-batches.

        Args:
            input_tensors: Batch of preprocessed image tensors (B, 3, H, W).
            target_classes: List of target class indices for each image.
                           If None, uses predicted classes.
            micro_batch_size: Size of micro-batches for processing (default: 8).
                            Reduce this if encountering OOM errors.
            clear_cache: Whether to clear CUDA cache between micro-batches (default: True).

        Returns:
            List of heatmaps, each of shape (h, w) with values in [0, 1].

        Example:
            >>> # Process 100 images without OOM
            >>> gradcam = GradCAM(model, target_layer, device)
            >>> heatmaps = gradcam.generate_batch(
            ...     input_tensors,  # (100, 3, 224, 224)
            ...     micro_batch_size=8  # Process 8 at a time
            ... )
        """
        if target_classes is None:
            target_classes = [None] * input_tensors.size(0)

        batch_size = input_tensors.size(0)
        heatmaps = []

        # Process in micro-batches to avoid OOM
        for start_idx in range(0, batch_size, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, batch_size)

            # Process micro-batch
            for i in range(start_idx, end_idx):
                img_tensor = input_tensors[i].unsqueeze(0)
                target_class = target_classes[i]

                heatmap = self.generate(
                    img_tensor,
                    target_class=target_class
                )
                heatmaps.append(heatmap)

                # Clear intermediate tensors
                del img_tensor

            # Clear CUDA cache between micro-batches
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return heatmaps

    def generate_with_metrics(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        compute_metrics: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate Grad-CAM heatmap with automated sanity metrics.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class: Class index to generate the heatmap for.
            compute_metrics: Whether to compute sanity metrics (default: True).

        Returns:
            Tuple of (heatmap, metrics_dict).
            If compute_metrics=False, metrics_dict will be empty.

        Example:
            >>> heatmap, metrics = gradcam.generate_with_metrics(input_tensor, target_class=5)
            >>> if metrics['confidence_drop_pct'] < 10:
            >>>     print("Warning: Unreliable explanation")
        """
        # Generate heatmap
        heatmap = self.generate(input_tensor, target_class)

        metrics = {}

        if compute_metrics:
            # Import here to avoid circular dependency
            from .metrics import GradCAMMetrics

            # Determine target class if not provided
            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor.to(self.device))
                    target_class = output.argmax(dim=1).item()

            # Compute all metrics
            metrics = GradCAMMetrics.compute_all_metrics(
                self.model,
                input_tensor,
                heatmap,
                target_class,
                device=self.device
            )

        return heatmap, metrics

    def remove_hooks(self):
        """Remove the registered hooks to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        """Ensure hooks are removed when the object is destroyed."""
        try:
            self.remove_hooks()
        except (AttributeError, RuntimeError):
            # Hooks may already be removed or not initialized
            pass
