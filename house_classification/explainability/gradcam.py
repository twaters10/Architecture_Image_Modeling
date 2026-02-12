#!/usr/bin/env python3
"""
Grad-CAM: Gradient-weighted Class Activation Mapping.

Production-ready implementation for generating visual explanations of
CNN predictions by highlighting important regions in input images.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (2017)
"""

from typing import Optional, List, Tuple
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
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate the Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class: Class index to generate the heatmap for.
                         If None, uses the predicted (argmax) class.

        Returns:
            np.ndarray: Heatmap of shape (h, w) with values in [0, 1].

        Raises:
            RuntimeError: If no gradients are captured (check model architecture).
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
            raise RuntimeError(
                "No gradients captured. Ensure the target layer is part of "
                "the forward pass and the model supports gradient computation."
            )

        # Global average pool the gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3)).squeeze(0)  # (C,)
        activations = self.activations.squeeze(0)  # (C, H, W)

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

        return cam

    def generate_batch(
        self,
        input_tensors: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM heatmaps for a batch of images.

        Args:
            input_tensors: Batch of preprocessed image tensors (B, 3, H, W).
            target_classes: List of target class indices for each image.
                           If None, uses predicted classes.

        Returns:
            List of heatmaps, each of shape (h, w) with values in [0, 1].
        """
        if target_classes is None:
            target_classes = [None] * input_tensors.size(0)

        heatmaps = []
        for i, (img_tensor, target_class) in enumerate(zip(input_tensors, target_classes)):
            heatmap = self.generate(
                img_tensor.unsqueeze(0),
                target_class=target_class
            )
            heatmaps.append(heatmap)

        return heatmaps

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
