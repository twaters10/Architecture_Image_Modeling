#!/usr/bin/env python3
"""
Target Layer Selection for Grad-CAM.

Provides utilities to automatically select the appropriate convolutional
layer for Grad-CAM visualization based on model architecture.
"""

from typing import Dict, Callable
import torch.nn as nn


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the target convolutional layer for Grad-CAM based on model architecture.

    For best results, Grad-CAM typically uses the last convolutional layer
    before the classification head, as it has:
    - High-level semantic information
    - Reasonable spatial resolution
    - Direct gradient flow from the output

    Args:
        model: The PyTorch model.
        model_name: Model architecture name (e.g., 'resnet18', 'efficientnet_b0').

    Returns:
        nn.Module: The target convolutional layer.

    Raises:
        ValueError: If the model architecture is not supported.

    Example:
        >>> model = get_pretrained_model('resnet18', num_classes=10)
        >>> target_layer = get_target_layer(model, 'resnet18')
        >>> print(type(target_layer))  # <class 'torchvision.models.resnet.BasicBlock'>
    """
    target_layers: Dict[str, Callable[[nn.Module], nn.Module]] = {
        # VanillaCNN: last Conv2d in features Sequential (index 12)
        "vanilla": lambda m: m.features[12],

        # ResNet: last residual block in stage 4 (layer4)
        "resnet18": lambda m: m.layer4[-1],
        "resnet34": lambda m: m.layer4[-1],
        "resnet50": lambda m: m.layer4[-1],
        "resnet101": lambda m: m.layer4[-1],
        "resnet152": lambda m: m.layer4[-1],

        # VGG: last conv layer before classifier
        "vgg11": lambda m: m.features[20],
        "vgg13": lambda m: m.features[24],
        "vgg16": lambda m: m.features[29],
        "vgg19": lambda m: m.features[36],

        # EfficientNet: last feature block (before average pooling)
        "efficientnet_b0": lambda m: m.features[-1],
        "efficientnet_b1": lambda m: m.features[-1],
        "efficientnet_b2": lambda m: m.features[-1],
        "efficientnet_b3": lambda m: m.features[-1],
        "efficientnet_b4": lambda m: m.features[-1],

        # MobileNetV2: last feature block (InvertedResidual)
        "mobilenet_v2": lambda m: m.features[-1],
        "mobilenet_v3_small": lambda m: m.features[-1],
        "mobilenet_v3_large": lambda m: m.features[-1],
    }

    model_name_lower = model_name.lower()

    if model_name_lower not in target_layers:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {list(target_layers.keys())}"
        )

    layer = target_layers[model_name_lower](model)
    return layer


def get_layer_info(layer: nn.Module) -> str:
    """
    Get human-readable information about a layer.

    Args:
        layer: The PyTorch layer.

    Returns:
        str: Description of the layer type and shape information.
    """
    layer_type = type(layer).__name__

    # Try to get output channel info if available
    info = f"{layer_type}"

    if hasattr(layer, 'out_channels'):
        info += f" (out_channels={layer.out_channels})"
    elif hasattr(layer, 'num_features'):
        info += f" (num_features={layer.num_features})"

    return info
