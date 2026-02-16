#!/usr/bin/env python3
"""
Target Layer Selection for Grad-CAM.

Provides utilities to automatically select the appropriate convolutional
layer for Grad-CAM visualization based on model architecture.

Enhanced for production with:
- Vision Transformer (ViT) support
- Automatic architecture detection
- Robust fallback mechanisms
"""

from typing import Dict, Callable, Optional, List, Tuple
import warnings
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
        "efficientnet_b5": lambda m: m.features[-1],
        "efficientnet_b6": lambda m: m.features[-1],
        "efficientnet_b7": lambda m: m.features[-1],

        # MobileNetV2/V3: last feature block (InvertedResidual)
        "mobilenet_v2": lambda m: m.features[-1],
        "mobilenet_v3_small": lambda m: m.features[-1],
        "mobilenet_v3_large": lambda m: m.features[-1],

        # Vision Transformer (ViT): last encoder block
        # Note: Grad-CAM for ViT requires special handling - use attention rollout instead
        # These are provided for compatibility but may not give optimal results
        "vit_b_16": lambda m: m.encoder.layers[-1].ln_1,  # Layer norm before last attention
        "vit_b_32": lambda m: m.encoder.layers[-1].ln_1,
        "vit_l_16": lambda m: m.encoder.layers[-1].ln_1,
        "vit_l_32": lambda m: m.encoder.layers[-1].ln_1,

        # Swin Transformer: last stage
        "swin_t": lambda m: m.features[-1][-1],  # Last block of last stage
        "swin_s": lambda m: m.features[-1][-1],
        "swin_b": lambda m: m.features[-1][-1],

        # DenseNet: last dense block
        "densenet121": lambda m: m.features.denseblock4,
        "densenet161": lambda m: m.features.denseblock4,
        "densenet169": lambda m: m.features.denseblock4,
        "densenet201": lambda m: m.features.denseblock4,
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


def auto_detect_model_architecture(model: nn.Module) -> Optional[str]:
    """
    Automatically detect model architecture by inspecting module structure.

    Args:
        model: PyTorch model.

    Returns:
        Detected architecture name (lowercase) or None if unknown.

    Example:
        >>> model = torchvision.models.resnet50()
        >>> arch = auto_detect_model_architecture(model)
        >>> print(arch)  # "resnet50"
    """
    model_class = type(model).__name__.lower()

    # Check for common architecture patterns
    architecture_patterns = {
        'resnet': 'resnet',
        'efficientnet': 'efficientnet',
        'mobilenet': 'mobilenet',
        'vgg': 'vgg',
        'densenet': 'densenet',
        'inception': 'inception',
        'vit': 'vit',  # Vision Transformer
        'swin': 'swin',  # Swin Transformer
        'vanilla': 'vanilla',
    }

    for pattern, arch_name in architecture_patterns.items():
        if pattern in model_class:
            return arch_name

    # Check module names as fallback
    module_names = [name for name, _ in model.named_modules()]
    for pattern, arch_name in architecture_patterns.items():
        if any(pattern in name.lower() for name in module_names):
            return arch_name

    warnings.warn(
        f"Could not auto-detect architecture for {model_class}. "
        "Please specify model_name explicitly or use find_last_conv_layer().",
        UserWarning
    )
    return None


def find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Find the last convolutional/attention layer in the model automatically.

    Useful for unknown architectures. Searches for the last:
    - Conv2d layer (for CNNs)
    - Attention/MultiheadAttention layer (for Transformers)

    Args:
        model: PyTorch model.

    Returns:
        Last conv/attention layer or None if not found.

    Example:
        >>> model = load_custom_model()
        >>> target_layer = find_last_conv_layer(model)
        >>> if target_layer is None:
        >>>     raise ValueError("No suitable layer found")
    """
    last_conv = None
    last_attention = None

    for name, module in model.named_modules():
        # Check for convolutional layers
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            last_conv = module

        # Check for transformer attention layers
        if isinstance(module, nn.MultiheadAttention):
            last_attention = module

        # Check for ViT-specific layers
        if 'attention' in name.lower() or 'attn' in name.lower():
            if hasattr(module, 'proj') or hasattr(module, 'qkv'):
                last_attention = module

    # Prefer conv layers for CNNs, attention for Transformers
    if last_attention is not None:
        return last_attention
    elif last_conv is not None:
        return last_conv
    else:
        warnings.warn(
            "Could not find any Conv2d or Attention layers. "
            "Grad-CAM may not work with this model.",
            UserWarning
        )
        return None


def get_target_layer_robust(
    model: nn.Module,
    model_name: Optional[str] = None,
    fallback_to_auto: bool = True
) -> nn.Module:
    """
    Robust layer selection with automatic detection and fallback.

    Recommended for production pipelines where model architecture may vary.

    Args:
        model: PyTorch model.
        model_name: Model architecture name (optional).
                   If None, will attempt auto-detection.
        fallback_to_auto: If True, use automatic layer finding if model_name
                         lookup fails (default: True).

    Returns:
        Target layer for Grad-CAM.

    Raises:
        ValueError: If no suitable layer can be found.

    Example:
        >>> # Model-agnostic pipeline
        >>> target_layer = get_target_layer_robust(model)
        >>> gradcam = GradCAM(model, target_layer, device)
    """
    # Try explicit model_name first
    if model_name is not None:
        try:
            return get_target_layer(model, model_name)
        except ValueError as e:
            if not fallback_to_auto:
                raise
            warnings.warn(
                f"Failed to get layer for '{model_name}': {e}. "
                "Attempting automatic detection.",
                UserWarning
            )

    # Auto-detect architecture
    detected_arch = auto_detect_model_architecture(model)
    if detected_arch is not None:
        try:
            return get_target_layer(model, detected_arch)
        except ValueError:
            pass  # Fall through to automatic layer finding

    # Final fallback: automatic layer finding
    if fallback_to_auto:
        layer = find_last_conv_layer(model)
        if layer is not None:
            warnings.warn(
                f"Using automatically detected layer: {type(layer).__name__}",
                UserWarning
            )
            return layer

    raise ValueError(
        "Could not find suitable target layer. "
        "Please specify model_name or ensure model has Conv2d layers."
    )


def get_all_conv_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Get all convolutional layers in the model.

    Useful for debugging or experimenting with different target layers.

    Args:
        model: PyTorch model.

    Returns:
        List of (name, layer) tuples.

    Example:
        >>> layers = get_all_conv_layers(model)
        >>> for name, layer in layers:
        >>>     print(f"{name}: {type(layer).__name__}")
    """
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            conv_layers.append((name, module))
    return conv_layers
