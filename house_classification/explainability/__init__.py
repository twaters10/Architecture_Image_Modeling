#!/usr/bin/env python3
"""
Explainability Module for Architectural Style Classification.

Production-ready interpretability tools including:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Automated sanity metrics
- Efficient metadata storage
- Visualization utilities
- Robust layer selection (including ViT support)
"""

from .gradcam import GradCAM
from .visualizers import GradCAMVisualizer
from .layer_selectors import (
    get_target_layer,
    get_target_layer_robust,
    auto_detect_model_architecture,
    find_last_conv_layer,
    get_all_conv_layers
)
from .metrics import GradCAMMetrics
from .metadata_storage import (
    ExplainabilityMetadata,
    MetadataStorage,
    extract_heatmap_statistics,
    extract_spatial_distribution,
    extract_top_activations
)

__all__ = [
    # Core Grad-CAM
    "GradCAM",
    "GradCAMVisualizer",

    # Layer selection
    "get_target_layer",
    "get_target_layer_robust",
    "auto_detect_model_architecture",
    "find_last_conv_layer",
    "get_all_conv_layers",

    # Metrics
    "GradCAMMetrics",

    # Metadata storage
    "ExplainabilityMetadata",
    "MetadataStorage",
    "extract_heatmap_statistics",
    "extract_spatial_distribution",
    "extract_top_activations",
]
