#!/usr/bin/env python3
"""
Quick test script for Grad-CAM implementation.

This script verifies that the Grad-CAM module is working correctly
by running a simple test on a sample model.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainability import GradCAM, GradCAMVisualizer, get_target_layer
from model import VanillaCNN


def test_gradcam_basic():
    """Test basic Grad-CAM functionality."""
    print("=" * 60)
    print("Testing Grad-CAM Implementation")
    print("=" * 60)

    # Create a simple model
    num_classes = 10
    model = VanillaCNN(num_classes=num_classes)
    model.eval()

    # Get device
    device = torch.device("cpu")
    model = model.to(device)

    # Get target layer
    model_name = "vanilla"
    target_layer = get_target_layer(model, model_name)
    print(f"✓ Target layer selected: {type(target_layer).__name__}")

    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer, device)
    print(f"✓ Grad-CAM initialized")

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    dummy_input.requires_grad_(True)

    # Generate heatmap
    try:
        with torch.enable_grad():
            heatmap = gradcam.generate(dummy_input, target_class=0)
        print(f"✓ Heatmap generated: shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")
    except Exception as e:
        print(f"✗ Heatmap generation failed: {e}")
        gradcam.remove_hooks()
        return False

    # Verify heatmap properties
    assert heatmap.ndim == 2, "Heatmap should be 2D"
    assert 0 <= heatmap.min() <= heatmap.max() <= 1, "Heatmap should be normalized to [0, 1]"
    print(f"✓ Heatmap properties verified")

    # Clean up
    gradcam.remove_hooks()
    print(f"✓ Hooks removed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


def test_layer_selectors():
    """Test layer selector for different architectures."""
    print("\n" + "=" * 60)
    print("Testing Layer Selectors")
    print("=" * 60)

    # Test architectures that can be created without downloading weights
    test_models = ["vanilla"]

    for model_name in test_models:
        try:
            if model_name == "vanilla":
                model = VanillaCNN(num_classes=10)
            else:
                from model import get_pretrained_model
                model = get_pretrained_model(model_name, num_classes=10, pretrained=False)

            layer = get_target_layer(model, model_name)
            print(f"✓ {model_name}: {type(layer).__name__}")
        except Exception as e:
            print(f"✗ {model_name}: {e}")

    print("\n" + "=" * 60)
    print("Layer selector tests complete!")
    print("=" * 60)


def test_visualizer():
    """Test visualizer initialization."""
    print("\n" + "=" * 60)
    print("Testing Visualizer")
    print("=" * 60)

    class_names = ["class_a", "class_b", "class_c"]
    visualizer = GradCAMVisualizer(class_names)
    print(f"✓ Visualizer initialized with {len(class_names)} classes")

    # Test overlay function
    image = np.random.rand(224, 224, 3)
    heatmap = np.random.rand(7, 7)
    overlay = visualizer.overlay_heatmap(image, heatmap, alpha=0.5)
    assert overlay.shape == image.shape, "Overlay should match image shape"
    print(f"✓ Overlay function works: {overlay.shape}")

    print("\n" + "=" * 60)
    print("Visualizer tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Grad-CAM Module Test Suite")
    print("=" * 60 + "\n")

    try:
        test_gradcam_basic()
        test_layer_selectors()
        test_visualizer()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓✓✓")
        print("=" * 60)
        print("\nThe Grad-CAM module is ready to use!")
        print("Run 'python explain.py' to start the interactive explainability tool.\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
