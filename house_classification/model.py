#!/usr/bin/env python3
"""
CNN Model Architectures for Architectural Style Classification.

This module provides neural network architectures for classifying architectural
styles from images. Includes both a vanilla CNN baseline and options for
transfer learning with pretrained models.

Models Available:
    1. VanillaCNN: Simple CNN built from scratch (baseline)
    2. get_pretrained_model: ResNet, VGG, EfficientNet with transfer learning

Usage:
    from model import VanillaCNN, get_pretrained_model

    # Vanilla CNN baseline
    model = VanillaCNN(num_classes=10)

    # Transfer learning with ResNet
    model = get_pretrained_model('resnet18', num_classes=10, pretrained=True)

Requirements:
    - torch
    - torchvision
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class VanillaCNN(nn.Module):
    """
    A vanilla CNN architecture for image classification baseline.

    Architecture:
        - 4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
        - Global average pooling
        - Fully connected classifier with dropout

    This provides a simple baseline to compare against more sophisticated
    architectures and transfer learning approaches.

    Args:
        num_classes: Number of output classes (architectural styles).
        input_channels: Number of input image channels. Defaults to 3 (RGB).
        dropout_rate: Dropout probability in classifier. Defaults to 0.5.

    Input:
        Tensor of shape (batch_size, 3, 224, 224)

    Output:
        Tensor of shape (batch_size, num_classes) - raw logits (no softmax)
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        dropout_rate: float = 0.5
    ):
        super(VanillaCNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 14x14x256 -> 7x7x512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling: 7x7x512 -> 1x1x512
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def get_pretrained_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_features: bool = False
) -> nn.Module:
    """
    Get a pretrained model for transfer learning.

    Loads a pretrained model from torchvision and replaces the final
    classification layer to match the number of architectural style classes.

    Args:
        model_name: Name of the model architecture. Options:
            - 'resnet18', 'resnet34', 'resnet50'
            - 'vgg16', 'vgg19'
            - 'efficientnet_b0', 'efficientnet_b1'
            - 'mobilenet_v2'
        num_classes: Number of output classes (architectural styles).
        pretrained: Whether to load pretrained ImageNet weights. Defaults to True.
        freeze_features: Whether to freeze the feature extractor layers.
            If True, only the classifier will be trained. Defaults to False.

    Returns:
        nn.Module: The model with modified classifier for the target task.

    Raises:
        ValueError: If model_name is not recognized.
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=weights)
        model.classifier[-1] = nn.Linear(4096, num_classes)

    elif model_name == "vgg19":
        model = models.vgg19(weights=weights)
        model.classifier[-1] = nn.Linear(4096, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Supported: resnet18, resnet34, resnet50, vgg16, vgg19, "
            "efficientnet_b0, efficientnet_b1, mobilenet_v2"
        )

    # Freeze feature extractor if requested
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        dict: Dictionary with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen
    }


if __name__ == "__main__":
    print("=" * 50)
    print("Model Architecture Test")
    print("=" * 50)

    num_classes = 10  # Number of architectural styles

    # Test Vanilla CNN
    print("\n1. Vanilla CNN (Baseline):")
    model = VanillaCNN(num_classes=num_classes)
    params = count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")

"""
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")

    # Test pretrained models
    print("\n2. Pretrained Models (Transfer Learning):")
    for model_name in ["resnet18", "efficientnet_b0", "mobilenet_v2"]:
        model = get_pretrained_model(model_name, num_classes=num_classes)
        params = count_parameters(model)
        print(f"\n   {model_name}:")
        print(f"   Total parameters: {params['total']:,}")
        print(f"   Trainable: {params['trainable']:,}")

    # Test with frozen features
    print("\n3. ResNet18 with Frozen Features:")
    model = get_pretrained_model("resnet18", num_classes=num_classes, freeze_features=True)
    params = count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    print(f"   Frozen: {params['frozen']:,}")

    print("\nAll models ready!")
    
"""