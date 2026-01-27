#!/usr/bin/env python3
"""
Training Script for Architectural Style Classification CNN.

This script handles the complete training pipeline including:
- Model initialization (vanilla CNN or pretrained)
- Training loop with validation
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training history logging

Usage:
    # Train vanilla CNN baseline
    python 04_train.py --model vanilla --epochs 50 --batch_size 32

    # Train with transfer learning
    python 04_train.py --model resnet18 --epochs 30 --lr 0.001

    # Resume training from checkpoint
    python 04_train.py --resume checkpoints/best_model.pth

Requirements:
    - torch
    - torchvision
    - tqdm (optional, for progress bars)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local imports
from config import load_config, get_training_config, get_data_paths
from data_loaders import get_data_loaders
from model import VanillaCNN, get_pretrained_model, count_parameters


def get_device() -> torch.device:
    """
    Get the best available device for training.

    Returns:
        torch.device: CUDA if available, MPS for Apple Silicon, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for updating weights.
        device: Device to run training on.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)} - "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model on the validation set.

    Args:
        model: The neural network model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run validation on.

    Returns:
        Tuple of (average_loss, accuracy) for the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_dir: str = "checkpoints",
    device: Optional[torch.device] = None
) -> Dict:
    """
    Complete training loop with validation and early stopping.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Maximum number of epochs to train. Defaults to 50.
        learning_rate: Initial learning rate. Defaults to 0.001.
        weight_decay: L2 regularization strength. Defaults to 1e-4.
        patience: Epochs to wait for improvement before early stopping. Defaults to 10.
        checkpoint_dir: Directory to save model checkpoints. Defaults to "checkpoints".
        device: Device to train on. Defaults to best available.

    Returns:
        dict: Training history with losses and accuracies per epoch.
    """
    if device is None:
        device = get_device()

    print(f"\nTraining on: {device}")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler - reduce on plateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": []
    }

    # Early stopping variables
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path / "best_model.pth")
            print(f"  Saved new best model (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"\nTraining completed in {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, checkpoint_path / "final_model.pth")

    # Save training history
    with open(checkpoint_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def main():
    """Main entry point for training script."""
    # Load config for defaults
    config = load_config()
    train_config = get_training_config()
    paths = get_data_paths()

    parser = argparse.ArgumentParser(
        description="Train CNN for architectural style classification. "
                    "Default values are loaded from conf/data.yaml."
    )
    parser.add_argument(
        "--model", type=str, default=config.get("model", {}).get("default", "vanilla"),
        choices=["vanilla", "resnet18", "resnet34", "resnet50",
                 "efficientnet_b0", "mobilenet_v2"],
        help="Model architecture to use (default from config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=train_config["epochs"],
        help=f"Number of training epochs (default: {train_config['epochs']} from config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.get("data_loader", {}).get("batch_size", 32),
        help="Batch size for training (default from config)"
    )
    parser.add_argument(
        "--lr", type=float, default=train_config["learning_rate"],
        help=f"Learning rate (default: {train_config['learning_rate']} from config)"
    )
    parser.add_argument(
        "--patience", type=int, default=train_config["early_stopping_patience"],
        help=f"Early stopping patience (default: {train_config['early_stopping_patience']} from config)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom config YAML file (default: conf/data.yaml)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Architectural Style Classification - Training")
    print("=" * 60)
    print("\nConfiguration (from conf/data.yaml, overridden by CLI args):")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Checkpoint dir: {paths['checkpoints']}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, _, class_names = get_data_loaders(
        batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == "vanilla":
        model = VanillaCNN(num_classes=num_classes)
    else:
        model = get_pretrained_model(args.model, num_classes=num_classes)

    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Train - use checkpoint path from config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = paths["checkpoints"] / f"{args.model}_{timestamp}"

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=str(checkpoint_dir)
    )

    print(f"\nCheckpoints saved to: {checkpoint_dir}/")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()