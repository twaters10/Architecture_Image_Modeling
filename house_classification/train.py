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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Local imports
from utils.config import (
    load_config, get_training_config, get_data_paths,
    get_normalization_config, get_visualization_config, get_model_config,
    get_mlflow_config
)
from utils.data_loaders import get_data_loaders, save_sample_batches, save_training_diagnostics, append_training_runtime
from model import VanillaCNN, get_pretrained_model, count_parameters
from utils.mlflow_training import log_training_run, TrainingMetricsMonitor


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
    device: torch.device,
    epoch: int = 0
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for updating weights.
        device: Device to run training on.
        epoch: Current epoch number (for progress bar display).

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar for batches
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for images, labels in pbar:
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

        # Update progress bar with current metrics
        current_acc = 100.0 * correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0
) -> Tuple[float, float]:
    """
    Validate the model on the validation set.

    Args:
        model: The neural network model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run validation on.
        epoch: Current epoch number (for progress bar display).

    Returns:
        Tuple of (average_loss, accuracy) for the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar for validation batches
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar with current metrics
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

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
) -> Tuple[Dict, float]:
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
        Tuple of (history, total_time) where:
            - history: dict with losses and accuracies per epoch
            - total_time: total training time in seconds
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

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)

    for epoch in epoch_pbar:
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch + 1
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch=epoch + 1)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%',
            'lr': f'{current_lr:.6f}'
        })

        # Check for improvement
        improvement_msg = ""
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
            improvement_msg = f" [New best model saved!]"
        else:
            epochs_without_improvement += 1
            improvement_msg = f" [No improvement: {epochs_without_improvement}/{patience}]"

        # Write detailed epoch info above progress bar
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - "
                   f"Train: {train_loss:.4f}/{train_acc:.2f}% - "
                   f"Val: {val_loss:.4f}/{val_acc:.2f}% - "
                   f"Time: {epoch_time:.1f}s{improvement_msg}")

        # Early stopping
        if epochs_without_improvement >= patience:
            tqdm.write(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    epoch_pbar.close()

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

    return history, total_time


def prompt_mlflow_tracking() -> bool:
    """
    Prompt user whether to enable MLflow tracking.

    Returns:
        bool: True if MLflow should be enabled, False otherwise.
    """
    print("\n" + "=" * 70)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 70)
    print("MLflow tracks all training metrics, parameters, and artifacts.")
    print("You can view results later with: mlflow ui")
    print("=" * 70)
    print("1. Yes - Enable MLflow tracking (recommended)")
    print("2. No - Skip MLflow tracking")
    print("=" * 70)

    while True:
        try:
            choice = input("\nEnable MLflow tracking? (1-2): ").strip()
            if choice == "1":
                print("✓ MLflow tracking enabled")
                return True
            elif choice == "2":
                print("✗ MLflow tracking disabled")
                return False
            else:
                print("Please enter 1 or 2.")
        except (ValueError, KeyboardInterrupt):
            print("\nDefaulting to no MLflow tracking")
            return False


def prompt_model_selection() -> str:
    """
    Prompt user to select a model architecture interactively.

    Returns:
        str: Selected model name.
    """
    models = {
        1: ("vanilla", "Vanilla CNN - Simple baseline model built from scratch"),
        2: ("resnet18", "ResNet-18 - Pretrained, good balance of speed and accuracy"),
        3: ("resnet34", "ResNet-34 - Pretrained, more parameters than ResNet-18"),
        4: ("resnet50", "ResNet-50 - Pretrained, largest ResNet for best accuracy"),
        5: ("efficientnet_b0", "EfficientNet-B0 - Pretrained, efficient architecture"),
        6: ("mobilenet_v2", "MobileNet-V2 - Pretrained, lightweight for deployment")
    }

    print("\n" + "=" * 70)
    print("SELECT MODEL ARCHITECTURE")
    print("=" * 70)
    for num, (name, description) in models.items():
        print(f"{num}. {description}")
    print("=" * 70)

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            choice_num = int(choice)
            if choice_num in models:
                selected_model = models[choice_num][0]
                print(f"Selected: {models[choice_num][1]}")
                return selected_model
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(models)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(models)}.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            exit(0)


def main():
    """Main entry point for training script."""
    # Load config for defaults
    config = load_config()
    train_config = get_training_config()
    model_config = get_model_config()
    paths = get_data_paths()

    parser = argparse.ArgumentParser(
        description="Train CNN for architectural style classification. "
                    "Default values are loaded from conf/training_config.yaml."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["vanilla", "resnet18", "resnet34", "resnet50",
                 "efficientnet_b0", "mobilenet_v2"],
        help="Model architecture to use (will prompt if not specified)"
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
        help="Path to custom config YAML file (default: conf/training_config.yaml)"
    )
    parser.add_argument(
        "--freeze_features", action="store_true",
        default=model_config["freeze_features"],
        help=f"Freeze pretrained layers and only train classifier head (default: {model_config['freeze_features']} from config)"
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Enable MLflow experiment tracking"
    )
    parser.add_argument(
        "--no_mlflow", action="store_true",
        help="Disable MLflow experiment tracking"
    )
    parser.add_argument(
        "--mlflow_tracking_uri", type=str, default=None,
        help="MLflow tracking server URI (default: from config or local ./mlruns)"
    )
    parser.add_argument(
        "--mlflow_experiment", type=str, default=None,
        help="MLflow experiment name (default: from config)"
    )

    args = parser.parse_args()

    # Load MLflow configuration
    mlflow_config = get_mlflow_config(config)

    # Determine if MLflow should be enabled (priority: CLI > interactive > config)
    mlflow_enabled = False
    if args.no_mlflow:
        # Explicitly disabled via CLI
        mlflow_enabled = False
    elif args.mlflow:
        # Explicitly enabled via CLI
        mlflow_enabled = True
    elif mlflow_config.get("enabled", True):
        # Config says enabled, prompt user
        mlflow_enabled = prompt_mlflow_tracking()
    else:
        # Config says disabled
        mlflow_enabled = False

    # Use config defaults if not provided via CLI
    mlflow_tracking_uri = args.mlflow_tracking_uri or mlflow_config.get("tracking_uri")
    mlflow_experiment = args.mlflow_experiment or mlflow_config.get("experiment_name", "architectural-style-training")

    # Prompt for model selection if not provided via CLI
    if args.model is None:
        args.model = prompt_model_selection()

    print("\n" + "=" * 60)
    print("Architectural Style Classification - Training")
    print("=" * 60)
    print("\nConfiguration (from conf/training_config.yaml, overridden by CLI args):")
    print(f"  Model: {args.model}")
    print(f"  Freeze features: {args.freeze_features}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Checkpoint dir: {paths['checkpoints']}")
    print(f"  MLflow tracking: {'enabled' if mlflow_enabled else 'disabled'}")

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
        model = get_pretrained_model(
            args.model,
            num_classes=num_classes,
            freeze_features=args.freeze_features
        )
        if args.freeze_features:
            print("  Feature layers frozen - only training classifier head")

    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Train - use checkpoint path from config (overwrites previous runs with same hyperparameters)
    checkpoint_dir = paths["checkpoints"] / f"{args.model}_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}"

    # Generate sample batch visualizations
    norm_config = get_normalization_config()
    viz_config = get_visualization_config()
    save_sample_batches(
        train_loader=train_loader,
        class_names=class_names,
        output_dir=checkpoint_dir,
        samples_per_class=viz_config.get("samples_per_class", 12),
        norm_mean=norm_config.get("mean", [0.485, 0.456, 0.406]),
        norm_std=norm_config.get("std", [0.229, 0.224, 0.225])
    )

    # Save training diagnostics
    _, _, test_loader_for_diag, _ = get_data_loaders(batch_size=args.batch_size)
    save_training_diagnostics(
        output_dir=checkpoint_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader_for_diag,
        class_names=class_names,
        model_name=args.model,
        model_params=params,
        config=config
    )

    # Train the model with system metrics monitoring (CPU, GPU, memory, etc.)
    system_metrics_summary = None
    with TrainingMetricsMonitor(interval=5.0, enable_gpu=True) as metrics_monitor:
        history, total_time = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            checkpoint_dir=str(checkpoint_dir)
        )
        system_metrics_summary = metrics_monitor.get_summary()

    # Append training runtime to diagnostics file
    append_training_runtime(checkpoint_dir, total_time)

    # Log to MLflow if enabled
    if mlflow_enabled:
        print("\n" + "=" * 60)
        print("Logging to MLflow...")
        print("=" * 60)

        training_config = {
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": train_config.get("weight_decay", 1e-4),
            "epochs": args.epochs,
            "patience": args.patience,
            "freeze_features": args.freeze_features
        }

        log_training_run(
            model_name=args.model,
            model=model,
            history=history,
            config=training_config,
            checkpoint_dir=checkpoint_dir,
            class_names=class_names,
            total_time=total_time,
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment,
            system_metrics_summary=system_metrics_summary
        )

    print(f"\nCheckpoints saved to: {checkpoint_dir}/")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()