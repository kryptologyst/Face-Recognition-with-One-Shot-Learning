"""Main training script for face recognition with one-shot learning."""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_default_config, load_config
from src.utils.device import get_device, set_seed
from src.models.siamese import SiameseNetwork, ContrastiveLoss
from src.models.clip_models import CLIPSiameseNetwork
from src.data.dataset import create_data_loaders, OneShotDataset
from src.train.trainer import Trainer


def create_model(config: DictConfig) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration.
        
    Returns:
        Created model.
    """
    model_name = config.model.name.lower()
    
    if model_name == "siamese_resnet":
        model = SiameseNetwork(
            backbone=config.model.backbone,
            embedding_dim=config.model.embedding_dim,
            dropout=config.model.dropout,
            pretrained=config.model.pretrained,
        )
        model.criterion = ContrastiveLoss()
        
    elif model_name == "clip_siamese":
        model = CLIPSiameseNetwork()
        model.criterion = ContrastiveLoss()
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.device = args.device
    
    # Set device
    if config.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.device)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    print(f"Configuration:")
    print(f"  Data directory: {config.data.data_dir}")
    print(f"  Model: {config.model.name}")
    print(f"  Device: {device}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        target_size=tuple(config.data.input_size),
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    print(f"Test metrics: {test_metrics}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
